"""Bounded instrumentation for one authentic FlexGEMM import worker.

No Torch/Triton imports occur when this module is loaded. Native ELF
constructors are outside these Python guards and require host containment.
A policy violation terminates the attempt; callers must not catch and retry.
"""
import importlib
import os
from pathlib import Path
import sys
import threading


NATIVE_BINDINGS = (
    "hashmap_insert", "hashmap_lookup", "hashmap_insert_3d", "hashmap_lookup_3d",
    "hashmap_insert_3d_idx_as_val", "z_order_encode", "z_order_decode",
    "hilbert_encode", "hilbert_decode",
    "hashmap_build_grid_sample_3d_nearest_neighbor_map",
    "hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight",
    "hashmap_build_submanifold_conv_neighbour_map", "hashmap_build_sparse_conv_out_coords",
    "expand_unique_build_sparse_conv_out_coords", "hashmap_build_sparse_conv_neighbour_map",
    "neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd",
    "neighbor_map_post_process_for_masked_implicit_gemm_1",
    "neighbor_map_post_process_for_masked_implicit_gemm_2",
)


class ImportPolicyError(RuntimeError):
    pass


class ImportPolicy:
    def __init__(self, scratch_root, writable_roots=()):
        self.writable_roots = tuple(Path(root).resolve() for root in (scratch_root, *writable_roots))
        self.phase = "native_import"
        self.blocked = {}
        self.tensor_operations_allowed = 0
        self._patches = []
        self._tensor_patches = []
        self._audit_installed = False
        self._profile_installed = False

    def reject(self, boundary, detail):
        self.blocked[boundary] = self.blocked.get(boundary, 0) + 1
        raise ImportPolicyError(boundary + ": " + str(detail))

    def snapshot(self):
        return {"phase": self.phase, "blocked": dict(self.blocked),
                "tensor_operations_allowed": self.tensor_operations_allowed,
                "native_calls": self.blocked.get("native_function", 0),
                "rejected_calls": sum(self.blocked.values())}

    def set_phase(self, phase):
        if phase == "import":
            phase = "native_import"
        if phase not in ("native_import", "cpu_meta"):
            raise ValueError("Only native_import and cpu_meta phases are admitted")
        self.phase = phase

    @staticmethod
    def _leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from ImportPolicy._leaves(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from ImportPolicy._leaves(item)
        else:
            yield value

    def _check_device(self, device):
        if device is not None:
            kind = getattr(device, "type", str(device).split(":", 1)[0])
            if kind not in ("cpu", "meta"):
                self.reject("tensor_operation", "Unadmitted device " + str(device))

    def dispatch(self, function, args, kwargs, torch):
        kwargs = kwargs or {}
        if self.phase == "native_import":
            self.reject("tensor_operation", function)
        for value in self._leaves((args, kwargs)):
            if isinstance(value, torch.Tensor):
                self._check_device(value.device)
        self._check_device(kwargs.get("device"))
        for index, argument in enumerate(getattr(getattr(function, "_schema", None), "arguments", ())):
            if argument.name == "device":
                device = kwargs.get("device", args[index] if index < len(args) else None)
                # Dispatcher factory defaults are CPU. An explicit device is checked
                # before the operation; GPU tensor inputs are independently rejected.
                self._check_device(device)
        result = function(*args, **kwargs)
        for value in self._leaves(result):
            if isinstance(value, torch.Tensor):
                self._check_device(value.device)
        self.tensor_operations_allowed += 1
        return result

    def tensor_mode(self, torch, dispatch_type=None):
        if dispatch_type is None:
            dispatch_type = importlib.import_module("torch.utils._python_dispatch").TorchDispatchMode
        policy = self
        class GuardedTensorDispatch(dispatch_type):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return policy.dispatch(function, args, kwargs, torch)
        return GuardedTensorDispatch()

    def install_tensor_guards(self, torch):
        """Check frontend device requests before Torch can call _lazy_init."""
        if self._tensor_patches:
            raise RuntimeError("Tensor frontend guards already installed")
        device_context = sys.modules.get("torch.utils._device")
        if device_context is None and hasattr(torch, "__version__"):
            device_context = importlib.import_module("torch.utils._device")
        if device_context is not None:
            # DeviceContext compares function identities. Populate its existing
            # cache before wrappers replace module attributes, preserving the
            # genuine original factory's implicit meta/CPU device routing.
            device_context._device_constructors()
        factory_names = ("empty", "empty_strided", "zeros", "ones", "full", "tensor", "as_tensor", "asarray",
                         "from_numpy", "arange", "rand", "randn", "randint", "eye", "linspace", "logspace",
                         "randperm", "scalar_tensor", "empty_like", "zeros_like", "ones_like", "full_like",
                         "rand_like", "randn_like", "randint_like")
        methods = ("cuda", "to", "new_empty", "new_empty_strided", "new_tensor", "new_zeros", "new_ones", "new_full")
        targets = [(torch, name, False) for name in factory_names if callable(getattr(torch, name, None))]
        targets += [(torch.Tensor, name, True) for name in methods if callable(getattr(torch.Tensor, name, None))]
        for owner, name, method in targets:
            original = getattr(owner, name)
            def guarded(*args, _name=name, _original=original, _method=method, **kwargs):
                if self.phase == "native_import":
                    self.reject("tensor_operation", "Frontend " + _name)
                if _name == "cuda":
                    self.reject("tensor_operation", "Tensor.cuda")
                for value in self._leaves((args, kwargs)):
                    if isinstance(value, torch.Tensor):
                        self._check_device(value.device)
                device = kwargs.get("device")
                if _name == "to" and len(args) > 1:
                    requested = args[1]
                    if isinstance(requested, (str, int)) or hasattr(requested, "type"):
                        device = requested
                    elif isinstance(requested, torch.Tensor):
                        device = requested.device
                self._check_device(device)
                if not _method and device is None:
                    context_device = getattr(device_context, "CURRENT_DEVICE", None)
                    if context_device is not None:
                        self._check_device(context_device)
                    else:
                        default_device = getattr(getattr(torch, "_C", None), "_get_default_device", None)
                        if not callable(default_device):
                            self.reject("tensor_operation", "Cannot attest default factory device")
                        self._check_device(default_device())
                return _original(*args, **kwargs)
            self._tensor_patches.append((owner, name, original))
            setattr(owner, name, guarded)

    def restore_tensor_guards(self):
        for owner, name, original in reversed(self._tensor_patches):
            setattr(owner, name, original)
        self._tensor_patches.clear()

    def profile(self, frame, event, argument):
        if event == "c_call":
            module = getattr(argument, "__module__", "") or ""
            if module == "flex_gemm.kernels.cuda":
                self.reject("native_function", getattr(argument, "__name__", "unknown"))
        elif event == "call":
            module, name = frame.f_globals.get("__name__", "") or "", frame.f_code.co_name
            if module == "flex_gemm.utils.autotuner" and name in (
                    "__call__", "run", "warmup", "_benchmark", "_bench",
                    "load_autotune_cache", "save_autotune_cache", "get_autotune_cache"):
                self.reject("autotuner_execution", name)
            if ((module.startswith("flex_gemm.ops") and name not in ("<module>", "__init__", "set_algorithm", "set_hashmap_ratio")
                 and (name in ("forward", "backward", "sparse_conv3d", "sparse_submanifold_conv3d", "grid_sample_3d")
                      or name.startswith("_compute_neighbor") or name.startswith("_sparse_")))
                    or (module.startswith("trellis2.") and name in ("forward", "decode", "decode_latent"))
                    or (module == "torch.nn.modules.module" and name == "_call_impl")):
                self.reject("forward_execution", module + "." + name)

    def install_profile(self):
        if self._profile_installed:
            raise RuntimeError("Profile guard already installed")
        self._old_profile, self._old_thread_profile = sys.getprofile(), threading.getprofile()
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)
        self._profile_installed = True

    def restore_profile(self):
        if self._profile_installed:
            sys.setprofile(self._old_profile)
            threading.setprofile(self._old_thread_profile)
            self._profile_installed = False

    def _write_path(self, path, dir_fd=None):
        if dir_fd not in (None, -1):
            self.reject("filesystem_write", "Relative directory descriptor is not admitted")
        if isinstance(path, int):
            try:
                path = os.readlink("/proc/self/fd/" + str(path))
            except OSError:
                self.reject("filesystem_write", "Unresolved file descriptor")
        resolved = Path(os.fsdecode(path)).resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in self.writable_roots):
            self.reject("filesystem_write", resolved)

    def audit(self, event, args):
        if event.startswith("subprocess.") or event in ("os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.posix_spawnp", "os.fork", "os.forkpty", "pty.spawn"):
            self.reject("process_execution", event)
        if event.startswith("socket."):
            self.reject("network", event)
        if event == "open":
            path, mode, flags = args
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if (isinstance(mode, str) and any(char in mode for char in "wax+")) or (flags and flags & write_flags):
                self._write_path(path)
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime"):
            # CPython supplies dir_fd last; chmod/chown/utime may also include
            # additional scalar arguments. Reject relative dir_fd explicitly.
            dir_fd = args[-1] if len(args) > 1 and isinstance(args[-1], int) else None
            if event == "os.mkdir":
                dir_fd = args[2] if len(args) > 2 else None
            self._write_path(args[0], dir_fd)
        elif event in ("os.rename", "os.link", "os.symlink"):
            source_fd = args[2] if event != "os.symlink" and len(args) > 2 else None
            destination_fd = args[3] if len(args) > 3 else (args[2] if event == "os.symlink" and len(args) > 2 else None)
            self._write_path(args[0], source_fd)
            self._write_path(args[1], destination_fd)

    def install_audit_hook(self):
        if self._audit_installed:
            raise RuntimeError("Audit guard already installed")
        sys.addaudithook(self.audit)
        self._audit_installed = True

    def install_runtime_guards(self, triton):
        if self._patches:
            raise RuntimeError("Runtime guards already installed")
        driver_module = importlib.import_module("triton.runtime.driver")
        if driver_module.driver._active is not None or driver_module.driver._default is not None:
            self.reject("runtime_guard", "Triton driver was already activated")
        build = importlib.import_module("triton.runtime.build")
        amd = importlib.import_module("triton.backends.amd.driver")
        compiler = importlib.import_module("triton.compiler.compiler")
        targets = [(triton, "compile"), (triton.compiler, "compile"), (compiler, "compile"),
                   (driver_module, "_create_driver"), (build, "compile_module_from_src"),
                   (build, "_build"), (amd, "compile_module_from_src"),
                   (triton.runtime.JITFunction, "run"), (triton.runtime.JITFunction, "warmup"),
                   (triton.runtime.Autotuner, "run"), (triton.runtime.Autotuner, "warmup"),
                   (triton.runtime.Autotuner, "_bench"), (triton.testing, "do_bench"),
                   (triton.testing, "do_bench_cudagraph")]
        originals = []
        for owner, name in targets:
            original = getattr(owner, name, None)
            if not callable(original):
                self.reject("runtime_guard", "Missing execution boundary " + name)
            originals.append((owner, name, original))
        for owner, name, original in originals:
            def deny(*args, _name=name, **kwargs):
                self.reject("triton_execution", _name)
            self._patches.append((owner, name, original))
            setattr(owner, name, deny)

    def restore_runtime_guards(self):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()
