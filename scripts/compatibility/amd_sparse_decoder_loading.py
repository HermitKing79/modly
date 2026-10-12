"""Single-shot offline sparse-decoder CPU load, ROCm placement and release.

No pipeline/model factory, additional checkpoints, module forward calls or
compute kernels. The parent authorizes Gate 2 only after Gate 1 and observer PASS.
"""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_decoder_contract as contract
import amd_bounded_gpu_proof as gpu_proof
import amd_model_metadata as model_metadata

SOURCE = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD")
VENV = Path("D:/AI/Trellis2AMD/venv")
PREFIX = Path("D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
ARCHIVED = ROOT / "docs/compatibility/amd-offline-model-audit-evidence/final-assets/ss_dec_conv3d_16l8_fp16.metadata.json"
EVENT = "DECODER_EVENT="
RESULT = "DECODER_RESULT="


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    config = model_metadata.json_metadata(PREFIX.with_suffix(".json"))
    current = model_metadata.safetensors_metadata(PREFIX.with_suffix(".safetensors"))
    source = SOURCE / "trellis2/models/sparse_structure_vae.py"
    current.update(config=config, source_module=str(source), source_module_sha256=digest(source))
    archived = json.loads(ARCHIVED.read_text())
    contract.validate_identity(archived, current)
    current["checkpoint_sha256"] = digest(PREFIX.with_suffix(".safetensors"))
    current["config_sha256"] = digest(PREFIX.with_suffix(".json"))
    return current


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_bytes": counters.WorkingSetSize, "private_commit_bytes": counters.PrivateUsage,
            "peak_working_set_bytes": counters.PeakWorkingSetSize, "peak_commit_bytes": counters.PeakPagefileUsage}


def tensor_manifest(state):
    dtypes = {"torch.float16": "F16", "torch.float32": "F32"}
    return {key: {"shape": list(value.shape), "dtype": dtypes.get(str(value.dtype), str(value.dtype)),
                  "device": str(value.device), "is_meta": value.is_meta} for key, value in state.items()}


def verify_payload(model, verified):
    result = {}
    with PREFIX.with_suffix(".safetensors").open("rb") as stream:
        for key, value in model.state_dict().items():
            descriptor = verified["tensors"][key]
            begin, end = descriptor["data_offsets"]
            stream.seek(8 + verified["header_bytes"] + begin)
            raw = stream.read(end - begin)
            expected = hashlib.sha256(raw).hexdigest()
            actual = hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
            if len(raw) != end - begin or expected != actual:
                raise ValueError("Loaded CPU parameter bytes differ from checkpoint: " + key)
            result[key] = {"sha256": actual, "bytes": end - begin}
    return result


def gpu_memory(torch):
    free, total = torch.cuda.mem_get_info(0)
    return {"allocated_bytes": torch.cuda.memory_allocated(0), "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "available_bytes": free, "total_bytes": total, "sampled_global_used_bytes": total - free}


def worker(scratch):
    record = {"status": "FAIL", "stage": "startup", "gate_one": {"status": "NOT ATTEMPTED"},
              "gate_two": {"status": "NOT ATTEMPTED"}, "forward_calls_prohibited": True, "sdk_discovery": []}
    torch = model = reference = None
    allow_gpu = False
    def stage(name):
        record["stage"] = name
        print("DECODER_STAGE=" + name, flush=True)
    try:
        scratch = scratch.resolve()
        scratch.relative_to(ROOT)
        env = gpu_proof.scratch_environment(scratch)
        for key in gpu_proof.CACHE_KEYS:
            if key != "FLEX_GEMM_AUTOTUNE_CACHE_PATH":
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        audit = gpu_proof.preflight.load_audit()
        record["sdk_verification"] = audit.block_external_actions(
            sdk_executable=VENV / "Scripts/rocm-sdk.exe", discovery_records=record["sdk_discovery"], scratch_dir=scratch)
        os.environ.update(env)
        checkpoint = PREFIX.with_suffix(".safetensors").resolve()
        def access_guard(event, args):
            if event.startswith("socket."):
                raise RuntimeError("Decoder proof prohibits network: " + event)
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0]))
                if path.suffix.lower() in (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf") and path.resolve() != checkpoint:
                    raise RuntimeError("Decoder proof prohibits any other model file: " + str(path))
        sys.addaudithook(access_guard)
        record["identity"] = verified = identity()
        import torch
        from services import amd_runtime
        Path(sys.executable).resolve().relative_to(VENV)
        Path(torch.__file__).resolve().relative_to(VENV / "Lib/site-packages")
        if (sys.version.split()[0] != "3.12.10" or torch.__version__ != "2.13.0+rocm10.0.0"
                or not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized()):
            raise RuntimeError("Unexpected AMD runtime or existing GPU context")
        record.update(python=sys.version, executable=sys.executable, torch=torch.__version__, torch_path=torch.__file__,
                      hip=torch.version.hip, cuda=torch.version.cuda, gpu_initialized_start=False,
                      source_identity=amd_runtime.source_identity(SOURCE))
        record["source_identity"].pop("upstream_imported", None)
        torch.set_num_threads(1)
        original_init = torch.cuda._lazy_init
        def forbidden(*args, **kwargs):
            raise RuntimeError("Decoder proof prohibits model forward/inference, pickle loading and compilation")
        def forbid_init(*args, **kwargs):
            raise RuntimeError("Gate 1 prohibits GPU initialization")
        torch.cuda._lazy_init = forbid_init
        torch.load = forbidden
        torch.nn.Module.__call__ = forbidden
        from torch.utils._python_dispatch import TorchDispatchMode
        class PlacementOnly(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                def contains_gpu(value):
                    if isinstance(value, torch.Tensor):
                        return value.device.type == "cuda"
                    if isinstance(value, torch.device):
                        return value.type == "cuda"
                    if isinstance(value, (list, tuple)):
                        return any(contains_gpu(item) for item in value)
                    if isinstance(value, dict):
                        return any(contains_gpu(item) for item in value.values())
                    return isinstance(value, str) and value.startswith("cuda")
                on_gpu = contains_gpu(args) or contains_gpu(kwargs)
                if on_gpu and (not allow_gpu or not contract.gpu_operation_allowed(str(func))):
                    raise RuntimeError("Decoder proof forbids GPU computation: " + str(func))
                return func(*args, **kwargs)
        with PlacementOnly():
            stage("real-decoder-import")
            sys.path.insert(0, str(SOURCE))
            cls = importlib.import_module("trellis2.models.sparse_structure_vae").SparseStructureDecoder
            if Path(inspect.getfile(cls)).resolve() != (SOURCE / "trellis2/models/sparse_structure_vae.py").resolve():
                raise RuntimeError("Decoder class provenance mismatch")
            record["class"] = {"name": cls.__name__, "module": cls.__module__, "path": inspect.getfile(cls)}
            stage("CPU-load")
            memory_before = process_memory()
            started = time.monotonic()
            with torch.device("meta"):
                model = cls(**verified["config"]["args"])
            contract.validate_tensors(verified["tensors"], tensor_manifest(model.state_dict()), "meta")
            import safetensors.torch
            record["dependencies"] = {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "numpy")}
            record["imported_source"] = {name: {"path": module.__file__, "sha256": digest(Path(module.__file__))}
                for name, module in sorted(sys.modules.items()) if name.startswith("trellis2")
                and getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"}
            original_load = safetensors.torch.load_file
            def only_decoder(path, device="cpu", **kwargs):
                if Path(path).resolve() != checkpoint or device != "cpu" or kwargs:
                    raise RuntimeError("Only the exact decoder may be loaded, on CPU, once")
                if record.get("load_calls", 0):
                    raise RuntimeError("Automatic checkpoint load retries are prohibited")
                record["load_calls"] = 1
                return original_load(str(checkpoint), device="cpu")
            safetensors.torch.load_file = only_decoder
            raw_state = only_decoder(checkpoint)
            contract.validate_tensors(verified["tensors"], tensor_manifest(raw_state), "cpu")
            state = {key: value.clone() for key, value in raw_state.items()}
            del raw_state
            incompatible = model.load_state_dict(state, strict=True, assign=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("Strict decoder load returned incompatible keys")
            del state
            model.eval().requires_grad_(False)
            model.forward = forbidden
            actual = tensor_manifest(model.state_dict())
            contract.validate_tensors(verified["tensors"], actual, "cpu")
            payload_hashes = verify_payload(model, verified)
            if torch.cuda.is_initialized():
                raise RuntimeError("Gate 1 unexpectedly initialized GPU")
            record["gate_one"] = {"status": "PASS", "elapsed_seconds": time.monotonic() - started,
                                  "memory_before": memory_before, "memory_after": process_memory(),
                                  "tensors": actual, "payload_hashes": payload_hashes,
                                  "missing_keys": [], "unexpected_keys": [], "strict": True, "assign": True,
                                  "parameter_bytes": sum(row["bytes"] for row in payload_hashes.values()),
                                  "gpu_initialized": False}
            print(EVENT + json.dumps({"event": "gate-one-ready", "gate_one": record["gate_one"],
                                     "checkpoint_sha256": verified["checkpoint_sha256"]}), flush=True)
            stage("waiting-for-observer")
            if sys.stdin.readline().strip() != "PLACE_DECODER_ON_AMD":
                raise RuntimeError("Gate 2 not authorized by the healthy parent observer")
            stage("GPU-metadata")
            allow_gpu = True
            torch.cuda._lazy_init = original_init
            properties = torch.cuda.get_device_properties(0)
            if "RX 9070 XT" not in properties.name or getattr(properties, "gcnArchName", "").split(":")[0] != "gfx1201":
                raise RuntimeError("Unexpected AMD GPU identity")
            record["gpu"] = {"name": properties.name, "architecture": properties.gcnArchName,
                             "total_memory_bytes": properties.total_memory}
            torch.cuda.reset_peak_memory_stats(0)
            before = gpu_memory(torch)
            if before["allocated_bytes"] or before["reserved_bytes"] or before["available_bytes"] < verified["tensor_data_bytes"] + 256 * 1024**2:
                raise RuntimeError("Unexpected baseline allocations or insufficient available VRAM")
            record["gate_two"] = {"status": "STARTED", "before_placement": before}
            stage("CPU-to-AMD-placement")
            started = time.monotonic()
            model.to(device="cuda:0")
            torch.cuda.synchronize(0)
            contract.validate_tensors(verified["tensors"], tensor_manifest(model.state_dict()), "cuda:0")
            placed = gpu_memory(torch)
            if placed["allocated_bytes"] < verified["tensor_data_bytes"]:
                raise RuntimeError("Tracked placed tensor bytes are smaller than the decoder payload")
            record["gate_two"].update(placement_seconds=time.monotonic() - started, after_placement=placed, status="PASS")
            stage("release")
            reference = weakref.ref(model)
            model = None
            gc.collect()
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            record["release"] = {**gpu_memory(torch), "model_collected": reference() is None, "cpu_memory": process_memory()}
            contract.validate_release(record["release"])
            stage("complete")
            record["status"] = "PASS"
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        model = None
        gc.collect()
        if torch is not None:
            record["gpu_initialized_final"] = torch.cuda.is_initialized()
        print(RESULT + json.dumps(record), flush=True)
    return 0 if record["status"] == "PASS" else 1


def reap_owned_child(child, process):
    """Always verify exit; on an error kill only the owned process tree, then reap."""
    if child.poll() is None:
        process["forced_cleanup"] = True
        try:
            process["tree_cleanup"] = gpu_proof.preflight.run_child(
                [str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"), "/PID", str(child.pid), "/T", "/F"], 10)
        except BaseException as exc:
            process["tree_cleanup_error"] = repr(exc)
        if child.poll() is None:
            try:
                child.kill()
            except BaseException as exc:
                process["kill_error"] = repr(exc)
    try:
        child.wait(timeout=5)
        process.update(returncode=child.returncode, worker_exit_verified=True)
    except BaseException as exc:
        process.update(cleanup_unverified=True, worker_exit_verified=False, reap_error=repr(exc))


def supervise(output):
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise RuntimeError("Single controlled attempt only; existing run evidence must never be overwritten")
    output.mkdir(parents=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    verified = None
    healthy = False
    record = {"status": "NOT ATTEMPTED", "error": "Observer unhealthy before launch"}
    process = {"returncode": None, "timed_out": False}
    try:
        verified = identity()
        save("checkpoint-before.json", verified)
        before = gpu_proof.windows_observation(start)
        save("windows-before.json", before)
        healthy = gpu_proof.healthy_observation(before)
    except BaseException as exc:
        process["error"] = "Pre-launch verification: " + repr(exc)
    if healthy:
        scratch = output / "scratch"
        scratch.mkdir()
        command = [str(VENV / "Scripts/python.exe"), "-B", "-I", str(Path(__file__).resolve()), "--worker", "--output", str(scratch)]
        child = None
        stdout, stderr, messages = [], [], queue.Queue()
        def read_stdout():
            for line in iter(child.stdout.readline, b""):
                stdout.append(line)
                messages.put(line.decode("utf-8", errors="replace").strip())
            messages.put(None)
        def read_stderr():
            stderr.append(child.stderr.read())
        reader = threading.Thread(target=read_stdout, daemon=True)
        err_reader = threading.Thread(target=read_stderr, daemon=True)
        began = time.monotonic()
        deadline = began + 45
        authorized = False
        final_seen = False
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process["pid"] = child.pid
            reader.start()
            err_reader.start()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Decoder worker stage deadline expired")
                line = messages.get(timeout=remaining)
                if line is None:
                    break
                if line.startswith(EVENT):
                    event = json.loads(line[len(EVENT):])
                    if authorized or event.get("event") != "gate-one-ready":
                        raise RuntimeError("Unexpected or duplicate Gate 1 worker event")
                    contract.authorize_placement(event.get("gate_one", {}), verified, verified["checkpoint_sha256"], event.get("checkpoint_sha256"))
                    save("gate-one.json", event)
                    checkpoint_observer = gpu_proof.windows_observation(start)
                    save("windows-gate-one.json", checkpoint_observer)
                    healthy = healthy and gpu_proof.healthy_observation(checkpoint_observer)
                    child.stdin.write(b"PLACE_DECODER_ON_AMD\n" if healthy else b"STOP\n")
                    child.stdin.flush()
                    child.stdin.close()
                    authorized = healthy
                    deadline = min(began + 90, time.monotonic() + 30)
                if line.startswith(RESULT):
                    if final_seen:
                        raise RuntimeError("Duplicate final worker record")
                    final_seen = True
                    record = json.loads(line[len(RESULT):])
            child.wait(timeout=max(0.1, min(5, began + 90 - time.monotonic())))
            process.update(returncode=child.returncode, elapsed_seconds=time.monotonic() - began,
                           gpu_authorized=authorized, stage_timeouts_seconds={"cpu": 45, "placement_release": 30, "overall": 90})
        except BaseException as exc:
            process.update(timed_out=isinstance(exc, (TimeoutError, queue.Empty, subprocess.TimeoutExpired)), error=repr(exc))
        finally:
            if child is not None:
                reap_owned_child(child, process)
                if child.stdin is not None and not child.stdin.closed:
                    try:
                        child.stdin.close()
                    except OSError:
                        pass
            if reader.ident is not None:
                reader.join(timeout=5)
            if err_reader.ident is not None:
                err_reader.join(timeout=5)
            process["output_readers_completed"] = not reader.is_alive() and not err_reader.is_alive()
            (output / "worker.stdout.txt").write_bytes(b"".join(stdout))
            (output / "worker.stderr.txt").write_bytes(b"".join(stderr))
            process["command"] = command
    preserved = False
    try:
        after_observer = gpu_proof.windows_observation(start)
        save("windows-after-exit.json", after_observer)
        healthy = healthy and gpu_proof.healthy_observation(after_observer)
        after = identity()
        save("checkpoint-after.json", after)
        preserved = verified == after
    except BaseException as exc:
        process["postcheck_error"] = repr(exc)
        healthy = False
    result = {"classification": contract.classify(record, process["returncode"], process["timed_out"],
              healthy and not process.get("cleanup_unverified", False) and process.get("output_readers_completed", False)
              and not process.get("forced_cleanup", False) and "error" not in process
              and process.get("worker_exit_verified") is True and process.get("gpu_authorized") is True, preserved),
              "observer_healthy": healthy, "checkpoint_unchanged": preserved, "worker": record, "process": process}
    save("results.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("worker", "process")}), flush=True)
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(worker(args.output) if args.worker else supervise(args.output))
