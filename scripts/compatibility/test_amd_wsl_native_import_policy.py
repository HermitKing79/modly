"""CPU-only policy contracts; no Torch, Triton, native import, or WSL."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("amd_wsl_native_import_policy.py")


def temporary_root():
    return tempfile.TemporaryDirectory(dir=SCRIPT.parents[2])


def fake_function(module, name, body="return 7"):
    namespace = {"__name__": module}
    exec(compile("def " + name + "():\n    " + body, "policy-fixture.py", "exec"), namespace)
    return namespace[name]


class FakeDevice:
    def __init__(self,kind):self.type=kind


class FakeTensor:
    def __init__(self, device):
        self.device = types.SimpleNamespace(type=device)


class PolicyContracts(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), "Missing bounded native-import policy")
        spec = importlib.util.spec_from_file_location("native_import_policy", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def policy(self, module, root):
        return module.ImportPolicy(root)

    def test_import_phase_rejects_operation_before_callback(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            calls = []
            with self.assertRaises(module.ImportPolicyError):
                policy.dispatch(lambda *a, **k: calls.append(1), (), {}, types.SimpleNamespace(Tensor=FakeTensor))
            self.assertEqual(calls, [])
            self.assertEqual(policy.snapshot()["blocked"]["tensor_operation"], 1)

    def test_cpu_meta_phase_allows_inputs_and_rejects_gpu_before_callback(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            policy.set_phase("cpu_meta")
            calls = []
            operation = lambda *a, **k: calls.append(1) or FakeTensor("meta")
            torch = types.SimpleNamespace(Tensor=FakeTensor)
            self.assertEqual(policy.dispatch(operation, ([FakeTensor("cpu"), {"x": FakeTensor("meta")}],), {}, torch).device.type, "meta")
            for args, kwargs in (((FakeTensor("cuda"),), {}), ((), {"device": "cuda:0"}), ((), {"device": "hip"})):
                with self.assertRaises(module.ImportPolicyError):
                    policy.dispatch(operation, args, kwargs, torch)
            self.assertEqual(calls, [1])

    def test_positional_schema_device_is_rejected_before_callback(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            policy.set_phase("cpu_meta")
            calls = []
            def operation(*args, **kwargs):
                calls.append(1)
            operation._schema = types.SimpleNamespace(arguments=[types.SimpleNamespace(name="size"), types.SimpleNamespace(name="device")])
            with self.assertRaises(module.ImportPolicyError):
                policy.dispatch(operation, ([2], "cuda"), {}, types.SimpleNamespace(Tensor=FakeTensor))
            self.assertEqual(calls, [])

    def test_unknown_phase_is_refused(self):
        module = self.load()
        with temporary_root() as root:
            with self.assertRaises(ValueError):
                self.policy(module, root).set_phase("gpu")

    def test_frontend_device_guard_prevents_initialization_before_dispatch(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            self.assertTrue(hasattr(policy, "install_tensor_guards"), "Missing pre-dispatch device guard")
            calls = []
            def factory(*args, **kwargs):
                calls.append(kwargs.get("device", "cpu"))
                return FakeTensor(kwargs.get("device", "cpu"))
            torch = types.SimpleNamespace(Tensor=FakeTensor, device=FakeDevice, empty=factory,
                                          _C=types.SimpleNamespace(_get_default_device=lambda: "cpu"))
            class DeviceContext: pass
            torch.overrides=types.SimpleNamespace(_get_current_function_mode_stack=lambda:[])
            context=types.SimpleNamespace(DeviceContext=DeviceContext,_device_constructors=lambda:{factory})
            with patch.dict(sys.modules,{"torch.utils._device":context}):
                policy.install_tensor_guards(torch)
            try:
                with self.assertRaises(module.ImportPolicyError):
                    torch.empty((1,), device="cpu")
                policy.set_phase("cpu_meta")
                with self.assertRaises(module.ImportPolicyError):
                    torch.empty((1,), device="cuda")
                self.assertEqual(calls, [])
                self.assertEqual(torch.empty((1,), device="meta").device.type, "meta")
            finally:
                policy.restore_tensor_guards()
            self.assertEqual(calls, ["meta"])

    def test_profile_handles_shutdown_module_name_none(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            frame = types.SimpleNamespace(f_globals={"__name__": None},
                                          f_code=types.SimpleNamespace(co_name="_removeHandlerRef"))
            policy.profile(frame, "call", None)

    def test_factory_guards_preserve_device_context_constructor_identity(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            policy.set_phase("cpu_meta")
            calls = []
            class DeviceContext:
                def __init__(self,device):self.device=FakeDevice(device)
            active=DeviceContext("meta")
            context = types.SimpleNamespace(CURRENT_DEVICE=None,DeviceContext=DeviceContext)
            def factory(*args, **kwargs):
                calls.append(1)
                self.assertIn(factory, context.cached)
                return FakeTensor(active.device.type)
            def constructors():
                if not hasattr(context, "cached"):
                    context.cached = {torch.empty}
                return context.cached
            context._device_constructors = constructors
            torch = types.SimpleNamespace(Tensor=FakeTensor, device=FakeDevice, empty=factory,
                                          _C=types.SimpleNamespace(_get_default_device=lambda: "cpu"))
            torch.overrides=types.SimpleNamespace(_get_current_function_mode_stack=lambda:[active])
            with patch.dict(sys.modules, {"torch.utils._device": context}):
                policy.install_tensor_guards(torch)
                try:
                    self.assertTrue(hasattr(context, "cached"), "Original constructor identity was not preserved")
                    self.assertEqual(torch.empty((1,)).device.type, "meta")
                    active.device = FakeDevice("cuda")
                    with self.assertRaises(module.ImportPolicyError):
                        torch.empty((1,))
                    self.assertEqual(calls, [1])
                finally:
                    policy.restore_tensor_guards()

    def test_active_device_context_stack_overrides_stale_current_device_before_factory(self):
        module=self.load()
        class DeviceContext:
            def __init__(self,device):self.device=FakeDevice(device)
        with temporary_root() as root:
            policy=self.policy(module,root);policy.set_phase('cpu_meta');calls=[];stack=[]
            def factory(*a,**k):calls.append(1);return FakeTensor(k.get('device','cpu'))
            context=types.SimpleNamespace(CURRENT_DEVICE=None,DeviceContext=DeviceContext,_device_constructors=lambda:{factory})
            torch=types.SimpleNamespace(Tensor=FakeTensor,device=FakeDevice,empty=factory,_C=types.SimpleNamespace(_get_default_device=lambda:'cpu'),overrides=types.SimpleNamespace(_get_current_function_mode_stack=lambda:list(stack)))
            with patch.dict(sys.modules,{'torch.utils._device':context}):
                policy.install_tensor_guards(torch)
                try:
                    stack.append(DeviceContext('cuda'))
                    with self.assertRaises(module.ImportPolicyError):torch.empty((1,))
                    self.assertEqual(calls,[])
                    self.assertEqual(torch.empty((1,),device='cpu').device.type,'cpu')
                    stack.append(DeviceContext('meta'))
                    torch.empty((1,))
                    self.assertEqual(calls,[1,1])
                    stack[:]=[DeviceContext('cpu')];stack[0].device='cuda'
                    with self.assertRaises(module.ImportPolicyError):torch.empty((1,))
                    self.assertEqual(calls,[1,1])
                    class AlteredContext(DeviceContext):pass
                    stack[:]=[AlteredContext('cpu')]
                    with self.assertRaises(module.ImportPolicyError):torch.empty((1,),device='cpu')
                    stack[:]=[object()]
                    with self.assertRaises(module.ImportPolicyError):torch.empty((1,),device='cpu')
                    self.assertEqual(calls,[1,1])
                finally:policy.restore_tensor_guards()
            self.assertIs(torch.empty,factory)

    def test_profile_preserves_constructors_and_gpu_metadata_but_blocks_execution(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            permitted = [("flex_gemm.utils.autotuner", "__init__"), ("flex_gemm.kernels.triton.utils", "get_gpu_name"), ("torch.cuda", "get_device_properties")]
            forbidden = [("flex_gemm.utils.autotuner", "_benchmark"), ("flex_gemm.utils.autotuner", "run"), ("flex_gemm.utils.autotuner", "__call__"), ("flex_gemm.ops.spconv.submanifold_conv3d", "forward"), ("torch.nn.modules.module", "_call_impl"), ("trellis2.models.sparse_elastic_vae", "forward")]
            for namespace, name in permitted + forbidden:
                function = fake_function(namespace, name)
                policy.install_profile()
                try:
                    if (namespace, name) in permitted:
                        self.assertEqual(function(), 7)
                    else:
                        with self.assertRaises(module.ImportPolicyError):
                            function()
                finally:
                    policy.restore_profile()

    def test_all_native_bindings_are_refused_as_c_calls(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            self.assertEqual(len(module.NATIVE_BINDINGS), 18)
            for name in module.NATIVE_BINDINGS:
                with self.assertRaises(module.ImportPolicyError):
                    policy.profile(None, "c_call", types.SimpleNamespace(__module__="flex_gemm.kernels.cuda", __name__=name))

    def test_audit_blocks_process_network_and_external_writes(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            for event, args in (("subprocess.Popen", ("cmd",)), ("os.system", ("cmd",)), ("os.exec", ("cmd",)), ("os.spawn", ("cmd",)), ("socket.connect", (None,)), ("socket.getaddrinfo", ("example.com",)), ("open", (str(Path(root).parent / "foreign"), "w", 0))):
                with self.assertRaises(module.ImportPolicyError):
                    policy.audit(event, args)
            policy.audit("open", (str(Path(root).parent / "readable"), "r", 0))

    def test_installed_audit_hook_stops_real_open_and_process_before_effects(self):
        self.load()
        with temporary_root() as root:
            source = """
import importlib.util, pathlib, subprocess, sys
spec = importlib.util.spec_from_file_location('policy', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = pathlib.Path(sys.argv[2])
policy = module.ImportPolicy(root)
policy.install_audit_hook()
(root / 'owned.txt').write_text('allowed')
foreign = root.parent / (root.name + '-foreign.txt')
for call in (lambda: foreign.write_text('forbidden'),
             lambda: subprocess.Popen([sys.executable, '-c', 'raise SystemExit(0)'])):
    try:
        call()
    except module.ImportPolicyError:
        pass
    else:
        raise AssertionError('Prohibited side effect escaped')
assert not foreign.exists()
assert policy.snapshot()['rejected_calls'] == 2
print('audit enforcement passed')
"""
            result = subprocess.run([sys.executable, "-B", "-I", "-c", source, str(SCRIPT), root],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "audit enforcement passed")
            self.assertEqual((Path(root) / "owned.txt").read_text(), "allowed")

    def test_audit_allows_only_owned_scratch_and_exact_extra_root(self):
        module = self.load()
        with temporary_root() as root:
            scratch, evidence = Path(root) / "scratch", Path(root) / "evidence"
            scratch.mkdir()
            evidence.mkdir()
            policy = module.ImportPolicy(scratch, writable_roots=(evidence,))
            policy.audit("open", (str(scratch / "owned"), "w", 0))
            policy.audit("open", (str(evidence / "results.json"), "w", 0))
            for name in ("evidence-foreign", "scratch-other"):
                with self.assertRaises(module.ImportPolicyError):
                    policy.audit("open", (str(Path(root) / name / "file"), "w", 0))
            with self.assertRaises(module.ImportPolicyError):
                policy.audit("os.rename", (str(scratch / "owned"), str(Path(root) / "foreign"), -1, -1))

    def test_runtime_guards_block_callbacks_and_restore_originals(self):
        module = self.load()
        with temporary_root() as root:
            calls = []
            def original(*args, **kwargs):
                calls.append(1)
                return 9
            def make_namespace(*names):
                return types.SimpleNamespace(**{name: original for name in names})
            modules = {
                "triton.compiler.compiler": make_namespace("compile"),
                "triton.runtime.driver": make_namespace("_create_driver"),
                "triton.runtime.build": make_namespace("compile_module_from_src", "_build"),
                "triton.backends.amd.driver": make_namespace("compile_module_from_src"),
            }
            driver = types.SimpleNamespace(_active=None, _default=None)
            modules["triton.runtime.driver"].driver = driver
            triton = make_namespace("compile")
            triton.compiler = make_namespace("compile")
            triton.runtime = types.SimpleNamespace(JITFunction=make_namespace("run", "warmup"), Autotuner=make_namespace("run", "warmup", "_bench"))
            triton.testing = make_namespace("do_bench", "do_bench_cudagraph")
            policy = self.policy(module, root)
            with patch.dict(sys.modules, modules):
                policy.install_runtime_guards(triton)
                with self.assertRaises(module.ImportPolicyError):
                    triton.compile()
                with self.assertRaises(module.ImportPolicyError):
                    modules["triton.backends.amd.driver"].compile_module_from_src()
                with self.assertRaises(module.ImportPolicyError):
                    triton.runtime.JITFunction.run()
                self.assertEqual(calls, [])
                policy.restore_runtime_guards()
                self.assertEqual(triton.compile(), 9)
            self.assertEqual(calls, [1])

    def test_runtime_guards_refuse_preexisting_driver_activation(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            driver_module = types.SimpleNamespace(driver=types.SimpleNamespace(_active=object(), _default=None))
            with patch.dict(sys.modules, {"triton.runtime.driver": driver_module}):
                with self.assertRaises(module.ImportPolicyError):
                    policy.install_runtime_guards(types.SimpleNamespace())

    def test_injected_dispatch_mode_uses_live_policy_phase(self):
        module = self.load()
        with temporary_root() as root:
            policy = self.policy(module, root)
            torch = types.SimpleNamespace(Tensor=FakeTensor)
            mode = policy.tensor_mode(torch, dispatch_type=object)
            with self.assertRaises(module.ImportPolicyError):
                mode.__torch_dispatch__(lambda: 11, (), (), {})
            policy.set_phase("cpu_meta")
            self.assertEqual(mode.__torch_dispatch__(lambda: 11, (), (), {}), 11)


if __name__ == "__main__":
    unittest.main()
