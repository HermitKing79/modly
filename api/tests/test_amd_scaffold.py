import importlib.util
import json
import os
from pathlib import Path
import queue
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from services import amd_runtime
from services.extension_process import ExtensionProcess

MANIFEST = {"id": "trellis2-amd/generate", "ext_id": "trellis2-amd", "node_id": "generate",
            "runtime_contract": amd_runtime.CONTRACT, "experimental": True}


class AMDScaffoldTests(unittest.TestCase):
    def test_explicit_opt_in_and_config_are_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "opt-in"):
                amd_runtime.runtime_config(MANIFEST)
        with patch.dict(os.environ, {"MODLY_ENABLE_EXPERIMENTAL_AMD": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "absolute"):
                amd_runtime.runtime_config(MANIFEST)

    def test_unknown_contract_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            amd_runtime.is_amd({"runtime_contract": "unknown"})

    def test_unc_and_device_paths_are_rejected_before_filesystem_access(self):
        for value in (r"\\server\share\weights", r"\\?\C:\weights", r"\\.\C:\weights"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "local"):
                amd_runtime.local_path(value, "model")

    def test_blocked_transport_write_stops_worker(self):
        proc = ExtensionProcess(Path("unused"), MANIFEST)
        event = __import__("threading").Event()
        with patch.object(proc, "_write_message", side_effect=lambda *a: event.wait(1)), \
             patch.object(proc, "stop") as stop, patch("services.extension_process.AMD_SEND_TIMEOUT", 0.01):
            try:
                with self.assertRaises(TimeoutError):
                    proc._send({"action": "load"})
            finally:
                event.set()
        stop.assert_called_once()

    def test_missing_dependency_fails_before_native_or_torch_import(self):
        config = {"python": str(Path(sys.executable).resolve())}
        with patch.object(amd_runtime.metadata, "distribution", side_effect=amd_runtime.metadata.PackageNotFoundError("torch")), \
             patch.object(sys, "version_info", (3, 12)), patch.object(sys, "platform", "win32"):
            with self.assertRaisesRegex(ValueError, "AMD dependency missing: torch; installation is disabled"):
                amd_runtime.worker_preflight(config, MANIFEST["id"])

    def test_unopted_extension_is_not_discovered_or_imported_in_host(self):
        from services import generator_registry
        root = Path(__file__).resolve().parents[2] / "experimental/extensions"
        registry = generator_registry.GeneratorRegistry()
        with patch.dict(os.environ, {"MODLY_ENABLE_EXPERIMENTAL_AMD": "0"}), \
             patch.object(generator_registry, "EXTENSIONS_DIR", root):
            registry.initialize()
        self.assertEqual(registry._generators, {})

    def test_opted_invalid_runtime_cannot_fall_back_to_direct_import(self):
        from services import generator_registry
        root = Path(__file__).resolve().parents[2] / "experimental/extensions"
        registry = generator_registry.GeneratorRegistry()
        with patch.dict(os.environ, {"MODLY_ENABLE_EXPERIMENTAL_AMD": "1", "MODLY_AMD_RUNTIME_CONFIG": "relative.json"}), \
             patch.object(generator_registry, "EXTENSIONS_DIR", root), \
             patch.object(generator_registry.importlib.util, "spec_from_file_location", side_effect=AssertionError("host import forbidden")):
            registry.initialize()
        self.assertEqual(registry._generators, {})
        errors = registry.load_errors()
        self.assertEqual(set(errors), {"trellis2-amd/generate", "trellis2-amd/refine"})
        self.assertTrue(all("absolute" in msg for msg in errors.values()))

    def test_identity_rejects_wrong_node_cuda_and_gpu_initialization(self):
        config = {"python": str(Path("runtime/python.exe").resolve())}
        identity = {"contract": amd_runtime.CONTRACT, "model_id": MANIFEST["id"],
                    "python": config["python"], "mock_only": True,
                    "gpu_initialized": False, "weights_loaded": False,
                    "torch": "2.13.0+rocm10.0.0", "hip": "7.15.26333", "cuda": None,
                    "python_version": "3.12.10", "format": "safetensors"}
        amd_runtime.verify_identity(identity, MANIFEST, config)
        for key, value in (("model_id", "trellis2-amd/refine"), ("cuda", "12.6"),
                           ("gpu_initialized", True), ("mock_only", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                amd_runtime.verify_identity({**identity, key: value}, MANIFEST, config)

    def test_offline_pairs_reject_gguf_missing_and_remote_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "checkpoint"
            prefix.with_suffix(".json").write_text('{"name":"fixture","args":{}}')
            header = b'{"__metadata__":{"fixture":"true"}}'
            prefix.with_suffix(".safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
            result = amd_runtime.validate_models({"generate": [str(prefix)]}, "generate")
            self.assertEqual(result[0]["prefix"], str(prefix.resolve()))
            for bad in (str(prefix) + ".gguf", "https://example/model", str(prefix) + "missing"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    amd_runtime.validate_models({"generate": [bad]}, "generate")
            prefix.with_suffix(".safetensors").write_bytes(b"GGUFbroken")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                amd_runtime.validate_models({"generate": [str(prefix)]}, "generate")

    def test_startup_timeout_stops_worker_without_auto_repair(self):
        proc = ExtensionProcess(Path("unused"), MANIFEST)
        with patch.object(proc, "_launch", side_effect=lambda *a: None), \
             patch.object(proc, "_recv", side_effect=TimeoutError("bounded")), \
             patch.object(proc, "stop") as stop, \
             patch.object(proc, "_install_missing_package") as install, \
             patch("services.amd_runtime.runtime_config", return_value={"python": __file__}):
            with self.assertRaises(TimeoutError):
                proc._start()
        stop.assert_called_once()
        install.assert_not_called()

    def test_generation_deadline_stops_worker(self):
        proc = ExtensionProcess(Path("unused"), MANIFEST)
        with patch("services.extension_process.time.monotonic", side_effect=[0, 31]), \
             patch.object(proc, "stop") as stop:
            with self.assertRaises(TimeoutError):
                proc._receive_generation("x", None, None)
        stop.assert_called_once()

    def test_cancel_failure_retains_process_handle(self):
        import threading
        proc = ExtensionProcess(Path("unused"), MANIFEST)
        class StuckProcess:
            def poll(self):
                return None
            def kill(self):
                raise OSError("cannot terminate")
        child = StuckProcess()
        proc._proc = child
        cancel = threading.Event()
        cancel.set()
        with patch("services.extension_process.time.monotonic", side_effect=[0, 1, 1, 5, 5]), \
             patch.object(proc, "_send"), patch.object(proc._queue, "get", side_effect=queue.Empty):
            with self.assertRaisesRegex(RuntimeError, "Could not stop"):
                proc._receive_generation("x", None, cancel)
        self.assertIs(proc._proc, child)

    def test_headless_adapter_import_has_no_torch_or_upstream_import(self):
        path = Path(__file__).resolve().parents[2] / "experimental/extensions/trellis2-amd/generator.py"
        spec = importlib.util.spec_from_file_location("amd_headless_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", wraps=__import__) as imports:
            spec.loader.exec_module(module)
        names = [call.args[0] for call in imports.call_args_list]
        self.assertFalse(any(n.split(".")[0] in ("torch", "trellis2", "comfy", "folder_paths") for n in names))
        generator = module.AMDScaffold(Path("unused"), Path("unused"))
        with self.assertRaisesRegex(ValueError, "mock"):
            generator.generate(b"image", {"mock": False})


if __name__ == "__main__":
    unittest.main()
