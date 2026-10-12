"""Real headless interface contract tests, no TRELLIS/native imports."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
spec = importlib.util.spec_from_file_location("headless", ROOT / "experimental/extensions/trellis2-amd/headless.py")
headless = importlib.util.module_from_spec(spec)
spec.loader.exec_module(headless)


class HeadlessInterfaceTests(unittest.TestCase):
    def test_worker_pass_requires_clean_exit_and_healthy_observer(self):
        spec = importlib.util.spec_from_file_location("headless_proof", ROOT / "scripts/compatibility/amd_headless_import_proof.py")
        proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proof)
        workers = [{"route": route, "status": "PASS", "process": {"returncode": 0, "timed_out": False}}
                   for route in ("generate", "refine")]
        self.assertEqual(proof.classify_workers(True, workers), "PASS")
        for process in ({"returncode": -1, "timed_out": False}, {"returncode": 0, "timed_out": True}):
            self.assertEqual(proof.classify_workers(True, [workers[0], {**workers[1], "process": process}]), "PARTIAL")
        self.assertEqual(proof.classify_workers(False, workers), "PARTIAL")
        self.assertEqual(proof.classify_workers(True, workers[:1]), "PARTIAL")
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            proof.supervise(ROOT)
        self.assertTrue(proof.is_version_query("torchvision._cuda_version.default", "torchvision::_cuda_version() -> int _0", (), {}))
        self.assertFalse(proof.is_version_query("aten.empty.memory_format", "aten::empty(int[] size) -> Tensor", (), {}))
        self.assertFalse(proof.is_version_query("torchvision._cuda_version.default", "torchvision::_cuda_version(Tensor x) -> int", (), {}))
        self.assertFalse(proof.is_version_query("torchvision._cuda_version.default", "torchvision::_cuda_version() -> int _0", (1,), {}))
        self.assertIn(("transformers.activations", "GELUActivation"), proof.STATELESS_ACTIVATIONS)
        self.assertNotIn(("transformers.activations", "XIELUActivation"), proof.STATELESS_ACTIVATIONS)
        self.assertNotIn(("transformers.models.dinov3_vit.modeling_dinov3_vit", "DINOv3ViTModel"), proof.STATELESS_ACTIVATIONS)

    def test_progress_is_observable_and_bounded(self):
        events = []
        progress = headless.ProgressBar(3, lambda current, total: events.append((current, total)))
        progress.update(1)
        progress.update_absolute(8)
        self.assertEqual(events, [(1, 3), (3, 3)])
        with self.assertRaises(NotImplementedError):
            progress.update_absolute(2, preview=b"preview")

    def test_invalid_progress_is_rejected(self):
        for value in (0, -1, None):
            with self.assertRaises(ValueError):
                headless.ProgressBar(value)
        progress = headless.ProgressBar(3)
        with self.assertRaises(ValueError):
            progress.update(-1)

    def test_interfaces_expose_only_implemented_contract(self):
        with patch.dict(sys.modules):
            for name in ("folder_paths", "comfy", "comfy.utils"):
                sys.modules.pop(name, None)
            headless.install_comfy_interfaces(ROOT)
            import folder_paths
            from comfy.utils import ProgressBar
            self.assertEqual(folder_paths.models_dir, str(ROOT))
            self.assertIs(ProgressBar, headless.ProgressBar)
            with self.assertRaises(AttributeError):
                folder_paths.get_full_path("unimplemented", "file")
            with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
                headless.install_comfy_interfaces(ROOT)

    def test_invalid_source_route_and_nonlocal_root_fail_before_import(self):
        with self.assertRaisesRegex(ValueError, "route"):
            headless.import_pipeline_class(ROOT, ROOT, "fallback")
        with self.assertRaisesRegex(ValueError, "source"):
            headless.import_pipeline_class(ROOT, ROOT, "generate")
        with self.assertRaises(ValueError):
            headless.install_comfy_interfaces("https://example.invalid/models")


if __name__ == "__main__":
    unittest.main()
