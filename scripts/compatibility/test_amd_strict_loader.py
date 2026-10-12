"""Tiny synthetic checkpoints only; no real TRELLIS weights or GPU use."""
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")]
from strict_loader import StrictLocalComponent
import torch
from safetensors.torch import save_file


class Tiny(torch.nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))


class StrictLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.root = Path(self.temp.name)
        self.config = self.root / "tiny.json"
        self.weights = self.root / "tiny.safetensors"
        self.config.write_text(json.dumps({"name": "Tiny", "args": {"width": 2}}))
        self.init_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU initialization forbidden"))
        self.init_guard.start()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.init_guard.stop()
        self.temp.cleanup()

    def component(self, values=None, factory=None):
        save_file(values if values is not None else {"weight": torch.tensor([3., 4.]), "scale": torch.tensor([5., 6.])}, str(self.weights))
        return StrictLocalComponent(str(self.config), str(self.weights), factory or (lambda config: Tiny(**config["args"])))

    def test_lazy_load_matches_parameters_and_persistent_buffers_then_releases(self):
        calls = []
        component = self.component(factory=lambda config: (calls.append(config) or Tiny(**config["args"])))
        self.assertFalse(component.loaded)
        self.assertEqual(calls, [])
        model = component.load_cpu()
        self.assertTrue(torch.equal(model.weight, torch.tensor([3., 4.])))
        self.assertTrue(torch.equal(model.scale, torch.tensor([5., 6.])))
        self.assertIs(component.load_cpu(), model)
        self.assertEqual(len(calls), 1)
        self.assertEqual(component.diagnostics["missing_keys"], [])
        self.assertEqual(component.diagnostics["unexpected_keys"], [])
        reference = weakref.ref(model)
        del model
        component.unload()
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(component.loaded)

    def test_missing_buffer_unexpected_key_wrong_shape_or_dtype_fail_closed(self):
        good = {"weight": torch.tensor([3., 4.]), "scale": torch.tensor([5., 6.])}
        cases = [{"weight": good["weight"]}, {**good, "extra": torch.tensor([1.])},
                 {**good, "scale": torch.tensor([1.])}, {**good, "scale": good["scale"].half()}]
        for values in cases:
            with self.subTest(keys=values.keys()):
                component = self.component(values)
                with self.assertRaisesRegex(ValueError, "state dictionary"):
                    component.load_cpu()
                self.assertFalse(component.loaded)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    component.load_cpu()

    def test_changed_configuration_rejected_before_loading(self):
        component = self.component()
        self.config.write_text(json.dumps({"name": "Tiny", "args": {"width": 3}}))
        with self.assertRaisesRegex(ValueError, "changed"):
            component.load_cpu()
        self.assertFalse(component.loaded)

    def test_unmaterialized_nonpersistent_buffer_is_not_silently_accepted(self):
        class WithCache(Tiny):
            def __init__(self, width=2):
                super().__init__(width)
                self.register_buffer("cache", torch.ones(1), persistent=False)
        component = self.component(factory=lambda config: WithCache(**config["args"]))
        with self.assertRaisesRegex(ValueError, "buffer.*metadata"):
            component.load_cpu()
        self.assertFalse(component.loaded)

    def test_remote_and_relative_paths_are_rejected(self):
        for value in ("owner/model", "https://example.invalid/model", "tiny.json"):
            with self.assertRaises(ValueError):
                StrictLocalComponent(value, str(self.weights), lambda unused: Tiny())


if __name__ == "__main__":
    unittest.main()
