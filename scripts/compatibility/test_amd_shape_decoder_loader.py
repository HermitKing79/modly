"""Tiny CPU checkpoints; genuine checkpoint/native imports are never exercised."""
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")]
import torch
from safetensors.torch import save_file

PROOF = json.loads((ROOT / "docs/compatibility/amd-shape-decoder-neural-evidence/cpu-probe-4/results.json").read_text())
CONFIG = json.loads((ROOT / "docs/compatibility/amd-offline-model-audit-evidence/final-assets/shape_dec_next_dc_f16c32_fp16.metadata.json").read_text())["config"]
# Hand-derived from the pinned 32 ConvNeXt blocks, four C2S blocks and two projections.
WIDEN = frozenset(
    [f"blocks.{stage}.{block}.norm.{field}"
     for stage, count in enumerate((4, 16, 8, 4)) for block in range(count) for field in ("weight", "bias")]
    + [f"blocks.{stage}.{block}.norm1.{field}"
       for stage, block in enumerate((4, 16, 8, 4)) for field in ("weight", "bias")]
    + [f"{projection}.{field}" for projection in ("from_latent", "output_layer") for field in ("weight", "bias")]
)
MANIFEST = {key: {"shape": [2], "dtype": row["dtype"], "device": "meta"}
            for key, row in PROOF["worker"]["parameters"].items()}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TinyParent(torch.nn.Module):
    def __init__(self, out_channels, **args):
        super().__init__()
        if out_channels != 7 or "resolution" in args:
            raise AssertionError("Parent boundary must discard only wrapper resolution")
        for key, row in MANIFEST.items():
            owner = self
            parts = key.split(".")
            for name in parts[:-1]:
                if name not in owner._modules:
                    owner.add_module(name, torch.nn.Module())
                owner = owner._modules[name]
            owner.register_parameter(parts[-1], torch.nn.Parameter(torch.zeros(
                2, dtype=torch.float32 if row["dtype"] == "F32" else torch.float16)))

    def forward(self, *args, **kwargs):
        raise AssertionError("No decoder forward is admitted")


class ShapeDecoderLoaderTests(unittest.TestCase):
    def setUp(self):
        try:
            self.loader = importlib.import_module("shape_decoder_loader")
        except ModuleNotFoundError:
            self.fail("Component-specific shape decoder loader is missing")
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.root = Path(self.temp.name)
        self.config = self.root / "shape.json"
        self.weights = self.root / "shape.safetensors"
        self.config.write_text(json.dumps(CONFIG))
        (self.root / "trellis2").mkdir()
        self.source_file = self.root / "trellis2/parent.py"
        self.source_file.write_bytes(b"synthetic source, never imported")
        self.source_pin = {"parent.py": hashlib.sha256(self.source_file.read_bytes()).hexdigest()}
        self.values = {key: torch.tensor([-0., 1.0009765625], dtype=torch.float16) for key in MANIFEST}
        self.patches = [
            patch.object(self.loader, "PARAMETER_MANIFEST_SHA256", fingerprint(MANIFEST)),
            patch.object(self.loader, "SOURCE_SHA256", self.source_pin),
            patch.object(self.loader, "_import_parent", return_value=TinyParent),
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU initialization forbidden")),
        ]
        for guard in self.patches:
            guard.start()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        for guard in reversed(getattr(self, "patches", [])):
            guard.stop()
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def component(self, values=None, native_gate=lambda: True):
        save_file(self.values if values is None else values, str(self.weights))
        return self.loader.ShapeDecoderComponent(
            str(self.config), str(self.weights), str(self.root), MANIFEST, native_gate=native_gate)

    def test_strict_original_bytes_then_only_76_exact_widenings_and_release(self):
        component = self.component()
        self.assertFalse(component.loaded)
        model = component.load_cpu()
        state = model.state_dict()
        self.assertEqual(len(state), 292)
        self.assertEqual(set(component.diagnostics["widened_keys"]), WIDEN)
        self.assertEqual(component.diagnostics["entries"], 292)
        self.assertEqual(component.diagnostics["dtype_counts"], {"F16": 216, "F32": 76})
        self.assertFalse(model.training)
        for key, value in state.items():
            self.assertEqual(value.dtype, torch.float32 if key in WIDEN else torch.float16)
            self.assertEqual(value.half().view(torch.uint8).numpy().tobytes(),
                             self.values[key].view(torch.uint8).numpy().tobytes())
        self.assertIs(component.load_cpu(), model)
        reference = weakref.ref(model)
        del state, value, model
        component.unload()
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(component.loaded)
        self.assertFalse(component.diagnostics["weights_loaded"])

    def test_native_gate_blocks_construction_and_failure_is_sticky(self):
        for result in (False, 1, None, {"status": "PASS"}):
            with self.subTest(result=result):
                component = self.component(native_gate=lambda: result)
                with self.assertRaisesRegex(RuntimeError, "native"):
                    component.load_cpu()
                self.assertFalse(component.loaded)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    component.load_cpu()

    def test_two_argument_api_requires_completed_import_then_uses_pinned_manifest(self):
        proof = self.root / "proof.json"
        proof.write_text(json.dumps({"worker": {"parameters": MANIFEST}}))
        save_file(self.values, str(self.weights))
        with patch.object(self.loader, "SOURCE", self.root), patch.object(self.loader, "PARAMETER_PROOF", proof):
            for module in (None, SimpleNamespace(__spec__=None), SimpleNamespace(__spec__=SimpleNamespace(_initializing=True))):
                with self.subTest(module=module), patch.dict(sys.modules, {"flex_gemm": module}):
                    component = self.loader.ShapeDecoderComponent(str(self.config), str(self.weights))
                    with self.assertRaisesRegex(RuntimeError, "native"):
                        component.load_cpu()
            module = SimpleNamespace(__spec__=SimpleNamespace(_initializing=False))
            with patch.dict(sys.modules, {"flex_gemm": module}):
                component = self.loader.ShapeDecoderComponent(str(self.config), str(self.weights))
                model = component.load_cpu()
                self.assertEqual(model.from_latent.weight.dtype, torch.float32)
                component.unload()

    def test_cold_loader_import_does_not_import_torch_or_native_packages(self):
        script = "import sys; sys.path[:0]=sys.argv[1:]; import shape_decoder_loader; assert not {'torch','triton','flex_gemm','trellis2'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-c", script, str(ROOT / "api"),
                                 str(ROOT / "experimental/extensions/trellis2-amd")],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_constructor_shape_dtype_keys_and_buffers_fail_closed(self):
        for change in ("shape", "dtype", "missing", "buffer"):
            class WrongParent(TinyParent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if change == "shape": self.from_latent.weight = torch.nn.Parameter(torch.zeros(1))
                    if change == "dtype": self.from_latent.weight.data = self.from_latent.weight.half()
                    if change == "missing": del self.from_latent.weight
                    if change == "buffer": self.register_buffer("foreign", torch.ones(1))
            with self.subTest(change=change), patch.object(self.loader, "_import_parent", return_value=WrongParent):
                component = self.component()
                with self.assertRaisesRegex(ValueError, "constructor"):
                    component.load_cpu()
                self.assertFalse(component.loaded)

    def test_original_checkpoint_missing_extra_wrong_shape_and_dtype_are_rejected(self):
        for change in ("missing", "extra", "shape", "dtype"):
            values = dict(self.values)
            if change == "missing": values.pop("from_latent.weight")
            if change == "extra": values["foreign"] = torch.ones(2, dtype=torch.float16)
            if change == "shape": values["from_latent.weight"] = torch.ones(1, dtype=torch.float16)
            if change == "dtype": values["from_latent.weight"] = values["from_latent.weight"].float()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.component(values).load_cpu()

    def test_manifest_config_and_source_changes_are_rejected(self):
        bad = copy.deepcopy(MANIFEST); bad["from_latent.weight"]["shape"] = [1]
        save_file(self.values, str(self.weights))
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.loader.ShapeDecoderComponent(str(self.config), str(self.weights), str(self.root), bad, native_gate=lambda: True)
        component = self.component()
        config = copy.deepcopy(CONFIG); config["args"]["use_fp16"] = False
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration"):
            component.load_cpu()
        self.config.write_text(json.dumps(CONFIG))
        component = self.component()
        self.source_file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source"):
            component.load_cpu()

    def test_failed_roundtrip_unloads_and_does_not_expose_partial_model(self):
        component = self.component()
        original = torch.Tensor.half
        def corrupt_roundtrip(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            if value.device.type == "cpu" and value.dtype == torch.float32:
                result[0] = 2
            return result
        with patch.object(torch.Tensor, "half", corrupt_roundtrip), self.assertRaisesRegex(ValueError, "round.trip"):
            component.load_cpu()
        self.assertFalse(component.loaded)
        self.assertFalse(component.diagnostics["weights_loaded"])
        with self.assertRaisesRegex(RuntimeError, "failed"):
            component.load_cpu()


if __name__ == "__main__":
    unittest.main()
