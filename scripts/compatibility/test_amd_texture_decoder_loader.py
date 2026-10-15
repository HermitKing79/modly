"""Texture-only CPU contracts: tiny strict checkpoints and genuine meta construction."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")]
import torch
from safetensors.torch import save_file

AUDIT = json.loads((ROOT / "docs/compatibility/amd-offline-model-audit-evidence/final-assets/tex_dec_next_dc_f16c32_fp16.metadata.json").read_text())
F32 = frozenset(
    [f"blocks.{stage}.{block}.norm.{field}"
     for stage, count in enumerate((4, 16, 8, 4)) for block in range(count) for field in ("weight", "bias")]
    + [f"blocks.{stage}.{block}.norm1.{field}"
       for stage, block in enumerate((4, 16, 8, 4)) for field in ("weight", "bias")]
    + [f"{projection}.{field}" for projection in ("from_latent", "output_layer") for field in ("weight", "bias")]
)
TINY = {key: {"shape": [2], "dtype": "F32" if key in F32 else "F16", "device": "meta"}
        for key in AUDIT["tensors"]}


class TinyParent(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        if args != AUDIT["config"]["args"]:
            raise AssertionError("Texture constructor must receive the unmodified original args")
        self.dtype = torch.float16
        for key, row in TINY.items():
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


class TextureDecoderLoaderTests(unittest.TestCase):
    def setUp(self):
        try:
            self.loader = importlib.import_module("texture_decoder_loader")
        except ModuleNotFoundError:
            self.fail("Component-specific texture decoder loader is missing")
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.root = Path(self.temp.name)
        self.config = self.root / "texture.json"
        self.weights = self.root / "texture.safetensors"
        self.config.write_text(json.dumps(AUDIT["config"]))
        (self.root / "trellis2").mkdir()
        source = self.root / "trellis2/parent.py"
        source.write_bytes(b"synthetic source, never imported")
        self.source = source
        self.values = {key: torch.tensor([-0., 1.0009765625], dtype=torch.float16) for key in TINY}
        self.patches = [
            patch.object(self.loader, "schema", side_effect=lambda device="meta": {k: dict(v, device=device) for k, v in TINY.items()}),
            patch.object(self.loader, "SOURCE_SHA256", {"parent.py": hashlib.sha256(source.read_bytes()).hexdigest()}),
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
        return self.loader.TextureDecoderComponent(str(self.config), str(self.weights), str(self.root), native_gate=native_gate)

    def test_strict_284_original_payloads_before_only_76_exact_widenings(self):
        component = self.component()
        model = component.load_cpu()
        self.assertEqual(component.diagnostics["entries"], 284)
        self.assertEqual(component.diagnostics["dtype_counts"], {"F16": 208, "F32": 76})
        self.assertEqual(set(component.diagnostics["widened_keys"]), F32)
        self.assertTrue(component.diagnostics["roundtrip_verified"])
        self.assertEqual(model.dtype, torch.float16)
        for key, value in model.state_dict().items():
            self.assertEqual(value.dtype, torch.float32 if key in F32 else torch.float16)
            raw = self.values[key].view(torch.uint8).numpy().tobytes()
            self.assertEqual(value.half().view(torch.uint8).numpy().tobytes(), raw)
            self.assertEqual(component.diagnostics["payload_sha256"][key], hashlib.sha256(raw).hexdigest())
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertIs(component.load_cpu(), model)
        reference = weakref.ref(model)
        del model
        component.unload()
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(component.loaded)

    def test_native_gate_blocks_constructor_and_failure_is_sticky(self):
        for outcome in (False, 1, None):
            with self.subTest(outcome=outcome):
                component = self.component(native_gate=lambda: outcome)
                with self.assertRaisesRegex(RuntimeError, "native"):
                    component.load_cpu()
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    component.load_cpu()

    def test_missing_extra_wrong_shape_and_nonoriginal_dtype_fail_closed(self):
        for change in ("missing", "extra", "shape", "dtype"):
            values = dict(self.values)
            if change == "missing": values.pop("from_latent.weight")
            if change == "extra": values["foreign"] = torch.ones(2, dtype=torch.float16)
            if change == "shape": values["from_latent.weight"] = torch.ones(1, dtype=torch.float16)
            if change == "dtype": values["from_latent.weight"] = values["from_latent.weight"].float()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.component(values)

    def test_source_configuration_and_constructor_changes_fail_closed(self):
        component = self.component()
        bad = copy.deepcopy(AUDIT["config"]); bad["args"]["pred_subdiv"] = True
        self.config.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "configuration"):
            component.load_cpu()
        self.config.write_text(json.dumps(AUDIT["config"]))
        component = self.component(); self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source"):
            component.load_cpu()
        self.source.write_bytes(b"synthetic source, never imported")
        for change in ("shape", "dtype", "buffer"):
            class WrongParent(TinyParent):
                def __init__(self, **args):
                    super().__init__(**args)
                    if change == "shape": self.from_latent.weight = torch.nn.Parameter(torch.zeros(1))
                    if change == "dtype": self.from_latent.weight.data = self.from_latent.weight.half()
                    if change == "buffer": self.register_buffer("foreign", torch.ones(1))
            with self.subTest(change=change), patch.object(self.loader, "_import_parent", return_value=WrongParent):
                with self.assertRaisesRegex(ValueError, "constructor"):
                    self.component().load_cpu()

    def test_original_body_payload_corruption_is_rejected_before_any_widening(self):
        class CorruptParent(TinyParent):
            def load_state_dict(self, state, **kwargs):
                result = super().load_state_dict(state, **kwargs)
                self.blocks.get_submodule("0.0.conv").weight.data[0] = 2
                return result
        with patch.object(self.loader, "_import_parent", return_value=CorruptParent):
            component = self.component()
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                component.load_cpu()
            self.assertEqual(component.diagnostics["widened_keys"], [])
            self.assertFalse(component.loaded)

    def test_failed_widening_roundtrip_releases_partial_model(self):
        original = torch.Tensor.half
        def corrupt(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            if value.device.type == "cpu" and value.dtype == torch.float32:
                result[0] = 2
            return result
        component = self.component()
        with patch.object(torch.Tensor, "half", corrupt), self.assertRaisesRegex(ValueError, "round.trip"):
            component.load_cpu()
        self.assertFalse(component.loaded)
        self.assertFalse(component.diagnostics["weights_loaded"])
        self.assertFalse(component._component.loaded)


class GenuineTextureMetaTests(unittest.TestCase):
    def test_every_finite_f16_bit_pattern_survives_f32_widening_exactly(self):
        # Widening is exact for the entire finite F16 domain, including signed
        # zero and subnormals, not just the particular toy checkpoint values.
        bits = torch.arange(65536, dtype=torch.int32)
        finite = bits[(bits & 0x7c00) != 0x7c00].to(torch.uint16)
        original = finite.view(torch.float16)
        self.assertEqual(original.numel(), 63488)
        self.assertTrue(torch.isfinite(original).all())
        self.assertEqual(original.view(torch.uint8).numpy().tobytes(),
                         original.float().half().view(torch.uint8).numpy().tobytes())

    def test_cold_import_is_stdlib_only(self):
        script = "import sys;sys.path[:0]=sys.argv[1:];import texture_decoder_loader;assert not {'torch','triton','flex_gemm','trellis2','numpy','safetensors'} & set(sys.modules)"
        run = subprocess.run([sys.executable, "-c", script, str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_hand_derived_schema_matches_all_original_header_shapes(self):
        try:
            loader = importlib.import_module("texture_decoder_loader")
        except ModuleNotFoundError:
            self.fail("Component-specific texture decoder loader is missing")
        rows = loader.schema()
        self.assertEqual(set(rows), set(AUDIT["tensors"]))
        self.assertEqual(set(loader.FP32_KEYS), F32)
        for key, row in rows.items():
            self.assertEqual(row, {"shape": AUDIT["tensors"][key]["shape"], "dtype": "F32" if key in F32 else "F16", "device": "meta"})
        self.assertEqual(sum(row["dtype"] == "F16" for row in rows.values()), 208)

    def test_genuine_meta_factory_has_40_convolutions_four_external_guides_and_no_native(self):
        script = '''import sys,json
sys.path[:0]=sys.argv[1:]
import torch
def deny(*a,**k):raise AssertionError('CPU proof forbids GPU, weights, compilation and forward')
torch.cuda._lazy_init=deny;torch.load=deny;torch.compile=deny
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('flex_gemm','cumesh','o_voxel','nvdiffrast'):deny()
    if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).lower().endswith(('.safetensors','.pt','.pth','.ckpt','.gguf')):deny()
sys.addaudithook(audit)
torch.nn.Module.__call__=deny
import texture_decoder_loader as t
model=t.meta_factory()
assert t.manifest(model)==t.schema()
assert not list(model.named_buffers())
assert model.dtype==torch.float16 and model.use_fp16 and not model.pred_subdiv
assert model.out_channels==6 and model.from_latent.in_features==32
assert sum(type(m).__name__=='SparseConv3d' for m in model.modules())==40
assert sum(type(m).__name__=='SparseChannel2Spatial' for m in model.modules())==4
assert all(not hasattr(res[-1],'to_subdiv') for res in model.blocks[:-1])
assert 'flex_gemm' not in sys.modules and not torch.cuda.is_initialized()
assert all(p.is_meta for p in model.parameters())
# The pinned norm explicitly promotes F16 activation values to F32 and returns
# to F16. Exercise that genuine implementation on synthetic CPU values only.
from trellis2.modules.norm import LayerNorm32
norm=LayerNorm32(4,eps=1e-6)
x=torch.tensor([[0.,-0.,0.00006103515625,-0.00006103515625],[1000.,-1000.,.5,-.5]],dtype=torch.float16)
actual=norm._forward(x)
expected=torch.nn.functional.layer_norm(x.float(),(4,),norm.weight,norm.bias,1e-6).half()
assert norm.weight.dtype==torch.float32 and actual.dtype==torch.float16
assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)) and torch.isfinite(actual).all()
# Metadata helpers restore the original backend state even when validation
# fails, so proof construction cannot replace the runtime native backend.
from trellis2.modules.sparse.conv import conv
prior_backend=conv.config.CONV;prior_impl=dict(conv._backends)
original_schema=t.schema;t.schema=lambda device='meta':{}
try:
    try:t.meta_factory()
    except ValueError:pass
    else:raise AssertionError('Wrong constructor schema accepted')
finally:t.schema=original_schema
assert conv.config.CONV==prior_backend and conv._backends==prior_impl
import inspect,hashlib
from pathlib import Path
parent_path=Path(inspect.getfile(type(model)))
print(json.dumps({'entries':284,'F16':208,'F32':76,'convolutions':40,
    'class':type(model).__module__+'.'+type(model).__name__,
    'parent_source':str(parent_path),'parent_source_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    'parent_class_source_sha256':hashlib.sha256(inspect.getsource(type(model)).encode()).hexdigest(),
    'parameters':t.manifest(model),'buffers':{},
    'convolution_schedule':{name:{'in_channels':m.in_channels,'out_channels':m.out_channels,'weight_shape':list(m.weight.shape),'dtype':str(m.weight.dtype)} for name,m in model.named_modules() if type(m).__name__=='SparseConv3d'},
    'c2s_modules':[name for name,m in model.named_modules() if type(m).__name__=='SparseChannel2Spatial'],
    'normalization_cpu_equivalence':True,'backend_state_restored_on_failure':True,
    'native_imported':False,'weights_loaded':False,'checkpoint_payload_read':False,
    'gpu_initialized':False,'model_forward_calls':0,'torch':str(torch.__version__),'python':sys.version}))
'''
        run = subprocess.run([sys.executable, "-c", script, str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('"convolutions": 40', run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
