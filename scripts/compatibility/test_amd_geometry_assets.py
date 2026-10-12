"""Local metadata validation tests using synthetic files only."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "experimental/extensions/trellis2-amd")]
from model_assets import GEOMETRY_ROLES, MODEL_PATHS, validate_geometry_assets


def weights(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    header, data = {}, b""
    for name, shape in tensors.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + size]}
        data += bytes(size)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


class GeometryAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        root = Path(self.temp.name)
        self.source, self.models = root / "source", root / "models"
        model_dir = self.source / "trellis2/models"
        model_dir.mkdir(parents=True)
        classes = {"SparseStructureDecoder": "toy", "SparseStructureFlowModel": "toy", "SLatFlowModel": "toy", "FlexiDualGridVaeDecoder": "toy", "SparseUnetVaeDecoder": "toy"}
        (model_dir / "__init__.py").write_text('__attributes = ' + repr(classes))
        (model_dir / "toy.py").write_text('\n'.join('class ' + name + ': pass' for name in classes))
        self.pipeline_dir = self.models / "microsoft/TRELLIS.2-4B"
        self.pipeline_dir.mkdir(parents=True)
        args = {"models": {name: ('microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16' if name == 'sparse_structure_decoder' else 'ckpts/' + Path(path).name) for name, path in MODEL_PATHS.items()},
                "image_cond_model": {"name": "DinoV3FeatureExtractor", "args": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"}},
                "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}},
                "default_pipeline_type": "1024_cascade"}
        for name in ('sparse_structure', 'shape_slat', 'tex_slat'):
            args[name + '_sampler'] = {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 0.00001}, "params": {"steps": 12}}
        for name in ('shape_slat', 'tex_slat'):
            args[name + '_normalization'] = {"mean": [0.] * 32, "std": [1.] * 32}
        self.pipeline = self.pipeline_dir / "pipeline.json"
        self.pipeline.write_text(json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": args}))
        for role, path in MODEL_PATHS.items():
            prefix = self.models / path
            name = 'SparseStructureFlowModel' if role == 'sparse_structure_flow_model' else 'SLatFlowModel' if 'flow_model' in role else 'SparseStructureDecoder' if role == 'sparse_structure_decoder' else 'SparseUnetVaeDecoder' if role == 'tex_slat_decoder' else 'FlexiDualGridVaeDecoder'
            cfg = {"name": name, "args": {"model_channels": 2, "in_channels": 1, "out_channels": 1, "cond_channels": 2}}
            prefix.parent.mkdir(parents=True, exist_ok=True)
            Path(str(prefix) + '.json').write_text(json.dumps(cfg))
            weights(Path(str(prefix) + '.safetensors'), {"input_layer.weight": [2, 1], "out_layer.weight": [1, 2]})
        self.dino = self.models / "facebook/dinov3-vitl16-pretrain-lvd1689m"
        weights(self.dino / 'model.safetensors', {'embeddings.patch_embeddings.weight': [2, 3, 2, 2]})
        (self.dino / 'config.json').write_text(json.dumps({'model_type': 'dinov3_vit', 'architectures': ['DINOv3ViTModel'], 'hidden_size': 2, 'num_channels': 3, 'patch_size': 2}))
        (self.dino / 'preprocessor_config.json').write_text('{}')
        self.config = {'models_root': str(self.models), 'pipeline_dir': str(self.pipeline_dir), 'dino_dir': str(self.dino)}

    def tearDown(self):
        self.temp.cleanup()

    def test_all_declared_maps_are_local_and_geometry_requirements_explicit(self):
        assets = validate_geometry_assets(self.config, self.source)
        self.assertEqual(set(assets['geometry_roles']), set(GEOMETRY_ROLES))
        self.assertEqual(set(assets['models']), set(MODEL_PATHS))
        self.assertTrue(all(Path(row['weights']).is_absolute() for row in assets['models'].values()))
        self.assertEqual(assets['dino']['directory'], str(self.dino))
        self.assertFalse(assets['weights_loaded'])

    def test_missing_checkpoint_fails_before_preparation(self):
        Path(str(self.models / MODEL_PATHS['shape_slat_decoder']) + '.safetensors').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_geometry_assets(self.config, self.source)

    def test_remote_or_escaping_checkpoint_reference_rejected(self):
        for bad in ('https://example.invalid/model', '../outside/model'):
            value = json.loads(self.pipeline.read_text())
            value['args']['models']['shape_slat_decoder'] = bad
            self.pipeline.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_geometry_assets(self.config, self.source)

    def test_dino_remote_code_or_conditioning_width_mismatch_rejected(self):
        path = self.dino / 'config.json'
        original = json.loads(path.read_text())
        for changes in ({'auto_map': {'AutoModel': 'repo.Custom'}}, {'hidden_size': 3}):
            path.write_text(json.dumps({**original, **changes}))
            with self.assertRaises(ValueError):
                validate_geometry_assets(self.config, self.source)

    def test_relative_root_and_missing_normalization_rejected(self):
        with self.assertRaises(ValueError):
            validate_geometry_assets({**self.config, 'models_root': 'models'}, self.source)
        value = json.loads(self.pipeline.read_text())
        del value['args']['shape_slat_normalization']
        self.pipeline.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            validate_geometry_assets(self.config, self.source)

    def test_malformed_containers_and_missing_model_dimensions_are_validation_errors(self):
        original = self.pipeline.read_text()
        for value in ([], {'args': []}, {'name': 'Trellis2ImageTo3DPipeline', 'args': {'models': []}}):
            self.pipeline.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_geometry_assets(self.config, self.source)
        self.pipeline.write_text(original)
        cfg = Path(str(self.models / MODEL_PATHS['sparse_structure_flow_model']) + '.json')
        cfg.write_text(json.dumps({'name': 'SparseStructureFlowModel', 'args': {'in_channels': 1, 'out_channels': 1}}))
        with self.assertRaises(ValueError):
            validate_geometry_assets(self.config, self.source)

    def test_missing_source_registry_is_a_validation_error(self):
        (self.source / 'trellis2/models/__init__.py').write_text('unrelated = {}')
        with self.assertRaises(ValueError):
            validate_geometry_assets(self.config, self.source)

    def test_component_class_substitution_and_invalid_source_rejected(self):
        cfg = Path(str(self.models / MODEL_PATHS['shape_slat_decoder']) + '.json')
        original = cfg.read_text()
        value = json.loads(original)
        value['name'] = 'SparseStructureDecoder'
        cfg.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            validate_geometry_assets(self.config, self.source)
        cfg.write_text(original)
        (self.source / 'trellis2/models/toy.py').write_text('class bad syntax')
        with self.assertRaisesRegex(ValueError, 'source syntax'):
            validate_geometry_assets(self.config, self.source)


if __name__ == '__main__':
    unittest.main()
