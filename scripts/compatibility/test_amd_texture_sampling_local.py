"""Actual Torch meta construction; no checkpoint loading or GPU initialization."""
import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def forbidden(*args, **kwargs):
    raise RuntimeError('Texture CPU proof forbids GPU, weights, and compilation')


assert not torch.cuda.is_initialized()
torch.cuda._lazy_init = forbidden
torch.load = forbidden
torch.compile = forbidden
torch.set_num_threads(1)
os.environ['ATTN_BACKEND'] = 'sdpa'

import amd_texture_sampling_contract as c
import amd_texture_sampling_local as local


class TextureLocalTests(unittest.TestCase):
    def test_actual_meta_schema_and_thirty_genuine_cpu_rope_attributes(self):
        local.verify_sources()
        with torch.device('meta'):
            model = local.factory(c.CONFIG)
        c.validate_manifest(local.manifest(dict(model.named_parameters())),
                            local.manifest(dict(model.named_buffers())), 'meta')
        self.assertEqual(len(dict(model.named_parameters())), 640)
        self.assertEqual(sum(p.dtype == torch.float32 for p in model.parameters()), 220)
        self.assertEqual(model.in_channels, 64)
        reference = 1 / (10000 ** (torch.arange(21, dtype=torch.float32) / 21))
        for block in model.blocks:
            frequencies = block.self_attn.rope.freqs
            self.assertEqual(str(frequencies.device), 'cpu')
            self.assertEqual(frequencies.dtype, torch.float32)
            self.assertTrue(torch.equal(frequencies, reference))
        with torch.device('meta'):
            storage = local.factory(c.CONFIG, storage=True)
        self.assertTrue(all(p.dtype == torch.bfloat16 and p.is_meta for p in storage.parameters()))
        self.assertEqual(dict(storage.named_buffers()), {})
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_architecture_fails_before_model_import(self):
        with patch.object(local, 'import_class', side_effect=AssertionError('must fail first')):
            bad = {**c.CONFIG, 'args': {**c.CONFIG['args'], 'in_channels': 32}}
            with self.assertRaises(ValueError):
                local.factory(bad)

    def test_saved_denormalized_shape_feature_and_pipeline_identities(self):
        feature, normalized, coords = local.verify_inputs()
        self.assertTrue(feature['verified'])
        self.assertEqual(feature['file_sha256'], local.FEATURE_SHA)
        self.assertEqual(normalized.shape, (3743, 32))
        self.assertEqual(normalized.dtype, np.float32)
        c.validate_coords(coords, c.TOKEN_COUNT)
        saved = np.load(local.SHAPE_DENORMALIZED, allow_pickle=False)
        with torch.inference_mode():
            expected = ((torch.from_numpy(saved) - torch.tensor(c.SHAPE_NORMALIZATION['mean'], dtype=torch.float32))
                        / torch.tensor(c.SHAPE_NORMALIZATION['std'], dtype=torch.float32)).numpy()
        np.testing.assert_array_equal(normalized, expected)
        self.assertEqual(local.verify_sampler()['normalization'], c.TEXTURE_NORMALIZATION)
        self.assertFalse(torch.cuda.is_initialized())

    def test_existing_reversible_widening_checks_original_payload_bytes(self):
        model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16))
        raw = model.weight.detach().view(torch.uint8).numpy().tobytes()
        schema = {'weight': {'shape': [1, 2], 'dtype': 'F32'}}
        self.assertEqual(local.widen(model, {'weight': hashlib.sha256(raw).hexdigest()}, schema), ['weight'])
        self.assertEqual(model.weight.dtype, torch.float32)
        bad = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            local.widen(bad, {'weight': '0' * 64}, schema)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TextureLocalTests))
    assert not torch.cuda.is_initialized()
    print({'gpu_initialized': False, 'real_weights_loaded': False, 'real_model_forwards': 0, 'torch': torch.__version__})
    raise SystemExit(not result.wasSuccessful())
