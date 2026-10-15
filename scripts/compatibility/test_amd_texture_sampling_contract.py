"""CPU-only texture architecture, source schedule, and normalization acceptance."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import amd_shape_slat_contract as shape
import amd_texture_sampling_contract as c


class TextureContractTests(unittest.TestCase):
    def test_exact_64_in_32_out_640_bf16_checkpoint(self):
        expected = shape.schema()
        expected['input_layer.weight']['shape'] = [1536, 64]
        self.assertEqual(c.schema(), expected)
        self.assertEqual(len(c.schema()), 640)
        self.assertEqual(sum(r['dtype'] == 'F32' for r in c.schema().values()), 220)
        self.assertEqual(c.CONFIG['args']['in_channels'], 64)
        self.assertEqual(c.CONFIG['args']['out_channels'], 32)
        header = {k: {**v, 'dtype': 'BF16'} for k, v in c.schema().items()}
        c.validate_header(c.CONFIG, header)
        with self.assertRaises(ValueError):
            c.validate_header(shape.CONFIG, header)
        for name, field, value in [('input_layer.weight', 'shape', [1536, 32]),
                                   ('out_layer.weight', 'shape', [64, 1536]),
                                   ('blocks.0.norm2.weight', 'dtype', 'F32')]:
            bad = copy.deepcopy(header)
            bad[name][field] = value
            with self.assertRaises(ValueError):
                c.validate_header(c.CONFIG, bad)
        manifest = {k: {**v, 'device': 'meta', 'is_meta': True} for k, v in c.schema().items()}
        c.validate_manifest(manifest, {}, 'meta')
        with self.assertRaises(ValueError):
            c.validate_manifest(manifest, {'foreign': {}}, 'meta')

    def test_canonical_twelve_positive_forwards_even_inside_interval(self):
        self.assertEqual(c.PARAMS, dict(steps=12, rescale_t=3., guidance_strength=1.,
                                      guidance_rescale=0., guidance_interval=(.6, .9), dino_lock=0.))
        self.assertEqual(c.SIGMA_MIN, .00001)
        self.assertEqual(c.SEED, 20261015)
        self.assertEqual(c.times()[3], .9)
        self.assertEqual(c.times()[8], .6)
        self.assertEqual([t['step'] for t in c.tickets()], list(range(1, 13)))
        self.assertEqual([t['role'] for t in c.tickets()], ['positive'] * 12)
        c.require_count(12, 12, 0)
        for count in [(21, 12, 9), (13, 12, 1), (12, 11, 1), (True, 12, 0)]:
            with self.assertRaises(ValueError):
                c.require_count(*count)
        budget = c.Budget()
        with self.assertRaises(RuntimeError):
            budget.next_ticket()
        for a, b in zip(c.times(), c.times()[1:]):
            budget.begin_step(a, b)
            ticket = budget.next_ticket()
            with self.assertRaises(RuntimeError):
                budget.begin_forward(copy.deepcopy(ticket))
            budget.begin_forward(ticket)
            with self.assertRaises(RuntimeError):
                budget.end_step()
            budget.end_forward(ticket)
            with self.assertRaises(RuntimeError):
                budget.next_ticket()
            budget.end_step()
        budget.require_complete()
        with self.assertRaises(RuntimeError):
            budget.begin_step(1., c.times()[1])
        for pair in [(float('nan'), .9), (1., float('inf')), (True, .9), (.9, .8)]:
            with self.assertRaises(RuntimeError):
                c.Budget().begin_step(*pair)

    def test_shape_once_texture_distinct_denormalization(self):
        raw = np.ascontiguousarray(np.linspace(-3, 3, 64, dtype=np.float32).reshape(2, 32))
        sm = np.array(c.SHAPE_NORMALIZATION['mean'], np.float32)
        ss = np.array(c.SHAPE_NORMALIZATION['std'], np.float32)
        tm = np.array(c.TEXTURE_NORMALIZATION['mean'], np.float32)
        ts = np.array(c.TEXTURE_NORMALIZATION['std'], np.float32)
        shape_denormalized = raw * ss + sm
        normalized = c.normalize_shape(shape_denormalized)
        np.testing.assert_array_equal(normalized, (shape_denormalized - sm) / ss)
        np.testing.assert_array_equal(c.denormalize_texture(raw), raw * ts + tm)
        self.assertFalse(np.array_equal(c.denormalize_texture(raw), shape_denormalized))
        with self.assertRaises(ValueError):
            c.normalize_shape(raw, c.TEXTURE_NORMALIZATION)
        with self.assertRaises(ValueError):
            c.denormalize_texture(raw, c.SHAPE_NORMALIZATION)
        for bad in [raw.astype(np.float64), raw[:, ::-1], np.full((2, 32), np.nan, np.float32)]:
            with self.assertRaises(ValueError):
                c.normalize_shape(bad)

    def test_transfer_keeps_noise_and_shape_as_distinct_32_channel_inputs(self):
        for name in ('x', 'shape_cond'):
            c.validate_transfer(name, dict(shape=[7, 32], dtype='torch.float32', device='cpu'), 'cuda:0', 'inputs', 7)
        c.validate_transfer('seqlen', dict(shape=[1], dtype='torch.int64', device='cpu'), 'cuda:0', 'inputs', 7)
        for name, row in [('neg_cond', dict(shape=[1, 1029, 1024], dtype='torch.float32', device='cpu')),
                          ('x', dict(shape=[7, 64], dtype='torch.float32', device='cpu'))]:
            with self.assertRaises(ValueError):
                c.validate_transfer(name, row, 'cuda:0', 'inputs', 7)


if __name__ == '__main__':
    unittest.main()
