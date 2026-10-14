"""Host-only arithmetic and fail-closed ordering tests; never import torch."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import amd_wsl_decoder_admission as a


class AdmissionTests(unittest.TestCase):
    def test_exact_40_sites_and_four_logical_subdivisions(self):
        sites = a.convolution_sites()
        self.assertEqual(len(sites), 40)
        self.assertEqual([sum(s['stage'] == i for s in sites) for i in range(4)], [6, 18, 10, 6])
        self.assertEqual(len(a.subdivision_sites()), 4)
        self.assertEqual(sites[5]['support_stage'], 1)
        self.assertEqual((sites[4]['in_channels'], sites[4]['out_channels']), (1024, 4096))
        root = Path(__file__).resolve().parents[2]
        header = json.loads((root / 'docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json').read_text())['checkpoint_tensors']
        actual = {k: v['shape'] for k, v in header.items() if k.endswith(('.conv.weight', '.conv1.weight', '.conv2.weight'))}
        self.assertEqual(actual, {s['weight']: [s['out_channels'], 3, 3, 3, s['in_channels']] for s in sites})

    def test_visible_explicit_formula_and_uint64_delta(self):
        plan = a.explicit_visible(site='blocks.0.0.conv', tokens=2049, spatial_volume=64**3)
        self.assertEqual(plan['visible_bytes'], 16*2049 + 21*2049*27 + 4*2049*27*1024 + 2*2049*1024)
        large = a.explicit_visible(site='blocks.0.0.conv', tokens=2049, spatial_volume=2**32)
        self.assertEqual(large['hash_key_itemsize'], 8)
        self.assertEqual(large['visible_bytes']-plan['visible_bytes'], 8*2049)
        self.assertFalse(plan['whole_call_bound'])

    def test_hash_sentinel_and_native_index_limits(self):
        self.assertEqual(a.hash_layout(1, 2**32-1)['key_itemsize'], 4)
        self.assertEqual(a.hash_layout(1, 2**32)['key_itemsize'], 8)
        for tokens, volume in ((0, 1), (1_000_001, 1), (1, 0), (1, 2**64), (True, 1)):
            with self.assertRaises(ValueError): a.hash_layout(tokens, volume)

    def test_foreign_convolution_and_count_rejected(self):
        for site, tokens in (('foreign', 1), ('blocks.4.0.conv', 1), ('blocks.0.0.conv', -1)):
            with self.assertRaises(ValueError): a.explicit_visible(site=site, tokens=tokens, spatial_volume=64**3)

    def test_subdivision_growth_and_per_call_visible_buffers(self):
        plan = a.subdivision_visible(stage=0, parent_tokens=2049, child_tokens=16392, input_channels=4096)
        self.assertEqual(plan['visible_bytes'], 32*2049 + 104*16392 + 2*16392*512)
        self.assertEqual(a.subdivision_visible(stage=0, parent_tokens=2049, child_tokens=16392, input_channels=4096, cached=True)['visible_bytes'], 16*16392 + 2*16392*512)
        for values in (dict(child_tokens=16393), dict(stage=4), dict(input_channels=512), dict(cached=1)):
            arguments=dict(stage=0,parent_tokens=2049,child_tokens=16392,input_channels=4096); arguments.update(values)
            with self.assertRaises(ValueError): a.subdivision_visible(**arguments)

    def test_unknown_workspace_blocks_before_any_original_callback(self):
        calls=[]
        memory=a.Memory(0,0,16*a.GIB)
        with self.assertRaises(a.UnknownWorkspace):
            a.before_explicit_convolution(lambda: calls.append('native'), memory=memory,site='blocks.0.0.conv',tokens=2049,spatial_volume=64**3)
        with self.assertRaises(a.UnknownWorkspace):
            a.before_subdivision(lambda: calls.append('gather'),memory=memory,stage=0,parent_tokens=2049,child_tokens=4098,input_channels=4096)
        self.assertEqual(calls, [])

    def test_budget_rejection_also_precedes_original_callback(self):
        calls=[]
        with self.assertRaises(ValueError):
            a.before_explicit_convolution(lambda: calls.append('native'),memory=a.Memory(11*a.GIB,11*a.GIB,2*a.GIB),site='blocks.0.4.conv1',tokens=1_000_000,spatial_volume=64**3)
        self.assertEqual(calls, [])

    def test_no_override_can_claim_full_workspace_admission(self):
        with self.assertRaises(TypeError):
            a.before_explicit_convolution(lambda: None,memory=a.Memory(0,0,16*a.GIB),site='blocks.0.0.conv',tokens=1,spatial_volume=64**3,workspace_bytes=0)


if __name__ == '__main__': unittest.main()
