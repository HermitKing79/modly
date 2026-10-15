"""Focused CPU preparation regressions; no model, Torch, native or GPU imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoder_cpu as candidate
import numpy as np


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = candidate.prepare(candidate.PROFILE)

    def test_exact_optin_and_authenticated_contract_required(self):
        for value in (None, '', 'brick-1957891-v2', True):
            with self.assertRaises(ValueError):
                candidate.authenticate(value, self.prepared['contract'].document())
        document = self.prepared['contract'].document()
        for key, value in [('maximum_final_rows', 1842968), ('source_job_sha256', 'f'*64),
                           ('large_convolution_tiling', False), ('initial_count', 4431),
                           ('output_directory', '/home/modly/foreign/geometry')]:
            with self.assertRaises(ValueError):
                candidate.authenticate(candidate.PROFILE, dict(document, **{key: value}))

    def test_genuine_saved_guides_and_texture_order_are_exact(self):
        proof = self.prepared['proof']
        self.assertEqual(proof['parent_rows'], [4430, 21391, 97313, 438684])
        self.assertEqual(proof['child_rows'], [21391, 97313, 438684, 1957891])
        self.assertTrue(proof['all_four_masks_and_original_child_orders_verified'])
        self.assertEqual(proof['initial_coordinates_sha256'], candidate.COORDS_SHA)
        self.assertFalse(self.prepared['texture_coords'].flags.writeable)

    def test_nonfinite_or_reordered_guides_fail(self):
        bits = self.prepared['guides'][0]['logits_bf16'].copy()
        bits[0, 0] = 0x7f80
        with self.assertRaises(ValueError):
            candidate.inputs.bf16_to_f32(bits)
        stages = [(g['coords'], g['logits_f32']) for g in self.prepared['guides']]
        altered = stages[0][1].copy()
        altered[0, 0] = np.float32(-1 if altered[0, 0] > 0 else 1)
        with self.assertRaises(ValueError):
            candidate.contracts.validate_chain([(stages[0][0], altered), *stages[1:]],
                self.prepared['geometry_coords'], 4430, contract=self.prepared['contract'])
        coords = stages[0][0].copy()
        coords[[0, 1]] = coords[[1, 0]]
        with self.assertRaises(ValueError):
            candidate.contracts.validate_chain([(coords, stages[0][1]), *stages[1:]],
                self.prepared['geometry_coords'], 4430, contract=self.prepared['contract'])

    def test_exact_f16_lower_bound_blocks_final_only_tiling(self):
        report = candidate.memory_profile(self.prepared['contract'])
        self.assertEqual(report['runtime_model_parameter_bytes'], 948567448)
        self.assertEqual(report['decisive_conv1']['visible_bytes'], 8511785652)
        self.assertEqual(report['decisive_conv1']['padded_bytes'], 10639732065)
        self.assertEqual(report['decisive_conv1']['unavoidable_retained_lower_bound_bytes'], 1173173656)
        self.assertEqual(report['decisive_conv1']['minimum_cap_excess_bytes'], 1745657)
        self.assertFalse(report['final_only_tiling_admissible'])
        self.assertEqual((report['final']['chunks'], report['final']['last_chunk_rows']), (60, 24579))
        self.assertEqual(len(report['convolution_plans']), 40)
        self.assertEqual(len(report['c2s_plans']), 8)
        self.assertEqual(report['expected_native_map_calls'], 5)

    def test_historical_admission_and_artifact_consumers_reject_brick(self):
        historical, field = candidate.historical, candidate.field
        self.assertEqual((historical.FINAL_ROWS, historical.FINAL_PARENTS, field.MAX_ROWS),
                         (1842968, 378891, 1842968))
        with self.assertRaises(historical.AdmissionStop):
            historical.final_c2s_plan(438684, 512, 128, 1957891, 64)
        with self.assertRaises(historical.AdmissionStop):
            historical.chunked_conv_plan(1957891, 32768, 512**3)
        synthetic = np.broadcast_to(np.zeros((1, 6), dtype='<f4'), (1957891, 6))
        with self.assertRaises(ValueError):
            field._row_count(synthetic)
        header = field._json_bytes({'raw': {'shape': [1957891, 6]}, 'features': {},
                                    'coordinates': {}, '__metadata__': {}})
        header += b' ' * (-len(header) % 8)
        with self.assertRaisesRegex(ValueError, 'row count exceeds bound'):
            field._container(len(header).to_bytes(8, 'little') + header + b'0')
        # Exercise the public writer/independent-reader entrypoints too. The
        # writer refuses before any file is created; the malformed reader input
        # is only a tiny explicitly SYNTHETIC header, never a material output.
        approved = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688-tests').resolve()
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-pbr-consumer-', dir=approved) as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(approved))
            with self.assertRaisesRegex(ValueError, 'bounded nonempty'):
                field.write(root, synthetic, synthetic, self.prepared['geometry_coords'],
                            {'mode':'production','synthetic':True})
            self.assertEqual(list(root.iterdir()), [])
            (root/field.ARTIFACT_NAME).write_bytes(len(header).to_bytes(8, 'little') + header + b'0')
            (root/field.METADATA_NAME).write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'row count exceeds bound'):
                field.verify(root/field.ARTIFACT_NAME, root)

    def test_original_finite_and_exact_output_transform_checks_retained(self):
        raw = np.zeros((2, 6), dtype='<f4')
        features = raw * np.float32(.5) + np.float32(.5)
        candidate.field._array('raw', raw, 2)
        candidate.field._transform(raw, features)
        raw[0, 0] = np.inf
        with self.assertRaises(ValueError):
            candidate.field._array('raw', raw, 2)
        raw[0, 0] = 0
        features[0, 0] = 0
        with self.assertRaises(ValueError):
            candidate.field._transform(raw, features)


if __name__ == '__main__':
    unittest.main(verbosity=2)
