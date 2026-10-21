"""Independent synthetic saved-query qualification; no assets or Blender."""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).with_name('independent.py')
independent = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('independent_native_query_test', MODULE)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)

REGIONS = ('global', 'brick', 'foliage', 'rear_x', 'backfacing_x', 'boundary',
           'nonmanifold', 'high_variation', *(f'octant_{i}' for i in range(8)))


def fixture():
    data = {'vertices': np.zeros((3, 3), np.float32), 'faces': np.array([[0, 1, 2]], np.int32)}
    oracle = np.array([[0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255]], np.uint8)
    for count, held in ((56910, False), (2048, True)):
        bary = np.tile(np.array([.5, .5, 0.], np.float64), (count, 1))
        raw = np.tile(np.array([.7353569830524495] * 3 + [.00011082693761409814, .999995986777544, 1], np.float32), (count, 1))
        faces = np.zeros(count, np.int32)
        kinds = np.resize(np.array([4, 5] if held else [0, 1, 2, 3, 4, 5], np.int8), count)
        groups = {name: {'query_start': i * count // 16, 'query_stop': (i + 1) * count // 16}
                  for i, name in enumerate(REGIONS)}
        if held:
            data['resampled'] = {'holdout-query-faces': faces, 'holdout-barycentrics': bary,
                                 'holdout-query-kinds': kinds, 'holdout_source_spatial': raw,
                                 'holdout_weight_sum': np.ones(count, np.float32)}
            data['holdout_frozen'] = {'cohorts': groups}
        else:
            data['prior'] = {'query-faces': faces, 'barycentrics': bary, 'query-kinds': kinds,
                             'source_spatial': raw, 'raw_weight_sum': np.ones(count, np.float32)}
            data['frozen'] = {'cohorts': groups}
    return data, oracle


class Availability(unittest.TestCase):
    def test_independent_verifier_exists(self):
        self.assertIsNotNone(independent, 'Independent native query verifier not implemented')


@unittest.skipIf(independent is None, 'Independent verifier pending')
class NativeQueries(unittest.TestCase):
    def test_all_original_and_holdout_queries_and_112_rows_are_required(self):
        result = independent.verify(*fixture())
        self.assertTrue(result['all_gates_pass'])
        self.assertEqual(len(result['original_queries']['cohorts']), 64)
        self.assertEqual(len(result['holdout_queries']['cohorts']), 48)
        predicted = result['original_queries']['predicted_RGB']
        self.assertEqual(predicted.shape, (56910, 3))
        self.assertEqual(predicted.dtype, np.float64)
        self.assertLess(np.max(np.abs(predicted - .7353569830524495)), 1e-15)

    def test_rgb_and_each_scalar_gate_cannot_be_hidden_by_other_channels(self):
        for channel in (0, 3, 4, 5):
            data, oracle = fixture()
            data['resampled']['holdout_source_spatial'][:128, channel] = 0 if channel in (0, 4, 5) else 1
            result = independent.verify(data, oracle)
            self.assertFalse(result['all_gates_pass'])
            self.assertFalse(result['holdout_queries']['cohorts']['global:all']['pass'])
            self.assertTrue(result['original_queries']['all_gates_pass'])

    def test_queries_cohorts_positive_support_and_oracle_schema_fail_closed(self):
        changes = [lambda d, o: d['prior']['barycentrics'].__setitem__((0, 0), -1),
                   lambda d, o: d['prior']['barycentrics'].__setitem__((0, 0), np.nan),
                   lambda d, o: d['prior']['query-faces'].__setitem__(0, 1),
                   lambda d, o: d['prior']['raw_weight_sum'].__setitem__(0, 0),
                   lambda d, o: d['resampled']['holdout-query-kinds'].__setitem__(0, 0),
                   lambda d, o: d['frozen']['cohorts'].pop('brick'),
                   lambda d, o: d['frozen']['cohorts']['brick'].update(query_start=0),
                   lambda d, o: d['holdout_frozen']['cohorts']['global'].update(query_stop=0),
                   lambda d, o: o.__setitem__((0, 3), 0)]
        for change in changes:
            data, oracle = fixture(); change(data, oracle)
            with self.subTest(change=change), self.assertRaises(ValueError):
                independent.verify(data, oracle)

    def test_optional_actual_native_getter_rgb_replaces_the_iec_decode(self):
        data, oracle = fixture()
        native = np.array([[0, 0, 0], [.25, .25, .25], [0, 0, 0]], np.float32)
        result = independent.verify(data, oracle, decoded_linear=native)
        self.assertFalse(result['all_gates_pass'])
        self.assertLess(np.max(np.abs(result['original_queries']['predicted_RGB'] - .3885728590463344)), 1e-15)
        for raw in (data['prior']['source_spatial'], data['resampled']['holdout_source_spatial']):
            raw[:, :3] = .3885728590463344
        self.assertTrue(independent.verify(data, oracle, decoded_linear=native)['all_gates_pass'])
        for invalid in (native.astype(np.float64), np.full_like(native, np.nan), np.full_like(native, 2)):
            with self.assertRaises(ValueError):
                independent.verify(data, oracle, decoded_linear=invalid)

    def test_linear_p95_interpolates_adjacent_order_statistics(self):
        row = independent._stats(np.array([[0., .01], [.02, .03]], np.float64), np)
        self.assertAlmostEqual(row['rmse'], .018708286933869705, places=16)
        self.assertAlmostEqual(row['p95_abs'], .0285, places=16)
        self.assertEqual(row['max_abs'], .03)


if __name__ == '__main__':
    unittest.main(verbosity=2)
