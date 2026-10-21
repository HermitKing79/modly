"""Synthetic-only checks for original-face strata and bounded query plans."""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).with_name('strata.py')
if MODULE.is_file():
    spec = importlib.util.spec_from_file_location('full_colour_strata_tested', MODULE)
    strata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strata)
else:
    strata = None


def separate_triangles(count=200):
    vertices = []
    for i in range(count):
        x, y, z = (float((i >> b) & 1) * 4 - 2 for b in range(3))
        x += (i // 8) * .001
        vertices.extend([[x, y, z], [x, y + .2, z], [x, y, z + .2]])
    vertices = np.asarray(vertices, np.float32)
    faces = np.arange(3 * count, dtype=np.int32).reshape(-1, 3)
    # Half the triangles genuinely face negative X, with original winding kept.
    faces[::2] = faces[::2, ::-1]
    encoded = np.zeros_like(vertices)
    encoded[faces[5, 0], 0] = 1
    encoded[faces[7, 0], 0] = 1
    if count > 8:
        encoded[faces[8, 0], 0] = .9
    return vertices, faces, encoded


class StrataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(strata, 'Pure original-face strata implementation is required')

    def test_scan_distinguishes_global_boundary_and_nonmanifold_edges(self):
        v = np.array([[0,0,0], [0,1,0], [0,0,1], [0,0,-1], [1,0,0],
                      [3,3,3], [4,3,3], [3,4,3], [3,3,4]], np.float32)
        f = np.array([[0,1,2], [1,0,3], [0,1,4],
                      [5,7,6], [5,6,8], [5,8,7], [6,7,8]], np.int32)
        rgb = np.zeros_like(v); rgb[1] = [2, -1, .25]
        before = [a.copy() for a in (v, f, rgb)]
        result = strata.scan(v, f, rgb)
        np.testing.assert_array_equal(result['boundary'], [1,1,1,0,0,0,0])
        np.testing.assert_array_equal(result['nonmanifold'], [1,1,1,0,0,0,0])
        np.testing.assert_allclose(result['areas'], [.5,.5,.5,.5,.5,.5,np.sqrt(3)/2])
        self.assertAlmostEqual(float(result['normal_x'][0]), 1)
        self.assertAlmostEqual(float(result['normal_x'][1]), 1)
        self.assertEqual(float(result['span'][0]), 1)
        np.testing.assert_allclose(result['centers'][0], [0,1/3,1/3])
        for actual, original in zip((v, f, rgb), before):
            np.testing.assert_array_equal(actual, original)

    def test_scan_rejects_invalid_indices_degenerate_or_nonfinite_inputs(self):
        v, f, rgb = separate_triangles(8)
        cases = []
        bad = f.copy(); bad[0, 0] = -1; cases.append((v, bad, rgb))
        bad = f.copy(); bad[0, 0] = len(v); cases.append((v, bad, rgb))
        bad = f.copy(); bad[0, 1] = bad[0, 0]; cases.append((v, bad, rgb))
        bad = v.copy(); bad[0, 0] = np.nan; cases.append((bad, f, rgb))
        bad = rgb.copy(); bad[0, 0] = np.inf; cases.append((v, f, bad))
        cases.append((v.astype(np.float64), f, rgb))
        cases.append((v, f.astype(np.int64), rgb))
        for args in cases:
            with self.subTest(k=len(cases), dtype=args[1].dtype):
                with self.assertRaises(ValueError):
                    strata.scan(*args)

    def test_edge_incidence_is_global_across_geometry_batches(self):
        v = np.array([[0,0,0], [0,1,0], [0,0,1]], np.float32)
        f = np.tile(np.array([[0,1,2]], np.int32), (32769,1))
        result = strata.scan(v, f, np.zeros_like(v))
        self.assertFalse(result['boundary'].any())
        self.assertTrue(result['nonmanifold'].all())
        self.assertEqual(result['edge_counts']['maximum_incidence'], 32769)

    def test_area_weighted_choice_keeps_dominant_area_face(self):
        data = strata.scan(*separate_triangles(10))
        data['areas'][:] = 1e-30
        data['areas'][9] = 1
        budgets = {name: 0 for name in strata.DEFAULT_BUDGETS}
        budgets['global'] = 1
        result = strata.select(data, np.array([], np.int32), np.array([], np.int32), seed=12345, budgets=budgets)
        np.testing.assert_array_equal(result['cohorts']['global']['selected_face_ids'], [9])
        self.assertEqual(result['cohorts']['global']['selected_area'], 1)

    def test_high_variation_uses_full_input_rank_with_stable_original_ids(self):
        data = strata.scan(*separate_triangles())
        budgets = {name: 0 for name in strata.DEFAULT_BUDGETS}
        budgets.update(brick=2, high_variation=10)
        result = strata.select(data, np.array([3,1], np.int32), np.array([], np.int32), budgets=budgets)
        np.testing.assert_array_equal(result['cohorts']['brick']['selected_face_ids'], [1,3])
        row = result['cohorts']['high_variation']
        self.assertEqual(row['pool_count'], 2)
        self.assertEqual(row['selected_count'], 2)
        np.testing.assert_array_equal(row['selected_face_ids'], [5,7])
        self.assertEqual(result['total_queries'], 24)

    def test_query_blocks_keep_corner_identity_and_seeded_interior_points(self):
        data = strata.scan(*separate_triangles(40))
        budgets = {name: 0 for name in strata.DEFAULT_BUDGETS}
        budgets.update(global_=0)  # Invalid names must never silently add a cohort.
        with self.assertRaises(ValueError):
            strata.select(data, np.array([], np.int32), np.array([], np.int32), budgets=budgets)
        del budgets['global_']; budgets.update({'global': 7, 'brick': 2, 'foliage': 4})
        args = (data, np.array([9,2], np.int32), np.array([1,4,6], np.int32))
        result = strata.select(*args, seed=12345, budgets=budgets)
        repeated = strata.select(*args, seed=12345, budgets=budgets)
        self.assertEqual(result['total_queries'], 72)
        np.testing.assert_array_equal(result['query_face'], repeated['query_face'])
        np.testing.assert_array_equal(result['bary'], repeated['bary'])
        blocks = result['bary'].reshape(-1,6,3)
        np.testing.assert_array_equal(blocks[:, :3], np.broadcast_to(np.eye(3), blocks[:, :3].shape))
        np.testing.assert_allclose(blocks[:,3], np.full((12,3), 1/3))
        self.assertTrue(np.all(blocks[:,4:] > 0))
        np.testing.assert_allclose(blocks.sum(axis=2), 1, atol=2e-15)
        self.assertEqual(result['cohorts']['foliage']['pool_count'], 3)
        self.assertEqual(result['cohorts']['foliage']['selected_count'], 3)
        self.assertEqual(result['cohorts']['foliage']['query_stop'], 72)
        self.assertEqual(result['cohorts']['brick']['query_start'], 42)
        self.assertEqual(result['query_face'].dtype, np.int32)
        self.assertEqual(result['bary'].dtype, np.float64)

    def test_uniform_endpoints_cannot_create_corner_or_edge_probes(self):
        self.assertTrue(hasattr(strata, '_interior_barycentrics'), 'Strict interior endpoint guard required')
        uv = np.array([[[0.,0.], [np.nextafter(1.,0.),np.nextafter(1.,0.)]]])
        bary = strata._interior_barycentrics(uv)
        self.assertTrue(np.all(bary > 0))
        self.assertTrue(np.all(bary < 1))
        np.testing.assert_allclose(bary.sum(axis=2), 1, atol=2e-15)

    def test_geometric_pools_and_area_coverage_are_explicit(self):
        data = strata.scan(*separate_triangles(40))
        budgets = {name: 100 for name in strata.DEFAULT_BUDGETS}
        result = strata.select(data, np.array([0], np.int32), np.array([1], np.int32), budgets=budgets)
        rear = result['cohorts']['rear_x']
        # The quarter threshold includes all equal-X ties, not an arbitrary 10 IDs.
        self.assertEqual(rear['pool_count'], 12)
        self.assertLessEqual(data['centers'][rear['selected_face_ids'],0].max(), result['thresholds']['rear_x_q25'])
        back = result['cohorts']['backfacing_x']['selected_face_ids']
        self.assertTrue(np.all(data['normal_x'][back] < 0))
        self.assertTrue(np.all(data['centers'][back,0] < result['thresholds']['center_x_median']))
        all_octants = np.concatenate([result['cohorts'][f'octant_{i}']['selected_face_ids'] for i in range(8)])
        np.testing.assert_array_equal(np.sort(all_octants), np.arange(40))
        self.assertEqual(rear['area_fraction'], 1)
        self.assertAlmostEqual(rear['selected_area'], rear['pool_area'])

    def test_refuses_semantic_id_deletion_and_excess_query_budgets(self):
        data = strata.scan(*separate_triangles(20))
        budgets = {name: 0 for name in strata.DEFAULT_BUDGETS}
        budgets['brick'] = 1
        with self.assertRaises(ValueError):
            strata.select(data, np.array([0,1], np.int32), np.array([], np.int32), budgets=budgets)
        with self.assertRaises(ValueError):
            strata.select(data, np.array([0,0], np.int32), np.array([], np.int32))
        with self.assertRaises(ValueError):
            strata.select(data, np.array([20], np.int32), np.array([], np.int32))
        budgets['global'] = 65536
        with self.assertRaises(ValueError):
            strata.select(data, np.array([], np.int32), np.array([], np.int32), budgets=budgets)


if __name__ == '__main__':
    unittest.main()
