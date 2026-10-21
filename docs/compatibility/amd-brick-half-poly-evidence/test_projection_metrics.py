"""Synthetic CPU projection/raster fixtures; no model assets or Blender."""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).with_name('projection_metrics.py')
metrics = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('half_poly_projection_metrics_test', MODULE)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)


def fixture():
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    camera = dict(camera_type='ORTHO', camera_location=[0, 0, 3], camera_right=[1, 0, 0],
                  camera_up=[0, 1, 0], axis_direction=[0, 0, 1], ortho_scale=2, clip=[.1, 10])
    return vertices, faces, camera


class Availability(unittest.TestCase):
    def test_projection_metrics_exist(self):
        self.assertIsNotNone(metrics, 'Independent bounded projection metrics not implemented')


@unittest.skipIf(metrics is None, 'Projection implementation pending')
class RasterTests(unittest.TestCase):
    def test_exact_pixel_centres_both_windings_and_chunk_boundaries(self):
        v, f, camera = fixture(); projected = metrics.project(v, camera, size=8)
        expected = np.zeros((8, 8), np.bool_); expected[2:6, 2:6] = True
        first = metrics.raster(v, f, projected, foliage_faces=np.array([0, 1], np.int32), chunk_events=1)
        second = metrics.raster(v, f[:, ::-1], projected, foliage_faces=np.array([0, 1], np.int32))
        np.testing.assert_array_equal(first['mask'], expected)
        np.testing.assert_array_equal(first['mask'], second['mask'])
        np.testing.assert_array_equal(first['depth'], second['depth'])
        np.testing.assert_array_equal(first['visible_foliage'], expected)
        self.assertTrue(np.all(first['depth'][expected] == 3))
        self.assertTrue(np.all(first['depth'][~expected] == 0))
        result = metrics.compare(first, second, 1.)
        self.assertTrue(result['pass'])
        self.assertEqual(result['silhouette_IoU'], 1.)
        self.assertEqual(result['foliage_visible_pixel_retention'], 1.)

    def test_brick_fill_at_wrong_depth_cannot_hide_lost_visible_foliage(self):
        v, f, camera = fixture()
        v = np.concatenate((v + np.array([0, 0, .2], np.float32), v))
        projected = metrics.project(v, camera, size=8)
        source = metrics.raster(v, np.concatenate((f, f + 4)), projected,
                                foliage_faces=np.array([0, 1], np.int32))
        candidate = metrics.raster(v, f + 4, projected)
        row = metrics.compare(source, candidate, 1.)
        self.assertEqual(row['silhouette_IoU'], 1.)
        self.assertEqual(row['foliage_visible_pixel_retention'], 0.)
        self.assertFalse(row['pass'])

    def test_edge_on_3d_triangle_is_counted_without_invented_coverage(self):
        _, _, camera = fixture()
        v = np.array([[0, -.5, 0], [0, .5, 0], [0, 0, .5]], np.float32)
        row = metrics.raster(v, np.array([[0, 1, 2]], np.int32), metrics.project(v, camera, size=8))
        self.assertEqual(row['projected_edge_on_triangles'], 1)
        self.assertFalse(row['mask'].any())

    def test_invalid_clipping_nonfinite_3d_degeneracy_and_work_budgets_fail(self):
        v, f, camera = fixture()
        for size in (0, 8., True, 641):
            with self.subTest(size=size), self.assertRaises(ValueError):
                metrics.project(v, camera, size=size)
        for position in ([2, 0, 0], [0, 0, 3], [np.inf, 0, 0], [np.nan, 0, 0]):
            bad = v.copy(); bad[0] = position
            with self.subTest(position=position), self.assertRaises(ValueError):
                metrics.project(bad, camera, size=8)
        projected = metrics.project(v, camera, size=8)
        for bad_faces in (np.array([[0, 0, 1]], np.int32), np.array([[0, 1, 9]], np.int32), f.astype(np.float32)):
            with self.subTest(faces=bad_faces), self.assertRaises(ValueError):
                metrics.raster(v, bad_faces, projected)
        for kwargs in ({'max_events': 1}, {'max_events': 16777217}, {'chunk_events': 0}, {'chunk_events': 262145}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                metrics.raster(v, f, projected, **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
