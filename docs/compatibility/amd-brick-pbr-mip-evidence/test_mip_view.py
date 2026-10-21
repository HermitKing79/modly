"""Synthetic-only CPU projection checks; never opens original assets."""
import unittest

import numpy as np

from mip_view import project


def plane(x=0.0):
    vertices = np.array([[x, 0, 0], [x, 1, 0], [x, 1, 1], [x, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uv = np.array([[0, 0], [.25, 0], [.25, .25], [0, .25]], np.float32)
    return vertices, faces, uv


class ProjectionTests(unittest.TestCase):
    def test_pixel_centres_barycentric_uv_and_plus_z_up(self):
        vertices, faces, uv = plane()
        result = project(vertices, faces, uv, 32, frame=64)
        y, x = result['pixels_yx'].T
        expected = np.column_stack(((x + .5 - 16) / 32, (48 - y - .5) / 32)) * .25
        np.testing.assert_allclose(result['query_uv'], expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result['mask'].sum(), 1024)
        self.assertGreater(result['query_uv'][y.argmin(), 1], result['query_uv'][y.argmax(), 1])
        self.assertEqual(result['metadata']['actual_projected_height_pixels'], 32)
        self.assertEqual(result['query_uv'].dtype, np.float64)
        self.assertEqual(result['query_faces'].dtype, np.int32)

    def test_double_sided_winding_and_input_arrays_unchanged(self):
        vertices, faces, uv = plane()
        originals = [x.copy() for x in (vertices, faces, uv)]
        first = project(vertices, faces, uv, 32, frame=64)
        second = project(vertices, faces[:, ::-1].copy(), uv, 32, frame=64)
        np.testing.assert_array_equal(first['mask'], second['mask'])
        np.testing.assert_allclose(first['query_uv'], second['query_uv'], rtol=0, atol=1e-14)
        for actual, original in zip((vertices, faces, uv), originals):
            np.testing.assert_array_equal(actual, original)

    def test_plus_x_nearest_layer_wins_regardless_of_face_order(self):
        rear, faces, uv = plane(-2)
        front, _, _ = plane(1)
        vertices = np.vstack((rear, front))
        all_faces = np.vstack((faces, faces + 4))
        values = np.vstack((uv, uv + np.float32(.5)))
        result = project(vertices, all_faces, values, 32, frame=64)
        self.assertTrue(np.all(result['query_faces'] >= 2))
        self.assertTrue(np.all(result['query_uv'] > .5))
        reversed_result = project(vertices, all_faces[::-1].copy(), values, 32, frame=64)
        np.testing.assert_allclose(result['query_uv'], reversed_result['query_uv'], rtol=0, atol=1e-14)

    def test_fourfold_projected_reduction_adds_two_lod_and_preserves_isotropy(self):
        vertices, faces, uv = plane()
        near = project(vertices, faces, uv, 32, frame=64)
        far = project(vertices, faces, uv, 8, frame=64)
        np.testing.assert_allclose(near['query_lod'], 3, rtol=0, atol=1e-14)
        np.testing.assert_allclose(far['query_lod'], 5, rtol=0, atol=1e-14)
        self.assertAlmostEqual(far['metadata']['LOD_max'] - near['metadata']['LOD_max'], 2)
        self.assertAlmostEqual(near['metadata']['anisotropy_max'], 1)
        self.assertEqual(far['metadata']['fragment_count'], 64)

    def test_foreign_shapes_indices_nan_uv_frame_and_unsupported_lod_reject(self):
        vertices, faces, uv = plane()
        bad_cases = [
            (vertices.astype(np.int32), faces, uv, 32, 64),
            (vertices, faces.astype(np.int64), uv, 32, 64),
            (vertices, np.array([[0, 1, 99]], np.int32), uv, 32, 64),
            (vertices, np.array([[0, 0, 2]], np.int32), uv, 32, 64),
            (vertices * np.float32(np.nan), faces, uv, 32, 64),
            (vertices, faces, np.full((4, 2), np.nan), 32, 64),
            (vertices, faces, uv + 1, 32, 64),
            (vertices, faces, uv, 32, 641),
            (vertices, faces, uv, 64, 64),
            (vertices, faces, uv, 1, 64),
        ]
        for args in bad_cases:
            with self.subTest(shape=[np.shape(v) for v in args[:3]], height=args[3], frame=args[4]):
                with self.assertRaises(ValueError):
                    project(*args[:3], object_height=args[3], frame=args[4])

    def test_bbox_work_budget_rejects_before_raster_allocation(self):
        vertices, faces, uv = plane()
        many = np.tile(faces[:1], (70, 1))
        with self.assertRaisesRegex(ValueError, 'work budget'):
            project(vertices, many, uv, 512, frame=640)

    def test_edge_on_faces_are_counted_without_visible_surface_substitution(self):
        vertices, faces, uv = plane()
        vertices = np.vstack((vertices, [[0, 2, 0], [1, 2, 0], [0, 2, 1]])).astype(np.float32)
        faces = np.vstack((faces, [[4, 5, 6]])).astype(np.int32)
        uv = np.vstack((uv, [[.5, 0], [.6, 0], [.5, .1]])).astype(np.float32)
        result = project(vertices, faces, uv, 16, frame=64)
        self.assertEqual(result['metadata']['edge_on_projected_faces'], 1)
        self.assertFalse(np.any(result['query_faces'] == 2))


if __name__ == '__main__':
    unittest.main()
