"""Small synthetic fixtures; never load the production mesh or render."""
import unittest

import numpy as np

from projection import build_triangle_bounds, edge_counts, project, sample_rays, visible_edge_samples


CAMERA = {
    "camera_type": "ORTHO", "camera_location": [3., 0., 0.],
    "camera_right": [0., 1., 0.], "camera_up": [0., 0., 1.],
    "axis_direction": [1., 0., 0.], "ortho_scale": 2., "clip": [0.1, 10.],
}


class ProjectionTests(unittest.TestCase):
    def test_exact_axis_order_and_downward_png_y(self):
        vertices = np.array([[1., 0., 0.], [1., 1., 1.], [1., -1., -1.]])
        p = project(vertices, CAMERA, 100, 100)
        np.testing.assert_array_equal(p.screen, [[50., 50.], [100., 0.], [0., 100.]])
        np.testing.assert_array_equal(p.depth, [2., 2., 2.])

    def test_reject_unverified_projection_contracts(self):
        for bad in ({**CAMERA, "camera_type": "PERSP"},
                    {**CAMERA, "camera_right": [0., -1., 0.]}):
            with self.assertRaises(ValueError):
                project(np.zeros((1, 3)), bad, 100, 100)
        with self.assertRaises(ValueError):
            project(np.zeros((1, 3)), CAMERA, 100, 50)

    def test_actual_metadata_values_and_original_coordinate_order(self):
        camera = {**CAMERA, "camera_location": [2.797344923019409, -0.0018387436866760254,
                                                0.0007975846529006958],
                  "ortho_scale": 1.5038471221923828}
        center = np.array(camera["camera_location"])
        vertices = np.array([center + [-2., 0., 0.],
                             center + [-2., camera["ortho_scale"] / 4, 0.]])
        p = project(vertices, camera, 1024, 1024)
        np.testing.assert_allclose(p.screen, [[512., 512.], [768., 512.]])
        np.testing.assert_allclose(p.depth, [2., 2.])


class RayTests(unittest.TestCase):
    def setUp(self):
        # At x=1, +X-facing winding. Center ray barycentric = (.25,.25,.5).
        self.vertices = np.array([[1., -1., -1.], [1., 1., -1.], [1., 0., 1.]])
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.p = project(self.vertices, CAMERA, 100, 100)

    def test_pixel_centers_are_offset_once(self):
        r = sample_rays(self.vertices, self.faces, self.p, [[49, 49]])[0]
        self.assertEqual(r["sample_xy"], [49.5, 49.5])
        self.assertTrue(r["hit"])
        # Explicit subpixel geometry samples must not receive another +0.5.
        exact = sample_rays(self.vertices, self.faces, self.p, [[50., 50.]], pixel_centers=False)[0]
        np.testing.assert_allclose(exact["barycentric"], [.25, .25, .5])
        self.assertEqual(exact["depth"], 2.)
        np.testing.assert_allclose(exact["position"], [1., 0., 0.])

    def test_miss_is_distinct_from_dark_material_hit(self):
        raw = np.zeros((3, 6))
        records = sample_rays(self.vertices, self.faces, self.p, [[50., 50.], [1., 1.]],
                              raw, pixel_centers=False)
        self.assertTrue(records[0]["hit"])
        self.assertFalse(records[0]["empty_ray"])
        self.assertEqual(records[0]["raw_rgb"], [0., 0., 0.])
        self.assertFalse(records[1]["hit"])
        self.assertTrue(records[1]["empty_ray"])
        self.assertIsNone(records[1]["raw_rgb"])

    def test_interpolation_matches_per_vertex_native_gamma_conversion(self):
        rgb = np.eye(3)
        r = sample_rays(self.vertices, self.faces, self.p, [[50., 50.]], rgb,
                        pixel_centers=False)[0]
        np.testing.assert_allclose(r["raw_rgb"], [.25, .25, .5])
        np.testing.assert_allclose(r["linear_rgb"], [.25, .25, .5])
        self.assertFalse(np.allclose(r["linear_rgb"], np.array(r["raw_rgb"]) ** 2.2))
        self.assertAlmostEqual(r["normal_cos_camera"], 1.)

    def test_closest_hit_is_original_face_id_and_winding_is_signed(self):
        vertices = np.vstack((self.vertices, self.vertices + [0.5, 0., 0.]))
        faces = np.array([[0, 1, 2], [5, 4, 3]])
        p = project(vertices, CAMERA, 100, 100)
        r = sample_rays(vertices, faces, p, [[50., 50.]], pixel_centers=False, chunk_size=1)[0]
        self.assertEqual(r["face_id"], 1)
        self.assertAlmostEqual(r["depth"], 1.5)
        self.assertAlmostEqual(r["normal_cos_camera"], -1.)
        self.assertAlmostEqual(r["abs_normal_cos_camera"], 1.)
        self.assertEqual(r["intersection_count"], 2)

    def test_conversion_uses_original_float32_attribute_arithmetic(self):
        rgb = np.array([[.1234567, .2345678, .3456789],
                        [.4567891, .5678912, .6789123],
                        [.7891234, .8912345, .9123456]], dtype=np.float32)
        r = sample_rays(self.vertices, self.faces, self.p, [[50., 50.]], rgb,
                        pixel_centers=False)[0]
        expected = np.array([.25, .25, .5]) @ (np.clip(rgb, 0., 1.) ** np.float32(2.2)).astype(np.float64)
        np.testing.assert_array_equal(r["linear_rgb"], expected)

    def test_near_clip_prevents_hit(self):
        vertices = self.vertices.copy()
        vertices[:, 0] = 2.99
        p = project(vertices, CAMERA, 100, 100)
        r = sample_rays(vertices, self.faces, p, [[50., 50.]], pixel_centers=False)[0]
        self.assertFalse(r["hit"])
        self.assertEqual(r["clipped_intersection_count"], 1)
        self.assertIn("intersections_outside_clip", r["ambiguities"])

    def test_projected_degenerate_triangle_is_explicit(self):
        vertices = np.array([[1., 0., -1.], [2., 0., 0.], [1., 0., 1.]])
        p = project(vertices, CAMERA, 100, 100)
        r = sample_rays(vertices, self.faces, p, [[50., 50.]], pixel_centers=False)[0]
        self.assertFalse(r["hit"])
        self.assertEqual(r["projected_degenerate_candidate_count"], 1)
        self.assertIn("projected_degenerate_candidates", r["ambiguities"])

    def test_tied_faces_remain_explicit_and_selection_is_deterministic(self):
        faces = np.repeat(self.faces, 2, axis=0)
        r = sample_rays(self.vertices, faces, self.p, [[50., 50.]],
                        pixel_centers=False, chunk_size=1)[0]
        self.assertEqual(r["face_id"], 0)
        self.assertEqual(r["nearest_depth_tie_face_ids"], [0, 1])
        self.assertIn("coincident_nearest_hits", r["ambiguities"])

    def test_original_edge_incidence_and_geometric_edge_distance(self):
        # Opposite vertices 0,1,2 have edge incidences 3,2,1 respectively.
        keys = np.array([1, 2, 5], dtype=np.uint64)
        counts = np.array([1, 2, 3], dtype=np.uint32)
        incidence = (keys, counts, 3)
        self.assertEqual(edge_counts(np.array([[2, 1], [2, 0], [0, 1]]), incidence).tolist(), [3, 2, 1])
        r = sample_rays(self.vertices, self.faces, self.p, [[50., 100.]],
                        pixel_centers=False, edge_incidence=incidence)[0]
        self.assertTrue(r["hit"])
        self.assertEqual(r["face_edge_incidence_opposite_vertex"], [3, 2, 1])
        self.assertEqual(r["nearest_edge_vertices"], [0, 1])
        self.assertEqual(r["nearest_edge_incidence"], 1)
        self.assertEqual(r["nearest_edge_distance_px"], 0.)
        self.assertIn("on_triangle_edge", r["ambiguities"])

    def test_bounds_cache_is_conservative_and_reusable(self):
        bounds = build_triangle_bounds(self.faces, self.p, chunk_size=1)
        r = sample_rays(self.vertices, self.faces, self.p, [[50., 50.]],
                        pixel_centers=False, bounds=bounds)[0]
        self.assertTrue(r["hit"])
        self.assertEqual(bounds.minimum.dtype, np.dtype("float32"))

    def test_visibility_discards_rear_edge_and_keeps_front_edge(self):
        vertices = np.vstack((self.vertices, self.vertices + [-0.5, 0., 0.]))
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        p = project(vertices, CAMERA, 100, 100)
        records = visible_edge_samples(vertices, faces, p, np.array([[0, 1], [3, 4]]),
                                       np.array([1, 3]), (0., 0., 100., 101.), max_edges=4)
        self.assertEqual(len(records), 2)
        by_edge = {tuple(x["vertices"]): x for x in records}
        self.assertTrue(by_edge[(0, 1)]["visible"])
        self.assertFalse(by_edge[(3, 4)]["visible"])
        self.assertEqual(by_edge[(3, 4)]["occlusion_depth"], 0.5)


if __name__ == "__main__":
    unittest.main()
