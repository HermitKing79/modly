"""Small CPU fixtures only; these tests never read the private brick asset."""
import unittest
import numpy as np

from uv_layout import LayoutError, first_overlap, layout, validate_layout


class SeamLayoutTests(unittest.TestCase):
    def square(self):
        vertices = np.array([[0, 0, 0], [0, .02, 0],
                             [0, .02, .02], [0, 0, .02]], np.float32)
        faces = np.array([[0, 2, 1], [0, 3, 2]], np.int32)
        return vertices, faces

    def test_shared_edge_remains_one_island_and_complete_source_corners(self):
        vertices, faces = self.square()
        original = vertices.copy(), faces.copy()
        result = layout(vertices, faces, np.array([37, 12]),
                        source_vertex_ids=np.array([100, 101, 102, 103]), size=256)
        np.testing.assert_array_equal(result.vertices, vertices[faces].reshape(-1, 3))
        np.testing.assert_array_equal(result.faces, np.arange(6).reshape(2, 3))
        np.testing.assert_array_equal(result.source_face_ids, [37, 12])
        np.testing.assert_array_equal(result.source_vertex_ids,
                                      np.array([100, 101, 102, 103])[faces].ravel())
        np.testing.assert_array_equal(result.face_island_ids, [0, 0])
        np.testing.assert_array_equal(vertices, original[0])
        np.testing.assert_array_equal(faces, original[1])
        self.assertTrue((result.face_coverage >= 1).all())
        self.assertGreaterEqual(result.report['texels_per_world_unit'], 4096)
        self.assertIsNone(first_overlap(result.uv, result.faces))
        self.assertEqual(result.report['min_gutter_texels'], 4)
        self.assertEqual(result.report['positive_area_overlap_count'], 0)
        self.assertEqual(result.report['original_triangle_corners_bitwise_preserved'], True)

    def test_overlapping_connected_projection_is_split_at_uv_seam(self):
        # Both source triangles share edge 0->1 and overlap in YZ; no geometry edit.
        vertices = np.array([[0, 0, 0], [0, .02, 0], [0, 0, .02],
                             [.001, .005, .005]], np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], np.int32)
        result = layout(vertices, faces, np.array([5, 9]), size=256)
        self.assertEqual(len(result.island_rects), 2)
        self.assertNotEqual(result.face_island_ids[0], result.face_island_ids[1])
        self.assertFalse(np.array_equal(result.uv[0], result.uv[3]))
        np.testing.assert_array_equal(result.vertices[0], result.vertices[3])
        self.assertIsNone(first_overlap(result.uv, result.faces))
        for face_id in range(2):
            mask = result.raster_ids == face_id
            bary = result.barycentrics[mask].astype(np.float64)
            world = bary @ result.vertices[result.faces[face_id]].astype(np.float64)
            expected = bary @ vertices[faces[face_id]].astype(np.float64)
            np.testing.assert_array_equal(world, expected)
            np.testing.assert_allclose(bary.sum(1), 1, atol=1e-7)
        self.assertGreaterEqual(result.report['minimum_island_rectangle_separation_texels'], 8)

    def test_coincident_distinct_source_corners_and_duplicate_faces_not_dropped(self):
        v, f = self.square()
        vertices = np.concatenate((v, v))
        faces = np.vstack((f[0], f[0] + 4, f[0])).astype(np.int32)
        result = layout(vertices, faces, np.array([41, 42, 41]), size=256)
        self.assertEqual(len(result.faces), 3)
        self.assertEqual(len(result.island_rects), 3)
        np.testing.assert_array_equal(result.vertices[result.faces], vertices[faces])
        np.testing.assert_array_equal(result.source_face_ids, [41, 42, 41])
        self.assertTrue((result.face_coverage > 0).all())

    def test_overlap_detail_matches_analytic_clipped_area_and_allows_contact(self):
        uv = np.array([[0, 0], [1, 0], [0, 1], [.25, .25],
                       [1.25, .25], [.25, 1.25]], np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
        hit = first_overlap(uv, faces)
        self.assertEqual((hit['face_i'], hit['face_j']), (0, 1))
        self.assertAlmostEqual(hit['intersection_area'], .125)
        self.assertGreaterEqual(len(hit['intersection_polygon']), 3)
        square_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
        self.assertIsNone(first_overlap(square_uv, np.array([[0, 1, 2], [0, 2, 3]], np.int32)))

    def test_subtexel_positive_overlap_is_detected(self):
        uv = np.array([[.1000, .1000], [.1003, .1000], [.1000, .1003],
                       [.1001, .1001], [.1004, .1001], [.1001, .1004]], np.float64)
        hit = first_overlap(uv, np.arange(6, dtype=np.int32).reshape(2, 3))
        self.assertIsNotNone(hit)
        self.assertGreater(hit['intersection_area'], 1e-12)

    def test_zero_coverage_skinny_triangle_is_rejected_without_density_relaxation(self):
        vertices = np.array([[0, 0, 0], [0, .02, 0], [0, 0, 1e-10]], np.float32)
        with self.assertRaisesRegex(LayoutError, 'covered texel'):
            layout(vertices, np.array([[0, 1, 2]], np.int32), np.array([11]), size=256)

    def test_capacity_and_work_bounds_reject_and_do_not_rewrite_inputs(self):
        vertices, faces = self.square()
        with self.assertRaisesRegex(LayoutError, 'density floor'):
            layout(vertices, faces, np.array([1, 2]), size=256, min_density=100000)
        with self.assertRaisesRegex(LayoutError, 'face count'):
            layout(vertices, faces, np.array([1, 2]), size=256, max_faces=1)
        with self.assertRaisesRegex(LayoutError, 'island count'):
            layout(vertices, np.vstack((faces[0], faces[0])), np.array([1, 1]),
                   size=256, max_islands=1)
        with self.assertRaisesRegex(LayoutError, 'density floor'):
            layout(vertices, faces, np.array([1, 2]), size=256, min_density=1e300)

    def test_raster_work_is_predeclared_and_bounded(self):
        from unittest.mock import patch
        vertices, faces = self.square()
        with patch('uv_layout.MAX_RASTER_TEXELS', 1, create=True):
            with self.assertRaisesRegex(LayoutError, 'raster work'):
                layout(vertices, faces, np.array([1, 2]), size=256)

    def test_deterministic_and_validation_rejects_modified_uv(self):
        vertices, faces = self.square()
        one = layout(vertices, faces, np.array([2, 3]), size=256)
        two = layout(vertices, faces, np.array([2, 3]), size=256)
        np.testing.assert_array_equal(one.uv, two.uv)
        self.assertEqual(one.report, two.report)
        one.uv[one.faces[1]] = one.uv[one.faces[0]]
        with self.assertRaisesRegex(LayoutError, 'overlap'):
            validate_layout(one, vertices, faces)

    def test_source_face_storage_order_does_not_create_needless_disconnected_islands(self):
        vertices = np.array([[0, 0, 0], [0, .01, 0], [0, 0, .01],
                             [0, .01, .01], [0, .02, 0], [0, .02, .01]], np.float32)
        # First two stored triangles do not touch. Later ones connect the strip.
        faces = np.array([[0, 1, 2], [4, 5, 3], [1, 3, 2], [1, 4, 3]], np.int32)
        result = layout(vertices, faces, np.arange(4), size=256)
        self.assertEqual(len(result.island_rects), 1)
        np.testing.assert_array_equal(result.vertices[result.faces], vertices[faces])

    def test_signed_zero_corner_bits_and_opposite_winding_are_retained(self):
        vertices, faces = self.square()
        vertices[0, 0] = np.float32(-0.)
        faces[1] = faces[1, ::-1]
        result = layout(vertices, faces, np.array([8, 8]), size=256)
        np.testing.assert_array_equal(result.vertices.view(np.uint32),
                                      vertices[faces].reshape(-1, 3).view(np.uint32))
        self.assertEqual(len(result.island_rects), 2)

    def test_density_floor_accounts_for_x_slope_not_only_projected_yz(self):
        vertices = np.array([[0, 0, 0], [.02, .02, 0], [0, 0, .02]], np.float32)
        result = layout(vertices, np.array([[0, 1, 2]], np.int32), np.array([4]), size=256)
        projected_density = result.report['texels_per_world_unit']
        # A 45-degree sheet projects one sqrt(2)-long source direction shorter.
        self.assertGreaterEqual(projected_density / np.sqrt(2), 4096)
        self.assertGreaterEqual(result.report['minimum_3d_tangent_texels_per_world_unit'], 4096)
        with self.assertRaisesRegex(LayoutError, 'density floor'):
            layout(vertices, np.array([[0, 1, 2]], np.int32), np.array([4]), size=120)

    def test_revalidation_rejects_uv_scaled_below_actual_source_density_floor(self):
        vertices, faces = self.square()
        result = layout(vertices, faces, np.array([1, 2]), size=256)
        # Remain positive, bounded, nonoverlapping, and inside the same gutters.
        result.uv[:] = result.uv[0] + (result.uv - result.uv[0]) * np.float32(.01)
        with self.assertRaisesRegex(LayoutError, 'density floor'):
            validate_layout(result, vertices, faces)

    def test_revalidation_checks_source_id_correspondence(self):
        vertices, faces = self.square()
        sf, sv = np.array([18, 19]), np.array([71, 72, 73, 74])
        result = layout(vertices, faces, sf, sv, size=256)
        result.source_vertex_ids[0] = 999
        with self.assertRaisesRegex(LayoutError, 'source vertex ID'):
            validate_layout(result, vertices, faces, source_face_ids=sf, source_vertex_ids=sv)

    def test_invalid_geometry_winding_projection_and_metadata_are_rejected(self):
        vertices, faces = self.square()
        with self.assertRaisesRegex(LayoutError, 'F32'):
            layout(vertices.astype(float), faces, np.array([1, 2]))
        with self.assertRaisesRegex(LayoutError, 'source face'):
            layout(vertices, faces, np.array([1]))
        zero = vertices.copy(); zero[:, 2] = 0
        with self.assertRaisesRegex(LayoutError, 'degenerate YZ'):
            layout(zero, faces, np.array([1, 2]))
        with self.assertRaisesRegex(LayoutError, '1024'):
            layout(vertices, faces, np.array([1, 2]), size=2048)


if __name__ == '__main__':
    unittest.main()
