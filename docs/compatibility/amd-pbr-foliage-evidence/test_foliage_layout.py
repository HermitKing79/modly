"""Synthetic CPU fixtures only: no asset loading, GPU, Blender, or baking."""
import unittest
import numpy as np

try:
    from foliage_layout import build_layout, LayoutError
except ModuleNotFoundError:
    build_layout, LayoutError = None, ValueError


class FoliageLayoutTests(unittest.TestCase):
    def build(self, vertices, faces, sf=None, sv=None):
        self.assertIsNotNone(build_layout, 'The foliage feasibility helper is missing')
        return build_layout(vertices, faces, np.arange(len(faces)) if sf is None else sf,
                            np.arange(len(vertices)) if sv is None else sv)

    def leaf(self):
        angles = np.linspace(0, np.pi, 7)
        v = np.array([[.009 * np.cos(a), side, .009 * np.sin(a)]
                      for a in angles for side in (-.003, .003)], np.float32)
        f = np.array([[2*i, 2*i+1, 2*i+2] for i in range(6)] +
                     [[2*i+1, 2*i+3, 2*i+2] for i in range(6)], np.int32)
        return v, f

    def assert_go(self, result, vertices, faces, sf, sv):
        self.assertEqual(result['classification'], 'GO', result['report'])
        out = result['layout']
        np.testing.assert_array_equal(out.vertices.view(np.uint32),
                                      vertices[faces].reshape(-1, 3).view(np.uint32))
        np.testing.assert_array_equal(out.faces, np.arange(faces.size).reshape(-1, 3))
        np.testing.assert_array_equal(out.source_face_ids, sf)
        np.testing.assert_array_equal(out.source_vertex_ids, sv[faces].ravel())
        self.assertTrue((out.face_coverage >= 1).all())
        self.assertTrue((out.island_rects % 32 == 0).all())
        self.assertEqual(out.uv.dtype, np.float64)
        self.assertEqual(result['portable_uv_f32'].dtype, np.float32)
        self.assertGreaterEqual(result['report']['minimum_3d_tangent_density_f64'], 4096)
        self.assertTrue(result['report']['portable_f32_support']['all_continuous_LODs_protected'])
        self.assertEqual(result['report']['positive_area_overlap_count'], 0)

    def test_curved_connected_leaf_uses_multiple_axes_and_preserves_every_corner(self):
        v, f = self.leaf()
        v[0, 1] = np.float32(-0.)
        sf, sv = np.arange(len(f)) + 100, np.arange(len(v)) + 700
        before = v.copy(), f.copy()
        out = self.build(v, f, sf, sv)
        self.assert_go(out, v, f, sf, sv)
        self.assertEqual(set(out['report']['face_projection_axes']), {0, 2})
        np.testing.assert_array_equal(v.view(np.uint32), before[0].view(np.uint32))
        np.testing.assert_array_equal(f, before[1])
        again = self.build(v, f, sf, sv)
        np.testing.assert_array_equal(out['layout'].uv, again['layout'].uv)
        self.assertEqual(out['report'], again['report'])

    def test_curved_connected_tube_does_not_assume_yz_projection(self):
        v = np.array([[.006*np.cos(a)+.001*z*z, .006*np.sin(a), .008*z]
                      for z in range(3) for a in np.arange(8)*np.pi/4], np.float32)
        f = np.array([[8*z+i, 8*z+(i+1)%8, 8*(z+1)+i] for z in range(2) for i in range(8)] +
                     [[8*z+(i+1)%8, 8*(z+1)+(i+1)%8, 8*(z+1)+i]
                      for z in range(2) for i in range(8)], np.int32)
        out = self.build(v, f)
        self.assert_go(out, v, f, np.arange(len(f)), np.arange(len(v)))
        self.assertGreaterEqual(len(set(out['report']['face_projection_axes'])), 2)

    def test_nonmanifold_collision_seams_keep_duplicates_and_opposite_winding(self):
        v = np.array([[0, 0, 0], [0, .006, 0], [0, 0, .006],
                      [0, .002, .004]], np.float32)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2], [2, 1, 0]], np.int32)
        sf = np.array([8, 9, 8, 10])
        out = self.build(v, f, sf)
        self.assert_go(out, v, f, sf, np.arange(len(v)))
        self.assertEqual(out['report']['island_count'], 4)

    def test_no_texel_center_coverage_blocks_without_density_search(self):
        v = np.array([[0, 0, 0], [0, .02, 0], [0, 0, 1e-10]], np.float32)
        out = self.build(v, np.array([[0, 1, 2]], np.int32))
        self.assertEqual(out['classification'], 'BLOCKED')
        self.assertIn('covered texel', out['report']['reason'])
        self.assertNotIn('layout', out)

    def test_rectangle_capacity_lower_bound_and_island_bound_are_explicit(self):
        tri = np.array([[0, 0, 0], [0, .0001, 0], [0, 0, .0001]], np.float32)
        v = np.tile(tri, (257, 1))
        f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
        out = self.build(v, f)
        self.assertEqual(out['classification'], 'BLOCKED')
        self.assertEqual(out['report']['island_count'], 257)
        self.assertEqual(out['report']['rectangle_area_texels'], 1052672)
        self.assertTrue(out['report']['area_lower_bound_exceeds_atlas'])
        self.assertFalse(out['report']['packing_optimality_claimed'])
        v = np.array([[0, 0, 0], [0, .2, 0], [0, 0, .001]], np.float32)
        out = self.build(v, np.array([[0, 1, 2]], np.int32))
        self.assertEqual(out['classification'], 'BLOCKED')
        self.assertTrue(out['report']['rectangle_dimension_exceeds_atlas'])

    def test_shelf_failure_does_not_claim_area_lower_bound_or_optimality(self):
        # Hand-chosen extents produce three 608 x 512 rectangles: 933888 px².
        d = 4096*np.sqrt(3)*(1+1e-6)
        v = np.tile(np.array([[0, 0, 0], [0, 550/d, 0], [0, 0, 450/d]], np.float32), (3, 1))
        out = self.build(v, np.arange(9, dtype=np.int32).reshape(-1, 3))
        self.assertEqual(out['classification'], 'BLOCKED')
        self.assertEqual(out['report']['rectangle_area_texels'], 933888)
        self.assertIn('shelf candidate', out['report']['reason'])
        self.assertFalse(out['report']['area_lower_bound_exceeds_atlas'])
        self.assertFalse(out['report']['rectangle_dimension_exceeds_atlas'])
        self.assertFalse(out['report']['packing_optimality_claimed'])

    def test_equal_normal_components_keep_the_true_tangent_density_floor(self):
        v = np.array([[0, 0, 0], [.004, -.004, 0], [.004, 0, -.004]], np.float32)
        f = np.array([[0, 1, 2]], np.int32)
        out = self.build(v, f)
        self.assert_go(out, v, f, np.arange(1), np.arange(3))
        self.assertGreaterEqual(out['report']['portable_f32_checks']['minimum_3d_tangent_density'], 4096)

    def test_malformed_ids_geometry_and_more_than_6000_faces_are_rejected(self):
        v, f = self.leaf()
        for sf in (np.arange(len(f)-1), np.full(len(f), -1), np.arange(len(f), dtype=float)):
            with self.subTest(sf=sf.dtype), self.assertRaises(LayoutError):
                self.build(v, f, sf)
        for sv in (np.arange(len(v)-1), np.full(len(v), -1), np.ones(len(v), bool)):
            with self.subTest(sv=sv.dtype), self.assertRaises(LayoutError):
                self.build(v, f, sv=sv)
        with self.assertRaises(LayoutError):
            self.build(v.astype(np.float64), f)
        with self.assertRaises(LayoutError):
            self.build(v, f.astype(np.int64))
        with self.assertRaises(LayoutError):
            self.build(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.int32))
        with self.assertRaises(LayoutError):
            self.build(v, np.tile(f[0], (6001, 1)))
        with self.assertRaises(LayoutError):
            self.build(v, np.array([[0, 1, len(v)]], np.int32))
        with self.assertRaises(LayoutError):
            build_layout(v, f, np.arange(len(f)), None)


if __name__ == '__main__':
    unittest.main()
