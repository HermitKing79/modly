"""Synthetic atlas and seam fixtures; never load the private brick asset."""
import unittest
import numpy as np

from bake_math import footprint_owner, isolate_padding, seam_queries


class BakeMathTests(unittest.TestCase):
    def atlas(self):
        values = np.zeros((1024, 1024, 6), np.float32)
        ids = np.full((1024, 1024), -1, np.int32)
        ids[8:16, 8:16] = 0
        ids[8:16, 24:32] = 1
        values[ids == 0] = [.71, .31, .12, -.002, 1.002, .998]
        values[ids == 1] = [.05, .8, .3, .6, .2, .4]
        return values, ids, np.array([0, 1], np.int32), np.array([[4, 4, 20, 20], [20, 4, 36, 20]], np.int32)

    def test_isolated_native_padding_preserves_raw_and_packed_covered_channels(self):
        from common import originals
        values, ids, islands, rects = self.atlas()
        before = values.copy()
        result = isolate_padding(values, ids, islands, rects)
        old = originals()
        for island, (x0, y0, x1, y1) in enumerate(rects):
            mask = ids[y0:y1, x0:x1] >= 0
            base, mr, original_base, original_mr = old.source_padding(values[y0:y1, x0:x1], mask)
            np.testing.assert_array_equal(result['base_color_rgba'][y0:y1, x0:x1], base)
            np.testing.assert_array_equal(result['metallic_roughness_rgb'][y0:y1, x0:x1], mr)
            np.testing.assert_array_equal(result['original_base_color_rgba'][y0:y1, x0:x1], original_base)
            np.testing.assert_array_equal(result['original_metallic_roughness_rgb'][y0:y1, x0:x1], original_mr)
            self.assertTrue((result['ownership'][y0:y1, x0:x1] == island).all())
        np.testing.assert_array_equal(result['reference'][ids >= 0].view(np.uint32), values[ids >= 0].view(np.uint32))
        self.assertTrue((result['base_color_rgba'][ids == 0, 3] == 254).all())
        self.assertTrue((result['metallic_roughness_rgb'][..., 0] == 0).all())
        self.assertEqual(result['ownership'].dtype, np.int16)
        self.assertTrue((result['reference'][result['ownership'] < 0] == 0).all())
        self.assertTrue((result['base_color_rgba'][result['ownership'] < 0] == 0).all())
        y, x = 5, 5
        packed = np.r_[result['base_color_rgba'][y, x, :3],
                       result['metallic_roughness_rgb'][y, x, 2],
                       result['metallic_roughness_rgb'][y, x, 1],
                       result['base_color_rgba'][y, x, 3]].astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(result['reference'][y, x], packed)
        np.testing.assert_array_equal(values, before)

    def test_foreign_coverage_overlap_empty_rectangle_and_dimensions_reject(self):
        values, ids, islands, rects = self.atlas()
        foreign = ids.copy(); foreign[9, 9] = 2
        with self.assertRaisesRegex(ValueError, 'face ID'):
            isolate_padding(values, foreign, islands, rects)
        foreign[9, 9] = 1
        with self.assertRaisesRegex(ValueError, 'island'):
            isolate_padding(values, foreign, islands, rects)
        overlap = rects.copy(); overlap[1, 0] = 19
        with self.assertRaisesRegex(ValueError, 'overlap'):
            isolate_padding(values, ids, islands, overlap)
        empty = ids.copy(); empty[empty == 1] = -1
        with self.assertRaisesRegex(ValueError, 'coverage'):
            isolate_padding(values, empty, islands, rects)
        with self.assertRaisesRegex(ValueError, '1024'):
            isolate_padding(values[:16], ids, islands, rects)

    def test_all_four_bilinear_footprint_owners_required_including_clamp(self):
        owner = np.full((1024, 1024), -1, np.int16)
        owner[:20, :10] = 0; owner[:20, 10:20] = 1
        owner[-2:, -2:] = 2
        uv = np.array([[8.5, 8.5], [10., 8.5], [25.5, 8.5], [1024., 1024.]]) / 1024
        np.testing.assert_array_equal(footprint_owner(uv, np.array([0, 0, 0, 2]), owner),
                                      [True, False, False, True])
        with self.assertRaisesRegex(ValueError, 'finite'):
            footprint_owner(np.array([[np.nan, 0.]]), np.array([0]), owner)

    def test_seam_pair_world_points_match_despite_opposite_corner_order(self):
        ids = np.array([[10, 20, 30], [20, 10, 40]], np.int64)
        result = seam_queries(ids, np.array([0, 1], np.int32))
        self.assertEqual(len(result['query_faces']), 20)
        self.assertEqual(result['pairing'].shape, (5, 2))
        xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 0, 0], [0, -1, 0]]], float)
        world = (xyz[result['query_faces']] * result['bary'][:, :, None]).sum(1)
        np.testing.assert_array_equal(world[result['pairing'][:, 0]], world[result['pairing'][:, 1]])
        np.testing.assert_allclose(world[result['pairing'][:, 0], 0], [0, .25, .5, .75, 1])
        self.assertTrue((result['cohort'][result['pairing']] == 0).all())
        self.assertTrue((result['bary'][result['cohort'] == 1] > 0).all())
        np.testing.assert_allclose(result['bary'].sum(1), 1, atol=1e-15)
        np.testing.assert_array_equal(result['edge_source_ids'], np.tile([10, 20], (5, 1)))

    def test_nonmanifold_pairs_not_dropped_same_island_excluded_and_budget_rejected(self):
        ids = np.array([[10, 20, 30], [20, 10, 40], [10, 20, 50]], np.int64)
        result = seam_queries(ids, np.array([0, 1, 2], np.int32))
        self.assertEqual(len(result['query_faces']), 60)
        self.assertEqual(len(result['pairing']), 15)
        two = seam_queries(ids, np.array([0, 1, 1], np.int32))
        self.assertEqual(len(two['query_faces']), 40)
        with self.assertRaisesRegex(ValueError, 'budget'):
            seam_queries(ids, np.array([0, 1, 2], np.int32), max_queries=59)
        empty = seam_queries(ids, np.array([0, 0, 0], np.int32))
        self.assertEqual(empty['bary'].shape, (0, 3))
        np.testing.assert_array_equal(seam_queries(ids, np.array([0, 1, 2], np.int32))['bary'], result['bary'])


if __name__ == '__main__':
    unittest.main()
