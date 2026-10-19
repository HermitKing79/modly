"""Synthetic CPU sampler checks; never open original artifacts or native modules."""
import itertools
import math
import unittest

import numpy as np

try:
    import sampler
except ModuleNotFoundError:
    sampler = None


def scalar_oracle(coordinates, features, query, spatial_shape):
    """Independent float64 scalar oracle, small synthetic coordinate dictionary only."""
    table = {tuple(int(v) for v in row[1:]): features[i].astype(np.float64)
             for i, row in enumerate(coordinates)}
    result = []
    for point in query:
        point = tuple(float(v) for v in point)
        base = tuple(math.floor(v - 0.5) for v in point)
        weighted, total = [], 0.0
        # Deliberately use scalar z-fast traversal rather than production x-fast arrays.
        for offset in itertools.product((0, 1), repeat=3):
            cell = tuple(base[i] + offset[i] for i in range(3))
            if not all(0 <= cell[i] < spatial_shape[i] for i in range(3)):
                continue
            if cell not in table:
                continue
            weight = math.prod(1.0 - abs(point[i] - cell[i] - 0.5) for i in range(3))
            weighted.append((weight, table[cell]))
            total += weight
        result.append([sum(weight * value[c] for weight, value in weighted)
                       / max(total, 1e-12) for c in range(6)])
    return np.asarray(result, dtype=np.float64).reshape(-1, 6)


class SparseSamplerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(sampler, "The source-faithful CPU sampler is not implemented")

    def cube(self):
        coords = np.array([[0, x, y, z] for x, y, z in
                           itertools.product((1, 2), repeat=3)], dtype=np.int32)
        features = np.array([[.1*x + .01*y + .001*z, .05*x, -.01*y,
                              -.002*z, 1 + .001*x, 1.004 - .001*y]
                             for _, x, y, z in coords], dtype=np.float32)
        return coords, features

    def test_asymmetric_eight_neighbours_match_independent_scalar_oracle(self):
        coords, features = self.cube()
        # Change row order to ensure lookup returns original rows, not sorted positions.
        coords, features = coords[::-1].copy(), features[::-1].copy()
        queries = np.array([[1.7, 1.9, 2.1], [2.25, 1.55, 1.8]], dtype=np.float32)
        result = sampler.SparseSampler(coords, features, (4, 5, 6)).sample_grid(queries)
        np.testing.assert_allclose(result.values, scalar_oracle(coords, features, queries, (4, 5, 6)),
                                   rtol=0, atol=2e-6)
        expected_offsets = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                                     [0,0,1],[1,0,1],[0,1,1],[1,1,1]])
        for row, base in enumerate(np.floor(queries - np.float32(.5)).astype(int)):
            np.testing.assert_array_equal(coords[result.neighbour_rows[row], 1:],
                                          base + expected_offsets)
        np.testing.assert_array_equal(result.found_neighbour_count, [8, 8])
        np.testing.assert_array_equal(result.positive_weight_neighbour_count, [8, 8])
        self.assertEqual(result.values.dtype, np.float32)

    def test_exact_centres_preserve_all_six_unclamped_channels(self):
        coords, features = self.cube()
        result = sampler.SparseSampler(coords, features, (4, 5, 6)).sample_grid(
            coords[:, 1:].astype(np.float32) + np.float32(.5))
        np.testing.assert_array_equal(result.values.view(np.uint32), features.view(np.uint32))
        self.assertLess(float(result.values[:, 3].min()), 0)
        self.assertGreater(float(result.values[:, 4].max()), 1)
        np.testing.assert_array_equal(result.positive_weight_neighbour_count, np.ones(8))

    def test_partial_support_renormalizes_instead_of_zero_padding(self):
        coords, features = self.cube()
        coords, features = coords[[0, 2, 5]], features[[0, 2, 5]]
        query = np.array([[1.7, 1.9, 2.1]], dtype=np.float32)
        result = sampler.SparseSampler(coords, features, (4, 5, 6)).sample_grid(query)
        np.testing.assert_allclose(result.values, scalar_oracle(coords, features, query, (4, 5, 6)),
                                   rtol=0, atol=2e-6)
        self.assertEqual(int(result.found_neighbour_count[0]), 3)
        self.assertLess(float(result.raw_weight_sum[0]), 1)
        self.assertAlmostEqual(float(result.normalized_weights[0].sum()), 1, places=6)
        self.assertGreater(float(result.values[0, 5]), .99)

    def test_single_found_positive_weight_neighbour_preserves_its_material(self):
        coords, features = self.cube()
        query = np.array([[1.75, 1.75, 1.75]], dtype=np.float32)
        result = sampler.SparseSampler(coords[:1], features[:1], (4, 5, 6)).sample_grid(query)
        np.testing.assert_array_equal(result.values, features[:1])
        self.assertEqual(int(result.positive_weight_neighbour_count[0]), 1)

    def test_missing_support_yields_six_zeros_including_alpha(self):
        coords, features = self.cube()
        query = np.array([[3.5, 4.5, 5.5], [-10, 2, 2], [12, 2, 2]], dtype=np.float32)
        result = sampler.SparseSampler(coords, features, (4, 5, 6)).sample_grid(query)
        np.testing.assert_array_equal(result.values, np.zeros((3, 6), dtype=np.float32))
        np.testing.assert_array_equal(result.zero_support, [True, True, True])
        np.testing.assert_array_equal(result.neighbour_rows, np.full((3, 8), -1))

    def test_spatial_boundaries_exclude_outside_cells_but_renormalize_inside(self):
        coords = np.array([[0,0,1,2],[0,3,1,2]], dtype=np.int32)
        features = np.array([[.2,.3,.4,-.01,1.01,.998], [.8,.6,.5,.01,.99,1.004]], dtype=np.float32)
        query = np.array([[-.25,1.5,2.5], [4.25,1.5,2.5], [-.5,1.5,2.5],
                          [4.5,1.5,2.5]], dtype=np.float32)
        result = sampler.SparseSampler(coords, features, (4, 5, 6)).sample_grid(query)
        np.testing.assert_allclose(result.values, scalar_oracle(coords, features, query, (4,5,6)),
                                   rtol=0, atol=2e-6)
        np.testing.assert_array_equal(result.values[:2], features)
        np.testing.assert_array_equal(result.values[2:], np.zeros((2, 6)))
        np.testing.assert_array_equal(result.zero_support, [False, False, True, True])

    def test_denominator_floor_is_retained_for_tiny_positive_support(self):
        coords = np.array([[0,1,1,1]], dtype=np.int32)
        features = np.ones((1,6), dtype=np.float32)
        tiny = np.nextafter(np.float32(.5), np.float32(1))
        query = np.full((1,3), tiny, dtype=np.float32)
        result = sampler.SparseSampler(coords, features, (3,3,3)).sample_grid(query)
        self.assertGreater(float(result.raw_weight_sum[0]), 0)
        self.assertLess(float(result.raw_weight_sum[0]), 1e-12)
        self.assertLess(float(result.values[0,0]), 1e-6)
        np.testing.assert_allclose(result.values, scalar_oracle(coords, features, query, (3,3,3)),
                                   rtol=0, atol=1e-15)

    def test_world_transform_uses_original_XYZ_centres_and_preserves_inputs(self):
        coords, features = self.cube()
        original_coords, original_features = coords.copy(), features.copy()
        world = ((coords[:,1:].astype(np.float32)+np.float32(.5))*np.float32(1/512)
                 - np.float32(.5))
        original_world = world.copy()
        result = sampler.SparseSampler(coords, features, (4,5,6)).sample_world(world)
        np.testing.assert_array_equal(result.values.view(np.uint32), features.view(np.uint32))
        np.testing.assert_array_equal(coords, original_coords)
        np.testing.assert_array_equal(features.view(np.uint32), original_features.view(np.uint32))
        np.testing.assert_array_equal(world.view(np.uint32), original_world.view(np.uint32))

    def test_random_sparse_positive_and_negative_channels_match_scalar_oracle(self):
        rng = np.random.default_rng(1701)
        cells = np.array(list(itertools.product(range(4), range(5), range(6))), dtype=np.int32)
        cells = cells[rng.choice(len(cells), 55, replace=False)]
        coords = np.column_stack((np.zeros(len(cells), dtype=np.int32), cells))
        features = rng.uniform(-1.01, 1.01, (len(cells), 6)).astype(np.float32)
        query = rng.uniform([-1,-1,-1], [5,6,7], (300,3)).astype(np.float32)
        result = sampler.SparseSampler(coords, features, (4,5,6)).sample_grid(query)
        np.testing.assert_allclose(result.values, scalar_oracle(coords, features, query, (4,5,6)),
                                   rtol=0, atol=2e-6)

    def test_rejects_duplicate_malformed_out_of_bounds_or_unbounded_support(self):
        coords, features = self.cube()
        cases = [
            (coords.astype(np.int64), features, (4,5,6), {}),
            (coords, features.astype(np.float64), (4,5,6), {}),
            (coords[:,:3], features, (4,5,6), {}),
            (coords, features[:,:5], (4,5,6), {}),
            (coords, features[:-1], (4,5,6), {}),
            (coords[:0], features[:0], (4,5,6), {}),
            (coords, features, (0,5,6), {}),
            (coords, features, (4.,5,6), {}),
            (coords, features, (513,5,6), {}),
            (coords, features, (4,5), {}),
            (coords, features, (4,5,6), {"max_rows": 7}),
        ]
        duplicate = coords.copy(); duplicate[1] = duplicate[0]
        batch = coords.copy(); batch[0,0] = 1
        negative = coords.copy(); negative[0,1] = -1
        outside = coords.copy(); outside[0,2] = 5
        nonfinite = features.copy(); nonfinite[0,0] = np.nan
        cases += [(v, features, (4,5,6), {}) for v in [duplicate,batch,negative,outside]]
        cases.append((coords, nonfinite, (4,5,6), {}))
        for c, f, shape, kwargs in cases:
            with self.subTest(shape=shape, dtype=c.dtype, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    sampler.SparseSampler(c, f, shape, **kwargs)

    def test_queries_are_bounded_finite_float32_XYZ_and_allow_empty_batch(self):
        coords, features = self.cube()
        instance = sampler.SparseSampler(coords, features, (4,5,6))
        cases = [np.zeros((1,3), dtype=np.float64), np.zeros((1,4), dtype=np.float32),
                 np.array([[np.inf,0,0]], dtype=np.float32),
                 np.array([[1e20,0,0]], dtype=np.float32),
                 np.zeros((65537,3), dtype=np.float32)]
        for query in cases:
            with self.subTest(shape=query.shape, dtype=query.dtype):
                with self.assertRaises(ValueError):
                    instance.sample_grid(query)
        result = instance.sample_grid(np.empty((0,3), dtype=np.float32))
        self.assertEqual(result.values.shape, (0,6))
        self.assertEqual(result.neighbour_rows.shape, (0,8))

    def test_native_texture_packing_truncates_and_keeps_channel_semantics(self):
        values = np.array([[[.5,.25,1.5,-.1,1.01,.998],
                            [.001,.999,-.01,.125,.5,1.004]]], dtype=np.float32)
        original = values.copy()
        packed = sampler.pack_native_textures(values)
        np.testing.assert_array_equal(packed["base_color_rgba"],
                                      [[[127,63,255,254], [0,254,0,255]]])
        np.testing.assert_array_equal(packed["metallic_roughness_rgb"],
                                      [[[0,255,0], [0,127,31]]])
        np.testing.assert_array_equal(packed["alpha"], [[254,255]])
        self.assertEqual(packed["base_color_rgba"].dtype, np.uint8)
        np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))

    def test_packing_rejects_nonfinite_wrong_schema_and_unbounded_images(self):
        for values in [np.zeros((1,6), dtype=np.float64), np.zeros((1,5), dtype=np.float32),
                       np.full((1,6), np.nan, dtype=np.float32),
                       np.zeros((1048577,6), dtype=np.float32)]:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(ValueError):
                    sampler.pack_native_textures(values)


if __name__ == "__main__":
    unittest.main()
