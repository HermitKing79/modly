"""Synthetic CPU checks only: no private assets, Torch, Blender or GPU calls."""
import unittest

import numpy as np

from mip_math import conditioned, pyramid, sample, srgb_decode, srgb_encode


class MipMathTests(unittest.TestCase):
    def test_exact_box_levels_are_independent_hand_averages_and_leave_input_unchanged(self):
        values = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        original = values.copy()
        levels = pyramid(values, max_level=2)
        self.assertEqual(len(levels), 3)
        self.assertTrue(all(level.dtype == np.float64 for level in levels))
        np.testing.assert_array_equal(levels[1][:, :, 0], [[2.5, 4.5], [10.5, 12.5]])
        np.testing.assert_array_equal(levels[2][:, :, 0], [[7.5]])
        np.testing.assert_array_equal(values, original)
        self.assertFalse(np.shares_memory(values, levels[0]))

    def test_srgb_known_values_and_clipping(self):
        encoded = np.array([-.1, 0, .04045, .5, 1, 2])
        decoded = srgb_decode(encoded)
        np.testing.assert_allclose(decoded, [0, 0, .0031308049535603713,
                                             .21404114048223255, 1, 1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(srgb_encode(decoded), np.clip(encoded, 0, 1), rtol=0, atol=4e-8)
        self.assertEqual(decoded.dtype, np.float64)

    def test_linear_light_box_is_not_encoded_rgb_average(self):
        encoded = np.array([[[0.], [1.]], [[1.], [0.]]])
        level = pyramid(srgb_decode(encoded), max_level=1)[1]
        self.assertAlmostEqual(float(level[0, 0, 0]), .5, places=15)
        self.assertAlmostEqual(float(srgb_encode(level)[0, 0, 0]), .7353569830524495, places=14)
        self.assertGreater(float(srgb_encode(level)[0, 0, 0]), float(encoded.mean()))

    def test_integer_and_half_lod_use_clamped_bilinear_then_level_blend(self):
        levels = pyramid(np.array([[[0.], [4.]], [[8.], [12.]]]), max_level=1)
        uv = np.array([[0., 0.], [1., 1.], [.5, .5]])
        np.testing.assert_allclose(sample(levels, uv, .5)[:, 0], [3, 9, 6], rtol=0, atol=0)
        np.testing.assert_array_equal(sample(levels, uv, 0)[:, 0], [0, 12, 6])
        np.testing.assert_array_equal(sample(levels, uv, 1)[:, 0], [6, 6, 6])

    def test_bilinear_pixel_centres_V_up_and_outer_edge_clamp(self):
        levels = pyramid(np.array([[[0.], [2.]], [[10.], [12.]]]), max_level=0)
        uv = np.array([[.25, .25], [.25, .75], [.5, .25], [-4., -1.], [2., 9.]])
        np.testing.assert_array_equal(sample(levels, uv, 0)[:, 0], [0, 10, 1, 0, 12])

    def test_each_query_can_have_its_own_fractional_lod(self):
        levels = pyramid(np.array([[[0.], [4.]], [[8.], [12.]]]), max_level=1)
        uv = np.zeros((3, 2), dtype=np.float64)
        np.testing.assert_array_equal(sample(levels, uv, np.array([0., .25, 1.]))[:, 0], [0, 1.5, 6])

    def test_conditioning_normalizes_after_complete_trilinear_filter(self):
        weights = np.array([[[1.], [0.]], [[0.], [1.]]])
        numerator = np.array([[[.2], [0.]], [[0.], [1.]]])
        value, mass = conditioned(pyramid(numerator, 1), pyramid(weights, 1), np.array([[0., 0.]]), .5)
        self.assertAlmostEqual(float(mass[0, 0]), .75, places=15)
        self.assertAlmostEqual(float(value[0, 0]), 1 / 3, places=15)
        self.assertNotAlmostEqual(float(value[0, 0]), (.2 + .6) / 2, places=10)

    def test_ownership_planes_preserve_unit_total_and_expose_foreign_influence(self):
        labels = np.array([[0, 1], [0, 2]])
        planes = np.eye(3, dtype=np.float64)[labels]
        levels = pyramid(planes, 1)
        weights = sample(levels, np.array([[0., 0.], [.5, .5], [1., 1.]]), .5)
        np.testing.assert_allclose(weights.sum(1), 1, rtol=0, atol=0)
        self.assertGreater(float(weights[0, 1:].sum()), 0)
        np.testing.assert_array_equal(levels[1][0, 0], [.5, .25, .25])

    def test_zero_support_negative_or_multichannel_weight_chains_are_rejected(self):
        values = pyramid(np.ones((2, 2, 2)), 1)
        uv = np.zeros((1, 2))
        for weights in (pyramid(np.zeros((2, 2, 1)), 1),
                        pyramid(-np.ones((2, 2, 1)), 1),
                        pyramid(np.ones((2, 2, 2)), 1)):
            with self.assertRaises(ValueError):
                conditioned(values, weights, uv, .5)
        with self.assertRaises(ValueError):
            conditioned(values, pyramid(np.ones((2, 2, 1)), 0), uv, 0)

    def test_conditioned_nonfinite_result_is_rejected_instead_of_returned(self):
        values = pyramid(np.ones((2, 2, 1)), 1)
        weights = pyramid(np.full((2, 2, 1), 1e-320), 1)
        with self.assertRaises(ValueError):
            conditioned(values, weights, np.array([[0., 0.]]), .5)

    def test_pyramid_arithmetic_overflow_fails_clearly(self):
        with self.assertRaises(ValueError):
            pyramid(np.full((2, 2, 1), np.finfo(np.float64).max), 1)

    def test_pyramid_rejects_invalid_sizes_types_channels_levels_and_nonfinite(self):
        for invalid in (np.ones((3, 3, 1)), np.ones((2, 4, 1)), np.ones((2, 2)),
                        np.ones((2, 2, 9)), np.ones((2, 2, 1), dtype=np.int32),
                        np.full((2, 2, 1), np.inf), np.ones((2048, 2048, 1))):
            with self.assertRaises(ValueError):
                pyramid(invalid, 0)
        for invalid_level in (-1, 2, 1.5, True):
            with self.assertRaises(ValueError):
                pyramid(np.ones((2, 2, 1)), invalid_level)

    def test_sampling_rejects_malformed_chains_queries_and_lod_without_clamping(self):
        levels = pyramid(np.ones((2, 2, 1)), 1)
        uv = np.zeros((2, 2))
        bad_chains = [(), (levels[0], np.ones((2, 2, 1))),
                      (levels[0].astype(np.float32),), (np.full((2, 2, 1), np.nan),),
                      (levels[0], np.ones((1, 1, 2)))]
        for invalid in bad_chains:
            with self.assertRaises(ValueError):
                sample(invalid, uv, 0)
        for bad_uv in (np.zeros((0, 2)), np.zeros((65537, 2)), np.zeros((2, 3)),
                       np.full((2, 2), np.nan), np.zeros((2, 2), dtype=np.int32)):
            with self.assertRaises(ValueError):
                sample(levels, bad_uv, 0)
        for invalid_lod in (-.001, 1.001, np.nan, np.inf, np.array([0., .5, 1.]), np.array([[0., .5]])):
            with self.assertRaises(ValueError):
                sample(levels, uv, invalid_lod)

    def test_srgb_nonfinite_and_non_numeric_inputs_fail_clearly(self):
        for function in (srgb_decode, srgb_encode):
            for invalid in (np.array([np.nan]), np.array([np.inf]), np.array(['x'])):
                with self.assertRaises(ValueError):
                    function(invalid)


if __name__ == '__main__':
    unittest.main()
