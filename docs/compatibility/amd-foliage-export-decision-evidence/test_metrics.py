"""Synthetic CPU checks: python -I -B test_metrics.py."""

import sys
sys.dont_write_bytecode = True

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("foliage_metrics", Path(__file__).with_name("metrics.py"))
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class ColourPredictionTests(unittest.TestCase):
    def test_black_white_interpolation_occurs_in_portable_linear_colour(self):
        colours = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        predicted, linear = metrics.color_prediction(
            colours, faces, np.array([0], dtype=np.int32),
            np.array([[0.5, 0.5, 0]], dtype=np.float64),
        )
        np.testing.assert_allclose(predicted, [[0.7353569830524495] * 3], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(linear, colours)
        self.assertEqual(linear.dtype, np.dtype(np.float32))
        self.assertEqual(predicted.dtype, np.dtype(np.float64))
        self.assertGreater(predicted[0, 0] - 0.5, 0.23)

    def test_clipping_and_float32_linear_rounding_precede_interpolation(self):
        colours = np.array([[-1, 0.5, 2]], dtype=np.float32)
        before = colours.copy()
        predicted, linear = metrics.color_prediction(
            colours, np.array([[0, 0, 0]], dtype=np.int32),
            np.array([0], dtype=np.int32), np.array([[1, 0, 0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(linear, np.array([[0, 0.21404114048223255, 1]], dtype=np.float32))
        self.assertEqual(predicted[0, 0], 0)
        self.assertAlmostEqual(predicted[0, 1], 0.5, places=7)
        self.assertAlmostEqual(predicted[0, 2], 1.0, places=14)
        np.testing.assert_array_equal(colours, before)

    def test_malformed_colour_faces_queries_and_barycentrics_are_rejected(self):
        colours = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        query = np.array([0], dtype=np.int32)
        bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
        cases = [
            (colours.astype(np.float64), faces, query, bary),
            (colours, faces.astype(np.int64), query, bary),
            (colours, np.array([[0, 1, 3]], dtype=np.int32), query, bary),
            (colours, faces, query.astype(np.int64), bary),
            (colours, faces, np.array([1], dtype=np.int32), bary),
            (colours, faces, query, bary.astype(np.float32)),
            (colours, faces, query, np.array([[-0.1, 0.6, 0.5]], dtype=np.float64)),
            (colours, faces, query, np.array([[0.2, 0.3, 0.4]], dtype=np.float64)),
            (colours, faces, query, np.full((1, 3), np.nan, dtype=np.float64)),
        ]
        for args in cases:
            with self.subTest(dtypes=[value.dtype for value in args]):
                with self.assertRaises((TypeError, ValueError)):
                    metrics.color_prediction(*args)


class ChannelRangesTests(unittest.TestCase):
    def test_standalone_ranges_report_each_raw_clipped_range_and_counts(self):
        raw = np.array([[-0.25, 0.5, 1, -2, 0.4, 1.25],
                        [1.5, 0.75, 0, 0.5, 0.8, 0.5]], dtype=np.float32)
        before = raw.copy()
        ranges = metrics.channel_ranges(raw)
        self.assertEqual(ranges["red"], {"raw": [-0.25, 1.5], "clipped": [0, 1],
                                        "below_zero_count": 1, "above_one_count": 1, "clipped_count": 2})
        self.assertEqual(ranges["green"]["clipped_count"], 0)
        self.assertEqual(ranges["metallic"]["raw"], [-2, 0.5])
        self.assertEqual(ranges["alpha"]["clipped"], [0.5, 1])
        self.assertEqual(ranges["alpha"]["above_one_count"], 1)
        np.testing.assert_array_equal(raw, before)

    def test_standalone_ranges_reject_wrong_dtype_shape_nonfinite_and_empty(self):
        for raw in (np.zeros((2, 6), dtype=np.float64), np.zeros((2, 5), dtype=np.float32),
                    np.full((2, 6), np.nan, dtype=np.float32),
                    np.full((2, 6), np.inf, dtype=np.float32), np.empty((0, 6), dtype=np.float32)):
            with self.subTest(shape=raw.shape, dtype=raw.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    metrics.channel_ranges(raw)


class EvaluationTests(unittest.TestCase):
    def test_scalar_fit_means_clipped_samples_instead_of_clipping_the_raw_mean(self):
        raw = np.array([[0.5, 0.5, 0.5, -2, 0.4, 1.2],
                        [0.5, 0.5, 0.5, 0.2, 0.8, 0.6]], dtype=np.float32)
        result = metrics.evaluate(raw[:, :3].astype(np.float64), raw,
                                  {"fit": np.array([0, 1])}, np.array([0, 1]))
        for name, expected in (("metallic", 0.1), ("roughness", 0.6), ("alpha", 0.8)):
            self.assertAlmostEqual(result["constants"][name], expected, places=7)
        self.assertNotEqual(result["constants"]["metallic"], 0.0)
        self.assertAlmostEqual(result["metrics"]["fit"]["irreducible_scalar_rmse_floor"]["metallic"], 0.1, places=7)

    def test_each_scalar_channel_must_pass_without_pooled_channel_masking(self):
        raw = np.array([[0.5, 0.5, 0.5, 0, 0.4, 1],
                        [0.5, 0.5, 0.5, 0.03, 0.4, 1]], dtype=np.float32)
        result = metrics.evaluate(raw[:, :3].astype(np.float64), raw,
                                  {"check": np.array([1])}, np.array([0]))
        self.assertTrue(result["RGB_all_pass"])
        self.assertFalse(result["scalars_all_pass"])
        self.assertFalse(result["decision_pass"])
        scalars = result["metrics"]["check"]["scalars"]
        self.assertFalse(scalars["metallic"]["pass"])
        self.assertTrue(scalars["roughness"]["pass"])
        self.assertTrue(scalars["alpha"]["pass"])
        self.assertAlmostEqual(scalars["metallic"]["rmse"], 0.03, places=7)

    def test_rgb_rmse_pools_components_and_does_not_apply_a_per_channel_rmse_gate(self):
        raw = np.zeros((1, 6), dtype=np.float32)
        predicted = np.array([[0.03, 0, 0]], dtype=np.float64)
        result = metrics.evaluate(predicted, raw, {"all": np.array([0])}, np.array([0]))
        self.assertTrue(result["RGB_all_pass"])
        self.assertTrue(result["decision_pass"])
        self.assertAlmostEqual(result["metrics"]["all"]["RGB"]["rmse"], 0.01732050807568877)

    def test_rgb_p95_gate_rejects_when_rmse_and_maximum_pass(self):
        raw = np.zeros((100, 6), dtype=np.float32)
        predicted = np.zeros((100, 3), dtype=np.float64)
        predicted[:6] = 0.051
        result = metrics.evaluate(predicted, raw, {"all": np.arange(100)}, np.arange(100))
        rgb = result["metrics"]["all"]["RGB"]
        self.assertLess(rgb["rmse"], 0.02)
        self.assertGreater(rgb["p95_abs"], 0.05)
        self.assertLess(rgb["max_abs"], 0.15)
        self.assertFalse(result["RGB_all_pass"])

    def test_rgb_maximum_gate_rejects_when_rmse_and_p95_pass(self):
        raw = np.zeros((100, 6), dtype=np.float32)
        predicted = np.zeros((100, 3), dtype=np.float64)
        predicted[0, 0] = 0.16
        result = metrics.evaluate(predicted, raw, {"all": np.arange(100)}, np.arange(100))
        rgb = result["metrics"]["all"]["RGB"]
        self.assertLess(rgb["rmse"], 0.02)
        self.assertEqual(rgb["p95_abs"], 0)
        self.assertGreater(rgb["max_abs"], 0.15)
        self.assertFalse(result["RGB_all_pass"])

    def test_all_required_cohorts_use_the_same_fitted_constants(self):
        raw = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0.1, 0, 0]], dtype=np.float32)
        result = metrics.evaluate(raw[:, :3].astype(np.float64), raw,
                                  {"first": np.array([0]), "second": np.array([1])}, np.array([0, 1]))
        self.assertFalse(result["scalars_all_pass"])
        for name in ("first", "second"):
            self.assertAlmostEqual(result["metrics"][name]["scalars"]["metallic"]["rmse"], 0.05, places=7)
            self.assertEqual(result["metrics"][name]["irreducible_scalar_rmse_floor"]["metallic"], 0)
        self.assertAlmostEqual(result["metrics"]["second"]["optimistic_scalar_constants"]["metallic"], 0.1, places=7)

    def test_raw_clipped_ranges_and_clipping_counts_remain_explicit(self):
        raw = np.array([[-0.2, 0.1, 0.5, -2, 0.5, 1.2],
                        [1.2, 0.9, 0.2, 0.4, 0.8, 0.6]], dtype=np.float32)
        before = raw.copy()
        result = metrics.evaluate(np.clip(raw[:, :3], 0, 1).astype(np.float64), raw,
                                  {"all": np.array([0, 1])}, np.array([0, 1]))
        red = result["ranges"]["red"]
        self.assertAlmostEqual(red["raw"][0], -0.2, places=7)
        self.assertAlmostEqual(red["raw"][1], 1.2, places=7)
        self.assertEqual(red["clipped"], [0, 1])
        self.assertEqual(red["below_zero_count"], 1)
        self.assertEqual(red["above_one_count"], 1)
        self.assertEqual(red["clipped_count"], 2)
        self.assertEqual(result["ranges"]["metallic"]["clipped_count"], 1)
        np.testing.assert_array_equal(raw, before)

    def test_invalid_evaluation_shapes_dtypes_and_empty_required_sets_fail(self):
        predicted = np.zeros((2, 3), dtype=np.float64)
        raw = np.zeros((2, 6), dtype=np.float32)
        cohorts, fit = {"all": np.array([0, 1])}, np.array([0, 1])
        for args in (
            (predicted.astype(np.float32), raw, cohorts, fit),
            (predicted, raw.astype(np.float64), cohorts, fit),
            (predicted[:1], raw, cohorts, fit),
            (np.full((2, 3), np.nan), raw, cohorts, fit),
            (np.full((2, 3), 1.2), raw, cohorts, fit),
            (np.full((2, 3), -0.1), raw, cohorts, fit),
            (predicted, np.full((2, 6), np.inf, dtype=np.float32), cohorts, fit),
            (predicted, raw, {}, fit),
            (predicted, raw, {"empty": np.array([], dtype=np.int64)}, fit),
            (predicted, raw, {"float": np.array([0.0])}, fit),
            (predicted, raw, cohorts, np.array([-1])),
            (predicted, raw, cohorts, np.array([], dtype=np.int64)),
        ):
            with self.subTest(shapes=[getattr(value, "shape", None) for value in args]):
                with self.assertRaises((TypeError, ValueError)):
                    metrics.evaluate(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
