"""Synthetic CPU-only tests. Run with ``python -I -B test_diagnostics.py``."""

import sys
sys.dont_write_bytecode = True

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "bounded_foliage_diagnostics", Path(__file__).with_name("diagnostics.py")
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class TopologyTests(unittest.TestCase):
    def test_vertex_touch_does_not_join_face_edge_components(self):
        faces = np.array([[0, 1, 2], [2, 1, 3], [3, 4, 5]], dtype=np.int32)
        ids = np.array([100, 2, 10, 8, 70, 6], dtype=np.int64)
        faces_before, ids_before = faces.copy(), ids.copy()
        result = diagnostics.analyze_topology(faces, ids)
        self.assertEqual(result["face_edge_connected_components"], 2)
        self.assertEqual(result["face_edge_component_sizes"], [2, 1])
        self.assertEqual(result["face_vertex_connected_components"], 1)
        self.assertEqual(result["face_vertex_component_sizes"], [3])
        self.assertEqual(result["edge_incidence_1"], 7)
        self.assertEqual(result["edge_incidence_2"], 1)
        self.assertEqual(result["edge_incidence_gt2"], 0)
        self.assertIsNone(result["selected_edge_global_incidence"])
        self.assertIsNone(result["artificial_crop_boundary_edges"])
        self.assertIsNone(result["natural_boundary_edges"])
        self.assertEqual(result["selected_original_edge_keys"][0], 2 * 1957891 + 8)
        np.testing.assert_array_equal(faces, faces_before)
        np.testing.assert_array_equal(ids, ids_before)

    def test_nonmanifold_edge_connects_all_incident_faces(self):
        result = diagnostics.analyze_topology(
            np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32),
            np.arange(5, dtype=np.int64),
        )
        self.assertEqual(result["face_edge_component_sizes"], [3])
        self.assertEqual(result["edge_incidence_1"], 6)
        self.assertEqual(result["edge_incidence_2"], 0)
        self.assertEqual(result["edge_incidence_gt2"], 1)
        self.assertEqual(max(result["selected_edge_local_incidence"]), 3)

    def test_global_incidence_separates_natural_from_crop_boundaries(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        ids = np.array([9, 3, 5], dtype=np.int64)
        original = np.array(
            [[9, 3, 5], [3, 9, 8], [5, 3, 4], [100, 101, 102]],
            dtype=np.int32,
        )
        before = original.copy()
        result = diagnostics.analyze_topology(
            faces, ids, original, global_face_chunk_size=1
        )
        self.assertEqual(result["selected_original_edge_keys"], [5873678, 5873682, 9789464])
        self.assertEqual(result["selected_edge_global_incidence"], [2, 2, 1])
        self.assertEqual(result["artificial_crop_boundary_edges"], 2)
        self.assertEqual(result["natural_boundary_edges"], 1)
        self.assertEqual(result["unmatched_boundary_edges"], 0)
        np.testing.assert_array_equal(original, before)

    def test_global_edges_not_present_are_reported_as_unmatched(self):
        result = diagnostics.analyze_topology(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.arange(3, dtype=np.int64),
            np.array([[10, 11, 12]], dtype=np.int32),
        )
        self.assertEqual(result["selected_edge_global_incidence"], [0, 0, 0])
        self.assertEqual(result["unmatched_boundary_edges"], 3)
        self.assertEqual(result["artificial_crop_boundary_edges"], 0)
        self.assertEqual(result["natural_boundary_edges"], 0)

    def test_empty_selection_has_no_components_or_edges(self):
        result = diagnostics.analyze_topology(
            np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int32),
        )
        self.assertEqual(result["face_edge_connected_components"], 0)
        self.assertEqual(result["face_vertex_connected_components"], 0)
        self.assertEqual(result["selected_original_edge_keys"], [])
        self.assertEqual(result["selected_edge_global_incidence"], [])
        self.assertEqual(result["natural_boundary_edges"], 0)

    def test_local_faces_must_be_bounded_int32_triangles(self):
        ids = np.arange(3, dtype=np.int64)
        invalid = [
            np.array([[0, 1, 2]], dtype=np.int64),
            np.array([[0.0, 1.0, 2.0]]),
            np.zeros((1, 4), dtype=np.int32),
            np.zeros((6001, 3), dtype=np.int32),
            np.array([[-1, 1, 2]], dtype=np.int32),
            np.array([[0, 1, 3]], dtype=np.int32),
            np.array([[0, 0, 1]], dtype=np.int32),
        ]
        for faces in invalid:
            with self.subTest(shape=faces.shape, dtype=faces.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.analyze_topology(faces, ids)

    def test_original_ids_must_be_unique_bounded_int64(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        for ids in (
            np.arange(3, dtype=np.int32),
            np.array([0, 1, -1], dtype=np.int64),
            np.array([0, 1, 1957891], dtype=np.int64),
            np.array([0, 1, 1], dtype=np.int64),
            np.array([[0, 1, 2]], dtype=np.int64),
        ):
            with self.subTest(ids=ids):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.analyze_topology(faces, ids)

    def test_original_faces_and_chunk_size_are_validated(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        ids = np.arange(3, dtype=np.int64)
        for original in (
            np.array([[0.0, 1.0, 2.0]]),
            np.zeros((2, 4), dtype=np.int32),
            np.array([[0, 1, -1]], dtype=np.int32),
            np.array([[0, 1, 1957891]], dtype=np.int32),
        ):
            with self.subTest(original=original):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.analyze_topology(faces, ids, original)
        for chunk_size in (0, -1, 1.5, True):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.analyze_topology(
                        faces, ids, faces, global_face_chunk_size=chunk_size
                    )


class CoverageTests(unittest.TestCase):
    def test_union_counts_overlapping_triangles_once_in_each_proxy(self):
        # Two identical 2x2 squares represented by duplicated triangle pairs.
        square = [[0, 0], [2, 0], [2, 2]], [[0, 0], [2, 2], [0, 2]]
        xy = np.array([*square, *square], dtype=np.float64)
        before = xy.copy()
        result = diagnostics.coverage_proxies(
            xy, (0, 0, 4, 2), {"left": (0, 0, 2, 2), "right": (2, 0, 4, 2)}
        )
        self.assertEqual(result["covered_pixel_count"], 4)
        self.assertEqual(result["roi_pixel_count"], 8)
        self.assertEqual(result["proxy_covered_pixel_counts"], {"left": 4, "right": 0})
        self.assertEqual(result["proxies"]["left"]["covered_pixels"], 4)
        self.assertEqual(result["proxies"]["left"]["pixel_count"], 4)
        self.assertFalse(result["all_proxies_have_coverage"])
        np.testing.assert_array_equal(xy, before)

    def test_fixed_windows_both_receive_positive_projected_coverage(self):
        xy = np.array(
            [[[350, 399], [354, 399], [350, 403]],
             [[380, 394], [384, 394], [380, 398]]], dtype=np.float64
        )
        result = diagnostics.coverage_proxies(xy)
        self.assertEqual(result["covered_pixel_count"], 20)
        self.assertEqual(result["proxy_covered_pixel_counts"], {"leaf": 10, "vine": 10})
        self.assertTrue(result["all_proxies_have_coverage"])
        self.assertEqual(result["roi_pixel_count"], 2021)
        self.assertIn("manually nominated", result["interpretation"])

    def test_clockwise_triangles_and_edges_include_pixel_centres(self):
        result = diagnostics.coverage_proxies(
            np.array([[[0, 0], [0, 2], [2, 0]]], dtype=np.float64),
            (0, 0, 2, 2), {"all": (0, 0, 2, 2)},
        )
        self.assertEqual(result["covered_pixel_count"], 3)

    def test_degenerate_and_outside_triangles_cover_no_pixels(self):
        result = diagnostics.coverage_proxies(
            np.array([[[0, 0], [1, 1], [2, 2]],
                      [[10, 10], [11, 10], [10, 11]]], dtype=np.float64),
            (0, 0, 2, 2), {"all": (0, 0, 2, 2)},
        )
        self.assertEqual(result["covered_pixel_count"], 0)

    def test_coverage_rejects_wrong_dtype_shape_nonfinite_and_face_limit(self):
        for xy in (
            np.zeros((1, 3, 2), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float64),
            np.zeros((6001, 3, 2), dtype=np.float64),
            np.full((1, 3, 2), np.nan, dtype=np.float64),
            np.full((1, 3, 2), np.inf, dtype=np.float64),
        ):
            with self.subTest(shape=xy.shape, dtype=xy.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.coverage_proxies(xy)

    def test_coverage_rejects_invalid_or_unbounded_windows(self):
        xy = np.empty((0, 3, 2), dtype=np.float64)
        for roi, proxies in (
            ((0, 0, 0, 2), {}),
            ((0, 0, 48, 2), {}),
            ((0, 0, 2, 44), {}),
            ((0.0, 0, 2, 2), {}),
            ((0, 0, 2, 2), {"outside": (1, 1, 3, 3)}),
            ((0, 0, 2, 2), {"empty": (1, 1, 1, 2)}),
        ):
            with self.subTest(roi=roi, proxies=proxies):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.coverage_proxies(xy, roi, proxies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
