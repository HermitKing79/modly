"""Synthetic only: ``python -I -B test_chart_forest.py``."""

import sys
sys.dont_write_bytecode = True

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "connected_chart_forest", Path(__file__).with_name("chart_forest.py")
)
chart_forest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chart_forest)


class ForestTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        self.vertex_ids = np.array([40, 10, 30, 20, 50], dtype=np.int64)

    def assert_exact_corners(self, result, vertices, faces, face_ids, vertex_ids):
        np.testing.assert_array_equal(
            result["seam_vertices"][result["seam_faces"]].view(np.uint32),
            vertices[faces].view(np.uint32),
        )
        np.testing.assert_array_equal(result["source_face_ids"], face_ids)
        np.testing.assert_array_equal(
            result["source_vertex_ids"][result["seam_faces"]], vertex_ids[faces]
        )
        np.testing.assert_array_equal(
            result["source_local_vertex_ids"][result["seam_faces"]], faces
        )
        flat_corners = faces.reshape(-1)
        for new_vertex, first_corner in enumerate(result["source_corner_indices"]):
            self.assertEqual(
                result["source_local_vertex_ids"][new_vertex], flat_corners[first_corner]
            )

    def test_opposite_winding_edge_is_retained_without_changing_corners(self):
        faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
        face_ids = np.array([200, 100], dtype=np.int32)
        vertices_before, faces_before = self.vertices.copy(), faces.copy()
        result = chart_forest.forest(self.vertices, faces, face_ids, self.vertex_ids)
        np.testing.assert_array_equal(result["retained_source_edges"], [[10, 40]])
        np.testing.assert_array_equal(result["face_chart_ids"], [0, 0])
        np.testing.assert_array_equal(result["source_corner_indices"], [0, 1, 2, 5])
        self.assertEqual(result["report"]["chart_count"], 1)
        self.assertEqual(result["report"]["charts"][0]["vertex_count"], 4)
        self.assertEqual(result["report"]["charts"][0]["edge_count"], 5)
        self.assert_exact_corners(result, self.vertices, faces, face_ids, self.vertex_ids)
        np.testing.assert_array_equal(self.vertices.view(np.uint32), vertices_before.view(np.uint32))
        np.testing.assert_array_equal(faces, faces_before)

    def test_same_winding_edge_is_a_seam_and_chart_order_uses_source_face(self):
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        face_ids = np.array([200, 100], dtype=np.int32)
        result = chart_forest.forest(self.vertices, faces, face_ids, self.vertex_ids)
        self.assertEqual(result["retained_source_edges"].shape, (0, 2))
        np.testing.assert_array_equal(result["face_chart_ids"], [1, 0])
        np.testing.assert_array_equal(result["seam_faces"], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(result["report"]["orientation_inconsistent_edge_count"], 1)
        self.assert_exact_corners(result, self.vertices, faces, face_ids, self.vertex_ids)

    def test_nonmanifold_edge_is_never_glued(self):
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32)
        face_ids = np.array([30, 10, 20], dtype=np.int32)
        result = chart_forest.forest(self.vertices, faces, face_ids, self.vertex_ids)
        self.assertEqual(result["report"]["nonmanifold_edge_count"], 1)
        self.assertEqual(result["report"]["chart_count"], 3)
        self.assertEqual(result["report"]["retained_edge_count"], 0)
        np.testing.assert_array_equal(result["face_chart_ids"], [2, 0, 1])
        self.assert_exact_corners(result, self.vertices, faces, face_ids, self.vertex_ids)

    def test_disconnected_and_vertex_touching_triangles_remain_separate(self):
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
        face_ids = np.array([4, 5], dtype=np.int32)
        result = chart_forest.forest(self.vertices, faces, face_ids, self.vertex_ids)
        self.assertEqual(result["report"]["chart_count"], 2)
        self.assertEqual(result["seam_vertices"].shape, (6, 3))
        np.testing.assert_array_equal(result["source_corner_indices"], np.arange(6))

    def test_closed_tetrahedron_retains_longest_dual_tree_and_becomes_disk(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
        face_ids = np.array([8, 6, 7, 5], dtype=np.int32)
        vertex_ids = np.arange(4, dtype=np.int64)
        result = chart_forest.forest(vertices, faces, face_ids, vertex_ids)
        np.testing.assert_array_equal(result["retained_source_edges"], [[1, 2], [1, 3], [2, 3]])
        report = result["report"]
        self.assertEqual(report["cycle_cut_edge_count"], 3)
        self.assertEqual(report["chart_count"], 1)
        chart = report["charts"][0]
        for key, expected in {"face_count": 4, "vertex_count": 6, "edge_count": 9,
                              "boundary_edge_count": 6, "boundary_loop_count": 1,
                              "euler_characteristic": 1}.items():
            self.assertEqual(chart[key], expected)
        self.assert_exact_corners(result, vertices, faces, face_ids, vertex_ids)

    def test_equal_length_cycle_ties_use_original_edge_ids_deterministically(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=np.int32)
        face_ids = np.array([50, 30, 20, 40], dtype=np.int64)
        vertex_ids = np.arange(5, dtype=np.int64)
        first = chart_forest.forest(vertices, faces, face_ids, vertex_ids)
        np.testing.assert_array_equal(first["retained_source_edges"], [[0, 1], [0, 2], [0, 3]])
        # Reorder both local vertices and faces while retaining their source identity.
        perm = np.array([3, 0, 4, 2, 1])
        inverse = np.argsort(perm)
        face_perm = np.array([2, 0, 3, 1])
        reordered_faces = inverse[faces[face_perm]].astype(np.int32)
        second = chart_forest.forest(
            vertices[perm], reordered_faces, face_ids[face_perm], vertex_ids[perm]
        )
        np.testing.assert_array_equal(first["retained_source_edges"], second["retained_source_edges"])
        np.testing.assert_array_equal(first["retained_source_face_pairs"], second["retained_source_face_pairs"])
        self.assert_exact_corners(second, vertices[perm], reordered_faces,
                                  face_ids[face_perm], vertex_ids[perm])

    def test_coordinate_signed_zero_bits_are_preserved(self):
        vertices = self.vertices.copy()
        vertices[0, 0] = np.float32(-0.0)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        face_ids = np.array([9], dtype=np.int64)
        result = chart_forest.forest(vertices, faces, face_ids, self.vertex_ids)
        self.assert_exact_corners(result, vertices, faces, face_ids, self.vertex_ids)

    def test_invalid_dtypes_bounds_ids_and_geometry_are_rejected(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        face_ids = np.array([9], dtype=np.int64)
        cases = [
            (self.vertices.astype(np.float64), faces, face_ids, self.vertex_ids),
            (self.vertices, faces.astype(np.int64), face_ids, self.vertex_ids),
            (self.vertices, np.array([[0, 1, 5]], dtype=np.int32), face_ids, self.vertex_ids),
            (self.vertices, np.array([[0, 0, 2]], dtype=np.int32), face_ids, self.vertex_ids),
            (self.vertices, faces, face_ids, np.array([40, 10, 10, 20, 50], dtype=np.int64)),
            (self.vertices, faces, np.array([], dtype=np.int64), self.vertex_ids),
            (self.vertices, faces, face_ids, self.vertex_ids.astype(np.float64)),
            (np.full((5, 3), np.nan, dtype=np.float32), faces, face_ids, self.vertex_ids),
            (np.zeros((5, 3), dtype=np.float32), faces, face_ids, self.vertex_ids),
            (self.vertices, np.tile(faces, (6001, 1)), np.arange(6001), self.vertex_ids),
            (self.vertices, np.tile(faces, (2, 1)), np.array([9, 9]), self.vertex_ids),
        ]
        for vertices, selected, sf, sv in cases:
            with self.subTest(vertex_dtype=vertices.dtype, face_shape=selected.shape):
                with self.assertRaises((TypeError, ValueError)):
                    chart_forest.forest(vertices, selected, sf, sv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
