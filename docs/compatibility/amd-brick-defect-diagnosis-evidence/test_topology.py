"""Bounded, hand-counted CPU fixtures for original-index mesh topology."""

from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("topology.py")
analyze = None
if MODULE_PATH.is_file():
    specification = importlib.util.spec_from_file_location("brick_topology", MODULE_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    analyze = module.analyze


ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[4]
    / "artifacts/amd-brick-defect-diagnosis-1d1c1a6"
)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(analyze), "topology.analyze has not been implemented")
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="topology-fixture-", dir=ARTIFACT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_three_faces_sharing_one_edge_keep_all_triangle_multiplicities(self):
        # Losing any face's third edge, or merging face occurrences, breaks this.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32)
        result = analyze(vertices, faces, self.output)
        self.assertEqual(result["input_counts"], {
            "vertices": 5, "triangles": 3, "triangle_edge_occurrences": 9,
        })
        self.assertEqual(result["edges"], {
            "unique": 7, "boundary_incidence_1": 6, "manifold_incidence_2": 0,
            "nonmanifold_incidence_gt_2": 1, "max_incidence": 3, "self_edges": 0,
        })
        np.testing.assert_array_equal(
            np.load(result["private_cache"]["edge_keys"], mmap_mode="r"),
            [1, 2, 3, 4, 7, 8, 9],
        )
        np.testing.assert_array_equal(
            np.load(result["private_cache"]["incidence"], mmap_mode="r"),
            [3, 1, 1, 1, 1, 1, 1],
        )

    def test_components_use_shared_indices_and_count_unused_and_degenerate_faces(self):
        # Welding the equal-position vertices 0 and 3 incorrectly joins components.
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 0, 0], [0, 1, 0], [0, 0, 1],
            [99, 99, 99], [1, 1, 1],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [7, 7, 7]], dtype=np.int32)
        vertices_before, faces_before = vertices.copy(), faces.copy()
        vertices.flags.writeable = faces.flags.writeable = False
        result = analyze(vertices, faces, self.output)
        self.assertEqual(result["degeneracy"]["repeated_index_faces"], 1)
        self.assertEqual(result["degeneracy"]["zero_area_faces_float64"], 2)
        self.assertEqual(result["degeneracy"]["nonfinite_vertex_rows"], 0)
        self.assertEqual(result["components"]["vertex_components_including_unused"], 4)
        self.assertEqual(result["components"]["face_components"], 3)
        self.assertEqual(result["components"]["unused_vertices"], 1)
        self.assertEqual(result["components"]["tiny_face_components_le_10"], 3)
        self.assertEqual(result["components"]["largest"][0]["vertices"], 3)
        self.assertEqual(result["components"]["largest"][0]["faces"], 1)
        labels = np.load(result["private_cache"]["vertex_components"], mmap_mode="r")
        self.assertEqual(labels.dtype, np.dtype("int32"))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[0], labels[3])
        self.assertNotEqual(labels[6], labels[7])
        self.assertEqual(result["edges"]["self_edges"], 1)
        self.assertEqual(result["edges"]["nonmanifold_incidence_gt_2"], 1)
        np.testing.assert_array_equal(vertices, vertices_before)
        np.testing.assert_array_equal(faces, faces_before)

    def test_empty_face_set_keeps_isolated_vertices_and_empty_edge_caches(self):
        # Omitting unused vertices from the graph incorrectly reports zero components.
        result = analyze(np.zeros((2, 3), dtype=np.float32), np.empty((0, 3), np.int32), self.output)
        self.assertEqual(result["components"]["vertex_components_including_unused"], 2)
        self.assertEqual(result["components"]["face_components"], 0)
        self.assertEqual(result["components"]["unused_vertices"], 2)
        self.assertEqual(result["edges"]["unique"], 0)
        self.assertEqual(np.load(result["private_cache"]["edge_keys"]).shape, (0,))

    def test_invalid_indices_fail_before_writing_private_caches(self):
        # Negative indices must never silently address the last vertex through NumPy.
        with self.assertRaises(ValueError):
            analyze(np.zeros((3, 3), np.float32), np.array([[0, 1, -1]], np.int32), self.output)
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
