"""Synthetic CPU fixtures only; no production mesh, GPU, renderer, or export."""
import unittest
from types import SimpleNamespace

import numpy as np

from local_geometry import loop_metrics, multihit, screen_components, trace_component


def ray_fixture(depths=(2., 2., 3., 20.)):
    # +X outward, right +Y, up +Z. Every center hit is (.25, .25, .5).
    vertices = np.concatenate([np.array([[3-d, -1., -1.], [3-d, 1., -1.],
                                         [3-d, 0., 1.]]) for d in depths])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    if len(faces) > 1:
        faces[1] = faces[1, ::-1]  # Both original windings must survive.
    screen = np.column_stack((50 + vertices[:, 1] * 50, 50 - vertices[:, 2] * 50))
    projection = SimpleNamespace(screen=screen, depth=3-vertices[:, 0],
                                 outward=np.array([1., 0., 0.]), width=100,
                                 height=100, clip=(.1, 10.))
    xy = screen[faces]
    bounds = SimpleNamespace(minimum=np.nextafter(xy.min(axis=1).astype(np.float32),
                                                  -np.inf),
                             maximum=np.nextafter(xy.max(axis=1).astype(np.float32),
                                                  np.inf))
    return vertices, faces, projection, bounds


class LocalGeometryTests(unittest.TestCase):
    def test_screen_matches_full_traces_for_cycle_open_and_branching_components(self):
        pairs = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5],
                          [6, 7], [7, 8], [8, 6], [7, 9]])
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3])
        result = screen_components(pairs, labels, np.empty((0, 2), dtype=int))
        np.testing.assert_array_equal(result["degree"], [2, 2, 2, 1, 2, 1, 2, 3, 2, 1, 0])
        for key, expected in (("vertex_counts", [3, 3, 4, 0]), ("edge_counts", [3, 2, 4, 0]),
                              ("endpoints", [0, 2, 1, 0]), ("branches", [0, 0, 1, 0]),
                              ("closed_cycle_mask", [True, False, False, False])):
            np.testing.assert_array_equal(result[key], expected)
        for label, seed in enumerate(([0, 1], [3, 4], [6, 7])):
            traced = trace_component(pairs, seed)
            self.assertEqual(bool(result["closed_cycle_mask"][label]), traced["closed_simple"])
            self.assertEqual(result["vertex_counts"][label], traced["vertex_count"])
            self.assertEqual(result["edge_counts"][label], traced["edge_count"])

    def test_screen_counts_unique_nm_edges_once_per_touched_component(self):
        pairs = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
        labels = np.array([0, 0, 0, 1, 1, 1, 2])
        # Same-component endpoints and reversed duplicate rows count once.
        nm = np.array([[0, 1], [0, 3], [3, 0], [1, 0], [5, 6], [6, 6]])
        result = screen_components(pairs, labels, nm)
        np.testing.assert_array_equal(result["nm_touches"], [2, 2, 0])
        np.testing.assert_array_equal(result["closed_cycle_mask"], [True, True, False])
        np.testing.assert_array_equal(result["isolated_cycle_mask"], [False, False, False])

    def test_screen_preserves_separate_original_ids_even_at_coincident_coordinates(self):
        # Vertices 0:3 and 3:6 may have identical coordinates: IDs control topology.
        pairs = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
        result = screen_components(pairs, np.array([0, 0, 0, 1, 1, 1, 2]), np.array([[0, 6]]))
        np.testing.assert_array_equal(result["vertex_counts"], [3, 3, 0])
        np.testing.assert_array_equal(result["nm_touches"], [1, 0, 0])
        np.testing.assert_array_equal(result["isolated_cycle_mask"], [False, True, False])

    def test_screen_ignores_unused_isolated_vertex_labels_and_accepts_empty_graph(self):
        pairs = np.array([[0, 1], [1, 2], [2, 0]])
        labels = np.array([0, 0, 0, 1, 2, 3, 4, 5])
        result = screen_components(pairs, labels, np.empty((0, 2), dtype=int))
        np.testing.assert_array_equal(result["vertex_counts"], [3, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["isolated_cycle_mask"], [True, False, False, False, False, False])
        empty = screen_components(np.empty((0, 2), dtype=int), np.array([], dtype=int),
                                  np.empty((0, 2), dtype=int))
        self.assertTrue(all(value.size == 0 for value in empty.values()))

    def test_screen_rejects_inconsistent_labels_and_invalid_original_indices(self):
        for pairs, labels, nm in (([[0, 1]], [0, 1], []), ([[0, 2]], [0, 0], []),
                                  ([[0, 1]], [0, 0], [[0, 2]]),
                                  ([[0, 1]], [0, -1], []), ([[0, 1]], [0, 2], []),
                                  ([[0, 1]], ["0", "0"], [])):
            with self.subTest(pairs=pairs, labels=labels, nm=nm), self.assertRaises(ValueError):
                screen_components(np.array(pairs), np.array(labels), np.array(nm, dtype=int).reshape(-1, 2))

    def test_closed_loop_is_complete_deterministic_original_id_cycle(self):
        pairs = np.array([[9, 7], [3, 7], [9, 3], [20, 21]])
        result = trace_component(pairs, [7, 9])
        self.assertTrue(result["closed_simple"])
        self.assertEqual(result["edge_ids"], [0, 1, 2])
        self.assertEqual(result["edges"], [[9, 7], [3, 7], [9, 3]])
        self.assertEqual(result["vertices"], [3, 7, 9])
        self.assertEqual(result["degree_histogram"], {"2": 3})
        self.assertEqual(result["ordered_loop"], [3, 7, 9])

    def test_open_chain_has_no_fabricated_loop_order(self):
        result = trace_component(np.array([[2, 4], [4, 8]]), [4, 8])
        self.assertFalse(result["closed_simple"])
        self.assertIsNone(result["ordered_loop"])
        self.assertEqual(result["degree_histogram"], {"1": 2, "2": 1})

    def test_branching_junction_retains_every_component_edge(self):
        result = trace_component(np.array([[0, 1], [1, 2], [2, 0], [1, 3]]), [0, 1])
        self.assertEqual(len(result["edges"]), 4)
        self.assertEqual(result["degree_histogram"], {"1": 1, "2": 2, "3": 1})
        self.assertFalse(result["closed_simple"])
        self.assertIsNone(result["ordered_loop"])

    def test_invalid_seed_and_graph_inputs_fail_closed(self):
        for pairs, seed in [(np.array([[1, 2]]), [8, 9]),
                            (np.array([[1, 2]]), [1, 1]),
                            (np.array([[1., 2.]]), [1, 2]),
                            (np.array([[-1, 2]]), [-1, 2]),
                            (np.array([[1, 2]]), ["1", "2"]),
                            (np.zeros((600001, 2), dtype=np.int32), [0, 0])]:
            with self.subTest(shape=pairs.shape, seed=seed), self.assertRaises(ValueError):
                trace_component(pairs, seed)

    def test_coincident_coordinates_do_not_weld_original_ids(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
        result = trace_component(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), [0, 1])
        self.assertEqual(result["ordered_loop"], [0, 1, 2, 3])
        metrics = loop_metrics(vertices, result["ordered_loop"])
        self.assertTrue(metrics["projected_self_intersection"])

    def test_multihit_keeps_ties_both_windings_and_clip_exclusions(self):
        vertices, faces, projection, bounds = ray_fixture()
        result = multihit(vertices, faces, projection, [[50., 50.]], bounds,
                          pixel_centers=False)[0]
        self.assertEqual([hit["face_id"] for hit in result["hits"]], [0, 1, 2])
        self.assertEqual(result["intersection_count"], 3)
        self.assertEqual(result["clipped_intersection_count"], 1)
        self.assertIn("coincident_depth_hits", result["ambiguities"])
        self.assertIn("intersections_outside_clip", result["ambiguities"])
        first, second = result["hits"][:2]
        self.assertEqual(first["original_vertex_ids"], [0, 1, 2])
        np.testing.assert_allclose(first["barycentric"], [.25, .25, .5])
        np.testing.assert_allclose(first["position"], [1., 0., 0.])
        self.assertEqual(first["normal_cos_camera"], 1.)
        self.assertEqual(second["normal_cos_camera"], -1.)
        self.assertEqual(first["tied_depth_face_ids"], [0, 1])

    def test_multihit_pixel_center_edge_ambiguity_and_no_hit_truncation(self):
        vertices, faces, projection, bounds = ray_fixture((2.,))
        result = multihit(vertices, faces, projection, [[49, 49]], bounds)[0]
        self.assertEqual(result["sample_xy"], [49.5, 49.5])
        edge = multihit(vertices, faces, projection, [[50., 100.]], bounds,
                        pixel_centers=False)[0]
        self.assertIn("on_triangle_edge", edge["hits"][0]["ambiguities"])
        vertices, faces, projection, bounds = ray_fixture((2.,) * 129)
        with self.assertRaises(ValueError):
            multihit(vertices, faces, projection, [[50., 50.]], bounds,
                     pixel_centers=False)

    def test_nonnumeric_nonfinite_and_ray_budget_are_rejected(self):
        vertices, faces, projection, bounds = ray_fixture((2.,))
        for points in ([["bad", 1]], [[np.nan, 1]], [[50, 50]] * 122):
            with self.subTest(points=points[:1]), self.assertRaises(ValueError):
                multihit(vertices, faces, projection, points, bounds)
        for operation in (lambda: trace_component(np.array([[0, 1]]), 0),
                          lambda: loop_metrics(vertices, 0)):
            with self.assertRaises(ValueError):
                operation()
        for bad_vertices in ([["bad", 0, 0]] * 3, [[np.inf, 0, 0]] * 3):
            with self.assertRaises(ValueError):
                multihit(bad_vertices, faces, projection, [[50, 50]], bounds)
            with self.assertRaises(ValueError):
                loop_metrics(bad_vertices, [0, 1, 2])

    def test_loop_metrics_square_and_crossing_have_hand_checked_values(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
        result = loop_metrics(vertices, [0, 1, 2, 3])
        self.assertTrue(result["supported"])
        self.assertAlmostEqual(result["perimeter"], 8.)
        self.assertAlmostEqual(result["projected_shoelace_area"], 4.)
        self.assertEqual(result["bounds"], [[0., 0., 0.], [2., 2., 0.]])
        self.assertAlmostEqual(result["max_plane_deviation"], 0.)
        self.assertFalse(result["projected_self_intersection"])
        crossing = loop_metrics(vertices, [0, 2, 1, 3])
        self.assertTrue(crossing["projected_self_intersection"])
        self.assertEqual(crossing["projected_intersecting_segment_pairs"], [[0, 2]])

    def test_nonplanar_projection_cannot_prove_3d_intersection_and_size_is_bounded(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 1.], [0., 2., 0.]])
        result = loop_metrics(vertices, [0, 2, 1, 3])
        self.assertFalse(result["planar_within_tolerance"])
        self.assertIsNone(result["self_intersection_3d"])
        self.assertEqual(result["self_intersection_3d_qualification"],
                         "not_proven_for_nonplanar_loop")
        bounded = loop_metrics(np.zeros((513, 3)), np.arange(513))
        self.assertFalse(bounded["supported"])
        self.assertEqual(bounded["reason"], "loop_vertex_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
