"""Tiny synthetic patch contracts; never open a model array or run a worker."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))

import unittest
import numpy as np
from patch_contract import choose_component, complete_stars, extract, full_link, greedy_capacity, raw_edge_components


def edge_index(f, n):
    """Independent fixture encoder: occurrence = original face ID * 3 + side."""
    entries = []
    for face_id, face in enumerate(f):
        for side in range(3):
            a, b = int(face[side]), int(face[(side + 1) % 3])
            entries.append((min(a, b) * n + max(a, b), face_id * 3 + side))
    entries.sort()
    keys = np.array(sorted({key for key, _ in entries}), dtype=np.uint64)
    counts = [sum(key == int(source_key) for key, _ in entries) for source_key in keys]
    order = np.array([occurrence for _, occurrence in entries], dtype=np.int32)
    offsets = np.array([0] + list(np.cumsum(counts)), dtype=np.int32)
    ids = {int(key): i for i, key in enumerate(keys)}
    return keys, order, offsets, ids


def edge_ids(ids, n, pairs):
    return np.array([ids[min(a, b) * n + max(a, b)] for a, b in pairs], dtype=np.int32)


def ring_fixture():
    # 0--1 is the seed edge. Face 2 belongs to endpoint 0's complete star.
    # Faces 3, 4, 5, 6 are successive edge rings; face 7 touches only a vertex.
    f = np.array([
        [0, 1, 2], [1, 0, 3], [0, 3, 4], [3, 4, 5],
        [4, 5, 6], [5, 6, 7], [6, 7, 8], [2, 9, 10],
    ], dtype=np.int32)
    n = 11
    keys, order, offsets, ids = edge_index(f, n)
    return f, keys, order, offsets, ids[1], n


class PatchContractTests(unittest.TestCase):
    def test_raw_edge_components_do_not_join_vertex_only_contacts(self):
        self.assertEqual(raw_edge_components(np.array([[0,1,2],[2,1,3]],np.int32)),1)
        self.assertEqual(raw_edge_components(np.array([[0,1,2],[0,3,4]],np.int32)),2)

    def assert_i32(self, value, expected):
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.dtype, np.dtype(np.int32))
        self.assertEqual(value.ndim, 1)
        np.testing.assert_array_equal(value, expected)

    def test_largest_eligible_face_component_beats_more_edges_and_uses_lower_median(self):
        # Choosing by eligible-edge count would select the four-face tetrahedron.
        f = np.array([
            [0, 1, 2], [1, 0, 3], [0, 2, 3], [1, 3, 2],
            [10, 11, 12], [12, 11, 13], [12, 13, 14],
            [14, 13, 15], [14, 15, 16],
        ], dtype=np.int32)
        n = 17
        keys, order, offsets, ids = edge_index(f, n)
        tetra = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        chain = [(11, 12), (12, 13), (13, 14), (14, 15)]
        eligible = edge_ids(ids, n, list(reversed(chain + tetra)))
        result = choose_component(f, keys, order, offsets, eligible)
        self.assert_i32(result['component_face_ids'], [4, 5, 6, 7, 8])
        self.assert_i32(result['component_edge_ids'], edge_ids(ids, n, chain))
        self.assertEqual(result['seed_edge_id'], ids[12 * n + 13])
        self.assertEqual(result['eligible_component_count'], 2)

    def test_equal_face_count_tie_uses_original_face_id_and_ignores_ineligible_links(self):
        # The lower source-key edge is in the losing component. The final face
        # shares an ineligible edge with face 1 and must not enlarge that component.
        f = np.array([
            [10, 11, 12], [0, 1, 2], [2, 1, 3], [12, 11, 13], [0, 2, 4],
        ], dtype=np.int32)
        n = 14
        keys, order, offsets, ids = edge_index(f, n)
        eligible = edge_ids(ids, n, [(1, 2), (11, 12)])
        result = choose_component(f, keys, order, offsets, eligible)
        self.assert_i32(result['component_face_ids'], [0, 3])
        self.assert_i32(result['component_edge_ids'], [ids[11 * n + 12]])
        self.assertEqual(result['seed_edge_id'], ids[11 * n + 12])
        self.assertEqual(result['eligible_component_count'], 2)

    def test_complete_stars_preserve_original_face_ids_and_exclude_incomplete_vertex(self):
        f = np.array([[0, 2, 1], [1, 2, 3], [0, 4, 5]], dtype=np.int32)
        source_ids = np.array([7, 2, 11], dtype=np.int32)
        degree = np.array([2, 2, 3, 1, 1, 1], dtype=np.int32)
        stars = complete_stars(f, source_ids, degree)
        self.assertEqual(set(stars), {0, 1, 3, 4, 5})
        for vertex, expected in {0: [7, 11], 1: [2, 7], 3: [2], 4: [11], 5: [11]}.items():
            with self.subTest(vertex=vertex):
                self.assert_i32(stars[vertex], expected)

    def test_extract_uses_complete_endpoint_stars_then_exactly_three_synchronous_edge_rings(self):
        f, keys, order, offsets, seed, n = ring_fixture()
        result = extract(f, keys, order, offsets, seed, n, max_faces=6)
        self.assert_i32(result['seed_edge_face_ids'], [0, 1])
        self.assert_i32(result['seed_face_ids'], [0, 1, 2])
        self.assertEqual(len(result['ring_added_face_ids']), 3)
        for added, expected in zip(result['ring_added_face_ids'], ([3], [4], [5])):
            self.assert_i32(added, expected)
        self.assert_i32(result['source_face_ids'], [0, 1, 2, 3, 4, 5])

    def test_extract_fails_on_seed_or_later_cumulative_limit_instead_of_truncating(self):
        f, keys, order, offsets, seed, n = ring_fixture()
        for cap in (2, 5):
            with self.subTest(max_faces=cap), self.assertRaises(ValueError):
                extract(f, keys, order, offsets, seed, n, max_faces=cap)

    def test_full_link_accepts_open_pair_but_rejects_tetrahedron_common_opposite_edge(self):
        pair = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
        pair_star = np.array([0, 1], dtype=np.int32)
        self.assertTrue(full_link(pair, 0, 1, pair_star, pair_star))
        tetra = np.array([[0, 1, 2], [1, 0, 3], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
        self.assertFalse(full_link(tetra, 0, 1,
                                   np.array([0, 1, 2], dtype=np.int32),
                                   np.array([0, 1, 3], dtype=np.int32)))

    def test_full_link_uses_global_original_stars_to_reject_off_crop_common_neighbour(self):
        f = np.array([
            [9, 10, 11], [0, 1, 2], [1, 0, 3], [0, 4, 5], [1, 6, 4],
        ], dtype=np.int32)
        cropped = np.array([1, 2], dtype=np.int32)
        self.assertTrue(full_link(f, 0, 1, cropped, cropped))
        self.assertFalse(full_link(f, 0, 1,
                                   np.array([1, 2, 3], dtype=np.int32),
                                   np.array([1, 2, 4], dtype=np.int32)))

    def test_greedy_capacity_orders_source_keys_and_requires_disjoint_closed_neighbourhoods(self):
        # Edges 0--1 and 4--5 have disjoint endpoints but share neighbour 2.
        f = np.array([
            [0, 1, 2], [1, 0, 3], [4, 5, 2], [5, 4, 6], [7, 8, 9], [8, 7, 10],
        ], dtype=np.int32)
        original = f.copy()
        n = 11
        keys, _, _, ids = edge_index(f, n)
        eligible = edge_ids(ids, n, [(7, 8), (4, 5), (0, 1)])
        stars = {vertex: np.array(face_ids, dtype=np.int32) for vertex, face_ids in {
            0: [0, 1], 1: [0, 1], 4: [2, 3], 5: [2, 3], 7: [4, 5], 8: [4, 5],
        }.items()}
        selected = greedy_capacity(f, keys, eligible, stars, n)
        self.assert_i32(selected, edge_ids(ids, n, [(0, 1), (7, 8)]))
        incomplete = dict(stars)
        del incomplete[0]
        self.assert_i32(greedy_capacity(f, keys, eligible, incomplete, n),
                        edge_ids(ids, n, [(4, 5), (7, 8)]))
        np.testing.assert_array_equal(f, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
