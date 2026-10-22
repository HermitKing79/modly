"""Synthetic saved-readback attribution only; no MeshLib or real patch arrays."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
try:
    from analyze import analyze_arrays
except ModuleNotFoundError:
    analyze_arrays = None


def fixture(faces=None, vertices=None, failed=(1,), split=()):
    if faces is None:
        faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    f = np.array(faces, np.int32); v = np.array(vertices, np.float32)
    failed = np.array(sorted(failed), np.int32)
    vf = np.setdiff1d(np.arange(len(f), dtype=np.int32), failed)
    rf = f[vf].copy(); groups = {}
    for row, tri in enumerate(rf):
        for corner in range(3):
            pair = tuple(sorted((int(tri[corner]), int(tri[(corner + 1) % 3]))))
            groups.setdefault(pair, []).append((row, corner))
    eid = np.empty_like(rf); hid = np.empty_like(rf)
    left = np.repeat(vf[:, None], 3, axis=1); right = np.full_like(rf, -1)
    next_edge = 0
    for pair, occurrences in sorted(groups.items()):
        parts = [[x] for x in occurrences] if pair in split else [occurrences]
        for group in parts:
            for side, (row, corner) in enumerate(group):
                eid[row, corner] = next_edge; hid[row, corner] = 2 * next_edge + side
                if len(group) == 2: right[row, corner] = vf[group[1 - side][0]]
            next_edge += 1
    original = dict(vertices=v, faces=f, source_vertex_ids=np.arange(100, 100 + len(v), dtype=np.int32),
                    source_face_ids=np.arange(500, 500 + len(f), dtype=np.int32))
    saved = dict(vertices_readback=v.astype(np.float64), faces_readback=rf, valid_face_ids=vf,
                 valid_vertex_ids=np.unique(rf).astype(np.int32), failed_face_ids=failed,
                 native_undirected_edge_ids=eid, native_halfedge_ids=hid,
                 native_left_faces=left, native_right_faces=right,
                 vertices_after_native=v.copy(), faces_after_native=f.copy(),
                 source_vertex_ids=original['source_vertex_ids'].copy(), source_face_ids=original['source_face_ids'].copy())
    return original, saved, failed


class SavedAttribution(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(analyze_arrays, 'Attribution helper is not implemented')

    def test_manifold_face_loss_is_described_without_a_causal_claim(self):
        public, private = analyze_arrays(np, *fixture())
        self.assertEqual(public['conclusion'], 'CAUSE_UNRESOLVED_PARK')
        self.assertEqual(public['failed_face_count'], 1)
        witness = public['witnesses'][0]
        self.assertEqual(witness['label'], 'W1')
        self.assertEqual(witness['original_edge_class_counts'], dict(boundary=0, manifold=3, nonmanifold=0))
        self.assertTrue(witness['positive_area'])
        self.assertFalse(witness['prefix_is_native_insertion_state'])
        self.assertEqual(private['witnesses'][0]['original_local_face_id'], 1)

    def test_complete_vertex_stars_and_two_open_link_components_are_descriptive(self):
        original, saved, failed = fixture([[0, 1, 2], [0, 3, 4]],
                                          [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        public, private = analyze_arrays(np, original, saved, failed)
        star = next(x for x in private['witnesses'][0]['vertex_stars'] if x['local_vertex_id'] == 0)
        self.assertEqual(star['original_face_ids'], [0, 1])
        self.assertEqual(star['links']['original']['component_count'], 2)
        self.assertEqual(star['links']['original']['open_chain_components'], 2)
        self.assertEqual(star['links']['prefix']['component_count'], 1)
        self.assertEqual(star['links']['retained']['component_count'], 1)
        self.assertEqual(public['conclusion'], 'CAUSE_UNRESOLVED_PARK')
        self.assertFalse(public['native_rejection_cause_proven'])

    def test_co_directed_manifold_count_edge_and_earlier_competitor(self):
        args = fixture([[0, 1, 2], [0, 1, 3]],
                       [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]])
        public, private = analyze_arrays(np, *args)
        witness = public['witnesses'][0]
        self.assertEqual(witness['original_edge_class_counts'], dict(boundary=2, manifold=1, nonmanifold=0))
        self.assertEqual(witness['co_directed_original_edge_conflicts'], 1)
        self.assertEqual(witness['earlier_same_direction_edge_competitors'], 1)
        edge = next(x for x in private['witnesses'][0]['edge_stars'] if x['local_endpoint_ids'] == [0, 1])
        self.assertEqual(edge['earlier_same_direction_face_ids'], [0])

    def test_actual_split_endpoint_pair_and_lost_vertex_context(self):
        args = fixture([[0, 1, 2], [1, 0, 3], [4, 5, 6]],
                       [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
                       failed=(2,), split=((0, 1),))
        public, private = analyze_arrays(np, *args)
        self.assertEqual(public['split_endpoint_pair_count'], 1)
        self.assertEqual(public['split_pairs'][0], dict(label='S1', original_incidence=2, retained_incidence=2,
            native_edge_count=2, native_singleton_edges=2, native_paired_edges=0, original_co_directed_conflict=False))
        self.assertEqual(private['split_endpoint_pairs'][0]['label'], public['split_pairs'][0]['label'])
        self.assertEqual(len(private['split_endpoint_pairs'][0]['native_groups']), 2)
        self.assertEqual(public['lost_valid_vertex_count'], 3)
        self.assertTrue(all(x['original_face_ids'] == [2] and x['retained_face_ids'] == [] for x in private['lost_valid_vertices']))

    def test_zero_area_is_recorded_without_fix_or_native_cause(self):
        args = fixture([[0, 1, 2], [0, 1, 3]],
                       [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]])
        public, private = analyze_arrays(np, *args)
        self.assertFalse(public['witnesses'][0]['positive_area'])
        self.assertEqual(private['witnesses'][0]['area'], 0.0)
        self.assertIsNone(private['witnesses'][0]['unit_normal'])
        self.assertFalse(public['native_rejection_cause_proven'])

    def test_multiple_failed_faces_have_stable_source_order_labels(self):
        public, private = analyze_arrays(np, *fixture(failed=(3, 1)))
        self.assertEqual([x['label'] for x in public['witnesses']], ['W1', 'W2'])
        self.assertEqual([x['original_local_face_id'] for x in private['witnesses']], [1, 3])

    def test_inconsistent_failed_partition_rejects(self):
        original, saved, failed = fixture()
        with self.assertRaises(ValueError): analyze_arrays(np, original, saved, np.array([2], np.int32))

    def test_public_output_has_no_private_geometry_or_ids_and_inputs_unchanged(self):
        original, saved, failed = fixture()
        before = {('o', k): a.tobytes() for k, a in original.items()}
        before.update({('s', k): a.tobytes() for k, a in saved.items()})
        public, private = analyze_arrays(np, original, saved, failed)
        encoded = json.dumps(public, allow_nan=False)
        for forbidden in ['source_face_id', 'source_vertex_id', 'local_face_id', 'local_vertex_id', 'coordinates', 'unit_normal', 'native_halfedge_id']:
            self.assertNotIn(forbidden, encoded)
        self.assertTrue(private['context_faces'])
        for (kind, name), value in before.items(): self.assertEqual((original if kind == 'o' else saved)[name].tobytes(), value)

    def test_common_saved_array_handoff_keeps_authenticated_I32_failed_array(self):
        import common
        original, saved, _ = fixture(failed=(0, 1, 2, 3))
        with tempfile.TemporaryDirectory(prefix='synthetic-handoff-', dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp); directory = root / 'import/brick'; directory.mkdir(parents=True)
            inputs = {}; schemas = {}
            for name, value in original.items():
                path = root / (name + '.npy'); np.save(path, value, allow_pickle=False)
                inputs[name] = dict(path=str(path)); schemas[name] = dict(dtype=str(value.dtype), shape=list(value.shape))
            for name, value in saved.items(): np.save(directory / (name + '.npy'), value, allow_pickle=False)
            (directory / 'result.json').write_text(json.dumps(dict(files={name + '.npy': {} for name in saved})))
            plan = dict(patches=[dict(inputfiles=inputs, array_schemas=schemas)])
            with patch.object(common, 'PRIOR_P', root), patch.object(common.prior, 'load_plan', return_value=plan):
                actual_original, actual_saved, actual_failed = common.load_inputs(np)
                self.assertIsInstance(actual_failed, np.ndarray)
                self.assertEqual(actual_failed.dtype, np.int32)
                public, _ = analyze_arrays(np, actual_original, actual_saved, actual_failed)
                self.assertEqual(public['failed_face_count'], 4)
                del actual_original, actual_saved, actual_failed


if __name__ == '__main__': unittest.main(verbosity=2)
