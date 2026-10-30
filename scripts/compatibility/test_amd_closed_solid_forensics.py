"""Small CPU fixtures; no saved mesh or neural runtime required."""
import itertools
import io
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import amd_closed_solid_forensics as d


class ForensicTests(unittest.TestCase):
    def test_edge_oracle_and_direction(self):
        faces = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 4]], np.int32)
        edges, counts, signs, order = d.edge_table(faces)
        expected = {}
        for tri in faces:
            for a, b in zip(tri, np.roll(tri, -1)):
                expected.setdefault(tuple(sorted((int(a), int(b)))), []).append(1 if a < b else -1)
        self.assertEqual(edges.tolist(), [list(k) for k in sorted(expected)])
        self.assertEqual(counts.tolist(), [len(expected[k]) for k in sorted(expected)])
        self.assertEqual(signs.tolist(), [sum(expected[k]) for k in sorted(expected)])
        self.assertEqual(int(((counts == 2) & (abs(signs) == 2)).sum()), 2)
        self.assertEqual(sorted(order.tolist()), list(range(9)))

    def test_both_diagonals_preserve_perimeters(self):
        q = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], np.int32)
        e, c, s, _ = d.edge_table(q)
        for split in ([0, 1, 2, 0, 2, 3], [0, 1, 3, 3, 1, 2]):
            t = q[:, split].reshape(-1, 3)
            te, tc, ts, _ = d.edge_table(t)
            self.assertEqual(te[tc == 1].tolist(), e[c == 1].tolist())
            self.assertEqual(int(((tc == 2) & (abs(ts) == 2)).sum()), 0)

    def test_three_quad_junction_is_not_split_artifact(self):
        q = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 6, 7]], np.int32)
        for split in ([0, 1, 2, 0, 2, 3], [0, 1, 3, 3, 1, 2]):
            e, c, _, _ = d.edge_table(q[:, split].reshape(-1, 3))
            self.assertEqual(e[c > 2].tolist(), [[0, 1]])
            self.assertEqual(c[c > 2].tolist(), [3])

    def test_tuple_neighbours_shuffled_and_missing(self):
        coords = np.array(list(itertools.product(range(2), repeat=3)), np.int32)[::-1].copy()
        flags = np.ones((8, 3), bool)
        quads, anchors, axes, missing = d.tuple_quads(coords, flags)
        self.assertEqual(len(quads), 6)
        self.assertEqual(len(missing), 18)
        self.assertEqual(len(anchors), len(axes))
        for quad, anchor, axis in zip(quads, anchors, axes):
            np.testing.assert_array_equal(coords[quad], coords[anchor] + d.OFFSETS[axis])

    def test_constraint_cycle_contradiction(self):
        good = d.parity_graph(3, [(0, 1, 1), (1, 2, 0), (0, 2, 1)])
        bad = d.parity_graph(3, [(0, 1, 1), (1, 2, 0), (0, 2, 0)])
        self.assertEqual(good['contradictory_constraints'], 0)
        self.assertEqual(bad['contradictory_constraints'], 1)
        self.assertEqual(good['components'], 1)

    def test_only_proper_axis_rotations(self):
        rotations = d.rotations()
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(m.ravel()) for m in rotations}), 24)
        for matrix in rotations:
            self.assertAlmostEqual(np.linalg.det(matrix), 1)
            np.testing.assert_array_equal(matrix @ matrix.T, np.eye(3))
        self.assertFalse(any(np.array_equal(m, [[0,1,0],[1,0,0],[0,0,1]]) for m in rotations))

    def test_recover_source_pair_both_splits_and_reject(self):
        for t in ([[0,1,2],[0,2,3]], [[0,1,3],[3,1,2]]):
            np.testing.assert_array_equal(d.recover_quads(np.array(t, np.int32)), [[0,1,2,3]])
        with self.assertRaises(ValueError):d.recover_quads(np.array([[0,1,2],[3,2,1]], np.int32))

    def test_seams_last_boundary_and_interior_edge(self):
        q=np.array([[0,1,2,3],[1,4,5,2]],np.int32)
        e,c,s,o=d.edge_table(q)
        result=d.batch_seams(c,o,np.array([0,32768]))
        self.assertEqual(e[result].tolist(),[[1,2]])

    def test_default_inert_and_worker_requires_GO(self):
        with patch.object(d,'analyze',side_effect=AssertionError('No work')):
            self.assertEqual(d.main([]),0)
            with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))):
                with self.assertRaisesRegex(RuntimeError,'GO'):d.main(['--worker','analyze'])

    def test_reused_policy_denies_neural_children_and_source_writes(self):
        import amd_closed_solid_benchmark as b
        guard=b.cpu_guard(d.BASE)
        for event,args in [('import',('torch',)),('import',('o_voxel',)),('subprocess.Popen',('wsl.exe',)),
                           ('socket.connect',()),('os.remove',(str(d.RAW/'mesh/mesh.ply'),))]:
            with self.assertRaises(RuntimeError):guard(event,args)
        with self.assertRaises(ValueError):guard('open',(str(d.RAW/'mesh/mesh.ply'),'w',os.O_WRONLY))
        guard('open',(str(d.BASE/'safe-fixture.txt'),'x',os.O_CREAT|os.O_WRONLY))

    def test_bounded_owner_route_has_no_neural_stage(self):
        import amd_zombie_cpu_mesh as owner
        with patch.object(owner,'supervise',return_value=0) as launch,patch.object(owner,'BASE',owner.BASE),patch.object(owner,'__file__',owner.__file__):
            self.assertEqual(d.main(['--run']),0)
            launch.assert_called_once_with('analyze')
            self.assertEqual(owner.BASE,d.BASE)
            self.assertEqual(owner.CAP,6*1024**3);self.assertEqual(owner.DEADLINE,180)
        self.assertFalse(owner.admission(12*1024**3-1,2*1024**3)['admitted'])

if __name__ == '__main__':unittest.main(verbosity=2)
