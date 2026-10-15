"""Small independent connectivity checks; no Torch/GPU/native extractor."""
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import amd_flexible_dual_grid_cpu as fdg

OFFSETS=(
    ((0,0,0),(0,0,1),(0,1,1),(0,1,0)),
    ((0,0,0),(1,0,0),(1,0,1),(0,0,1)),
    ((0,0,0),(0,1,0),(1,1,0),(1,0,0)))


def oracle(coords,flags,weights):
    lookup={tuple(c):i for i,c in enumerate(coords.tolist())}
    faces=[];missing=0
    for i,c in enumerate(coords.tolist()):
        for axis in range(3):
            if not flags[i,axis]:continue
            corners=[tuple(c[j]+o[j] for j in range(3)) for o in OFFSETS[axis]]
            if not all(x in lookup for x in corners):missing+=1;continue
            q=[lookup[x] for x in corners]
            a=np.float32(weights[q[0]]*weights[q[2]])
            b=np.float32(weights[q[1]]*weights[q[3]])
            split=(0,1,2,0,2,3) if a>b else (0,1,3,3,1,2)
            faces.extend([[q[j] for j in split[:3]],[q[j] for j in split[3:]]])
    return np.asarray(faces,dtype=np.int32).reshape(-1,3),missing


class CanonicalFDG(unittest.TestCase):
    def test_exact_axis_order_winding_and_tie(self):
        c=np.array(list(itertools.product(range(2),repeat=3)),dtype=np.int32)
        for axis in range(3):
            flags=np.zeros((8,3),bool);flags[0,axis]=True
            lookup={tuple(x):i for i,x in enumerate(c.tolist())}
            q=[lookup[x] for x in OFFSETS[axis]]
            faces,stats=fdg.faces(c,flags,np.ones(8,np.float32),batch_rows=1)
            np.testing.assert_array_equal(faces,[[q[0],q[1],q[3]],[q[3],q[1],q[2]]])
            self.assertEqual(stats['split_2'],1)

    def test_both_splits_global_cross_batch_and_boundaries(self):
        c=np.array(list(itertools.product(range(3),repeat=3)),np.int32)
        c=c[np.random.default_rng(123).permutation(len(c))]
        flags=np.ones((len(c),3),bool)
        weights=(np.arange(len(c),dtype=np.float32)%7+1)/8
        expected,missing=oracle(c,flags,weights)
        for batch in (1,5,100):
            actual,stats=fdg.faces(c,flags,weights,batch_rows=batch)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(stats['missing_neighbour_quads'],missing)
            self.assertGreater(stats['split_1'],0);self.assertGreater(stats['split_2'],0)

    def test_missing_neighbour_and_no_linear_key_wrap(self):
        c=np.array([[0,0,511],[0,1,0],[1,1,0],[1,0,511]],np.int32)
        flags=np.zeros((4,3),bool);flags[0,1]=True
        actual,stats=fdg.faces(c,flags,np.ones(4,np.float32),batch_rows=1)
        self.assertEqual(actual.shape,(0,3));self.assertEqual(stats['missing_neighbour_quads'],1)

    def test_transforms_and_scale(self):
        f=np.zeros((2,7),np.float32);f[1,:3]=[100,-100,0];f[:,3:6]=[[0,1,-1],[1,-1,0]]
        d,flags,q=fdg.transform_features(f)
        np.testing.assert_array_equal(d[0],[.5,.5,.5])
        np.testing.assert_array_equal(d[1],[1.5,-.5,.5])
        np.testing.assert_array_equal(flags,[[False,True,False],[True,False,False]])
        np.testing.assert_allclose(q,np.float32(np.log(2)),rtol=1e-7)
        v=fdg.vertices(np.array([[0,0,0],[511,511,511]],np.int32),d)
        np.testing.assert_array_equal(v[0],np.full(3,np.float32(.5/512-.5)))
        self.assertEqual(v.dtype,np.float32)

    def test_invalid_inputs_fail(self):
        for c in (np.array([[0,0,0],[0,0,0]],np.int32),np.array([[512,0,0]],np.int32),np.array([[0,0,-1]],np.int32)):
            with self.assertRaises(ValueError):fdg.faces(c,np.ones((len(c),3),bool),np.ones(len(c),np.float32))
        with self.assertRaises(ValueError):fdg.transform_features(np.full((1,7),np.nan,np.float32))

    def test_binary_ply_preserves_all_vertices_and_faces(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[2,2,2]],np.float32)
        f=np.array([[0,1,2]],np.int32)
        evidence=Path(__file__).resolve().parents[2]/'docs/compatibility/amd-turret-cpu-mesh-extraction-evidence'
        # Preserve the tiny fixture under our workspace, never the host temp tree.
        p=Path(tempfile.mkdtemp(prefix='ply-fixture-',dir=evidence))/'mesh.ply'
        fdg.write_ply(p,v,f)
        actual_v,actual_f=fdg.read_ply(p)
        np.testing.assert_array_equal(v,actual_v);np.testing.assert_array_equal(f,actual_f)

if __name__=='__main__':unittest.main(verbosity=2)
