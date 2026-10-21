"""Synthetic diagnostics only; never selects or samples the genuine model."""
import sys
sys.dont_write_bytecode=True
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment import seam_queries


class SeamTests(unittest.TestCase):
    def test_shared_edge_and_inset_pairs_preserve_canonical_endpoint_order(self):
        f=np.array([[0,1,2],[1,0,3]],np.int32);ids=np.array([10,4,7,8],np.int64)
        q,b,c,p=seam_queries(f,ids,np.empty((0,2),np.int64))
        self.assertEqual(len(q),20);self.assertEqual(len(p),5)
        self.assertEqual((int((c==0).sum()),int((c==1).sum())),(10,10))
        np.testing.assert_allclose(b.sum(1),1,rtol=0,atol=1e-15)
        positions=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],float)
        xyz=(positions[f[q]]*b[:,:,None]).sum(1)
        np.testing.assert_array_equal(xyz[p[:,0]],xyz[p[:,1]])
        self.assertTrue((b[c==1]>0).all())
        self.assertEqual(len(seam_queries(f,ids,np.array([[4,10]],np.int64))[0]),0)
    def test_nonmanifold_multiplicity_is_not_discarded(self):
        f=np.array([[0,1,2],[1,0,3],[0,1,4]],np.int32);ids=np.arange(5,dtype=np.int64)
        q,b,c,p=seam_queries(f,ids,np.empty((0,2),np.int64))
        self.assertEqual(len(q),60);self.assertEqual(len(p),15)
    def test_foreign_retained_edge_rejected(self):
        f=np.array([[0,1,2],[1,0,3]],np.int32)
        with self.assertRaisesRegex(ValueError,'Foreign retained'):
            seam_queries(f,np.arange(4,dtype=np.int64),np.array([[999,1000]],np.int64))
    def test_shape_dtype_and_face_limits_fail_closed(self):
        f=np.array([[0,1,2]],np.int32);ids=np.arange(3,dtype=np.int64);empty=np.empty((0,2),np.int64)
        for bad in (f.astype(float),f.astype(np.int64),np.tile(f,(6001,1))):
            with self.subTest(shape=bad.shape),self.assertRaises((ValueError,TypeError)):
                seam_queries(bad,ids,empty)
    def test_query_work_budget_raises_without_truncating(self):
        f=np.array([[0,1,i+2] for i in range(100)],np.int32)
        with self.assertRaisesRegex(ValueError,'All seam queries'):
            seam_queries(f,np.arange(102,dtype=np.int64),np.empty((0,2),np.int64))


if __name__=='__main__':unittest.main()
