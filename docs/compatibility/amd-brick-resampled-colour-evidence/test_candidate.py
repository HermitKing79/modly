"""Bounded synthetic math and unchanged gate checks; no actual asset reads."""
import sys
sys.dont_write_bytecode=True
import importlib.util,unittest
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from candidate import interpolate,evaluate_holdout

def metrics():
    p=E.parent/'amd-foliage-export-decision-evidence/metrics.py'
    spec=importlib.util.spec_from_file_location('test_original_IEC',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class CandidateTests(unittest.TestCase):
    def test_original_ids_are_remapped_without_reordering_colours(self):
        ids=np.array([2,5,8],np.int32);faces=np.array([[8,2,5]],np.int32)
        rgb=np.array([[.2,.3,.4],[.3,.4,.5],[.7,.8,.9]],np.float32)
        qf=np.array([0,0],np.int32);bary=np.array([[1.,0,0],[.2,.3,.5]],np.float64)
        got,linear=interpolate(faces,qf,bary,ids,rgb,metrics())
        expected,_=metrics().color_prediction(rgb,np.array([[2,0,1]],np.int32),qf,bary)
        np.testing.assert_array_equal(got,expected);self.assertEqual(linear.dtype,np.float32)
        self.assertEqual(linear.shape,(3,3));self.assertGreater(got[0,0],got[1,0])

    def test_missing_unsorted_and_duplicate_source_ids_fail(self):
        f=np.array([[0,1,2]],np.int32);q=np.array([0],np.int32);b=np.array([[1/3]*3],np.float64);rgb=np.zeros((3,3),np.float32)
        for ids in (np.array([0,1,3],np.int32),np.array([2,0,1],np.int32),np.array([0,1,1],np.int32)):
            with self.assertRaises(ValueError):interpolate(f,q,b,ids,rgb,metrics())

    def test_holdout_reports_each_region_and_each_interior_stream(self):
        raw=np.ones((8,6),np.float32);raw[:,:3]=.5;raw[:,3]=.00011082693761409814;raw[:,4]=.999995986777544
        predicted=np.full((8,3),.5,np.float64);predicted[6,0]+=.2
        q=np.arange(8,dtype=np.int32);k=np.tile(np.array([4,5],np.int8),4)
        groups={'easy':dict(query_start=0,query_stop=4),'hard':dict(query_start=4,query_stop=8)}
        r=evaluate_holdout(predicted,raw,q,k,groups,np.zeros(8,np.bool_))
        self.assertTrue(r['regions']['easy']['pass']);self.assertFalse(r['regions']['hard']['pass'])
        self.assertEqual(r['failing_regions'],['hard']);self.assertFalse(r['regions']['hard']['interior0']['pass_gate'])

    def test_holdout_never_invents_corner_or_centroid_cohorts(self):
        raw=np.ones((2,6),np.float32);raw[:,:3]=.5;raw[:,3]=.00011082693761409814;raw[:,4]=.999995986777544
        r=evaluate_holdout(np.full((2,3),.5),raw,np.array([4,4],np.int32),np.array([4,5],np.int8),{'x':dict(query_start=0,query_stop=2)},np.zeros(2,np.bool_))
        self.assertEqual(set(r['regions']['x']),{'all','interior0','interior1','pass'})
        self.assertEqual(r['material_suitability'],'PASS')

    def test_unsupported_empty_and_nonfinite_holdout_reject(self):
        raw=np.ones((2,6),np.float32);raw[:,:3]=.5;raw[:,3]=.00011082693761409814;raw[:,4]=.999995986777544
        args=[np.full((2,3),.5),raw,np.array([4,4],np.int32),np.array([4,5],np.int8),{'x':dict(query_start=0,query_stop=2)},np.array([True,False])]
        self.assertEqual(evaluate_holdout(*args)['material_suitability'],'NO-GO')
        args[-1][:]=False;args[0][0,0]=np.nan
        with self.assertRaises(ValueError):evaluate_holdout(*args)

if __name__=='__main__':unittest.main()
