"""Actual pinned selector with the sole new nomination; bounded synthetic data."""
import sys
sys.dont_write_bytecode=True
import hashlib, importlib.util, json, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
E=Path(__file__).resolve().parent
path=E.parent/'amd-pbr-foliage-evidence/foliage_select.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='1e04a68faf7a9e5c23e4f23b1cd5b9b93b6d02b5aee77f89272b403ea358b57a'
spec=importlib.util.spec_from_file_location('test_original_fixed_selector',path)
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
selector.ROI=tuple(json.loads((E/'nomination.json').read_text())['ROI'])

class SelectionTests(unittest.TestCase):
    def fixture(self):
        xy=np.array([[selector.ROI[0]+3+6*x,selector.ROI[1]+3+6*y] for y in range(5) for x in range(5)],float)
        f=[]
        for y in range(4):
            for x in range(4):
                a=5*y+x;f.extend([[a,a+1,a+6],[a,a+6,a+5]])
        return np.column_stack((np.ones(25),xy)).astype(np.float32),np.array(f,np.int32),xy
    def test_nomination_frozen_and_within_original(self):
        path=E/'nomination.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'eecf4d0b880c28aab7fb660a6a573d8498291936ee14ac039e05b09c7572da25')
        self.assertEqual(selector.ROI,(346,393,393,436))
        self.assertTrue(332<=selector.ROI[0]<selector.ROI[2]<=400 and 384<=selector.ROI[1]<selector.ROI[3]<=448)
    def test_actual_front_selection_preserves_source_corners(self):
        v,f,xy=self.fixture();vv=np.vstack((v,v));vv[25:,0]=0;ff=np.vstack((f,f+25))
        original=vv.copy(),ff.copy()
        chosen,receipt=selector.visible_faces(vv,ff,SimpleNamespace(screen=np.vstack((xy,xy))))
        np.testing.assert_array_equal(chosen,np.arange(32))
        self.assertEqual(receipt['ROI'],list(selector.ROI))
        np.testing.assert_array_equal(vv,original[0]);np.testing.assert_array_equal(ff,original[1])
    def test_budget_refusal_and_duplicate_multiplicity_are_unchanged(self):
        v,f,xy=self.fixture()
        with self.assertRaisesRegex(ValueError,'Original ROI face budget: 32'):
            selector.visible_faces(v,f,SimpleNamespace(screen=xy),max_faces=20)
        chosen,_=selector.visible_faces(v,np.vstack((f,f)),SimpleNamespace(screen=xy))
        self.assertEqual(len(chosen),64)
        with self.assertRaisesRegex(ValueError,'Bounded fixed ROI candidate count'):
            selector.visible_faces(v,f,SimpleNamespace(screen=xy+1000))

if __name__=='__main__':unittest.main()
