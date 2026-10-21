import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from foliage_select import visible_faces,ROI

class SelectionTests(unittest.TestCase):
    def fixture(self):
        xy=np.array([[ROI[0]+4+6*x,ROI[1]+4+6*y] for y in range(5) for x in range(5)],float)
        faces=[]
        for y in range(4):
            for x in range(4):
                a=5*y+x;faces.extend([[a,a+1,a+6],[a,a+6,a+5]])
        f=np.array(faces,np.int32)
        v=np.column_stack((np.ones(25),xy)).astype(np.float32)
        return v,f,xy
    def test_original_front_faces_win_over_occluded_rear_without_face_changes(self):
        v,f,xy=self.fixture();vv=np.vstack((v.copy(),v.copy()));vv[25:,0]=0
        ff=np.vstack((f,f+25));picked,report=visible_faces(vv,ff,SimpleNamespace(screen=np.vstack((xy,xy))))
        np.testing.assert_array_equal(picked,np.arange(len(f)))
        self.assertEqual(report['selected_faces'],32)
        self.assertTrue(report['artificial_crop_boundaries'])
    def test_equal_depth_ties_choose_earliest_source_and_budget_fails_closed(self):
        v,f,xy=self.fixture();picked,_=visible_faces(v,np.vstack((f,f)),SimpleNamespace(screen=xy))
        # Neighbour ring legitimately includes duplicate original triangles.
        self.assertEqual(len(picked),64)
        with self.assertRaises(ValueError):visible_faces(v,f,SimpleNamespace(screen=xy),max_faces=20)
    def test_no_visible_complete_original_faces_rejected(self):
        v,f,xy=self.fixture()
        with self.assertRaises(ValueError):visible_faces(v,f,SimpleNamespace(screen=xy+1000))

if __name__=='__main__':unittest.main()
