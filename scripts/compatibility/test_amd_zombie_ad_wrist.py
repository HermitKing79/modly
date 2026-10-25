"""Small geometry witnesses for the read-only feasibility spike."""
import sys
from pathlib import Path
import unittest
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
import numpy as np
import amd_zombie_ad_wrist as subject

class Sections(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1],
                         [1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]],np.float32)
        self.f=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int32)
    def test_closed_tube_is_one_loop_independent_of_winding(self):
        a,p,s=subject.section(np,self.v,self.f,.123)
        b,_,_=subject.section(np,self.v,self.f[:,[0,2,1]],.123)
        self.assertTrue(a['simple_single_loop']);self.assertEqual(a,b)
        self.assertTrue(np.allclose(p[:,0],.123));self.assertEqual(len(s),8)
    def test_open_tube_detected(self):
        a,_,_=subject.section(np,self.v,self.f[:-1],.123)
        self.assertFalse(a['simple_single_loop']);self.assertEqual(a['degree_one'],2)
    def test_duplicate_face_is_branch_not_clean_loop(self):
        a,_,_=subject.section(np,self.v,np.concatenate((self.f,self.f[:1])),.123)
        self.assertFalse(a['simple_single_loop']);self.assertEqual(a['degree_over_two'],2)
    def test_nearest_distance_known_translation(self):
        a=np.array([[0.,0,0],[0,1,0]]);b=a+[0,0,.02]
        self.assertAlmostEqual(subject.nearest(np,a,b)['max'],.02)
    def test_no_generation_or_mesh_writer_interface(self):
        names=set(vars(subject))
        self.assertFalse(names & {'extract','write_ply','repair','remesh','generate','hybrid','inference'})

if __name__=='__main__':unittest.main()
