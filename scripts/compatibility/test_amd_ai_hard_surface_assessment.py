"""Tiny rotated-plane fixtures, never the procedural box or its dimensions."""
import io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_ai_hard_surface_assessment as a

class PlaneAssessment(unittest.TestCase):
    def test_parallel_tilted_planes_remain_separate(self):
        rng=np.random.default_rng(401);n=np.array([.31,.52,.79]);n/=np.linalg.norm(n)
        u=np.cross(n,[1.,0,0]);u/=np.linalg.norm(u);v=np.cross(n,u)
        xy=rng.uniform(-.3,.3,(800,2));p=xy[:,0,None]*u+xy[:,1,None]*v
        points=np.vstack((p+.2*n,p-.2*n));normals=np.tile(n,(len(points),1));normals[::2]*=-1
        settings=dict(maximum_planes=8,maximum_hypotheses=16,minimum_training_support=24,normal_bin_width=.05,normal_angle_degrees=10,point_to_plane_limit_over_AI_diagonal=.002,plane_refits=3)
        planes,labels=a.fit_planes(points,normals,1.,settings)
        self.assertEqual(len(planes),2);self.assertTrue((labels>=0).all())
        self.assertTrue(all(abs(abs(np.dot(r['normal'],n))-1)<1e-10 for r in planes))
        self.assertTrue(all(r['distance_p95']<1e-10 for r in planes))

    def test_curved_residuals_are_retained_not_dropped(self):
        t=np.linspace(-.3,.3,120);p=np.column_stack((t,t*t,np.sin(9*t)*.1));n=np.column_stack((np.ones(120),np.zeros(120),np.zeros(120)))
        s=dict(maximum_planes=1,maximum_hypotheses=1,minimum_training_support=24,normal_bin_width=.05,normal_angle_degrees=10,point_to_plane_limit_over_AI_diagonal=.00001,plane_refits=3)
        planes,labels=a.fit_planes(p,n,1.,s)
        self.assertTrue((labels<0).any());self.assertEqual(len(labels),len(p))

    def test_fit_does_not_change_input_points(self):
        p=np.column_stack((np.linspace(0,1,64),np.zeros(64),np.linspace(0,1,64)**2));n=np.tile([0.,1,0],(64,1));before=p.copy()
        s=dict(maximum_planes=2,maximum_hypotheses=4,minimum_training_support=24,normal_bin_width=.05,normal_angle_degrees=10,point_to_plane_limit_over_AI_diagonal=.002,plane_refits=3)
        a.fit_planes(p,n,1.,s);np.testing.assert_array_equal(p,before)

    def test_assessment_rejects_ground_truth_reads(self):
        guard=a.assessment_guard(a.BASE)
        for p in (a.ROOT/'scripts/compatibility/amd_closed_solid_benchmark.py',a.ROOT.parent/'artifacts/amd-closed-solid-benchmark-20261029-v1/source/mesh.ply',a.RAW/'raw-quality.json'):
            with self.assertRaisesRegex(RuntimeError,'Ground-truth|Non-AI'):guard('open',(str(p),'r',0))
        guard('open',(str(a.RAW/'mesh/vertices.npy'),'r',0))

    def test_default_inert_and_worker_needs_owned_GO(self):
        with patch.object(a,'assess',side_effect=AssertionError('No execution')):
            self.assertEqual(a.main([]),0)
            with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))):
                with self.assertRaisesRegex(RuntimeError,'GO'):a.main(['--worker','assess'])

    def test_frozen_criteria_rejects_changed_thresholds(self):
        self.assertEqual(a.criteria()['candidate_budget']['maximum_triangles'],1000)
        with patch.object(a,'CRITERIA',SimpleNamespace(read_bytes=lambda:b'{}')):
            with self.assertRaisesRegex(ValueError,'Frozen criteria'):a.criteria()

if __name__=='__main__':unittest.main(verbosity=2)
