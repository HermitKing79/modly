import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_mesh_aware_proof as run
sys.path.insert(0,str(run.SITE))
import numpy as np
from amd_mesh_aware_input import adjacency

class MeshAwareTests(unittest.TestCase):
    def test_source_only_artifact_read_scope(self):
        self.assertTrue(run.source_read_allowed(run.BASE/'input/mesh.bin'))
        self.assertTrue(run.source_read_allowed(run.PRIOR/'ai-input/samples.npz'))
        self.assertFalse(run.source_read_allowed(run.PRIOR/'official-sample/user_provided_planes_result.off'))
        self.assertFalse(run.source_read_allowed(run.BASE.parent/'amd-closed-solid-benchmark-v1/mesh.ply'))
        self.assertFalse(run.source_read_allowed(run.owner.ROOT/'scripts/compatibility/amd_closed_solid_benchmark.py'))
    def test_sign_invariant_plane_screen(self):
        from amd_cgal_ai_input import evaluate
        points=np.array([[0.,0,0],[1,1,0]])
        normals=np.array([[0.,0,1],[0,0,-1]])
        result,_,_=evaluate(np,points,normals,[dict(id=0,normal=[0,0,1],offset=0)],1)
        self.assertEqual(result['covered'],2)
    def test_distance_rejection_not_hidden_by_normal(self):
        from amd_cgal_ai_input import evaluate
        result,_,_=evaluate(np,np.array([[0.,0,.003]]),np.array([[0.,0,1]]),[dict(id=0,normal=[0,0,1],offset=0)],1)
        self.assertFalse(result['gate_pass'])
    def test_frozen_historical_gate_cannot_be_replaced(self):
        from amd_cgal_ai_input import evaluate
        p=np.zeros((13962,3));p[-91:,2]=.003
        n=np.tile([0.,0,1],(len(p),1))
        result,_,_=evaluate(np,p,n,[dict(id=0,normal=[0,0,1],offset=0)],1)
        self.assertEqual(result['covered'],13871);self.assertFalse(result['gate_pass'])
        p[-22:,2]=0
        result,_,_=evaluate(np,p,n,[dict(id=0,normal=[0,0,1],offset=0)],1)
        self.assertTrue(result['gate_pass'])
    def test_no_optimizer_route_after_unqualified_screen(self):
        self.assertNotIn('reconstruct',run.STAGES)
        self.assertIn('prepare-candidates',run.STAGES)
        self.assertFalse(run.spawn_allowed('reconstruct',('wsl.exe',[],str(run.BASE),{})))
    def test_native_domain_check_never_routes_to_python_data(self):
        self.assertNotIn('certify-supported-domains',run.DATA_ACTIONS)
        cmd=run.command('certify-supported-domains')
        self.assertTrue(run.spawn_allowed('certify-supported-domains',(cmd[0],cmd,str(run.BASE),{})))
        self.assertFalse(run.spawn_allowed('certify-supported-domains',(cmd[0],cmd+['--unsafe'],str(run.BASE),{})))
        self.assertNotIn('owned-reconstruct',run.DATA_ACTIONS)
    def test_analytic_triangle_distance(self):
        from amd_mesh_aware_result import point_triangle_distances
        tri=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        p=np.array([[.2,.2,1],[2,0,0],[.5,.5,0]])
        np.testing.assert_allclose(point_triangle_distances(np,p,tri),[1,1,0],atol=1e-14,rtol=0)
    def test_independent_batched_triangle_reference(self):
        from amd_mesh_aware_result import point_triangle_distances
        tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]]])
        points=np.array([[.2,.2,2],[.2,.2,3]])
        np.testing.assert_allclose(point_triangle_distances(np,points,tri),[2,2],atol=1e-14,rtol=0)
    def test_zero_area_reference_is_rejected(self):
        from amd_mesh_aware_result import point_triangle_distances
        with self.assertRaises(ValueError):point_triangle_distances(np,np.zeros((1,3)),np.zeros((3,3)))
    def test_boundary_distance_is_symmetric(self):
        from amd_mesh_aware_result import boundary,boundary_distance
        a=np.zeros((10,10),bool);a[2:6,2:6]=True;b=np.roll(a,1,axis=1)
        aa=boundary(np,a);bb=boundary(np,b)
        self.assertEqual(boundary_distance(np,aa,bb),1)
        self.assertEqual(boundary_distance(np,bb,aa),1)
    def test_shared_edge_and_boundaries(self):
        adj,c=adjacency(np,np.array([[0,1,2],[2,1,3]],dtype='i4'))
        self.assertEqual(c,dict(boundary_edges=4,nonmanifold_edges=0,two_face_edges=1))
        self.assertEqual(int((adj>=0).sum()),2)
    def test_three_way_junction_is_blocked(self):
        adj,c=adjacency(np,np.array([[0,1,2],[1,0,3],[0,1,4]],dtype='i4'))
        self.assertEqual(c['nonmanifold_edges'],1)
        self.assertTrue((adj==-1).all())
    def test_no_coordinate_welding(self):
        adj,c=adjacency(np,np.array([[0,1,2],[3,4,5]],dtype='i4'))
        self.assertEqual(c['boundary_edges'],6);self.assertTrue((adj==-1).all())
    def test_winding_does_not_change_adjacency(self):
        a,c=adjacency(np,np.array([[0,1,2],[1,2,3]],dtype='i4'))
        b,d=adjacency(np,np.array([[0,1,2],[3,2,1]],dtype='i4'))
        self.assertEqual(c,d);self.assertEqual(int((a>=0).sum()),int((b>=0).sum()))
    def test_exact_native_audit_windows_string(self):
        import subprocess
        cmd=run.command('segment')
        self.assertTrue(run.spawn_allowed('segment',(cmd[0],subprocess.list2cmdline(cmd),str(run.BASE),{})))
        self.assertFalse(run.spawn_allowed('segment',('wsl.exe',cmd,str(run.BASE),{})))
        self.assertFalse(run.spawn_allowed('segment',(cmd[0],cmd,str(run.PRIOR),{})))
        self.assertFalse(run.spawn_allowed('segment',(cmd[0],cmd+['--extra'],str(run.BASE),{})))
    def test_resource_profile_unchanged(self):
        self.assertFalse(run.owner.admission(12*1024**3-1,2*1024**3))
        self.assertFalse(run.owner.admission(12*1024**3,2*1024**3-1))
        self.assertEqual(run.owner.CAP,6*1024**3);self.assertEqual(run.owner.DEADLINE,180)

if __name__=='__main__':unittest.main()
