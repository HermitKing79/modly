"""Small synthetic admission fixtures only; never a simplification call."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest,numpy as np
import patch_contract as pc

def grid(size=12):
    v=np.array([(x,y,0) for y in range(size) for x in range(size)],np.float32)
    f=[]
    for y in range(size-1):
        for x in range(size-1):
            a=y*size+x;f.extend(((a,a+1,a+1+size),(a,a+1+size,a+size)))
    return v,np.array(f,np.int32)

def classify(v,f,full=None,N=None):
    full=f if full is None else full;N=len(v) if N is None else N
    keys,inc=np.unique(pc.edge_keys(full,N),return_counts=True)
    return pc.classify(v,f,np.arange(len(v),dtype=np.int32),f,np.array([],np.int32),keys,inc,N)

class PatchTests(unittest.TestCase):
    def test_grid_free_interior_survives_conservative_guard(self):
        v,f=grid();s,a=classify(v,f)
        self.assertGreater(s['strict_eligible_edges'],0);self.assertGreater(s['non_touching_certified_edges'],0)
        self.assertTrue(a['vertex_locks'][0]);self.assertFalse(a['vertex_locks'][6*12+6])
    def test_cropped_nm_edge_not_mistaken_for_manifold(self):
        v,f=grid(5);extra=np.array([[6,7,24]],np.int32)
        s,a=classify(v,f,np.concatenate((f,extra)))
        self.assertGreater(s['global_nonmanifold_patch_edges'],0)
        self.assertTrue(a['protected_vertices'][6]);self.assertTrue(a['protected_vertices'][7])
        self.assertTrue(a['protected_faces'][(f==6).any(1)].all())
    def test_misoriented_sheet_interface_is_protected(self):
        v,f=grid(5);f=f.copy();f[10]=f[10,[0,2,1]];s,a=classify(v,f)
        self.assertGreater(s['orientation_inconsistent_two_face_edges'],0);self.assertTrue(a['protected_faces'][10])
    def test_position_alias_lock_closure_and_signed_zero(self):
        v=np.array([[0,0,0],[-0.,0.,0.],[1,0,0]],np.float32)
        inverse,count=pc.same_position_classes(v)
        self.assertEqual(inverse[0],inverse[1]);self.assertEqual(count[inverse[0]],2)
    def test_invalid_positions_or_source_correspondence_rejected(self):
        v,f=grid(3);v[0,0]=np.nan
        with self.assertRaises(ValueError):classify(v,f)
        v,f=grid(3);k,i=np.unique(pc.edge_keys(f,len(v)),return_counts=True)
        with self.assertRaises(ValueError):pc.classify(v,f,np.arange(len(v),dtype=np.int32),f[:,[0,2,1]],[],k,i,len(v))
    def test_all_junction_faces_have_no_free_candidate(self):
        v=np.array([[0,0,0],[0,0,1],[1,0,.5],[0,1,.5],[-1,0,.5],[0,-1,.5]],np.float32)
        f=np.array([[0,1,2],[1,0,3],[0,1,4],[1,0,5]],np.int32);s,a=classify(v,f)
        self.assertEqual(s['protected_faces'],4);self.assertEqual(s['strict_eligible_edges'],0);self.assertFalse(s['eligible_interior_reduction_admission'])
    def test_tetrahedron_common_link_edge_is_not_eligible(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32);s,a=classify(v,f)
        self.assertEqual(s['consistent_two_face_interior_edges'],6)
        self.assertEqual(s['link_condition_rejected'],6);self.assertEqual(s['strict_eligible_edges'],0)
    def test_vertex_fans_can_be_separate_inside_one_global_sheet(self):
        v=np.array([[0,0,0],[1,0,0],[1,1,0],[-1,0,0],[-1,-1,0],[2,1,1],[-2,1,1]],np.float32)
        f=np.array([[0,1,2],[0,3,4],[2,1,5],[5,1,6],[5,6,3],[3,6,4]],np.int32)
        s,a=classify(v,f)
        self.assertEqual(s['sheet_components'],1);self.assertGreater(s['local_multiple_fan_vertices'],0);self.assertTrue(a['protected_vertices'][0])

if __name__=='__main__':unittest.main()
