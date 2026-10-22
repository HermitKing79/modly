"""Independent tiny durable-index negatives; no production data or graph engine."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import subprocess
import numpy as np
from verify_worker import verify_occurrences,verify_masks,link_from_stars,verify_distributions,FACE_MASKS,VERTEX_MASKS

def fixture():
    f=np.array([[0,1,2],[2,1,3]],np.int32);entries=[]
    for row,t in enumerate(f):
        for j in range(3):
            a,b=int(t[j]),int(t[(j+1)%3]);entries.append((min(a,b)*4+max(a,b),row*3+j,a<b))
    entries.sort();keys=np.array(sorted({x[0] for x in entries}),np.uint64)
    inc=np.array([sum(x[0]==k for x in entries) for k in keys],np.uint8)
    order=np.array([x[1] for x in entries],np.int32);off=np.r_[0,np.cumsum(inc)].astype(np.int32)
    consistent=np.array([len([x for x in entries if x[0]==k])==2 for k in keys])
    masks={name:np.zeros(4,np.bool_) for name in ('nonmanifold_vertices','alias_vertices','multiple_fan_vertices','multiple_sheet_vertices')}
    masks.update({name:np.ones(4,np.bool_) for name in ('boundary_vertices','protected_vertices','locked_vertices')})
    masks.update({name:np.ones(2,np.bool_) for name in ('boundary_touch_faces','boundary_edge_touch_faces','protected_faces')})
    masks.update({name:np.zeros(2,np.bool_) for name in ('nonmanifold_touch_faces','pre_guard_interior_faces','fully_unlocked_interior_faces','guard_affected_interior_faces','certified_edge_touch_faces')})
    return f,keys,inc,order,off,consistent,masks

class ReadbackTests(unittest.TestCase):
    def test_isolated_direct_module_resolves_exact_helper_without_running_main(self):
        worker=Path(__file__).resolve().with_name('verify_worker.py')
        code="import runpy;runpy.run_path("+repr(str(worker))+",run_name='qualification_only');import common;print(common.E)"
        result=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True,timeout=10,check=True)
        self.assertEqual(Path(result.stdout.strip()),worker.parent)
    def test_exact_permutation_keys_and_zero_endpoint_certificate(self):
        f,k,i,o,off,con,m=fixture()
        degree,conflict,_=verify_occurrences(f,k,i,o,off,con,4,lambda:None,rows=2)
        np.testing.assert_array_equal(degree,[1,2,2,1]);self.assertFalse(conflict.any())
        q=verify_masks(f,k,i,o,off,con,np.zeros(2,np.int32),m,degree,conflict,lambda:None)
        self.assertEqual(len(q),0)
    def test_duplicate_permutation_offsets_and_winding_corruption_fail(self):
        for kind in ('duplicate','offset','winding'):
            f,k,i,o,off,con,_=fixture()
            if kind=='duplicate':o[1]=o[0]
            elif kind=='offset':off[1]+=1
            else:con[:]=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):verify_occurrences(f,k,i,o,off,con,4,lambda:None,rows=2)
    def test_changed_protection_or_lock_is_rejected(self):
        for key in ('protected_faces','locked_vertices'):
            f,k,i,o,off,con,m=fixture();degree,conflict,_=verify_occurrences(f,k,i,o,off,con,4,lambda:None)
            m[key][0]=False
            with self.subTest(key=key),self.assertRaises(ValueError):verify_masks(f,k,i,o,off,con,np.zeros(2,np.int32),m,degree,conflict,lambda:None)
    def test_full_link_distinguishes_open_pair_from_tetrahedron(self):
        f=np.array([[0,1,2],[2,1,3]],np.int32)
        self.assertTrue(link_from_stars(1,2,f,f,0,3))
        tetra=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        self.assertFalse(link_from_stars(0,1,tetra[(tetra==0).any(1)],tetra[(tetra==1).any(1)],2,3))
    def test_cancellation_is_not_a_completed_verification(self):
        f,k,i,o,off,con,_=fixture()
        def stop():raise TimeoutError('synthetic cancellation')
        with self.assertRaises(TimeoutError):verify_occurrences(f,k,i,o,off,con,4,stop)
    def test_component_and_octant_denominators_recount_and_reject_change(self):
        f,k,i,o,off,con,m=fixture();v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
        names=('original_faces',)+FACE_MASKS+VERTEX_MASKS+('certified_edges',)
        totals={name:(2 if name=='original_faces' else int(m[name].sum()) if name in m else 0) for name in names}
        by_octant={str(j):{name:0 for name in names} for j in range(8)}
        for name in ('original_faces','boundary_touch_faces','boundary_edge_touch_faces','protected_faces'):
            by_octant['4'][name]=by_octant['7'][name]=1
        for name in ('boundary_vertices','protected_vertices','locked_vertices'):
            for j in range(4,8):by_octant[str(j)][name]=1
        record=dict(by_original_component={'0':totals},by_octant=by_octant,original_face_bearing_vertex_components=1,unused_source_vertices=0)
        verify_distributions(v,f,np.zeros(4,np.int32),m,np.empty(0,np.int32),k,np.array([4,7],np.uint8),record,lambda:None)
        record['by_octant']['4']['locked_vertices']=0
        with self.assertRaises(ValueError):verify_distributions(v,f,np.zeros(4,np.int32),m,np.empty(0,np.int32),k,np.array([4,7],np.uint8),record,lambda:None)

if __name__=='__main__':unittest.main(verbosity=2)
