"""Partial native output fixtures only; never loads MeshLib or actual patches."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
try:
    from verify_failure import verify_failure_arrays
except ModuleNotFoundError:
    verify_failure_arrays=None

def scalar(v):return dict(length=1,values=[v],getter=dict(ok=True,value=v))

def fixture():
    v=np.array([[-0.,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1],[1,1,0]],np.float32)
    f=np.array([[0,1,2],[0,1,4],[1,0,3],[2,1,5]],np.int32)
    valid=np.array([0,2,3],np.int32);rf=f[valid].copy()
    original=dict(vertices=v,faces=f,source_vertex_ids=np.arange(10,16,dtype=np.int64),source_face_ids=np.arange(100,104,dtype=np.int32))
    saved=dict(vertices_readback=v.astype(np.float64),faces_readback=rf,source_vertex_ids=original['source_vertex_ids'].copy(),source_face_ids=original['source_face_ids'].copy(),
               vertices_after_native=v.copy(),faces_after_native=f.copy(),valid_face_ids=valid,valid_vertex_ids=np.unique(rf).astype(np.int32),failed_face_ids=np.array([1],np.int32))
    edges={};eid=np.empty_like(rf);hid=np.arange(rf.size,dtype=np.int32).reshape(rf.shape)+100
    left=np.repeat(valid[:,None],3,axis=1);right=np.full_like(rf,-1)
    for row,tri in enumerate(rf):
        for corner in range(3):
            a,b=int(tri[corner]),int(tri[(corner+1)%3]);key=tuple(sorted((a,b)))
            group=edges.setdefault(key,[]);eid[row,corner]=list(edges).index(key)+20;group.append((row,corner,a,b))
    for group in edges.values():
        if len(group)==2:
            a,b=group;right[a[0],a[1]]=valid[b[0]];right[b[0],b[1]]=valid[a[0]]
    saved.update(native_undirected_edge_ids=eid,native_halfedge_ids=hid,native_left_faces=left,native_right_faces=right)
    counter=dict(before=scalar(-23456),settings_before=-23456,region_before=list(range(4)),settings_region_before=list(range(4)),after=scalar(1),settings_after=1,failed_ids=[1],settings_failed_ids=[1])
    return original,saved,counter

class FailureReadback(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(verify_failure_arrays,'Failure verifier is not implemented')
    def test_noncontiguous_original_FaceIds_and_allocated_invalid_point(self):
        report,witness=verify_failure_arrays(np,*fixture())
        self.assertEqual(report['brick_input_transport'],'NO-GO')
        self.assertEqual(report['missing_original_faces'],1)
        self.assertEqual(report['missing_valid_original_vertices'],1)
        self.assertEqual(witness['missing_native_face_ids'],[1])
        self.assertEqual(witness['missing_source_vertex_ids'],[14])
    def test_omitted_allocated_point_bits_still_required(self):
        original,saved,counter=fixture();saved['vertices_readback'][4,0]=1e-50
        with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_missing_faces_equal_both_failed_regions(self):
        original,saved,counter=fixture();saved['failed_face_ids'][0]=2;counter['failed_ids']=counter['settings_failed_ids']=[2]
        with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_counter_cannot_be_stale_or_wrong(self):
        for value in (-23456,0,2):
            original,saved,counter=fixture();counter['after']=scalar(value);counter['settings_after']=value
            with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_right_adjacency_cannot_reference_rejected_face(self):
        original,saved,counter=fixture();saved['native_right_faces'][0,0]=1
        with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_left_uses_real_noncontiguous_FaceIds(self):
        original,saved,counter=fixture();saved['native_left_faces'][1]=1
        with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_valid_vertices_equal_retained_references(self):
        original,saved,counter=fixture();saved['valid_vertex_ids']=np.array([0,1,2,3,4],np.int32)
        with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_retained_face_cyclic_orientation(self):
        original,saved,counter=fixture()
        for name in ('faces_readback','native_undirected_edge_ids','native_halfedge_ids','native_left_faces','native_right_faces'):
            saved[name]=np.roll(saved[name],-1,axis=1).copy()
        report,_=verify_failure_arrays(np,original,saved,counter)
        self.assertEqual(report['cyclic_rotation_counts'],{'0':0,'1':3,'2':0})
    def test_wrong_retained_row_or_winding_rejects(self):
        for value in ([0,1,4],[1,3,0]):
            original,saved,counter=fixture();saved['faces_readback'][1]=value
            with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)
    def test_original_heap_and_ID_copies_required(self):
        for name in ('vertices_after_native','faces_after_native','source_vertex_ids','source_face_ids'):
            original,saved,counter=fixture();saved[name].flat[0]+=1
            with self.assertRaises(ValueError):verify_failure_arrays(np,original,saved,counter)

if __name__=='__main__':unittest.main(verbosity=2)
