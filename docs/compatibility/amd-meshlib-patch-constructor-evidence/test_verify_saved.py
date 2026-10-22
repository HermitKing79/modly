"""Synthetic saved-array checks; no MeshLib import or real patch input."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
try:
    from verify_saved import verify_arrays,verify_counter
    import verify_saved as implementation
except ModuleNotFoundError:
    verify_arrays=verify_counter=None
    implementation=None

def fixture():
    v=np.array([[-0.,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],np.float32)
    f=np.array([[0,1,2],[1,0,3],[0,1,4]],np.int32)
    original=dict(vertices=v,faces=f,source_vertex_ids=np.arange(10,15,dtype=np.int64),source_face_ids=np.arange(100,103,dtype=np.int32))
    saved=dict(vertices_readback=v.astype(np.float64),faces_readback=f.copy(),source_vertex_ids=original['source_vertex_ids'].copy(),source_face_ids=original['source_face_ids'].copy(),
               vertices_after_native=v.copy(),faces_after_native=f.copy(),valid_face_ids=np.arange(3,dtype=np.int32),valid_vertex_ids=np.arange(5,dtype=np.int32),
               failed_face_ids=np.empty(0,np.int32),cyclic_rotations=np.zeros(3,np.int8))
    eid=np.empty((3,3),np.int32);hid=np.empty_like(eid);left=np.empty_like(eid);right=np.full_like(eid,-1)
    nxt=20
    for row,tri in enumerate(f):
        for corner in range(3):
            if corner==0:
                eid[row,corner]=10 if row<2 else 11
                hid[row,corner]=100+row
                if row<2:right[row,corner]=1-row
            else:
                eid[row,corner]=nxt;hid[row,corner]=200+nxt;nxt+=1
            left[row,corner]=row
    saved.update(native_undirected_edge_ids=eid,native_halfedge_ids=hid,native_left_faces=left,native_right_faces=right)
    return original,saved

def scalar(value):return dict(length=1,values=[value],getter=dict(ok=True,value=value))

class SavedVerification(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(verify_arrays,'Independent verifier is not implemented')
    def test_exact_roundtrip_and_native_split(self):
        report,detail=verify_arrays(np,*fixture())
        self.assertEqual(report['original_topology']['nonmanifold_source_edges'],1)
        self.assertEqual(report['original_topology']['edge_connected_face_components'],1)
        self.assertEqual(report['native_topology']['edge_connected_face_components'],2)
        self.assertEqual(report['sheet_correspondence'],'UNRESOLVED')
        self.assertEqual(len(detail),1)
    def test_only_canonical_relative_POSIX_manifest_names(self):
        self.assertTrue(hasattr(implementation,'relative_artifact_name'))
        for name in ('mesh.npy','brick/result.json'):
            self.assertEqual(implementation.relative_artifact_name(name),name)
        for name in ('','..\\foreign','../foreign','a/../x','./x','a//x','C:/x','a:b','/root','a\\b','a/'):
            with self.assertRaises(ValueError):implementation.relative_artifact_name(name)
    def test_cyclic_corners_and_metadata_alignment(self):
        original,saved=fixture()
        for name in ('faces_readback','native_undirected_edge_ids','native_halfedge_ids','native_left_faces','native_right_faces'):
            saved[name]=np.roll(saved[name],-1,axis=1).copy()
        saved['cyclic_rotations'][:]=1
        report,_=verify_arrays(np,original,saved)
        self.assertEqual(report['cyclic_rotation_counts'],{'0':0,'1':3,'2':0})
    def test_signed_zero_and_sub_F32_change_reject(self):
        for value in (0.,-1e-50,float('nan')):
            original,saved=fixture();saved['vertices_readback'][0,0]=value
            with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_reversed_winding_rejects(self):
        original,saved=fixture();saved['faces_readback'][0]=[0,2,1]
        with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_source_ID_copy_or_heap_mutation_rejects(self):
        for key in ('source_vertex_ids','source_face_ids','vertices_after_native','faces_after_native'):
            original,saved=fixture();saved[key].flat[0]+=1
            with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_native_edge_cannot_alias_other_endpoints(self):
        original,saved=fixture();saved['native_undirected_edge_ids'][0,1]=10
        with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_native_pairing_requires_reciprocal_faces(self):
        original,saved=fixture();saved['native_right_faces'][0,0]=-1
        with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_actual_left_face_and_halfedge_identity(self):
        for key,value in [('native_left_faces',2),('native_halfedge_ids',101)]:
            original,saved=fixture();saved[key][0,0]=value
            with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_dense_IDs_and_failed_region_reject(self):
        for key,value in [('valid_face_ids',np.array([0,2],np.int32)),('valid_vertex_ids',np.array([0,1,2,3],np.int32)),('failed_face_ids',np.array([0],np.int32))]:
            original,saved=fixture();saved[key]=value
            with self.assertRaises(ValueError):verify_arrays(np,original,saved)
    def test_counter_observed_write_and_failed_count(self):
        self.assertTrue(verify_counter(scalar(-23456),-23456,scalar(0),0,[],3))
        self.assertTrue(verify_counter(scalar(-23456),-23456,scalar(1),1,[2],3))
        for after,settings,failed in [(-23456,-23456,[]),(0,-23456,[]),(1,1,[]),(2,2,[1,1]),(1,1,[3]),(True,True,[])]:
            with self.assertRaises(ValueError):verify_counter(scalar(-23456),-23456,scalar(after),settings,failed,3)

if __name__=='__main__':unittest.main(verbosity=2)
