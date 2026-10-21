"""Small exact graph/arithmetic fixtures; never original quality-suite reruns."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest,numpy as np
from topology_witness import canonical,incidence,local_faces,face_records
from witness_material import select_material_query, decode, encode, local_hit, source_membership

class LocalWitnessTests(unittest.TestCase):
    def test_oriented_cycle_not_reverse(self):
        self.assertEqual(canonical([3,1,2]),(1,2,3));self.assertNotEqual(canonical([1,3,2]),(1,2,3))
    def test_lookup_absent_boundary_and_junction(self):
        keys=np.array([1,2,6],np.uint64);counts=np.array([1,9,2],np.uint8)
        self.assertEqual(incidence(keys,counts,2,0,4),9);self.assertEqual(incidence(keys,counts,1,2,4),2);self.assertEqual(incidence(keys,counts,2,3,4),0)
    def test_local_occurrences_preserve_duplicates(self):
        f=np.array([[0,1,2],[2,0,1],[1,2,3]],np.int32)
        self.assertEqual(local_faces(f,[0]).tolist(),[0,1]);self.assertEqual([canonical(t) for t in f[:2]],[(0,1,2)]*2)
    def test_face_source_ids_and_normal(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32);f=np.array([[0,1,2]],np.int32)
        r=face_records(v,f,[0],np.array([5,8,9],np.int32))[0]
        self.assertEqual(r['source_vertex_ids'],[5,8,9]);self.assertEqual(r['unit_normal'],[0,0,1]);self.assertEqual(r['area'],.5)
    def test_bad_witness_rejected(self):
        with self.assertRaises(ValueError):face_records(np.zeros((3,3),np.float32),np.array([[0,1,2]],np.int32),[0])

    def test_saved_query_tie_uses_query_then_channel(self):
        source=np.zeros((256,6),np.float32);predicted=np.zeros((256,3),np.float64)
        source[2,:2]=.3;source[128,2]=.3
        result=select_material_query(np,source,predicted,{'high_variation':{'samples':128},'octant_2':{'samples':128}})
        self.assertEqual((result['query'],result['channel'],result['region']),(2,0,'high_variation'))

    def test_native_codec_endpoints_and_F32_interpolation(self):
        rgba=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
        linear=decode(np,rgba)
        self.assertEqual(linear.dtype,np.float32);np.testing.assert_array_equal(linear,np.eye(3,dtype=np.float32))
        np.testing.assert_allclose(encode(np,linear),np.eye(3),rtol=0,atol=2e-16)
        np.testing.assert_array_equal(np.array([.25,.25,.5])@linear,np.array([.25,.25,.5]))

    def test_one_pixel_hit_is_two_sided_and_ties_lowest_row(self):
        v=np.array([[-.8,-.8,0],[.8,-.8,0],[0,.8,0],[-.8,-.8,-.4],[.8,-.8,-.4],[0,.8,-.4]],np.float32)
        f=np.array([[0,1,2],[2,1,0],[3,4,5]],np.int32)
        camera=dict(camera_type='ORTHO',camera_location=[0,0,3],camera_right=[1,0,0],camera_up=[0,1,0],axis_direction=[0,0,1],ortho_scale=2,clip=[.01,10])
        result=local_hit(np,v,f,camera,3,3,size=8)
        self.assertEqual(result['hit']['face'],0);self.assertEqual(result['hit']['depth'],3)
        self.assertEqual(result['containing_triangle_rows'],3)

    def test_one_pixel_bbox_miss_does_not_raster_other_pixels(self):
        v=np.array([[-.8,-.8,0],[.8,-.8,0],[0,.8,0]],np.float32);f=np.array([[0,1,2]],np.int32)
        camera=dict(camera_type='ORTHO',camera_location=[0,0,3],camera_right=[1,0,0],camera_up=[0,1,0],axis_direction=[0,0,1],ortho_scale=2,clip=[.01,10])
        result=local_hit(np,v,f,camera,0,0,size=8)
        self.assertIsNone(result['hit']);self.assertEqual(result['bbox_covering_triangle_rows'],0);self.assertFalse(result['full_raster_performed'])

    def test_original_triple_membership_distinguishes_cycle_reverse_new(self):
        faces=np.array([[0,1,2],[0,2,3]],np.int32)
        result=source_membership(np,faces,{'cycle':[1,2,0],'reverse':[0,2,1],'new':[1,2,3]},4)
        self.assertEqual(result['cycle']['oriented_original_face_ids'],[0]);self.assertEqual(result['cycle']['reverse_match_count'],0)
        self.assertEqual(result['reverse']['reverse_original_face_ids'],[0]);self.assertEqual(result['reverse']['oriented_match_count'],0)
        self.assertTrue(result['new']['no_oriented_original_triple_match']);self.assertFalse(result['new']['collapse_ancestry_inferred'])

if __name__=='__main__':unittest.main()
