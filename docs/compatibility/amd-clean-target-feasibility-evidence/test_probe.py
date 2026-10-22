"""Bounded CPU validator tests; never import MeshLib or create target geometry."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
from import_probe import compare_roundtrip

def sample():
    v=np.array([[-0.,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    f=np.array([[0,1,2],[2,1,3]],np.int32)
    return [np,v,f,np.array([10,20,30,40],np.int64),np.array([100,200],np.int32),v.astype(np.float64),f.copy(),np.arange(2,dtype=np.int32),np.arange(4,dtype=np.int32)]

class RoundtripTests(unittest.TestCase):
    def test_exact_and_cyclic_oriented_correspondence(self):
        args=sample();summary,rot=compare_roundtrip(*args);self.assertIsInstance(summary,dict);np.testing.assert_array_equal(rot,[0,0])
        args[6]=np.array([[1,2,0],[3,2,1]],np.int32)
        _,rot=compare_roundtrip(*args);np.testing.assert_array_equal(rot,[1,2])
    def test_reversed_winding_and_reordered_face_IDs_reject(self):
        for f in [np.array([[0,2,1],[2,1,3]],np.int32),np.array([[2,1,3],[0,1,2]],np.int32)]:
            args=sample();args[6]=f
            with self.assertRaises(ValueError):compare_roundtrip(*args)
    def test_skipped_faces_or_merged_vertices_reject(self):
        for index,value in [(7,np.array([0],np.int32)),(8,np.array([0,1,2],np.int32)),(6,np.array([[0,1,2]],np.int32))]:
            args=sample();args[index]=value
            with self.assertRaises(ValueError):compare_roundtrip(*args)
    def test_position_signed_zero_nonfinite_and_sub_F32_change_reject(self):
        for value in (0.,float('nan'),-1.e-50):
            args=sample();args[5][0,0]=value
            with self.assertRaises(ValueError):compare_roundtrip(*args)
    def test_source_alias_or_wrong_input_dimension_reject(self):
        for index,value in [(3,np.array([10,20,20,40],np.int64)),(4,np.array([100,100],np.int32)),(2,np.array([[0,1,2,3]],np.int32))]:
            args=sample();args[index]=value
            with self.assertRaises(ValueError):compare_roundtrip(*args)

if __name__=='__main__':unittest.main(verbosity=2)
