import unittest
import numpy as np
import contract as c

class TriangleContract(unittest.TestCase):
    def test_split_vertices_triangle_order_and_cyclic_rotation(self):
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype='f4')
        rgb=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype='u1')
        faces=np.array([[0,1,2],[0,3,1]],dtype='i4')
        corner=faces[::-1][:,[1,2,0]].reshape(-1)
        self.assertTrue(c.compare(xyz,faces,rgb,xyz[corner],np.arange(6,dtype='i4').reshape(2,3),rgb[corner],'POINT')['all_triangle_corners_exact_position_and_RGB'])
    def test_reversed_winding_missing_face_and_changed_RGB_rejected(self):
        xyz=np.eye(3,dtype='f4');rgb=np.eye(3,dtype='u1');faces=np.array([[0,1,2]],dtype='i4')
        for badfaces,badrgb in [(faces[:,::-1],rgb),(faces[:0],rgb),(faces,rgb+1)]:
            with self.assertRaises(AssertionError):c.compare(xyz,faces,rgb,xyz,badfaces,badrgb,'POINT')
    def test_duplicate_keys_and_degenerate_corner_rotations(self):
        a=np.array([[0,0,1],[1,0,0],[0,1,0],[2,1,2]],dtype='i4')
        self.assertTrue(np.array_equal(c.canonical_triangles(a),c.canonical_triangles(np.roll(a,1,axis=1))))
    def test_linear_RGB_not_merely_display_bytes(self):
        xyz=np.eye(3,dtype='f4');rgb=np.eye(3,dtype='u1');faces=np.array([[0,1,2]],dtype='i4')
        linear=rgb.astype('f4')/255
        self.assertEqual(c.compare(xyz,faces,rgb,xyz,faces,rgb,'POINT',linear,linear)['maximum_linear_RGB_error'],0)
        with self.assertRaises(AssertionError):c.compare(xyz,faces,rgb,xyz,faces,rgb,'POINT',linear,linear+np.float32(1e-4))

if __name__=='__main__':unittest.main()
