"""Standalone bounded NumPy tests; execute explicitly with python test_core.py."""
import unittest
import numpy as np
from core import edges, separate, stats


class FanSeparationTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                             (0, -1, 0), (0, 0, -1)], dtype='<f4')
        self.faces = np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3),
                               (0, 4, 1), (0, 1, 5), (0, 5, 4), (1, 4, 5)])
        self.rgba = np.arange(24, dtype='<f4').reshape(6, 4) / 24

    def check_exact(self, xyz, faces, rgba):
        out_xyz, out_faces, out_rgba, origin = separate(xyz, faces, rgba)
        self.assertEqual(out_xyz[out_faces].tobytes(), xyz[faces].tobytes())
        self.assertEqual(out_rgba[out_faces].tobytes(), rgba[faces].tobytes())
        np.testing.assert_array_equal(origin[out_faces], faces)
        self.assertLessEqual(int(edges(out_faces, len(out_xyz))[1].max()), 2)
        return out_xyz, out_faces, out_rgba, origin

    def test_alternate_endpoint_paths_keep_two_closed_tetrahedra(self):
        before, _ = stats(self.xyz, self.faces)
        self.assertEqual((before['components'], before['nonmanifold_gt2_edges']), (1, 1))
        xyz, faces, _, _ = self.check_exact(self.xyz, self.faces, self.rgba)
        after, ids = stats(xyz, faces)
        self.assertEqual((after['components'], after['boundary_edges']), (2, 0))
        self.assertEqual(after['triangles'], 8)
        self.assertEqual(ids.dtype, np.dtype('int32'))
        self.assertAlmostEqual(sum(before['component_areas']), sum(after['component_areas']))

    def test_duplicate_multiplicity_and_reversed_winding_survive(self):
        faces = np.array([(0, 1, 2), (0, 1, 2), (0, 2, 1)])
        xyz, out, _, _ = self.check_exact(self.xyz, faces, self.rgba)
        measured, _ = stats(xyz, out)
        self.assertEqual((measured['triangles'], measured['components'], measured['boundary_edges']), (3, 3, 9))

    def test_globally_blocked_local_two_face_edge_is_not_joined(self):
        faces = np.array([(0, 1, 2), (1, 0, 3)])
        keys, counts, per_face = edges(faces, len(self.xyz))
        np.testing.assert_array_equal(keys[counts == 2], [1])
        self.assertEqual(per_face.shape, (2, 3))
        xyz, out, _, _ = separate(self.xyz, faces, self.rgba, np.zeros(len(keys), bool))
        measured, _ = stats(xyz, out)
        self.assertEqual((measured['components'], measured['boundary_edges']), (2, 6))


if __name__ == '__main__':
    unittest.main()
