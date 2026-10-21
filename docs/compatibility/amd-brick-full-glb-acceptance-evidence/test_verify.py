"""Tiny independent verifier fixtures; no Blender or actual asset reads."""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).with_name('verify_import.py')
verifier = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('independent_acceptance_verifier_test', MODULE)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)


def fixture():
    vertices = np.array([[-2, -0., 3], [1, -3, -1], [0., 2, 1], [99, 99, 99], [4, 1, -2]], '<f4')
    faces = np.array([[0, 2, 1], [4, 1, 2], [0, 2, 1]], '<i4')
    colours = np.array([[0, .25, .5], [1, .5, .25], [.5, 0, 1], [.25, .5, .25], [1, 1, 0]], '<f4')
    oracle = np.array([[0, 137, 188, 255], [255, 188, 137, 255], [188, 0, 255, 255],
                       [137, 188, 137, 255], [255, 255, 0, 255]], np.uint8)
    actual_vertices = np.array([[-2, -0., 3], [1, -3, -1], [0., 2, 1], [4, 1, -2]], '<f4')
    actual_faces = np.array([[0, 2, 1], [3, 1, 2], [0, 2, 1]], '<i4')
    actual_bytes = np.array([[0, 137, 188, 255], [188, 0, 255, 255], [255, 188, 137, 255],
                             [255, 255, 0, 255], [255, 188, 137, 255], [188, 0, 255, 255],
                             [0, 137, 188, 255], [188, 0, 255, 255], [255, 188, 137, 255]], np.uint8)
    return [vertices, faces, colours, actual_vertices, actual_faces, actual_bytes, oracle]


class Availability(unittest.TestCase):
    def test_verifier_exists(self):
        self.assertIsNotNone(verifier, 'Independent acceptance verifier is not implemented')


@unittest.skipIf(verifier is None, 'Verifier implementation pending')
class ArrayVerification(unittest.TestCase):
    def test_only_orphan_is_omitted_all_oriented_triangles_and_corner_bytes_exact(self):
        report = verifier.verify_arrays(*fixture(), batch_rows=2)
        self.assertEqual(report['referenced_vertices'], 4)
        self.assertEqual(report['unreferenced_vertices_omitted'], 1)
        self.assertEqual(report['triangles'], 3)
        self.assertTrue(report['ordered_original_geometry_and_triangle_multiplicity_exact'])
        self.assertTrue(report['all_corner_RGBA_bytes_equal_native_oracle'])
        self.assertGreater(report['quantization_display_only']['linear_RGB']['max_abs'], 0)

    def test_geometry_color_oracle_alpha_and_schema_corruption_rejected(self):
        changes = [lambda a: a[3].__setitem__((0, 1), 0.),
                   lambda a: a[3].__setitem__((0, 0), np.nan),
                   lambda a: a[4].__setitem__((0, 0), 1),
                   lambda a: a[4].__setitem__((2, 0), 3),
                   lambda a: a[5].__setitem__((8, 0), 254),
                   lambda a: a[5].__setitem__((0, 3), 0),
                   lambda a: a[6].__setitem__((3, 3), 0),
                   lambda a: a[1].__setitem__((0, 0), -1),
                   lambda a: a[2].__setitem__((3, 0), np.inf),
                   lambda a: a.__setitem__(3, a[3].astype('<f8')),
                   lambda a: a.__setitem__(4, a[4].astype('<u4')),
                   lambda a: a.__setitem__(5, a[5].astype('<f4')),
                   lambda a: a.__setitem__(6, a[6][:4]),
                   lambda a: a.__setitem__(3, a[3][:3])]
        for change in changes:
            arrays = fixture(); change(arrays)
            with self.subTest(change=change), self.assertRaises(ValueError):
                verifier.verify_arrays(*arrays, batch_rows=2)

    def test_batch_size_is_bounded_and_typed(self):
        for size in (0, True, 1.5, 32769):
            with self.subTest(size=size), self.assertRaises(ValueError):
                verifier.verify_arrays(*fixture(), batch_rows=size)

    def test_matching_actual_and_oracle_bytes_do_not_hide_excess_quantization_loss(self):
        arrays = fixture()
        arrays[6][0, 0] = 1
        arrays[5][[0, 6], 0] = 1
        with self.assertRaisesRegex(ValueError, 'quantization'):
            verifier.verify_arrays(*arrays, batch_rows=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
