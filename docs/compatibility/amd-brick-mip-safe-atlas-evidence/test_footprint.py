"""Synthetic-only interval proof against the existing real mip sampler."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from footprint import axis_support, validate_isolation

path = Path(__file__).resolve().parent.parent / 'amd-brick-pbr-mip-evidence/mip_math.py'
spec = importlib.util.spec_from_file_location('existing_mip_math_for_support', path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class SupportTests(unittest.TestCase):
    def test_exact_zero_tap_fractional_tap_half_lod_and_outer_clamp(self):
        self.assertEqual(axis_support(16.5 / 64, 16.5 / 64, 64, 0, 0), (16, 16))
        self.assertEqual(axis_support(.4375, .4375, 64, 3, 3), (24, 31))
        self.assertEqual(axis_support(np.nextafter(.4375, 1), np.nextafter(.4375, 1), 64, 3, 3), (24, 39))
        self.assertEqual(axis_support(3.5 / 16, 3.5 / 16, 64, 2, 2), (12, 15))
        self.assertEqual(axis_support(3.5 / 16, 3.5 / 16, 64, 2.5, 2.5), (8, 23))
        self.assertEqual(axis_support(-.5, 0, 64, 0, 3), (0, 7))
        self.assertEqual(axis_support(1, 1.5, 64, 0, 3), (56, 63))

    def test_analytical_union_matches_direct_impulses_at_breakpoints_and_interiors(self):
        for lo, hi, first, last in ((.4375, .4375, 3, 3), (.21875, .21875, 2.5, 2.5),
                                   (.193, .438, 0, 3), (0, .02, .5, 2.5), (.98, 1.1, 0, 3)):
            positions = {lo, hi}
            for level in range(4):
                positions.update(x for x in (np.arange(64 >> level) + .5) / (64 >> level) if lo <= x <= hi)
            ordered = sorted(positions)
            positions.update((a + b) / 2 for a, b in zip(ordered, ordered[1:]))
            uv = np.column_stack((sorted(positions), np.full(len(positions), .5)))
            lods = sorted({first, last, *(x / 2 for x in range(7) if first <= x / 2 <= last)})
            positive = []
            for x in range(64):
                image = np.zeros((64, 64, 1), np.float64)
                image[:, x] = 1
                chain = old.pyramid(image, 3)
                if any((old.sample(chain, uv, lod) > 0).any() for lod in lods):
                    positive.append(x)
            actual = axis_support(lo, hi, 64, first, last)
            self.assertEqual(positive, list(range(actual[0], actual[1] + 1)))

    def test_all_triangle_ranges_isolated_until_positive_support_crosses_rectangle(self):
        triangles = np.array([[[.25, .25], [.4375, .25], [.25, .3]],
                              [[.75, .25], [.8, .25], [.75, .3]]], np.float64)
        islands = np.array([0, 1], np.int32)
        rects = np.array([[0, 0, 32, 64], [32, 0, 64, 64]], np.int32)
        before = triangles.copy()
        result = validate_isolation(triangles, islands, rects, 64, 0, 3)
        self.assertEqual(result['classification'], 'GO')
        self.assertEqual(result['triangle_count'], 2)
        np.testing.assert_array_equal(before, triangles)
        triangles[0, 1, 0] = np.nextafter(.4375, 1)
        with self.assertRaisesRegex(ValueError, 'triangle0'):
            validate_isolation(triangles, islands, rects, 64, 0, 3)

    def test_invalid_bounds_rectangles_and_ownership_fail_closed(self):
        for args in ((0, 1, 63, 0, 3), (0, 1, 64, 0, 7), (1, 0, 64, 0, 3),
                     (0, float('nan'), 64, 0, 3), (0, 1, 64, 2, 1)):
            with self.assertRaises(ValueError):
                axis_support(*args)
        triangle = np.array([[[.25, .25], [.3, .25], [.25, .3]]])
        with self.assertRaises(ValueError):
            validate_isolation(triangle, np.array([1], np.int32), np.array([[0, 0, 64, 64]], np.int32), 64, 0, 3)
        with self.assertRaises(ValueError):
            validate_isolation(np.repeat(triangle, 2, 0), np.array([0, 1], np.int32),
                               np.array([[0, 0, 48, 64], [32, 0, 64, 64]], np.int32), 64, 0, 3)


if __name__ == '__main__':
    unittest.main()
