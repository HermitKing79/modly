"""Pure CPU camera checks; never imports Blender or the private mesh."""
import importlib.util
import math
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("render_views.py")
spec = importlib.util.spec_from_file_location("brick_render_views", SOURCE)
renderer = importlib.util.module_from_spec(spec) if SOURCE.is_file() else None
if renderer is not None:
    spec.loader.exec_module(renderer)


class CameraTests(unittest.TestCase):
    def renderer(self):
        self.assertIsNotNone(renderer, "Camera implementation is missing")
        return renderer

    def test_axis_views_have_hand_checked_right_and_up_directions(self):
        basis = self.renderer().view_basis
        cases = (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        )
        for direction, expected_right, expected_up in cases:
            with self.subTest(direction=direction):
                right, up, forward = basis(direction)
                self.assertEqual(right, expected_right)
                self.assertEqual(up, expected_up)
                self.assertEqual(forward, tuple(-value for value in direction))

    def test_opposite_isometrics_look_toward_the_same_center(self):
        basis = self.renderer().view_basis
        positive = basis((1, -1, 1))
        negative = basis((-1, 1, -1))
        for actual, expected in zip(positive[2], negative[2]):
            self.assertAlmostEqual(actual, -expected)
        for right, up, forward in (positive, negative):
            for vector in (right, up, forward):
                self.assertAlmostEqual(sum(value * value for value in vector), 1)
            self.assertAlmostEqual(sum(a * b for a, b in zip(right, up)), 0)
            self.assertAlmostEqual(sum(a * b for a, b in zip(right, forward)), 0)
            self.assertAlmostEqual(sum(a * b for a, b in zip(up, forward)), 0)
            cross = (right[1] * up[2] - right[2] * up[1],
                     right[2] * up[0] - right[0] * up[2],
                     right[0] * up[1] - right[1] * up[0])
            for actual, expected in zip(cross, forward):
                self.assertAlmostEqual(actual, -expected)

    def test_shared_sphere_frame_contains_each_bbox_corner(self):
        center, radius, scale = self.renderer().sphere_frame(((-2, -1, 0), (2, 1, 6)))
        self.assertEqual(center, (0, 0, 3))
        self.assertAlmostEqual(radius, math.sqrt(14))
        self.assertAlmostEqual(scale, 2 * math.sqrt(14) * 1.08)
        # The sphere diameter fits in either image dimension with visible margin.
        self.assertLess(2 * radius, scale)

    def test_invalid_view_direction_and_degenerate_bounds_are_rejected(self):
        api = self.renderer()
        for direction in ((0, 0, 0), (float("nan"), 0, 1)):
            with self.assertRaises(ValueError):
                api.view_basis(direction)
        with self.assertRaises(ValueError):
            api.sphere_frame(((0, 0, 0), (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
