"""Exposure-only CPU contracts; no Blender, private mesh, or renderer launch."""
import copy
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("render_exposure.py")
spec = importlib.util.spec_from_file_location("brick_render_exposure", SOURCE)
renderer = importlib.util.module_from_spec(spec) if SOURCE.is_file() else None
if renderer is not None:
    spec.loader.exec_module(renderer)


class ExposureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(renderer, "Exposure comparison renderer is missing")
        return renderer

    def test_schedule_renders_only_same_positive_x_camera_at_zero_one_two_EV(self):
        self.assertEqual(self.api().exposure_schedule(), (
            ("pos-x", (1, 0, 0), 0),
            ("pos-x-ev1", (1, 0, 0), 1),
            ("pos-x-ev2", (1, 0, 0), 2),
        ))

    def test_exposure_is_the_only_permitted_settings_difference_without_mutation(self):
        validate = self.api().assert_exposure_only
        baseline = {"exposure": 0, "gamma": 1, "samples": 32, "world": [.12, .12, .12, 1], "view_transform": "AgX"}
        before = copy.deepcopy(baseline)
        for exposure in (0, 1, 2):
            actual = copy.deepcopy(baseline)
            actual["exposure"] = exposure
            snapshot = copy.deepcopy(actual)
            validate(baseline, actual, exposure)
            self.assertEqual(actual, snapshot)
        self.assertEqual(baseline, before)

    def test_any_other_settings_change_missing_field_or_wrong_EV_is_rejected(self):
        validate = self.api().assert_exposure_only
        baseline = {"exposure": 0, "gamma": 1, "samples": 32, "world": [.12, .12, .12, 1]}
        cases = [
            {**baseline, "exposure": 1, "gamma": 2.2},
            {**baseline, "exposure": 1, "samples": 16},
            {**baseline, "exposure": 1, "world": [.2, .12, .12, 1]},
            {**baseline, "exposure": 1, "extra": True},
            {"exposure": 1, "gamma": 1, "samples": 32},
            {**baseline, "exposure": 2},
        ]
        for actual in cases:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                validate(baseline, actual, 1)
        for exposure in (-1, 3, float("nan")):
            with self.subTest(exposure=exposure), self.assertRaises(ValueError):
                validate(baseline, {**baseline, "exposure": exposure}, exposure)


if __name__ == "__main__":
    unittest.main()
