"""Small known display-code fixtures; no render or model execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('exposure_composition', Path(__file__).with_name('compose_exposure.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StatisticsTests(unittest.TestCase):
    def test_known_fixed_region_clipping_and_dark_counts(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]],
                          [[128, 0, 0], [250, 0, 0]]], dtype=np.uint8)
        result = module.statistics(image, np.ones((2, 2), dtype=bool))
        self.assertEqual(result['pixels'], 4)
        self.assertEqual(result['any_channel_at_255_count'], 1)
        self.assertEqual(result['any_channel_at_least_250_count'], 2)
        self.assertEqual(result['all_channels_at_most_16_count'], 1)
        expected = ((255 + 128 + 250) * .2126 + 255 * (.7152 + .0722)) / (4 * 255)
        self.assertAlmostEqual(result['mean_encoded_luma'], expected)

    def test_empty_region_rejected(self):
        with self.assertRaises(ValueError):
            module.statistics(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=bool))


if __name__ == '__main__':
    unittest.main()
