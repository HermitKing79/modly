"""Small CPU fixtures only: no Blender, model, private input, or renderer launch."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).with_name("render_pbr.py")
spec = importlib.util.spec_from_file_location("brick_render_pbr", SOURCE)
renderer = importlib.util.module_from_spec(spec) if SOURCE.is_file() else None
if renderer is not None:
    spec.loader.exec_module(renderer)


class PreviewContractTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(renderer, "Learned PBR renderer is missing")
        return renderer

    def field(self, directory, mutate=None):
        header = {
            "coordinates": {"dtype": "I32", "shape": [2, 4], "data_offsets": [0, 32]},
            "features": {"dtype": "F32", "shape": [2, 6], "data_offsets": [32, 80]},
            "raw": {"dtype": "F32", "shape": [2, 6], "data_offsets": [80, 128]},
        }
        if mutate:
            mutate(header)
        encoded = json.dumps(header).encode("utf-8")
        path = Path(directory) / "fixture.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(128))
        return path, len(encoded)

    def test_header_maps_exact_rows_and_payload_offsets(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as directory:
            path, length = self.field(directory)
            base, tensors = api.read_field_header(path, rows=2)
            self.assertEqual(base, 8 + length)
            self.assertEqual(tensors["features"]["data_offsets"], [32, 80])
            self.assertEqual(tensors["coordinates"]["shape"], [2, 4])

    def test_header_rejects_overlapping_or_out_of_file_tensors(self):
        api = self.api()
        for offsets in ([16, 64], [104, 152]):
            with self.subTest(offsets=offsets), tempfile.TemporaryDirectory(dir=SOURCE.parent) as directory:
                path, _ = self.field(directory, lambda h: h["features"].update(data_offsets=offsets))
                with self.assertRaises(ValueError):
                    api.read_field_header(path, rows=2)

    def test_header_rejects_wrong_channel_count_and_oversized_length(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as directory:
            path, _ = self.field(directory, lambda h: h["features"].update(shape=[2, 5]))
            with self.assertRaises(ValueError):
                api.read_field_header(path, rows=2)
            path.write_bytes(struct.pack("<Q", 65537))
            with self.assertRaises(ValueError):
                api.read_field_header(path, rows=2)

    def test_native_gamma_transfer_clamps_for_shader_without_byte_quantization(self):
        api = self.api()
        rgb = np.array([[0, .5, 1], [-.1, .25, .75], [1.1, 0, .5]], dtype="<f4")
        original = rgb.copy()
        result = api.linear_rgba(rgb)
        np.testing.assert_allclose(result, [[0, .21763764, 1, 1],
                                           [0, .04736614, .53104923, 1],
                                           [1, 0, .21763764, 1]], rtol=2e-6, atol=1e-8)
        self.assertEqual(result.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(rgb, original)
        with self.assertRaises(ValueError):
            api.linear_rgba(np.array([[0, float("nan"), 1]], dtype="<f4"))

    def test_material_statistics_retain_out_of_range_values_and_alpha_distribution(self):
        api = self.api()
        features = np.array([[.2, .3, .4, -.01, 1.02, .998],
                             [.4, .6, .8, .02, .98, 1.004]], dtype="<f4")
        original = features.copy()
        statistics = api.channel_statistics(features, chunk_rows=1)
        self.assertEqual(statistics["metallic"]["outside_0_1"], 1)
        self.assertEqual(statistics["roughness"]["outside_0_1"], 1)
        self.assertEqual(statistics["alpha"]["outside_0_1"], 1)
        self.assertAlmostEqual(statistics["alpha"]["min"], .998, places=6)
        self.assertAlmostEqual(statistics["alpha"]["max"], 1.004, places=6)
        np.testing.assert_array_equal(features, original)
        features[0, 4] = np.nan
        with self.assertRaises(ValueError):
            api.channel_statistics(features)

    def test_json_writer_is_exclusive_and_rejects_nonfinite_receipts(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as directory:
            path = Path(directory) / "receipt.json"
            api.save_json(path, {"classification": "PASS", "alpha": .998})
            self.assertEqual(json.loads(path.read_text()), {"classification": "PASS", "alpha": .998})
            with self.assertRaises(FileExistsError):
                api.save_json(path, {"classification": "OVERWRITE"})
            with self.assertRaises(ValueError):
                api.save_json(Path(directory) / "bad.json", {"alpha": float("nan")})


if __name__ == "__main__":
    unittest.main()
