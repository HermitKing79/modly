"""Small CPU-only correspondence rejection tests; no real artifacts rewritten."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import verify_mapping as mapping


class CorrespondenceTests(unittest.TestCase):
    def fixture(self):
        coords = np.array([[0, 11, 12, 13], [0, 21, 22, 23], [0, 31, 32, 33]], dtype="<i4")
        features = np.array([[0, 1, -1, 0, 0, 0, 0], [2, -2, .25, 0, 0, 0, 0], [-3, 3, 0, 0, 0, 0, 0]], dtype="<f4")
        vertices = mapping.formula_vertices(coords, features)
        return coords, features, vertices

    def test_accepts_same_original_order(self):
        coords, features, vertices = self.fixture()
        mapping.check_rows(coords, coords.copy(), features, vertices.copy(), vertices.copy())

    def test_rejects_same_row_count_permuted_material(self):
        coords, features, vertices = self.fixture()
        with self.assertRaisesRegex(ValueError, "coordinates.*order differ"):
            mapping.check_rows(coords, coords[::-1].copy(), features, vertices, vertices)

    def test_rejects_reordered_mesh_and_single_bit_change(self):
        coords, features, vertices = self.fixture()
        with self.assertRaisesRegex(ValueError, "Formula/stored vertices"):
            mapping.check_rows(coords, coords, features, vertices[::-1].copy(), vertices)
        changed = vertices.copy()
        changed.view("<u4")[1, 1] ^= np.uint32(1)
        with self.assertRaisesRegex(ValueError, "Formula/PLY vertices"):
            mapping.check_rows(coords, coords, features, vertices, changed)

    def test_stats_retain_original_out_of_range_channels(self):
        features = np.array([[.1, .2, .3, -.01, 1.01, .998], [.4, .5, .6, .001, 1, 1.004]], dtype="<f4")
        original = features.copy()
        stats = mapping.channel_stats(features)
        self.assertEqual(stats["metallic"]["negative_count"], 1)
        self.assertEqual(stats["roughness"]["above_one_count"], 1)
        self.assertEqual(stats["alpha"]["above_one_count"], 1)
        self.assertTrue(np.array_equal(original.view("u4"), features.view("u4")))

    def test_header_rejects_payload_gap_before_mapping(self):
        with tempfile.TemporaryDirectory(prefix="mapping-test-", dir=mapping.E) as folder:
            path = Path(folder) / "gap.safetensors"
            header = json.dumps({"features": {"dtype": "F32", "shape": [1, 6], "data_offsets": [4, 28]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header))+header+b"\0"*28)
            with self.assertRaisesRegex(ValueError, "gap/overlap"):
                mapping.read_safetensors(path, {"features": ("F32", (1, 6))})


if __name__ == "__main__":
    unittest.main()
