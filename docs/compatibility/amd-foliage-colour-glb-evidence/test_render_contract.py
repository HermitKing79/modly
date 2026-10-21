"""Pure contracts; no Blender/model load, filesystem fixtures or renderer runs."""
import importlib.util
from pathlib import Path
import sys
import subprocess
import unittest

sys.dont_write_bytecode = True
import numpy as np

PATH = Path(__file__).with_name("render_worker.py")
WORKER = None
if PATH.exists():
    SPEC = importlib.util.spec_from_file_location("foliage_colour_render_contract", PATH)
    WORKER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(WORKER)


class RenderContractTests(unittest.TestCase):
    def worker(self):
        self.assertIsNotNone(WORKER, "The comparison worker must implement the pure contracts")
        return WORKER

    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="<f4")
        self.faces = np.array([[0, 1, 2], [1, 3, 2]], dtype="<i4")

    def test_cyclic_face_reordering_maps_back_to_original_corner_rows(self):
        # A vertex permutation and two cyclic rotations preserve oriented geometry.
        imported_vertices = self.vertices[[2, 0, 3, 1]]
        imported_faces = np.array([[2, 0, 3], [0, 1, 3]], dtype="<i4")
        got = self.worker().match_oriented_corners(imported_vertices, imported_faces, self.vertices, self.faces)
        np.testing.assert_array_equal(got, [[4, 5, 3], [2, 0, 1]])

    def test_reversed_face_winding_fails(self):
        reversed_faces = self.faces.copy()
        reversed_faces[0] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, "oriented"):
            self.worker().match_oriented_corners(self.vertices, reversed_faces, self.vertices, self.faces)

    def test_single_float32_position_bit_change_fails(self):
        changed = self.vertices.copy()
        changed[1, 0] = np.nextafter(np.float32(1), np.float32(2))
        with self.assertRaisesRegex(ValueError, "oriented"):
            self.worker().match_oriented_corners(changed, self.faces, self.vertices, self.faces)

    def test_duplicate_geometry_cannot_invent_source_lineage(self):
        faces = np.array([[0, 1, 2], [0, 1, 2]], dtype="<i4")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.worker().match_oriented_corners(self.vertices, faces, self.vertices, faces)

    def test_exact_srgb_encode_uses_linear_branch_and_unit_endpoint(self):
        got = self.worker().encode_srgb(np.array([0, .0031308, 1], dtype=np.float64))
        np.testing.assert_allclose(got, [0, .040449936, 1], rtol=0, atol=1e-12)

    def test_reference_schema_accepts_original_i32_face_ids_and_i64_vertex_ids(self):
        self.assertTrue(callable(getattr(self.worker(), "validate_reference", None)), "Original reference schema validator required")
        values = {
            "vertices.npy": np.zeros((3177, 3), dtype="<f4"),
            "faces.npy": np.tile(np.array([[0, 1, 2]], dtype="<i4"), (4531, 1)),
            "linear-COLOR0.npy": np.full((3177, 3), .125, dtype="<f4"),
            "source-vertex-ids.npy": np.arange(3177, dtype="<i8"),
            "source-face-ids.npy": np.arange(4531, dtype="<i4"),
        }
        self.assertIsNone(self.worker().validate_reference(values))
        values["source-face-ids.npy"] = np.arange(4531, dtype="<i8")
        with self.assertRaisesRegex(ValueError, "source IDs"):
            self.worker().validate_reference(values)

    def test_zero_emission_accepts_black_colour_or_zero_strength(self):
        self.assertTrue(callable(getattr(self.worker(), "emission_is_zero", None)), "Effective emission validator required")
        self.assertTrue(self.worker().emission_is_zero(1, [0, 0, 0]))
        self.assertTrue(self.worker().emission_is_zero(0, [1, 1, 1]))
        self.assertFalse(self.worker().emission_is_zero(1, [0, .1, 0]))

    def test_malformed_binary_go_exits_before_blender_access(self):
        self.worker()
        for token in (b"", b"GO\r\n", b"GO\x00\n"):
            with self.subTest(token=token):
                completed = subprocess.run([sys.executable, "-B", str(PATH)], input=token,
                                           capture_output=True, timeout=10, check=False)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(b"FOLIAGE_COLOUR_GLB_READY_GO", completed.stdout)
                self.assertIn(b"Exact binary GO admission rejected", completed.stderr)
                self.assertNotIn(b"No module named 'bpy'", completed.stderr)

    def test_image_gate_detects_local_maximum_even_when_rmse_is_small(self):
        reference = np.zeros((10, 10, 4), dtype=np.uint8)
        reference[1:9, 1:9, 3] = 255
        actual = reference.copy()
        actual[4, 4, 0] = 255
        result = self.worker().compare_images(reference, actual)
        self.assertFalse(result["pass"])
        self.assertEqual(result["silhouette_IoU"], 1)
        self.assertEqual(result["all_pixels_RGB"]["max_abs"], 1)

    def test_image_gate_detects_silhouette_loss_with_identical_rgb(self):
        reference = np.zeros((10, 10, 4), dtype=np.uint8)
        reference[1:9, 1:9, 3] = 255
        actual = reference.copy()
        actual[4, 4, 3] = 0
        result = self.worker().compare_images(reference, actual)
        self.assertFalse(result["pass"])
        self.assertEqual(result["all_pixels_RGB"]["max_abs"], 0)
        self.assertAlmostEqual(result["silhouette_IoU"], 63 / 64)

    def test_image_gate_accepts_identical_nonempty_silhouette(self):
        pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        pixels[0, 0] = [120, 50, 80, 255]
        result = self.worker().compare_images(pixels, pixels.copy())
        self.assertTrue(result["pass"])
        self.assertTrue(result["RGBA_pixels_exact"])


if __name__ == "__main__":
    unittest.main()
