"""CPU image-contract checks. No upstream, torch, or model imports."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("alpha_input", ROOT / "experimental/extensions/trellis2-amd/alpha_input.py")
alpha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alpha)


def fixture(size=(12, 10)):
    image = Image.new("RGBA", size, (80, 120, 160, 0))
    for y in range(2, size[1] - 2):
        for x in range(2, size[0] - 2):
            image.putpixel((x, y), (80, 120, 160, 255))
    return image


class AlphaInputTests(unittest.TestCase):
    def test_experimental_mock_mode_validates_and_recovers(self):
        import json
        import tempfile
        sys.path.insert(0, str(ROOT / "api"))
        sys.path.insert(0, str(ROOT / "experimental/extensions/trellis2-amd"))
        spec = importlib.util.spec_from_file_location("alpha_generator_test", ROOT / "experimental/extensions/trellis2-amd/generator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            generator = module.AMDScaffold(Path(directory), Path(directory))
            generator.MODEL_NODE_ID = "generate"
            generator._model = "mock-readiness-only"
            generator._checkpoints = []
            stream = io.BytesIO()
            fixture().save(stream, format="PNG")
            with self.assertRaisesRegex(ValueError, "alpha-only"):
                generator.generate(b"bad", {"mock": True, "image_mode": "alpha-only"})
            path = generator.generate(stream.getvalue(), {"mock": True, "image_mode": "alpha-only"})
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["image_mode"], "alpha-only")
            self.assertFalse(receipt["alpha_input"]["rembg_required"])
            self.assertFalse(receipt["weights_loaded"])
            self.assertFalse(receipt["asset_generated"])
            default = generator.generate(b"transport fixture", {"mock": True})
            self.assertEqual(json.loads(default.read_text())["image_mode"], "rembg")

    def test_valid_rgba_is_unchanged_and_reported(self):
        image = fixture()
        before = image.tobytes()
        result = alpha.validate_alpha_image(image)
        self.assertEqual(result["mode"], "alpha-only")
        self.assertEqual(result["foreground_bbox"], [2, 2, 9, 7])
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(image.mode, "RGBA")

    def test_opaque_rgb_empty_faint_and_degenerate_masks_fail(self):
        bad = [Image.new("RGB", (12, 10)), Image.new("RGBA", (12, 10), (0, 0, 0, 255)),
               Image.new("RGBA", (12, 10), (0, 0, 0, 0)),
               Image.new("RGBA", (12, 10), (0, 0, 0, 128)), fixture()]
        bad[-1].putalpha(255)
        thin = Image.new("RGBA", (12, 10))
        for x in range(2, 8):
            thin.putpixel((x, 4), (255, 255, 255, 255))
        bad.append(thin)
        for image in bad:
            with self.subTest(mode=image.mode, extrema=image.getextrema()), self.assertRaisesRegex(ValueError, "alpha-only"):
                alpha.validate_alpha_image(image)

    def test_faint_mask_with_transparent_background_fails(self):
        image = fixture()
        image.putalpha(image.getchannel("A").point(lambda value: 204 if value else 0))
        with self.assertRaisesRegex(ValueError, "foreground"):
            alpha.validate_alpha_image(image)

    def test_unsegmented_uniform_alpha_fails(self):
        image = Image.new("RGBA", (12, 10), (80, 120, 160, 254))
        with self.assertRaisesRegex(ValueError, "transparent background"):
            alpha.validate_alpha_image(image)

    def test_decode_is_explicit_rgba_and_rejects_invalid_bytes(self):
        stream = io.BytesIO()
        fixture().save(stream, format="PNG")
        image, detail = alpha.decode_alpha_image(stream.getvalue())
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(detail["size"], [12, 10])
        with self.assertRaisesRegex(ValueError, "alpha-only"):
            alpha.decode_alpha_image(b"not an image")

    def test_resource_bounds_precede_decoding_pixels(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            alpha.decode_alpha_image(b"x" * (alpha.MAX_IMAGE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "limit"):
            alpha.validate_alpha_image(Image.new("RGBA", (alpha.MAX_DIMENSION + 1, 1)))

    def test_unknown_mode_cannot_fall_back(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            alpha.validate_mode("automatic")
        self.assertEqual(alpha.validate_mode("rembg"), "rembg")
        self.assertEqual(alpha.validate_mode("alpha-only"), "alpha-only")

    def test_cpu_boundary_does_not_import_ai_packages(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "trellis2", "transformers")))


if __name__ == "__main__":
    unittest.main()
