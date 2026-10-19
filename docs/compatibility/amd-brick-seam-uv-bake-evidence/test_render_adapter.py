"""Tiny synthetic CPU checks; no bpy, real atlas, render or original field."""
import hashlib
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import render_adapter as adapter


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


class Pixels:
    def __init__(self, count):
        self.values = np.zeros(count, dtype=np.float32)

    def foreach_set(self, values):
        self.values[:] = values

    def foreach_get(self, values):
        values[:] = self.values


class Images:
    def __init__(self):
        self.created = []

    def new(self, name, **options):
        image = SimpleNamespace(name=name, size=[options["width"], options["height"]],
                                is_float=options["float_buffer"], colorspace_settings=SimpleNamespace(name="sRGB"),
                                pixels=Pixels(options["width"] * options["height"] * 4), options=options)
        self.created.append(image)
        return image


class AttributeData:
    def foreach_set(self, name, values):
        self.values = values.copy()

    def foreach_get(self, name, values):
        values[:] = self.values


class Attributes:
    def __init__(self):
        self.created = []

    def new(self, **options):
        attribute = SimpleNamespace(data=AttributeData(), options=options)
        self.created.append(attribute)
        return attribute


def socket(name):
    return SimpleNamespace(name=name, is_linked=False, default_value=1)


class Nodes:
    def __init__(self):
        self.created = []

    def new(self, kind):
        node = SimpleNamespace(bl_idname=kind,
                               outputs={name: socket(name) for name in ("UV", "Color", "Green", "Blue")},
                               inputs={name: socket(name) for name in ("Vector", "Color")})
        self.created.append(node)
        return node


class Links:
    def __init__(self):
        self.created = []

    def new(self, source, target):
        target.is_linked = True
        self.created.append((source.name, target.name))


class ReferenceAdapterTests(unittest.TestCase):
    def test_sRGB_breakpoint_known_values_and_clipping_use_piecewise_transfer(self):
        values = np.array([-.1, 0, .04045, .5, 1, 2], dtype=np.float64)
        expected = np.array([0, 0, .0031308049535603713, .21404114048223255, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(adapter.srgb_decode(values), expected)
        self.assertNotEqual(float(expected[3]), float(np.float32(.5 ** 2.2)))

    def test_atlas_preserves_V_up_rows_original_values_and_opaque_display_alpha(self):
        field = np.array([[[.5, .2, .9, -.1, 1.1, .998]], [[.1, .8, .4, .3, .7, 1.004]]], dtype=np.float32)
        original = field.copy()
        rgb, mr = adapter.atlas_payloads(field)
        np.testing.assert_array_equal(field, original)
        np.testing.assert_array_equal(rgb[:, :, :3], adapter.srgb_decode(field[:, :, :3]))
        np.testing.assert_array_equal(rgb[:, :, 3], 1)
        np.testing.assert_array_equal(mr[:, 0, :], [[1, 1, 0, 1], [1, np.float32(.7), np.float32(.3), 1]])
        self.assertGreater(rgb[0, 0, 0], rgb[1, 0, 0])

    def test_nonfinite_wrong_dtype_and_wrong_channel_count_are_rejected(self):
        for invalid in (np.zeros((2, 2, 6), dtype=np.float64), np.zeros((2, 2, 5), dtype=np.float32),
                        np.full((1, 1, 6), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                adapter.atlas_payloads(invalid)

    def test_float_image_constructor_prevents_byte_storage_or_second_sRGB_decode(self):
        bpy = SimpleNamespace(data=SimpleNamespace(images=Images()))
        pixels = np.array([[[.21404114, .0031308, 1, 1]], [[0, .25, .75, 1]]], dtype=np.float32)
        image, receipt = adapter.float_image(bpy, "test linear", pixels, array_digest)
        self.assertTrue(image.options["float_buffer"])
        self.assertTrue(image.options["alpha"])
        self.assertEqual(image.colorspace_settings.name, "Non-Color")
        np.testing.assert_array_equal(image.pixels.values.reshape(pixels.shape), pixels)
        self.assertEqual(receipt["payload_sha256"], array_digest(pixels))

    def test_reference_material_constructor_routes_float_RGB_and_MR_without_POINT_shader(self):
        bpy = SimpleNamespace(data=SimpleNamespace(images=Images()))
        obj = SimpleNamespace(data=SimpleNamespace(vertices=[0, 1], attributes=Attributes(), materials=[]))
        shader = SimpleNamespace(inputs={name: socket(name) for name in ("Alpha", "Base Color", "Roughness", "Metallic")})
        material, nodes, links = SimpleNamespace(name="test"), Nodes(), Links()
        legacy = SimpleNamespace(BAKE=adapter.PRIVATE / "bake", CHANNELS=("red", "green", "blue", "metallic", "roughness", "alpha"),
                                 new_material=lambda *args: (material, shader, nodes, links), array_digest=array_digest)
        field = np.arange(24, dtype=np.float32).reshape(2, 2, 6) / 24
        values = np.array([[.1, .2, .3, -.001, 1.001, .998], [.4, .5, .6, 0, 1, 1.004]], dtype=np.float32)
        with patch.object(adapter, "ATLAS_SIZE", 2), patch.object(adapter, "digest", return_value="test-payload"), patch("numpy.load", return_value=field):
            receipt = adapter.reference_material(legacy, bpy, obj, values)
        self.assertEqual(len(bpy.data.images.created), 2)
        self.assertEqual(len(obj.data.attributes.created), 6)
        for index, attribute in enumerate(obj.data.attributes.created):
            self.assertEqual((attribute.options["type"], attribute.options["domain"]), ("FLOAT", "POINT"))
            np.testing.assert_array_equal(attribute.data.values, values[:, index])
        self.assertIn(("Green", "Roughness"), links.created)
        self.assertIn(("Blue", "Metallic"), links.created)
        self.assertIn(("Color", "Base Color"), links.created)
        self.assertFalse(shader.inputs["Alpha"].is_linked)
        self.assertFalse(any(node.bl_idname in ("ShaderNodeAttribute", "ShaderNodeVertexColor") for node in nodes.created))
        self.assertFalse(receipt["raw_POINT_attributes_used_by_shader"])
        self.assertFalse(receipt["continuous_source_field_shader"])
        self.assertIn("padded texels are not independent learned source", receipt["reference_scope"])

    def test_receipt_relabels_old_prose_without_mutating_status_or_metrics(self):
        original = {"classification": "PASS", "RGB_semantic_difference": "old gamma22 point text", "comparison_RGB8_full_frame": {"MAE": 1.2}}
        result = adapter.receipt_with_reference(original, "adapter-pin")
        self.assertEqual(original["RGB_semantic_difference"], "old gamma22 point text")
        self.assertEqual(result["classification"], "PASS")
        self.assertEqual(result["comparison_RGB8_full_frame"], {"MAE": 1.2})
        self.assertEqual(result["RGB_semantic_difference"], adapter.RGB_COMPARISON)
        self.assertFalse(result["continuous_source_field_shader"])

    def test_pinned_renderer_configuration_keeps_positional_inputs_and_source_hashes(self):
        legacy = adapter.load_renderer()
        self.assertEqual(legacy.PRIVATE, adapter.PRIVATE)
        self.assertEqual(legacy.BAKE, adapter.BAKE)
        self.assertEqual(legacy.INPUT_NAMES[:4], ("vertices.npy", "faces.npy", "uv.npy", "direct.npy"))
        self.assertEqual(legacy.INPUT_NAMES[-2:], ("sampled-field.npy", "reference-field.npy"))
        self.assertEqual(legacy.BASELINE_HASHES[adapter.HISTORICAL], adapter.HISTORICAL_SHA256)
        self.assertIn(adapter.Path(adapter.__file__).resolve(), legacy.BASELINE_HASHES)


class StartupAdmissionTests(unittest.TestCase):
    def setUp(self):
        scratch = adapter.ROOT / "artifacts/amd-brick-seam-uv-bake-87f3755/unit-scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="startup-admission-", dir=scratch)
        self.private = Path(self.temporary.name) / "render-qualified"
        self.output = self.private / "export"
        self.scratch = self.output / "scratch"
        self.scratch.mkdir(parents=True)
        for name in ("stdout.txt", "stderr.txt"):
            (self.output / name).write_bytes(b"")
        self.patch_private = patch.object(adapter, "PRIVATE", self.private)
        self.patch_private.start()

    def tearDown(self):
        self.patch_private.stop()
        self.temporary.cleanup()

    def validate(self, live=None):
        self.assertTrue(callable(getattr(adapter, "validate_initial_output", None)), "Narrow startup validator is missing")
        return adapter.validate_initial_output("export", self.output, live_tempdir=live)

    def test_original_empty_scratch_is_accepted(self):
        self.assertEqual(self.validate()["output"], str(self.output.resolve()))

    def test_only_exact_live_empty_session_child_is_accepted(self):
        session = self.scratch / "blender_a00001"
        session.mkdir()
        self.assertEqual(self.validate(str(session))["live_session_directory"], str(session))
        for live in (None, str(self.scratch), str(self.private / "escaped")):
            with self.assertRaises(ValueError):
                self.validate(live)
        (self.scratch / "foreign").mkdir()
        with self.assertRaises(ValueError):
            self.validate(str(session))

    def test_nonempty_session_and_unexpected_stage_entries_are_rejected(self):
        session = self.scratch / "blender_a00001"
        session.mkdir()
        (session / "unexpected.bin").write_bytes(b"x")
        with self.assertRaises(ValueError):
            self.validate(str(session))
        (session / "unexpected.bin").unlink()
        (self.output / "patch.glb").write_bytes(b"prior attempt")
        with self.assertRaises(ValueError):
            self.validate(str(session))

    def test_session_symlink_and_reparse_metadata_are_rejected(self):
        session = self.scratch / "blender_a00001"
        session.mkdir()
        original_lstat = Path.lstat
        for symlink, reparse in ((True, False), (False, True)):
            def lstat(path, *args, **kwargs):
                result = original_lstat(path, *args, **kwargs)
                if path == session:
                    return SimpleNamespace(st_mode=stat.S_IFLNK if symlink else result.st_mode,
                                           st_file_attributes=0x400 if reparse else 0)
                return result
            with patch.object(Path, "lstat", lstat):
                with self.assertRaises(ValueError):
                    self.validate(str(session))

    def test_GATE_is_consumed_once_before_validation_and_invalid_gate_never_runs_body(self):
        self.assertTrue(callable(getattr(adapter, "install_output_admission", None)), "Verified GO wrapper is missing")
        calls = []
        legacy = SimpleNamespace(await_host=lambda: calls.append("GO"))
        def body(stage, output, save_blend=False):
            legacy.private_output(stage, output)
            legacy.await_host()
            calls.append("BODY")
        legacy.main = body
        adapter.install_output_admission(legacy)
        with self.assertRaises(ValueError):
            legacy.private_output("export", self.output)
        legacy.main("export", self.output)
        self.assertEqual(calls, ["GO", "BODY"])
        with self.assertRaises(ValueError):
            legacy.main("export", self.output)
        bad = SimpleNamespace(await_host=lambda: (_ for _ in ()).throw(ValueError("bad GO")), main=lambda *args: calls.append("UNSAFE"))
        adapter.install_output_admission(bad)
        with self.assertRaises(ValueError):
            bad.main("export", self.output)
        self.assertNotIn("UNSAFE", calls)


if __name__ == "__main__":
    unittest.main()
