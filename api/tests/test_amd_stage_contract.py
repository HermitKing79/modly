"""Small real-byte fixtures for the CPU stage artifact trust boundary."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from services.amd_completed_result import CompletionContext
try:
    from services import amd_stage_contract as contract
except ImportError:
    contract = None


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class StageContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(contract, "CPU stage artifact contract is not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.context = CompletionContext.new("job-stage", b"input", {}, model_id="amd-test")

    def fixture(self, tensors=None, *, context=None, header_change=None, suffix=b""):
        tensors = tensors or {"features": ("F32", [2, 1], struct.pack("<2f", 1, -2)),
                              "coordinates": ("I32", [2, 4], struct.pack("<8i", 0, 1, 2, 3, 0, 4, 5, 6))}
        header = {"__metadata__": {"product_context": json.dumps(context or self.context.to_dict())}}
        payload = b""
        specs = {}
        for name, (dtype, shape, data) in tensors.items():
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
            payload += data
            specs[name] = {"dtype": dtype, "shape": shape}
        if header_change:
            header_change(header)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        blob = struct.pack("<Q", len(encoded)) + encoded + payload + suffix
        path = self.root / "field.safetensors"
        path.write_bytes(blob)
        schema = {"tensors": specs, "max_rows": 2, "coordinate_tensor": "coordinates" if "coordinates" in specs else None}
        return contract.Artifact(path.name, sha(blob), len(blob), "safetensors", schema)

    def test_actual_tensor_bytes_and_coordinate_order_are_verified(self):
        artifact = self.fixture()
        coordinate_sha = sha(struct.pack("<8i", 0, 1, 2, 3, 0, 4, 5, 6))
        result = contract.validate_artifact(self.root, artifact, self.context, expected_coordinates_sha256=coordinate_sha)
        self.assertEqual(result["tensors"]["features"]["shape"], [2, 1])
        self.assertEqual(result["coordinates_sha256"], coordinate_sha)
        self.assertEqual(contract.Artifact.from_dict(artifact.to_dict()), artifact)
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context, expected_coordinates_sha256="0" * 64)

    def test_each_float_encoding_rejects_infinity_and_nan(self):
        for dtype, finite, bad in (("F32", struct.pack("<f", 1), struct.pack("<I", 0x7f800000)),
                                   ("F16", struct.pack("<H", 0x3c00), struct.pack("<H", 0xfc00)),
                                   ("BF16", struct.pack("<H", 0x3f80), struct.pack("<H", 0x7fc1))):
            with self.subTest(dtype=dtype):
                artifact = self.fixture({"features": (dtype, [1, 1], finite)})
                self.assertEqual(contract.validate_artifact(self.root, artifact, self.context)["tensors"]["features"]["dtype"], dtype)
                artifact = self.fixture({"features": (dtype, [1, 1], bad)})
                with self.assertRaises(ValueError):
                    contract.validate_artifact(self.root, artifact, self.context)

    def test_header_offsets_shape_and_extra_or_missing_bytes_fail(self):
        mutations = [lambda h: h["features"].update(data_offsets=[0, 1000]),
                     lambda h: h["coordinates"].update(data_offsets=[4, 36]),
                     lambda h: h["features"].update(shape=[3, 1]),
                     lambda h: h["features"].update(shape=[2, True]),
                     lambda h: h["features"].update(dtype="F64"),
                     lambda h: h.pop("coordinates")]
        for change in mutations:
            artifact = self.fixture(header_change=change)
            with self.assertRaises(ValueError):
                contract.validate_artifact(self.root, artifact, self.context)
        artifact = self.fixture(suffix=b"trailing")
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)

    def test_nonfinite_after_first_stream_chunk_is_rejected(self):
        data = struct.pack("<I", 0x3f800000) * (262144 + 1)
        artifact = self.fixture({"features": ("F32", [1, 262145], data)})
        self.assertEqual(contract.validate_artifact(self.root, artifact, self.context)["tensors"]["features"]["shape"], [1, 262145])
        artifact = self.fixture({"features": ("F32", [1, 262145], data[:-4] + struct.pack("<I", 0xff800000))})
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)

    def test_missing_pillow_reports_preprocessing_dependency_blocker(self):
        with patch.dict("sys.modules", {"PIL": None}):
            with self.assertRaisesRegex(RuntimeError, "Pillow.*blocked"):
                contract.admitted_image(b"bounded-image", preprocess={"image_mode": "alpha-only", "resolution": 512})
        artifact = self.fixture()
        path = self.root / artifact.relative_path
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)

    def test_host_row_bound_and_actual_coordinate_values_fail_closed(self):
        for values in ((0, 1, 2, 3, 0, 1, 2, 3), (1, 1, 2, 3, 0, 4, 5, 6), (0, -1, 2, 3, 0, 4, 5, 6), (0, 512, 2, 3, 0, 4, 5, 6)):
            artifact = self.fixture({"coordinates": ("I32", [2, 4], struct.pack("<8i", *values))})
            with self.assertRaises(ValueError):
                contract.validate_artifact(self.root, artifact, self.context)
        artifact = self.fixture()
        artifact.schema["max_rows"] = 1
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)

    def test_artifact_context_and_confined_paths_are_required(self):
        artifact = self.fixture(context=dict(self.context.to_dict(), nonce="foreign"))
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)
        value = artifact.to_dict()
        for name in ("../field.safetensors", "/field.safetensors", "a\\field.safetensors"):
            with self.assertRaises(ValueError):
                contract.Artifact.from_dict(dict(value, relative_path=name))
        with self.assertRaises(ValueError):
            contract.Artifact.from_dict(dict(value, injected=True))

    def test_bounded_json_reopens_hash_and_rejects_nonfinite_or_duplicate_keys(self):
        for blob, good in ((json.dumps({"product_context": self.context.to_dict(), "actual": 7}).encode(), True),
                           (b'{"x":1,"x":2}', False), (b'{"x":NaN}', False)):
            (self.root / "result.json").write_bytes(blob)
            artifact = contract.Artifact("result.json", sha(blob), len(blob), "json", {})
            if good:
                self.assertEqual(contract.validate_artifact(self.root, artifact, self.context)["document"]["actual"], 7)
            else:
                with self.assertRaises(ValueError):
                    contract.validate_artifact(self.root, artifact, self.context)

    def test_json_numeric_exponent_overflow_is_rejected(self):
        blob = ('{"product_context":' + json.dumps(self.context.to_dict()) + ',"nested":[{"value":1e999}]}').encode()
        (self.root / "result.json").write_bytes(blob)
        artifact = contract.Artifact("result.json", sha(blob), len(blob), "json", {})
        with self.assertRaises(ValueError):
            contract.validate_artifact(self.root, artifact, self.context)

    def test_actual_alpha_image_is_decoded_and_not_promoted_to_conditioning(self):
        from PIL import Image
        image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        image.paste((11, 22, 33, 255), (2, 2, 6, 6))
        out = io.BytesIO()
        image.save(out, format="PNG")
        blob = out.getvalue()
        result = contract.admitted_image(blob, preprocess={"image_mode": "alpha-only", "resolution": 512})
        self.assertEqual(result["input_sha256"], sha(blob))
        self.assertEqual(result["pixel_sha256"], sha(image.tobytes()))
        self.assertEqual(result["source_size"], [8, 8])
        self.assertEqual(result["foreground_bbox"], [2, 2, 5, 5])
        self.assertFalse(result["preprocessing_complete"])
        for request in ({"image_mode": "rembg", "resolution": 512}, {"image_mode": "alpha-only", "resolution": 1024}, {}):
            with self.assertRaises(ValueError):
                contract.admitted_image(blob, preprocess=request)
        for image in (Image.new("RGB", (8, 8)), Image.new("RGBA", (8, 8), (1, 2, 3, 255))):
            out = io.BytesIO()
            image.save(out, format="PNG")
            with self.assertRaises(ValueError):
                contract.admitted_image(out.getvalue(), preprocess={"image_mode": "alpha-only", "resolution": 512})


if __name__ == "__main__":
    unittest.main()
