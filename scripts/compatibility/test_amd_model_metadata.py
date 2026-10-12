import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("model_metadata", Path(__file__).with_name("amd_model_metadata.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(path, header, payload=b"\0\0\0\0"):
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class MetadataTests(unittest.TestCase):
    def test_cli_is_importable_in_isolated_python_without_ai_packages(self):
        run = subprocess.run([sys.executable, "-B", "-I",
                              str(Path(__file__).with_name("amd_offline_model_audit.py")), "--help"],
                             capture_output=True, timeout=5)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_tensor_metadata_is_read_without_tensor_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            fixture(path, {"weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}})
            original_open = Path.open
            reads = []
            limit = path.stat().st_size - 4
            class HeaderOnlyReader:
                def __init__(self, stream):
                    self.stream = stream
                def __enter__(self):
                    return self
                def __exit__(self, *unused):
                    self.stream.close()
                def read(self, size=-1):
                    if size < 0 or self.stream.tell() + size > limit:
                        raise AssertionError("Tensor payload read attempted")
                    reads.append(size)
                    return self.stream.read(size)
            def guarded_open(selected, *args, **kwargs):
                return HeaderOnlyReader(original_open(selected, *args, **kwargs))
            with patch.object(Path, "open", guarded_open):
                result = audit.safetensors_metadata(path)
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["parameter_count"], 2)
            self.assertEqual(result["tensor_data_bytes"], 4)
            self.assertEqual(result["bytes_read"], path.stat().st_size - 4)
            self.assertFalse(result["metadata_only_fixture"])
            self.assertEqual(reads, [8, limit - 8])

    def test_empty_scaffold_fixture_cannot_pass_real_model_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.safetensors"
            fixture(path, {"__metadata__": {"fixture_only": "true"}}, b"")
            result = audit.safetensors_metadata(path)
            self.assertTrue(result["metadata_only_fixture"])
            with self.assertRaisesRegex(ValueError, "Gate 1"):
                audit.require_gate_one({"essential_assets_verified": False})

    def test_invalid_layout_dtype_shape_and_truncation_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.safetensors"
            for tensor in (
                {"dtype": "F16", "shape": [3], "data_offsets": [0, 4]},
                {"dtype": "unknown", "shape": [2], "data_offsets": [0, 4]},
                {"dtype": "F16", "shape": [-2], "data_offsets": [0, 4]},
                {"dtype": "F16", "shape": [2], "data_offsets": [1, 5]},
            ):
                with self.subTest(tensor=tensor):
                    fixture(path, {"weight": tensor})
                    with self.assertRaises(ValueError):
                        audit.safetensors_metadata(path)
            fixture(path, {"weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}, b"\0")
            with self.assertRaises(ValueError):
                audit.safetensors_metadata(path)

    def test_duplicate_keys_overlaps_and_header_bombs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", 2**40))
            with self.assertRaises(ValueError):
                audit.safetensors_metadata(path)
            raw = b'{"w":{},"w":{}}  '
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit.safetensors_metadata(path)
            tensor = {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
            fixture(path, {"a": tensor, "b": tensor})
            with self.assertRaises(ValueError):
                audit.safetensors_metadata(path)

    def test_flow_shape_mismatch_is_not_a_compatible_config(self):
        config = {"name": "SLatFlowModel", "args": {"model_channels": 4, "in_channels": 2, "out_channels": 3}}
        tensors = {"input_layer.weight": {"shape": [4, 2]}, "out_layer.weight": {"shape": [3, 4]}}
        self.assertEqual(audit.flow_shape_checks(config, tensors), {"input_layer.weight": [4, 2], "out_layer.weight": [3, 4]})
        tensors["input_layer.weight"]["shape"] = [4, 5]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            audit.flow_shape_checks(config, tensors)

    def test_remote_or_escape_references_never_resolve_to_offline_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value in ("https://example/model", "../outside", r"\\server\share\weights"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    audit.local_reference(root, value)


if __name__ == "__main__":
    unittest.main()
