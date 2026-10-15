"""Static/stdlib CPU checks; no WSL start, AI import, model or checkpoint load."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

E = Path(__file__).resolve().parent
ROOT = E.parents[2]


def load(name):
    path = E / (name + ".py")
    if not path.is_file():
        raise AssertionError("Texture candidate is missing: " + name)
    spec = importlib.util.spec_from_file_location("texture_candidate_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TexturePreflightTests(unittest.TestCase):
    def test_cold_launcher_and_bridge_are_stdlib_only(self):
        before = set(sys.modules)
        load("decode"); load("artifact_io")
        self.assertFalse({"torch", "triton", "flex_gemm", "numpy", "safetensors"} & (set(sys.modules) - before))

    def test_deadlines_and_owned_guest_command_preserve_limits(self):
        module = load("decode")
        command = module.worker_command("/mnt/c/proof/decode.py", "/mnt/c/proof/run/guest")
        self.assertIn("450s", command)
        self.assertNotIn("90s", command)
        self.assertEqual(command[command.index("-d") + 1], "Modly-AMD-Ubuntu-24.04")
        self.assertEqual(module.NATIVE_DEADLINE, 60)
        self.assertEqual(module.CPU_DEADLINE, 120)
        self.assertEqual(module.GPU_DEADLINE, 240)
        self.assertEqual(module.HOST_DEADLINE, 480)
        self.assertEqual(module.HOST_RAM, 12 * 1024**3)
        self.assertEqual(module.CPU_RAM, 6972719480)
        module.check_deadlines(479, 0, None)
        with self.assertRaises(TimeoutError): module.check_deadlines(480, 0, None)

    def test_texture_identity_and_no_shape_inputs_or_bf16_conversion(self):
        source = (E / "decode.py").read_text()
        tree = ast.parse(source)
        self.assertNotIn("shape_decoder_loader", source)
        self.assertNotIn("shape-latent-denormalized.npy", source)
        self.assertNotIn("shape-latent-provenance.json", source)
        self.assertNotIn("len(changes)!=216", source)
        self.assertIn("TextureDecoderComponent", source)
        self.assertIn("read_inputs", source)
        self.assertIn("validate_committed_candidate", source)
        self.assertNotIn("base.SCRIPT =", source)
        self.assertNotIn("base.SCRIPT=", source)
        module = load("decode")
        self.assertEqual(module.CHECKPOINT_SHA256, "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c")

    def test_worker_reads_input_metadata_from_the_actual_dict_api(self):
        tree = ast.parse((E / "decode.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "read_inputs"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(isinstance(target, ast.Name) for target in calls[0].targets))

    def test_runtime_version_is_plain_json_string_for_the_strict_writer(self):
        tree = ast.parse((E / "decode.py").read_text())
        runtime = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                               and target.slice.value == 'runtime' for target in node.targets))
        value = runtime.value.values[runtime.value.keys.index(next(key for key in runtime.value.keys if isinstance(key, ast.Constant) and key.value == 'torch'))]
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Name)
        self.assertEqual(value.func.id, 'str')

    def test_artifact_bridge_publishes_exclusively_and_rejects_wrong_ack(self):
        module = load("artifact_io")
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            path = Path(directory) / "receipt.json"
            module.durable_json(path, {"status": "PASS"})
            self.assertEqual(json.loads(path.read_text()), {"status": "PASS"})
            with self.assertRaises(FileExistsError): module.durable_json(path, {})
            self.assertFalse(path.with_name(path.name + ".publishing").exists())
        with self.assertRaises(RuntimeError):
            module.validate_ack({"status": "PASS", "artifact_sha256": "bad"}, {"artifact_sha256": "a" * 64, "artifact_bytes": 100})
        module.validate_ack({"status": "PASS", "artifact_sha256": "a" * 64, "artifact_bytes": 100}, {"artifact_sha256": "a" * 64, "artifact_bytes": 100})

    def test_shared_checkpoint_helper_uses_texture_audit_and_rejects_wrong_bytes(self):
        module = load("decode")
        audit = ROOT / 'docs/compatibility/amd-turret-texture-route-evidence/local-assets.json'
        original = json.loads(audit.read_text())
        checkpoint = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=948458812, st_mtime_ns=1791274770477513600),
                                     is_file=lambda: True)
        checkpoint.resolve = lambda strict=True: SimpleNamespace(as_posix=lambda: 'D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors')
        def digest(path):
            return ('9f764cece1ae4630ec437b2a45d8619ff894c3312190e86813350a1490c2d481'
                    if isinstance(path, Path) else module.CHECKPOINT_SHA256)
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            fake_root = Path(directory)
            target = fake_root / 'docs/compatibility/amd-turret-texture-route-evidence/local-assets.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(original))
            with patch.object(module, 'ROOT', fake_root), patch.object(module, 'Path', return_value=checkpoint), patch.object(module, 'digest', side_effect=digest):
                self.assertEqual(module.checkpoint_record(checkpoint)['freshly_streamed_sha256'], module.CHECKPOINT_SHA256)
                original['model_checkpoints']['tex_dec_next_dc_f16c32_fp16']['bytes'] = 948490494
                target.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, 'record differs: bytes'):
                    module.checkpoint_record(checkpoint)

    def test_host_receipt_and_worker_wait_are_bounded_by_30_seconds(self):
        module = load("artifact_io")
        self.assertEqual(module.IO_DEADLINE, 30)
        source = (E / "decode.py").read_text()
        self.assertIn("independent_artifact_verification_after_exit", source)
        self.assertIn("verify_on_host", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
