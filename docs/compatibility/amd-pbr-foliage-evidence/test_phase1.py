"""Focused CPU-only admission, transport and containment tests; no Blender run."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))
import phase1_worker as worker
import phase1_host as host
import numpy as np


class AdmissionTests(unittest.TestCase):
    def test_exact_binary_gate_rejects_variations_and_eof(self):
        for token in (b"", b"GO", b"GO\r\n", b"GO_START\n", b"go\n", b"GO \n"):
            with self.subTest(token=token), patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(token))), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "exact binary token"):
                    worker.await_host()
        with patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"GO\n"))), patch("builtins.print"):
            worker.await_host()

    def test_only_exact_empty_live_temp_child_is_admitted(self):
        with tempfile.TemporaryDirectory(dir=worker.P) as directory:
            out = Path(directory) / "brick"
            out.mkdir()
            scratch = out / "scratch"
            scratch.mkdir()
            (out / "stdout.txt").touch()
            (out / "stderr.txt").touch()
            live = scratch / "blender_session"
            live.mkdir()
            with patch.object(worker, "OUT", out):
                result = worker.initial_output(out, str(live))
                self.assertEqual(result["live_session_directory"], str(live))
                with self.assertRaisesRegex(ValueError, "exact immediate"):
                    worker.initial_output(out, str(scratch / "foreign"))
                (live / "foreign.txt").touch()
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    worker.initial_output(out, str(live))

    def test_prior_outputs_are_never_reused(self):
        with tempfile.TemporaryDirectory(dir=worker.P) as directory:
            out = Path(directory)
            (out / "result.json").write_text("{}")
            with patch.object(worker, "OUT", out), self.assertRaisesRegex(ValueError, "No prior outputs"):
                worker.initial_output(out, None)

    def test_no_Blender_module_was_imported_by_tested_modules(self):
        self.assertNotIn("bpy", sys.modules)


class ContainmentTests(unittest.TestCase):
    def test_exact_GO_is_sent_only_after_successful_attach(self):
        calls = []
        stream = SimpleNamespace(write=lambda data: calls.append(("write", data)),
                                 flush=lambda: calls.append(("flush",)), close=lambda: calls.append(("close",)))
        child = SimpleNamespace(stdin=stream)
        job = SimpleNamespace(attach=lambda target: calls.append(("attach", target)))
        record = {}
        host.attach_and_go(job, child, record)
        self.assertEqual(calls, [("attach", child), ("write", b"GO\n"), ("flush",), ("close",)])
        self.assertTrue(record["attached_before_GO"])
        self.assertTrue(record["exact_binary_GO_sent"])

    def test_attach_failure_sends_no_GO(self):
        stream = io.BytesIO()
        def fail(child):
            raise OSError("attach rejected")
        with self.assertRaises(OSError):
            host.attach_and_go(SimpleNamespace(attach=fail), SimpleNamespace(stdin=stream), {})
        self.assertEqual(stream.getvalue(), b"")

    def test_success_requires_natural_cleanup_and_verified_receipt(self):
        success = dict(returncode=0, exact_binary_GO_sent=True, attached_before_GO=True, sources_unchanged=True,
                       worker_checkpoint_verified=True, worker_execution_seconds=21,
                       peak_process_committed_bytes=200_000_000, peak_job_committed_bytes=200_000_000,
                       job_cleanup=dict(owned_tree_exit_verified=True, forced_job_cleanup=False, active_after_cleanup=0))
        self.assertTrue(host.passed(success))
        for key, value in (("worker_checkpoint_verified", False), ("sources_unchanged", False),
                           ("returncode", 1), ("worker_execution_seconds", 180.001),
                           ("peak_job_committed_bytes", host.CAP + 1), ("error", "failed")):
            with self.subTest(key=key):
                self.assertFalse(host.passed({**success, key: value}))
        forced = copy.deepcopy(success)
        forced["job_cleanup"]["forced_job_cleanup"] = True
        self.assertFalse(host.passed(forced))


class GeometryAndRigTests(unittest.TestCase):
    def test_oriented_bits_admit_only_cyclic_rotation_and_face_order(self):
        vertices = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [1, 2, 3]], dtype="<f4")
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype="<i4")
        expected = worker.canonical_bits(vertices, faces)
        self.assertEqual(expected, worker.canonical_bits(vertices, np.array([[3, 2, 1], [2, 0, 1]], dtype="<i4")))
        self.assertNotEqual(expected, worker.canonical_bits(vertices, faces[:, ::-1]))
        changed = vertices.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(1))
        self.assertNotEqual(expected, worker.canonical_bits(changed, faces))

    def test_same_lighting_and_only_fourfold_ortho_scale_change(self):
        settings = dict(device="CPU", engine="CYCLES", samples=32, resolution=[640, 640, 100],
                        denoising_use_gpu=False, lights=[dict(energy=10)], camera=dict(ortho_scale=.4, location=[1, 0, 0]))
        near, far = copy.deepcopy(settings), copy.deepcopy(settings)
        far["camera"]["ortho_scale"] *= 4
        views = [dict(name="near", settings=near), dict(name="far", settings=far)]
        worker.validate_two_views(views)
        far["lights"][0]["energy"] = 20
        with self.assertRaisesRegex(ValueError, "Lighting"):
            worker.validate_two_views(views)
        far["lights"][0]["energy"] = 10
        far["camera"]["ortho_scale"] = near["camera"]["ortho_scale"]
        with self.assertRaisesRegex(ValueError, "fourfold"):
            worker.validate_two_views(views)


class AuthenticContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = worker.load_legacy()
        cls.vertices, cls.faces, cls.uv = [np.load(worker.BAKE / name, allow_pickle=False)
                                          for name in ("vertices.npy", "faces.npy", "uv.npy")]
        cls.reference = cls.legacy.canonical_triangles(cls.vertices, cls.faces, cls.uv)
        cls.maps = {name: cls.legacy.png_pixels((worker.BAKE / name).read_bytes())
                    for name in ("base-color.png", "MR.png")}

    def test_authentic_receipt_pins_and_original_corner_arrays(self):
        bake, result = worker.verify_source_pins(self.legacy)
        self.assertEqual(bake["triangle_count"], 781)
        self.assertEqual(result["GLB"]["sha256"], worker.GLB_SHA)
        self.assertEqual(self.vertices.shape, (2343, 3))
        self.assertEqual(self.uv.dtype, np.float64)
        self.assertLessEqual(np.max(np.abs(self.uv - self.uv.astype("<f4"))), 1.5e-8)

    def test_real_container_geometry_embedded_PNG_values_and_sampler(self):
        result = worker.container_metadata(self.legacy, self.reference, self.maps)
        self.assertEqual(result["sha256"], worker.GLB_SHA)
        self.assertEqual(result["geometry"]["triangles"], 781)
        self.assertEqual(result["glTF_sampler_metadata"], worker.SAMPLER)
        for value in result["maps"].values():
            self.assertTrue(value["all_decoded_channels_exact"])
            self.assertTrue(value["full_encoded_PNG_bytes_exact"])
            self.assertNotIn("CPU_filter_metric", value)

    def test_changed_sampler_minification_is_rejected(self):
        payload = worker.GLB.read_bytes()
        length = struct.unpack_from("<I", payload, 12)[0]
        document = json.loads(payload[20:20 + length])
        document["samplers"][0]["minFilter"] = 9729
        encoded = json.dumps(document, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        binary = payload[28 + length:]
        changed = (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
                   + struct.pack("<II", len(encoded), 0x4e4f534a) + encoded
                   + struct.pack("<II", len(binary), 0x004e4942) + binary)
        with tempfile.TemporaryDirectory(dir=worker.P) as directory:
            path = Path(directory) / "changed.glb"
            path.write_bytes(changed)
            with patch.object(worker, "GLB", path), self.assertRaisesRegex(ValueError, "sampler metadata"):
                worker.container_metadata(self.legacy, self.reference, self.maps)

    def test_encoded_alpha_identity_is_checked_even_with_opaque_shader(self):
        altered = {name: pixels.copy() for name, pixels in self.maps.items()}
        altered["base-color.png"][0, 0, 3] ^= 1
        with self.assertRaisesRegex(ValueError, "Full embedded PNG channel values"):
            worker.container_metadata(self.legacy, self.reference, altered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
