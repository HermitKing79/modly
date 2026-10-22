import io
import json
import platform
import queue
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

from services.extension_process import ExtensionProcess, _venv_python


def _make_proc() -> ExtensionProcess:
    return ExtensionProcess(ext_dir=None, manifest={"id": "demo"})  # type: ignore[arg-type]


class ExtensionProcessTests(unittest.TestCase):
    def test_generation_envelope_pins_worker_model_id(self) -> None:
        proc = _make_proc()
        sent = []
        proc._send = sent.append
        proc._receive_generation = lambda *args: Path("result.glb")
        proc._generate_request({"image_b64": ""}, {}, None, None)
        self.assertEqual(sent[0]["model_id"], "demo")

    def test_generate_artifact_sends_typed_scene_without_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            scene = workspace / "Workflows" / "room"
            scene.mkdir(parents=True)
            manifest = scene / "scene-manifest.json"
            manifest.write_text(json.dumps({
                "schema": "modly.scene-manifest.v1", "sceneRoot": ".", "assets": [],
            }))
            proc = _make_proc()
            calls = []
            proc._generate_request = lambda payload, params, progress, cancel: calls.append((payload, params)) or manifest
            with patch("services.generator_registry.WORKSPACE_DIR", workspace):
                result = proc.generate_artifact("scene", manifest, {"quality": "high"})
            self.assertEqual(result, manifest)
            self.assertEqual(calls, [({"input": {"kind": "scene", "path": str(manifest.resolve())}}, {"quality": "high"})])

    def test_read_loop_writes_sentinel_to_own_queue_only(self) -> None:
        proc = _make_proc()

        old_queue: queue.Queue = queue.Queue()
        new_queue: queue.Queue = queue.Queue()
        proc._queue = new_queue

        fake_proc = type("FakeProc", (), {"stdout": io.StringIO("")})()

        proc._read_loop(fake_proc, old_queue)

        self.assertFalse(old_queue.empty())
        self.assertTrue(new_queue.empty())

    def test_stop_kills_and_verifies_subprocess_exit(self) -> None:
        proc = _make_proc()

        class FakeProcess:
            def __init__(self) -> None:
                self.alive = True
                self.kill_called = False
                self.wait_called = False

            def poll(self):
                return None if self.alive else -9

            def kill(self) -> None:
                self.kill_called = True
                self.alive = False

            def wait(self, timeout: float):
                self.wait_called = True
                return -9

        child = FakeProcess()
        proc._proc = child  # type: ignore[assignment]
        proc._loaded = True

        proc.stop()

        self.assertTrue(child.kill_called)
        self.assertTrue(child.wait_called)
        self.assertIsNone(proc._proc)
        self.assertFalse(proc._loaded)

    def test_stop_failure_keeps_live_process_reference_and_raises(self) -> None:
        proc = _make_proc()

        class StuckProcess:
            def poll(self):
                return None

            def kill(self) -> None:
                raise PermissionError("cannot kill")

            def wait(self, timeout: float):
                raise AssertionError("wait must not run after kill failure")

        child = StuckProcess()
        proc._proc = child  # type: ignore[assignment]
        proc._loaded = True

        with self.assertRaisesRegex(RuntimeError, "Could not stop"):
            proc.stop()

        self.assertIs(proc._proc, child)
        self.assertFalse(proc._loaded)


class CompletedResultProtocolTests(unittest.TestCase):
    def _proc(self):
        from services.amd_completed_result import CONTRACT
        return ExtensionProcess(Path("unused"), {
            "id": "demo/generate", "completed_output_contract": CONTRACT,
        })

    def test_completion_requires_manifest_and_explicit_host_verifier(self):
        proc = _make_proc()
        self.assertFalse(getattr(proc, "completed_results_enabled", True))
        with self.assertRaises(ValueError):
            proc.configure_completion_verifier(lambda context, candidate: None)

    def test_manifest_alone_cannot_enable_completed_generation(self):
        proc = self._proc()
        self.assertFalse(proc.completed_results_enabled)
        with self.assertRaisesRegex(ValueError, "verifier"):
            proc.generate_completed(b"image", {}, "job-1")

    def test_correlated_completion_uses_host_authority_and_no_legacy_path(self):
        from services.amd_completed_result import CompletedWorkerOutput
        proc = self._proc()
        candidate = CompletedWorkerOutput(str(Path("receipt.json").resolve()), "a" * 64)
        sent, progress, verified = [], [], []
        authority, authenticated = object(), object()

        def verifier(context, output):
            verified.append((context, output))
            return authority

        def send(message):
            sent.append(message)
            rid = message["id"]
            for kind in ("progress", "done", "error", "cancelled"):
                proc._queue.put({"type": kind, "id": "stale", "pct": 99, "output_path": "wrong.glb"})
            proc._queue.put({"type": "progress", "id": rid, "pct": 42, "step": "Verifying"})
            proc._queue.put({"type": "done", "id": rid, "completion": candidate.to_dict()})

        proc._send = send
        proc.configure_completion_verifier(verifier)
        with patch("services.amd_completed_result.authenticate_completion", return_value=authenticated) as authenticate:
            result = proc.generate_completed(b"image", {"quality": "high"}, "job-1", lambda *v: progress.append(v))
        self.assertIs(result, authenticated)
        self.assertEqual(progress, [(42, "Verifying")])
        self.assertEqual(len(verified), 1)
        context, output = verified[0]
        self.assertEqual(output, candidate)
        self.assertEqual(context.job_id, "job-1")
        self.assertEqual(context.model_id, "demo/generate")
        self.assertEqual(sent[0]["id"], context.request_id)
        self.assertEqual(sent[0]["completed_result_context"], context.to_dict())
        self.assertEqual(sent[0]["params"], {"quality": "high"})
        authenticate.assert_called_once_with(context, candidate, authority, None)

    def test_matching_done_without_descriptor_cannot_fall_back_to_path(self):
        proc = self._proc()
        proc.configure_completion_verifier(lambda context, candidate: self.fail("malformed output reached authority"))
        proc._send = lambda message: proc._queue.put({
            "type": "done", "id": message["id"], "output_path": "legacy.glb",
        })
        with self.assertRaises(ValueError):
            proc.generate_completed(b"image", {}, "job-1")

    def test_replay_mode_requires_explicit_experimental_manifest(self):
        proc = self._proc()
        with self.assertRaises(ValueError):
            proc.configure_completion_verifier(lambda *args: None, mode="replay_fixture")
        proc.manifest["completed_result_mode"] = "replay_fixture"
        with self.assertRaises(ValueError):
            proc.configure_completion_verifier(lambda *args: None, mode="replay_fixture")
        proc.manifest["experimental"] = True
        proc.configure_completion_verifier(lambda *args: None, mode="replay_fixture")
        self.assertTrue(proc.completed_results_enabled)

    def test_cancel_after_dequeue_wins_over_completed_result(self):
        import threading
        from services.amd_completed_result import CompletedWorkerOutput
        from services.generators.base import GenerationCancelled
        proc = self._proc()
        cancel = threading.Event()
        candidate = CompletedWorkerOutput(str(Path("receipt.json").resolve()), "a" * 64)

        class CancellingQueue(queue.Queue):
            def get(self, *args, **kwargs):
                value = super().get(*args, **kwargs)
                cancel.set()
                return value

        proc._queue = CancellingQueue()
        proc._send = lambda message: proc._queue.put({
            "type": "done", "id": message["id"], "completion": candidate.to_dict(),
        })
        proc.configure_completion_verifier(lambda context, output: self.fail("cancelled output reached authority"))
        with self.assertRaises(GenerationCancelled):
            proc.generate_completed(b"image", {}, "job-1", cancel_event=cancel)

    def test_cancelled_completed_generation_stops_and_reaps_its_child(self):
        import threading
        from services.generators.base import GenerationCancelled
        proc = self._proc()
        cancel = threading.Event()

        class Child:
            alive = True
            reaped = False
            def poll(self): return None if self.alive else -9
            def kill(self): self.alive = False
            def wait(self, timeout): self.reaped = True; return -9

        class CancellingQueue(queue.Queue):
            def get(self, *args, **kwargs):
                value = super().get(*args, **kwargs)
                cancel.set()
                return value

        child = Child()
        proc._proc = child
        proc._loaded = True
        proc._queue = CancellingQueue()
        proc._send = lambda message: proc._queue.put({"type": "progress", "id": message["id"], "pct": 10})
        proc.configure_completion_verifier(lambda *args: self.fail("cancelled result reached verifier"))
        with self.assertRaises(GenerationCancelled):
            proc.generate_completed(b"image", {}, "job-1", cancel_event=cancel)
        self.assertFalse(child.alive)
        self.assertTrue(child.reaped)
        self.assertIsNone(proc._proc)
        self.assertFalse(proc._loaded)

    def test_cancel_during_authentication_wins_over_completed_result(self):
        import threading
        from services.amd_completed_result import CompletedWorkerOutput
        from services.generators.base import GenerationCancelled
        proc = self._proc()
        cancel = threading.Event()
        candidate = CompletedWorkerOutput(str(Path("receipt.json").resolve()), "a" * 64)
        proc._send = lambda message: proc._queue.put({
            "type": "done", "id": message["id"], "completion": candidate.to_dict(),
        })
        proc.configure_completion_verifier(lambda *args: object())

        def authenticate(*args):
            cancel.set()
            return object()

        with patch("services.amd_completed_result.authenticate_completion", side_effect=authenticate):
            with self.assertRaises(GenerationCancelled):
                proc.generate_completed(b"image", {}, "job-1", cancel_event=cancel)

    def test_authentication_cancellation_uses_generator_cancel_protocol(self):
        import threading
        from services.amd_completed_result import CompletedWorkerOutput
        from services.generators.base import GenerationCancelled
        proc = self._proc()
        cancel = threading.Event()
        candidate = CompletedWorkerOutput(str(Path("receipt.json").resolve()), "a" * 64)
        proc._send = lambda message: proc._queue.put({
            "type": "done", "id": message["id"], "completion": candidate.to_dict(),
        })
        proc.configure_completion_verifier(lambda *args: object())

        def authenticate(*args):
            cancel.set()
            raise InterruptedError("cancelled during receipt verification")

        with patch("services.amd_completed_result.authenticate_completion", side_effect=authenticate):
            with self.assertRaises(GenerationCancelled):
                proc.generate_completed(b"image", {}, "job-1", cancel_event=cancel)

    def test_legacy_generation_still_returns_path(self):
        proc = _make_proc()
        proc._queue.put({"type": "done", "output_path": "legacy.glb"})
        self.assertEqual(proc._receive_generation("legacy", None, None), Path("legacy.glb"))


class VenvPythonTests(unittest.TestCase):
    def test_resolves_interpreter_path_for_current_platform(self) -> None:
        result = _venv_python(Path("/tmp/ext"))
        if platform.system() == "Windows":
            self.assertEqual(result, Path("/tmp/ext") / "venv" / "Scripts" / "python.exe")
        else:
            self.assertEqual(result, Path("/tmp/ext") / "venv" / "bin" / "python")


class BuildEnvTests(unittest.TestCase):
    def test_forces_utf8_stdio_on_worker(self) -> None:
        proc = _make_proc()
        env = proc._build_env()
        self.assertEqual(env.get("PYTHONUTF8"), "1")

    def test_sets_worker_model_dir_when_known(self) -> None:
        proc = _make_proc()
        proc.model_dir = Path("/tmp/models/ext/node")
        env = proc._build_env()
        self.assertEqual(env.get("MODEL_DIR"), str(Path("/tmp/models/ext/node")))

    def test_sets_explicit_node_identity_and_shared_weight_dirs(self) -> None:
        proc = ExtensionProcess(
            ext_dir=Path("/tmp/extensions/ext"),
            manifest={"id": "ext/quality", "node_id": "quality"},
        )
        proc.model_dir = Path("/tmp/models/ext/quality")
        proc.shared_model_dirs = {"base": Path("/tmp/models/ext/_shared/base")}

        env = proc._build_env()

        self.assertEqual(env["MODEL_ID"], "ext/quality")
        self.assertEqual(env["MODEL_NODE_ID"], "quality")
        self.assertEqual(
            json.loads(env["SHARED_MODEL_DIRS"]),
            {"base": str(Path("/tmp/models/ext/_shared/base"))},
        )


class MissingModuleExtractionTests(unittest.TestCase):
    def test_extracts_module_name_from_message(self) -> None:
        proc = _make_proc()
        name = proc._extract_missing_module({"message": "No module named 'PIL'"})
        self.assertEqual(name, "PIL")

    def test_extracts_module_name_from_traceback(self) -> None:
        proc = _make_proc()
        name = proc._extract_missing_module(
            {"message": "boom", "traceback": "...\nModuleNotFoundError: No module named \"numpy\"\n"}
        )
        self.assertEqual(name, "numpy")

    def test_returns_none_when_no_missing_module(self) -> None:
        proc = _make_proc()
        self.assertIsNone(proc._extract_missing_module({"message": "some other error"}))


class AutoRepairPackageTests(unittest.TestCase):
    """Safety: only known modules map to a package; never guess arbitrary names."""

    def test_maps_known_module_to_package(self) -> None:
        proc = _make_proc()
        self.assertEqual(proc._resolve_auto_repair_package("PIL"), "Pillow")

    def test_maps_known_module_via_root_package(self) -> None:
        proc = _make_proc()
        self.assertEqual(proc._resolve_auto_repair_package("PIL.Image"), "Pillow")

    def test_returns_none_for_unknown_module(self) -> None:
        proc = _make_proc()
        self.assertIsNone(proc._resolve_auto_repair_package("totally_unknown_pkg"))


class StderrDecodingTests(unittest.TestCase):
    """tqdm draws partial blocks with U+258D/U+258F, whose UTF-8 bytes are
    undefined in cp1252. Decoding the child's pipes with the locale codec killed
    _stderr_loop mid-run; nothing then drained stderr and the child blocked
    forever on write once the pipe filled (a 3D generation froze at 80%)."""

    def test_tqdm_partial_blocks_survive_the_loop(self) -> None:
        proc = _make_proc()
        bar = "Volume Decoding:   1%|█▍▏| 123/13827\r"
        fake_proc = type("FakeProc", (), {"stderr": io.StringIO(bar)})()

        proc._stderr_loop(fake_proc)  # must not raise

    def test_child_is_told_to_write_utf8(self) -> None:
        proc = _make_proc()
        self.assertEqual(proc._build_env().get("PYTHONIOENCODING"), "utf-8")


class RecvTests(unittest.TestCase):
    def test_returns_message_from_queue(self) -> None:
        proc = _make_proc()
        proc._queue.put({"type": "ready"})
        self.assertEqual(proc._recv(timeout=1.0), {"type": "ready"})

    def test_none_sentinel_raises_runtime_error(self) -> None:
        proc = _make_proc()
        proc._queue.put(None)
        with self.assertRaises(RuntimeError):
            proc._recv(timeout=1.0)

    def test_empty_queue_raises_timeout_error(self) -> None:
        proc = _make_proc()
        with self.assertRaises(TimeoutError):
            proc._recv(timeout=0.05)


class GenerateErrorLoadedFlagTests(unittest.TestCase):
    """
    Issue #239: a generation that fails during lazy texture setup leaves the
    worker without a model. If _loaded stays True, GeneratorRegistry.get_active()
    skips load() forever and every later run reuses the broken worker.
    """

    def _failing_generate(self, error_msg: dict) -> ExtensionProcess:
        proc = _make_proc()
        proc._loaded = True
        proc._send = lambda msg: None  # type: ignore[assignment]
        proc._queue.put(error_msg)
        with self.assertRaises(RuntimeError):
            proc.generate(b"", {})
        return proc

    def test_clears_loaded_when_worker_reports_model_lost(self) -> None:
        proc = self._failing_generate(
            {"type": "error", "message": "No module named 'xatlas'", "loaded": False}
        )
        self.assertFalse(proc._loaded)

    def test_keeps_loaded_when_worker_still_has_its_model(self) -> None:
        proc = self._failing_generate(
            {"type": "error", "message": "bad input image", "loaded": True}
        )
        self.assertTrue(proc._loaded)

    def test_keeps_loaded_when_worker_reports_no_state(self) -> None:
        proc = self._failing_generate({"type": "error", "message": "boom"})
        self.assertTrue(proc._loaded)

    def test_error_still_propagates_the_original_cause(self) -> None:
        proc = _make_proc()
        proc._loaded = True
        proc._send = lambda msg: None  # type: ignore[assignment]
        proc._queue.put(
            {"type": "error", "message": "short", "traceback": "full traceback here",
             "loaded": False}
        )
        with self.assertRaises(RuntimeError) as ctx:
            proc.generate(b"", {})
        self.assertIn("full traceback here", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
