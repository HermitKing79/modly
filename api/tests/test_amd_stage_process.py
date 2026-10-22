"""CPU-only ownership contracts; every process/OS boundary is mocked."""
import io
import ctypes
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
from contextlib import redirect_stdout

try:
    from services import amd_stage_process as stage
except ImportError:
    stage = None
try:
    from services import amd_wsl_session as linux
except ImportError:
    linux = None
try:
    from services import amd_stage_bootstrap as bootstrap
except ImportError:
    bootstrap = None


class FakeJob:
    fail_attach = False
    instances = []

    def __init__(self):
        self.attached = False
        self.forced = False
        self.instances.append(self)

    def attach(self, proc):
        if self.fail_attach:
            raise OSError("attachment refused")
        self.attached = True

    def snapshot(self):
        return {"active": 0, "peak_job_bytes": 1234, "process_identities": []}

    def terminate(self):
        self.forced = True

    def close(self):
        pass


class FakeInput(io.BytesIO):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc

    def write(self, value):
        message = json.loads(value)
        if message["action"] == "GO":
            assert FakeJob.instances[-1].attached, "stage gate opened before Job attachment"
            self.proc.go = True
        elif message["action"] == "CANCEL" and not self.proc.go:
            self.proc.returncode = 0
        elif message["action"] == "START":
            self.proc.started = True
        return super().write(value)


class FakeProcess:
    linux_ack = True
    terminal = "succeeded"
    forced = False
    corrupt_nonce = False
    cancel_on_exit = None
    bootstrap_returncode = 0
    instances = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.spec = json.loads(Path(argv[-1]).read_text())
        self.pid = 456
        self.returncode = None
        self.go = False
        self.started = False
        self.ready = False
        self.stdin = FakeInput(self)
        self.stdout = None
        self.stderr = None
        self.polls = 0
        self.events = kwargs["stdout"]
        self.instances.append(self)

    def poll(self):
        self.polls += 1
        if self.go and self.returncode is None:
            identity = {"uid": 1000, "leader": {"pid": 777, "start_ticks": 234,
                                                   "session": 777, "pgrp": 777, "uid": 1000},
                        "broker": {"pid": 776, "start_ticks": 233, "uid": 1000},
                        "memory": {"rlimit_as_bytes": 6 * 1024**3,
                                   "aggregate_rss_limit_bytes": 6 * 1024**3,
                                   "aggregate_rss_peak_bytes": 1024}}
            if self.spec["kind"] == "wsl" and not self.ready:
                ready = {"event": "ready", "stage_id": self.spec["stage_id"],
                         "nonce": self.spec["nonce"], "context": self.spec["context"], **identity}
                self.events.write((json.dumps(ready) + "\n").encode())
                self.events.flush()
                self.ready = True
                return None
            record = {"event": "terminal", "stage_id": self.spec["stage_id"],
                      "nonce": self.spec["nonce"], "context": self.spec["context"],
                      "returncode": 0, "terminal": self.terminal,
                      "forced": self.forced, "natural": not self.forced}
            if self.corrupt_nonce:
                record["nonce"] = "foreign-nonce"
            if self.spec["kind"] == "wsl" and self.linux_ack:
                record.update(linux_settled=True, reaped_all=True, owned_remaining=[], **identity)
            self.events.write((json.dumps(record) + "\n").encode())
            self.events.flush()
            self.returncode = self.bootstrap_returncode
            if self.cancel_on_exit is not None:
                self.cancel_on_exit.set()
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = 1


class StageOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(stage, "owned stage backend has not been implemented")
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = {"request_id": "request-1", "nonce": "c" * 32,
                        "job_id": "job-1", "extension_id": "ext/node"}
        FakeJob.instances = []
        FakeJob.fail_attach = False
        FakeProcess.instances = []
        FakeProcess.linux_ack = True
        FakeProcess.terminal = "succeeded"
        FakeProcess.forced = False
        FakeProcess.corrupt_nonce = False
        FakeProcess.cancel_on_exit = None
        FakeProcess.bootstrap_returncode = 0
        self.patches = [patch.object(stage, "_OwnedJob", FakeJob),
                        patch.object(stage, "_require_windows", lambda: None),
                        patch.object(stage, "_host_health", lambda path: {
                            "available_ram_bytes": 13 * 1024**3, "free_disk_bytes": 3 * 1024**3}),
                        patch.object(stage.subprocess, "Popen", FakeProcess)]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def command(self, kind="windows"):
        return stage.StageCommand("prepare", kind, ("pinned-python", "fixture.py"),
                                  self.root, self.context, timeout_s=2)

    def test_windows_gate_follows_attachment_and_receipt_binds_context(self):
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), threading.Event())
        self.assertEqual(result["terminal"], "succeeded")
        self.assertTrue(result["windows_settled"])
        self.assertIsNone(result["linux_settled"])
        self.assertEqual(result["context"], self.context)
        self.assertEqual(FakeProcess.instances[0].argv[1:4], ["-B", "-I", "-S"])
        self.assertTrue(Path(result["receipt_file"]).is_file())

    def test_attachment_failure_never_opens_stage_gate(self):
        FakeJob.fail_attach = True
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), threading.Event())
        self.assertEqual(result["terminal"], "failed")
        self.assertFalse(FakeProcess.instances[0].go)

    def test_python_observer_permission_failure_has_private_phase_and_traceback(self):
        with patch.object(stage, '_host_health', side_effect=PermissionError('actual diagnostic fixture error')):
            result = stage.OwnedStageRunner(self.root / 'evidence').run(self.command(), threading.Event())
        detail = result['diagnostic_errors'][0]
        self.assertEqual(detail['stage_id'], 'prepare')
        self.assertEqual(detail['phase'], 'normal.observe')
        self.assertFalse(detail['during_shutdown'])
        self.assertIsNone(detail['native'])
        self.assertIsNone(detail['winerror'])
        self.assertIn('_host_health', detail['traceback'])
        self.assertTrue(detail['python_frames'])

    def test_wsl_client_exit_without_linux_ack_blocks(self):
        FakeProcess.linux_ack = False
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command("wsl"), threading.Event())
        self.assertEqual(result["terminal"], "blocked")
        self.assertFalse(result["linux_settled"])

    def test_complete_wsl_settlement_is_separate_from_windows_job(self):
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command("wsl"), threading.Event())
        self.assertEqual(result["terminal"], "succeeded")
        self.assertTrue(result["linux_settled"])
        self.assertEqual(result["linux"]["leader"]["start_ticks"], 234)
        self.assertFalse(result["limits"]["windows_job_caps_linux_memory"])

    def test_forced_exit_cannot_be_reported_natural_success(self):
        FakeProcess.forced = True
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), threading.Event())
        self.assertEqual(result["terminal"], "failed")
        self.assertTrue(result["forced"])
        self.assertFalse(result["natural"])

    def test_pre_cancelled_command_does_not_launch(self):
        cancel = threading.Event()
        cancel.set()
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), cancel)
        self.assertEqual(result["terminal"], "cancelled")
        self.assertEqual(FakeProcess.instances, [])
        self.assertTrue(result["natural"])
        self.assertTrue(result["no_stage_launched"])

    def test_observer_failure_closes_gate(self):
        def failed_tick():
            raise RuntimeError("observer failed")
        result = stage.OwnedStageRunner(self.root / "evidence").run(
            self.command(), threading.Event(), failed_tick)
        self.assertEqual(result["terminal"], "failed")
        self.assertFalse(FakeProcess.instances)

    def test_wrong_context_nonce_cannot_authenticate_terminal(self):
        FakeProcess.corrupt_nonce = True
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), threading.Event())
        self.assertEqual(result["terminal"], "failed")
        self.assertIn("nonce mismatch", result["error"]["message"])

    def test_unverified_linux_lease_blocks_reusing_the_runner(self):
        FakeProcess.linux_ack = False
        runner = stage.OwnedStageRunner(self.root / "evidence")
        result = runner.run(self.command("wsl"), threading.Event())
        self.assertEqual(result["terminal"], "blocked")
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            runner.run(self.command(), threading.Event())

    def test_cancel_racing_a_natural_exit_requires_cooperative_ack_for_clean(self):
        cancel = threading.Event()
        FakeProcess.cancel_on_exit = cancel
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), cancel)
        self.assertEqual(result["terminal"], "cancelled")
        self.assertFalse(result["natural"])

    def test_successful_terminal_cannot_hide_failed_bootstrap_transport(self):
        FakeProcess.bootstrap_returncode = 1
        result = stage.OwnedStageRunner(self.root / "evidence").run(self.command("wsl"), threading.Event())
        self.assertEqual(result["terminal"], "failed")
        self.assertEqual(result["bootstrap_returncode"], 1)

    def test_failing_resource_measurement_is_retained_in_blocked_receipt(self):
        measurement = {"available_ram_bytes": 11 * 1024**3, "free_disk_bytes": 1024**3}
        with patch.object(stage, "_host_health", side_effect=stage._HostResourceError(measurement)):
            result = stage.OwnedStageRunner(self.root / "evidence").run(self.command(), threading.Event())
        self.assertEqual(result["terminal"], "blocked")
        self.assertEqual(result["observations"][0]["available_ram_bytes"], measurement["available_ram_bytes"])
        self.assertIn(str(measurement["free_disk_bytes"]), result["error"]["message"])
        self.assertEqual(FakeProcess.instances, [])


class LinuxOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(linux, "Linux stage session ownership has not been implemented")

    def test_proc_stat_parser_preserves_start_identity_after_parenthesized_name(self):
        fields = ["S", "20", "30", "40"] + ["0"] * 15 + ["555", "0", "12"]
        row = linux._parse_stat("90 (name with ) spaces) " + " ".join(fields), 1000)
        self.assertEqual((row["pid"], row["ppid"], row["session"], row["start_ticks"], row["rss_pages"]),
                         (90, 20, 40, 555, 12))

    def test_subreaper_ancestry_includes_descendant_that_changed_session(self):
        rows = {1: {"pid": 1, "ppid": 0, "uid": 1000, "session": 1},
                2: {"pid": 2, "ppid": 1, "uid": 1000, "session": 2},
                3: {"pid": 3, "ppid": 2, "uid": 1000, "session": 3},
                4: {"pid": 4, "ppid": 99, "uid": 1000, "session": 4}}
        self.assertEqual(set(linux._owned_rows(rows, 1, 2)), {2, 3})

    def test_identity_change_prevents_any_signal(self):
        original = {"pid": 11, "start_ticks": 200, "uid": 1000}
        changed = {**original, "start_ticks": 201}
        with patch.object(linux, "_read_identity", return_value=changed), \
                patch.object(linux.os, "pidfd_open", create=True) as opened, \
                patch.object(linux.signal, "pidfd_send_signal", create=True) as signalled:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                linux._signal_exact(original, 15)
        opened.assert_not_called()
        signalled.assert_not_called()

    def test_stale_numeric_parent_link_cannot_authorize_a_foreign_child(self):
        broker = {"pid": 10, "start_ticks": 100, "uid": 1000, "ppid": 1}
        target = {"pid": 30, "start_ticks": 300, "uid": 1000, "ppid": 20}
        foreign_parent = {"pid": 20, "start_ticks": 200, "uid": 1000, "ppid": 99}
        foreign_root = {"pid": 99, "start_ticks": 50, "uid": 1000, "ppid": 1}
        rows = {10: broker, 30: target, 20: foreign_parent, 99: foreign_root}
        with patch.object(linux, "_read_identity", side_effect=lambda pid: rows[pid]), \
                patch.object(linux.os, "pidfd_open", return_value=100, create=True), \
                patch.object(linux.os, "close"), \
                patch.object(linux.select, "select", return_value=([], [], [])), \
                patch.object(linux.signal, "pidfd_send_signal", create=True) as signalled:
            with self.assertRaisesRegex(RuntimeError, "ancestry"):
                linux._signal_exact(target, 15, broker)
        signalled.assert_not_called()

    def test_no_linux_session_is_settled_from_client_returncode(self):
        self.assertFalse(stage._linux_settled({"returncode": 0}))

    def test_reused_session_number_does_not_authorize_foreign_process(self):
        rows = {1: {"pid": 1, "ppid": 0, "uid": 1000, "session": 1},
                2: {"pid": 2, "ppid": 99, "uid": 1000, "session": 2}}
        self.assertEqual(linux._owned_rows(rows, 1, 2), {})

    def test_linux_ready_requires_a_genuine_new_session_and_memory_limit(self):
        ready = {"uid": 1000, "broker": {"pid": 11, "start_ticks": 101, "uid": 1000},
                 "leader": {"pid": 12, "start_ticks": 102, "uid": 1000, "session": 99, "pgrp": 12},
                 "memory": {"rlimit_as_bytes": 6 * 1024**3}}
        self.assertFalse(stage._linux_ready(ready))
        ready["leader"]["session"] = 12
        self.assertTrue(stage._linux_ready(ready))


class BootstrapOwnershipTests(unittest.TestCase):
    def test_native_daemon_reader_uses_raw_stdin_without_shutdown_buffer_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as name:
            child = Mock(pid=123)
            child.returncode = 0
            child.stdin = io.BytesIO()
            child.poll.return_value = child.wait.return_value = 0
            raw = io.BytesIO()
            spec = {"stage_id": "prepare", "nonce": "c" * 32, "context": {},
                    "timeout_s": 180, "argv": ["never-launched"], "cwd": name,
                    "stdout_file": str(Path(name) / "stdout"), "stderr_file": str(Path(name) / "stderr")}
            with patch.object(bootstrap.subprocess, "Popen", return_value=child), \
                    patch.object(bootstrap.threading, "Thread") as thread, \
                    patch.object(bootstrap.sys, "stdin", Mock(buffer=Mock(raw=raw))), \
                    patch.object(bootstrap, "read_control", side_effect=ValueError("closed")) as read, \
                    redirect_stdout(io.StringIO()):
                bootstrap.native_stage(spec)
                thread.call_args.kwargs["target"]()
            read.assert_called_once_with(raw, spec)

    def test_native_stage_cancels_when_host_heartbeat_stalls(self):
        self.assertIsNotNone(bootstrap)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as name:
            child = Mock()
            child.pid = 123
            child.returncode = 0
            child.stdin = io.BytesIO()
            child.poll.side_effect = [None, None, 0]
            child.wait.return_value = 0
            spec = {"stage_id": "prepare", "nonce": "c" * 32, "context": {},
                    "timeout_s": 180, "argv": ["never-launched"], "cwd": name,
                    "stdout_file": str(Path(name) / "stdout"), "stderr_file": str(Path(name) / "stderr")}
            protocol = io.StringIO()
            with patch.object(bootstrap.subprocess, "Popen", return_value=child), \
                    patch.object(bootstrap.threading, "Thread"), \
                    patch.object(bootstrap.time, "monotonic", side_effect=[0, 16, 20]), \
                    redirect_stdout(protocol):
                bootstrap.native_stage(spec)
            child.kill.assert_called_once()
            terminal = json.loads(protocol.getvalue().splitlines()[-1])
            self.assertTrue(terminal["forced"])
            self.assertFalse(terminal["natural"])


class WindowsJobContractTests(unittest.TestCase):
    def test_resource_error_preserves_measured_values_and_required_floors(self):
        api = Mock()

        def memory(pointer):
            pointer._obj.available = 11 * 1024**3
            return 1

        api.GlobalMemoryStatusEx.side_effect = memory
        with patch.object(stage.ctypes, "WinDLL", return_value=api, create=True), \
                patch.object(stage.shutil, "disk_usage", return_value=Mock(free=1024**3)):
            with self.assertRaises(stage._HostResourceError) as failure:
                stage._host_health(Path.cwd())
        self.assertEqual(failure.exception.observation, {
            "available_ram_bytes": 11 * 1024**3, "free_disk_bytes": 1024**3})
        self.assertIn(str(12 * 1024**3), str(failure.exception))
        self.assertIn(str(2 * 1024**3), str(failure.exception))

    def test_private_job_enforces_both_memory_caps_without_breakaway(self):
        class Function:
            def __init__(self, action=lambda *args: 1):
                self.action = action

            def __call__(self, *args):
                return self.action(*args)

        api = Mock()
        captured = {}

        def set_limits(handle, category, pointer, size):
            captured["limits"] = stage._Limits.from_buffer_copy(ctypes.string_at(pointer, size))
            self.assertEqual(category, 9)
            return 1

        for name in ("CreateJobObjectW", "AssignProcessToJobObject", "QueryInformationJobObject",
                     "TerminateJobObject", "CloseHandle", "OpenProcess", "GetProcessTimes",
                     "QueryFullProcessImageNameW", "IsProcessInJob"):
            setattr(api, name, Function())
        api.SetInformationJobObject = Function(set_limits)
        with patch.object(stage.ctypes, "WinDLL", return_value=api, create=True), \
                patch.object(stage, "_require_windows", lambda: None):
            job = stage._OwnedJob()
            job.close()
        limits = captured["limits"]
        self.assertEqual(limits.ProcessMemory, 6 * 1024**3)
        self.assertEqual(limits.JobMemory, 6 * 1024**3)
        self.assertEqual(limits.Basic.Flags, 0x2000 | 0x100 | 0x200)


if __name__ == "__main__":
    unittest.main()
