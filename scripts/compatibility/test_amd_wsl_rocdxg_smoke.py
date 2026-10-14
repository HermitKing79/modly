"""Focused CPU-only contracts; never imports Torch or launches WSL."""
from concurrent.futures import Future
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("amd_wsl_rocdxg_smoke.py")


def healthy():
    return {"returncode": 0, "timed_out": False,
            "observation": {"observer": {"system_log_probe": {"RecordId": 1}},
                            "controllers": [{"Name": "AMD Radeon RX 9070 XT", "Status": "OK"}],
                            "events": []}}


def runtime_identity():
    return {"torch": "2.9.1+rocm7.2.1.gitff65f5bc",
            "distribution_version": "2.9.1+rocm7.2.1.lw.gitff65f5bc",
            "git_version": "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf",
            "hip": "7.2.53211-e1a6bc5663", "cuda": None, "python": "3.12.3",
            "prefix": "/opt/modly-wsl/venv", "gpu_initialized_before": False}


class SmokeContracts(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), "Missing narrow WSL smoke-test implementation")
        spec = importlib.util.spec_from_file_location("wsl_smoke", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_fixture_has_256_varied_exact_dyadic_sums(self):
        smoke = self.load()
        a, b, expected = smoke.fixture()
        self.assertEqual((len(a), len(b), len(expected)), (256, 256, 256))
        self.assertEqual(a[:4], [-8.0, -7.5, -7.0, -6.5])
        self.assertEqual(b[:4], [-2.0, -1.125, -0.25, 0.625])
        self.assertEqual(expected[:4], [-10.0, -8.625, -7.25, -5.875])
        self.assertEqual((a[255], b[255], expected[255]), (4.0, -2.0, 2.0))
        self.assertGreater(len(set(expected)), 100)
        # Integers scaled by 1/8 independently check every reference entry.
        self.assertEqual([int(v * 8) for v in expected],
                         [4 * (i % 33 - 16) + ((i * 7) % 35 - 16) for i in range(256)])

    def test_cpu_output_contract_rejects_wrong_shape_dtype_nonfinite_or_values(self):
        smoke = self.load()
        expected = [(4 * (i % 33 - 16) + ((i * 7) % 35 - 16)) / 8 for i in range(256)]
        smoke.validate_output(expected, [256], "float32")
        wrong = expected.copy()
        wrong[31] += 0.125
        nan = expected.copy()
        nan[0] = float("nan")
        inf = expected.copy()
        inf[0] = float("inf")
        for values, shape, dtype in ((expected[:-1], [256], "float32"),
                                     (expected, [16, 16], "float32"),
                                     (expected, [256], "float64"),
                                     (wrong, [256], "float32"),
                                     (nan, [256], "float32"), (inf, [256], "float32")):
            with self.subTest(shape=shape, dtype=dtype):
                with self.assertRaises(smoke.NumericalError):
                    smoke.validate_output(values, shape, dtype)

    def test_attempt_marker_is_exclusive_and_survives_output_refusal(self):
        smoke = self.load()
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[2]) as temporary:
            base = Path(temporary)
            marker, output = base / "attempt.marker", base / "one"
            smoke.reserve_attempt(output, marker)
            self.assertTrue(marker.is_file())
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                smoke.reserve_attempt(base / "two", marker)
            self.assertFalse((base / "two").exists())
            occupied = base / "occupied"
            occupied.mkdir()
            with self.assertRaises(FileExistsError):
                smoke.reserve_attempt(occupied, base / "second.marker")
            self.assertTrue((base / "second.marker").is_file())

    def test_command_can_only_launch_named_distro_as_modly_with_guest_timeout(self):
        smoke = self.load()
        command = smoke.worker_command("/mnt/c/proof/smoke.py", "/mnt/c/proof/evidence/guest")
        self.assertEqual(command, ["wsl.exe", "-d", "Modly-AMD-Ubuntu-24.04", "-u", "modly",
                                   "--exec", "/usr/bin/env", "HSA_ENABLE_DXG_DETECTION=1",
                                   "LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:/opt/rocm/lib:/usr/lib/wsl/lib",
                                   "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s",
                                   "90s", "/opt/modly-wsl/venv/bin/python", "-B", "-I",
                                   "/mnt/c/proof/smoke.py", "--worker", "/mnt/c/proof/evidence/guest"])
        with self.assertRaises(ValueError):
            smoke.worker_command("relative.py", "/mnt/c/proof/evidence/guest")

    def test_worker_gate_accepts_only_exact_go_and_refuses_eof(self):
        smoke = self.load()
        smoke.await_go(io.BytesIO(b"GO\n"))
        for content in (b"", b"NO\n", b"GO extra\n", b"GO", b"GO\r\n", b"GO\nX", b"\n"):
            with self.subTest(content=content):
                with self.assertRaises(RuntimeError):
                    smoke.await_go(io.BytesIO(content))

    def test_worker_gate_rejects_text_stream_and_bounds_extra_data_read(self):
        smoke = self.load()
        with self.assertRaises(RuntimeError):
            smoke.await_go(io.StringIO("GO\n"))
        stream = io.BytesIO(b"GO\n" + b"X" * 4096)
        with self.assertRaises(RuntimeError):
            smoke.await_go(stream)
        self.assertEqual(stream.tell(), 4)

    def test_runtime_identity_accepts_exact_lightweight_distribution_and_runtime(self):
        smoke = self.load()
        self.assertTrue(hasattr(smoke, "validate_runtime_identity"), "Missing exact runtime/distribution identity guard")
        smoke.validate_runtime_identity(runtime_identity())

    def test_runtime_identity_refuses_different_versions_git_hip_or_initialization(self):
        smoke = self.load()
        self.assertTrue(hasattr(smoke, "validate_runtime_identity"), "Missing exact runtime/distribution identity guard")
        for key, value in (("torch", "2.9.1+rocm7.2.1.lw.gitff65f5bc"),
                           ("distribution_version", "2.9.1+rocm7.2.1.gitff65f5bc"),
                           ("git_version", "ff65f5bc"), ("hip", "7.2.53211"),
                           ("cuda", "12.8"), ("python", "3.13.3"),
                           ("prefix", "/usr"), ("gpu_initialized_before", True)):
            bad = dict(runtime_identity(), **{key: value})
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    smoke.validate_runtime_identity(bad)
        with self.assertRaises(RuntimeError):
            smoke.validate_runtime_identity({})

    def test_host_never_sends_go_without_job_and_both_healthy_observations(self):
        smoke = self.load()
        for attached, before, after in ((False, healthy(), healthy()),
                                        (True, {}, healthy()), (True, healthy(), {})):
            pipe = io.BytesIO()
            with self.assertRaises(RuntimeError):
                smoke.authorize_go(pipe, attached, before, after)
            self.assertEqual(pipe.getvalue(), b"")
        pipe = io.BytesIO()
        smoke.authorize_go(pipe, True, healthy(), healthy())
        self.assertEqual(pipe.getvalue(), b"GO\n")

    def test_cleanup_refuses_live_refs_nonzero_memory_or_excessive_peak(self):
        smoke = self.load()
        record = {"owned_gpu_tensor_weakrefs_collected": True,
                  "memory_before": {"allocated": 0, "reserved": 0},
                  "memory_peak": {"allocated": 3072, "reserved": 2097152},
                  "memory_release": {"allocated": 0, "reserved": 0}}
        smoke.validate_cleanup(record)
        for key, value in (("owned_gpu_tensor_weakrefs_collected", False),
                           ("memory_before", {"allocated": 1024, "reserved": 0}),
                           ("memory_release", {"allocated": 0, "reserved": 1}),
                           ("memory_peak", {"allocated": 0, "reserved": 67108865})):
            bad = dict(record, **{key: value})
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    smoke.validate_cleanup(bad)
        with self.assertRaises(RuntimeError):
            smoke.validate_cleanup({})

    def test_supervisor_deadlines_fail_at_host_or_gpu_boundary(self):
        smoke = self.load()
        smoke.check_deadlines(119.9, 0.0, None)
        smoke.check_deadlines(69.9, 0.0, 10.0)
        for now, began, go in ((120.0, 0.0, None), (70.0, 0.0, 10.0)):
            with self.assertRaises(TimeoutError):
                smoke.check_deadlines(now, began, go)

    def monitor(self, smoke):
        pending = []
        def submit(function):
            future = Future()
            pending.append(future)
            return future
        return smoke.HealthMonitor(lambda: healthy(), smoke.healthy_observation, submit), pending

    def test_monitor_overlaps_observers_and_enforces_no_start_gap(self):
        smoke = self.load()
        monitor, pending = self.monitor(smoke)
        monitor.poll(0.0)
        monitor.poll(5.0)
        self.assertEqual(len(pending), 2)
        pending[0].set_result((9.0, healthy()))
        monitor.poll(9.0)
        self.assertTrue(monitor.healthy_since(-0.01))
        self.assertFalse(monitor.healthy_since(0.0))  # Same clock tick cannot prove an after-gate start.
        self.assertFalse(monitor.healthy_since(5.0))
        pending[1].set_result((9.5, healthy()))
        monitor.poll(10.0)
        self.assertTrue(monitor.healthy_since(4.99))
        self.assertFalse(monitor.healthy_since(5.0))
        pending[2].set_result((10.1, healthy()))
        with self.assertRaises(TimeoutError):
            monitor.poll(20.01)

    def test_monitor_fails_closed_for_late_missing_or_unhealthy_observation(self):
        smoke = self.load()
        for ended, capture in ((10.01, healthy()), (0.1, {}),
                                (0.1, dict(healthy(), observation={"events": [{"Id": 4101}]}))):
            monitor, pending = self.monitor(smoke)
            monitor.poll(0.0)
            pending[0].set_result((ended, capture))
            with self.subTest(ended=ended, capture=capture):
                with self.assertRaises((RuntimeError, TimeoutError)):
                    monitor.poll(min(ended, 9.0))
        monitor, pending = self.monitor(smoke)
        monitor.poll(0.0)
        monitor.poll(5.0)
        with self.assertRaises(TimeoutError):
            monitor.poll(10.0)

    def test_durable_journal_and_result_preserve_parseable_phase_evidence(self):
        smoke = self.load()
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[2]) as temporary:
            base = Path(temporary)
            smoke.journal(base, "gate-wait", gpu_allowed=False)
            smoke.journal(base, "gate-authorized", gpu_allowed=True)
            rows = [json.loads(line) for line in (base / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([row["phase"] for row in rows], ["gate-wait", "gate-authorized"])
            self.assertEqual([row["gpu_allowed"] for row in rows], [False, True])
            smoke.save_json(base / "results.json", {"classification": "UNKNOWN"})
            self.assertEqual(json.loads((base / "results.json").read_text()), {"classification": "UNKNOWN"})
            with self.assertRaises(FileExistsError):
                smoke.save_json(base / "results.json", {})

    def replay_supervisor(self, mode):
        """Real supervisor/files/clock and a real CPU child; replace only external WSL/Job/observer."""
        smoke = self.load()
        helper = smoke.load_sibling("amd_bounded_gpu_proof")
        real_popen = subprocess.Popen
        expected = [(4 * (i % 33 - 16) + ((i * 7) % 35 - 16)) / 8 for i in range(256)]
        fixture = {"classification": "PASS", "gate_authorized": True,
                   "computation": {"operation": "vector-add", "gpu_arithmetic_operations": 1,
                                   "shape": [256], "dtype": "float32", "values": expected,
                                   "torch_cpu_equal": True},
                   "owned_gpu_tensor_weakrefs_collected": True,
                   "memory_before": {"allocated": 0, "reserved": 0},
                   "memory_peak": {"allocated": 3072, "reserved": 2097152},
                   "memory_release": {"allocated": 0, "reserved": 0}}
        cpu_code = ("import json,pathlib,sys; line=sys.stdin.readline(); "
                    "print('CPU_GATE_LINE='+repr(line),flush=True); "
                    "sys.exit(2) if line!='GO\\n' else None; "
                    "out=pathlib.Path(sys.argv[1]); out.mkdir(); "
                    "(out/'results.json').write_text(sys.argv[2],encoding='utf-8')")
        state = {"attached": False, "terminated": False, "commands": [], "child": None}
        class CPUJob:
            active_reads = 0
            def attach(self, child):
                if mode == "attach-failure":
                    raise RuntimeError("CPU replay: Job attachment refused")
                state["attached"] = True
            def active(self):
                self.active_reads += 1
                if mode == "grace-query-failure" and self.active_reads == 1:
                    raise RuntimeError("CPU replay: Job identity query failed")
                if mode == "delayed-job-exit": return 2 if self.active_reads <= 2 else 0
                return 2 if mode == "persistent-job" else 0
            def finish(self):
                child = state["child"]
                if child is not None:
                    child.wait(timeout=2)
                return {"owned_tree_exit_verified": True, "active_after_cleanup": 0,
                        "forced_job_cleanup": mode == "persistent-job" or
                        (mode == "delayed-job-exit" and self.active_reads <= 2)}
        def observe(since):
            if ((mode == "unhealthy" and state["attached"])
                    or (mode == "post-exit-unhealthy" and state["terminated"])):
                return {"returncode": 0, "timed_out": False,
                        "observation": {"events": [{"Id": 4101}]}}
            return healthy()
        def launch(command, **kwargs):
            self.assertNotIn("text", kwargs, "Production stdin must stay binary")
            self.assertNotIn("encoding", kwargs, "Production stdin must not translate newlines")
            self.assertEqual(command[:6], ["wsl.exe", "-d", "Modly-AMD-Ubuntu-24.04", "-u", "modly", "--exec"])
            self.assertEqual(command[-2:], ["--worker", smoke.guest_path(output / "guest")])
            state["commands"].append(command)
            state["child"] = real_popen([sys.executable, "-B", "-I", "-c", cpu_code,
                                         str(output / "guest"), json.dumps(fixture)], **kwargs)
            return state["child"]
        def terminate(command, **kwargs):
            self.assertEqual(command, ["wsl.exe", "--terminate", "Modly-AMD-Ubuntu-24.04"])
            self.assertEqual(kwargs, {"capture_output": True, "timeout": 10})
            state["commands"].append(command)
            state["terminated"] = True
            return subprocess.CompletedProcess(command, 1 if mode == "termination-failure" else 0, b"", b"")
        observer_helper = types.SimpleNamespace(windows_observation=observe,
                                                healthy_observation=helper.healthy_observation)
        def sibling(name):
            if name == "amd_bounded_gpu_proof":
                return observer_helper
            if name == "amd_forward_owned_job":
                return types.SimpleNamespace(OwnedJob=CPUJob)
            raise AssertionError("Unexpected supervisor helper: " + name)
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[2]) as temporary:
            base = Path(temporary)
            output = base / "attempt"
            with patch.object(smoke, "ATTEMPT_MARKER", base / "attempt.marker"), \
                    patch.object(smoke, "OBSERVATION_INTERVAL", 0.01), \
                    patch.object(smoke, "load_sibling", sibling), \
                    patch.object(smoke.subprocess, "Popen", launch), \
                    patch.object(smoke.subprocess, "run", terminate):
                returncode = smoke.supervise(output)
            result = json.loads((output / "results.json").read_text())
            phases = [json.loads(line)["phase"] for line in (output / "phases.jsonl").read_text().splitlines()]
            stdout = (output / "worker.stdout.txt").read_text()
            return returncode, result, phases, stdout, state["commands"]

    def test_supervisor_cpu_replay_attaches_authorizes_and_requires_clean_exit(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("success")
        self.assertEqual((returncode, result["classification"]), (0, "PASS"))
        self.assertEqual(result["worker_returncode"], 0)
        self.assertIn("CPU_GATE_LINE='GO\\n'", stdout)
        self.assertLess(phases.index("windows-job-attached"), phases.index("go-sent"))
        self.assertTrue(result["observer_healthy"])
        self.assertEqual(commands[-1], ["wsl.exe", "--terminate", "Modly-AMD-Ubuntu-24.04"])

    def test_supervisor_attach_failure_never_sends_go_and_only_terminates_owned_distro(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("attach-failure")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertNotIn("go-sent", phases)
        self.assertIn("CPU_GATE_LINE=''", stdout)
        self.assertEqual(commands[-1], ["wsl.exe", "--terminate", "Modly-AMD-Ubuntu-24.04"])
        self.assertEqual(len(commands), 2)

    def test_supervisor_unhealthy_observer_refuses_go_and_pass(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("unhealthy")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertNotIn("go-sent", phases)
        self.assertIn("CPU_GATE_LINE=''", stdout)
        self.assertFalse(result["observer_healthy"])
        self.assertEqual(commands[-1], ["wsl.exe", "--terminate", "Modly-AMD-Ubuntu-24.04"])

    def test_supervisor_post_exit_observer_failure_prevents_pass(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("post-exit-unhealthy")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertEqual(result["worker"]["classification"], "PASS")
        self.assertFalse(result["observer_healthy"])
        self.assertIn("distro_cleanup_or_post_observer_error", result)

    def test_supervisor_owned_distro_termination_failure_prevents_pass(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("termination-failure")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertEqual(result["worker"]["classification"], "PASS")
        self.assertEqual(result["distro_termination"]["returncode"], 1)

    def test_supervisor_waits_for_delayed_normal_helper_exit_before_job_finish(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("delayed-job-exit")
        self.assertEqual((returncode, result["classification"]), (0, "PASS"))
        self.assertTrue(result["windows_job_grace"]["empty"])
        self.assertEqual(result["windows_job_grace"]["limit_seconds"], 5.0)
        self.assertFalse(result["windows_job_cleanup"]["forced_job_cleanup"])

    def test_supervisor_persistent_job_still_forces_cleanup_and_refuses_pass(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("persistent-job")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertFalse(result["windows_job_grace"]["empty"])
        self.assertLess(result["windows_job_grace"]["seconds"], 5.2)
        self.assertTrue(result["windows_job_cleanup"]["forced_job_cleanup"])

    def test_supervisor_failed_job_query_retains_cleanup_monitoring_and_refuses_pass(self):
        returncode, result, phases, stdout, commands = self.replay_supervisor("grace-query-failure")
        self.assertEqual((returncode, result["classification"]), (1, "UNKNOWN"))
        self.assertIn("windows_job_grace_error", result)
        self.assertTrue(result["observer_healthy"])
        self.assertTrue(result["windows_job_cleanup"]["owned_tree_exit_verified"])


if __name__ == "__main__":
    unittest.main()
