"""CPU-only failure cleanup acceptance; external WSL/Job/observer are stubs."""
import importlib.util
import io
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

PROBE = Path(__file__).with_name("amd_wsl_stdin_probe.py")
REPLAY_OLD_MONITOR_CONDITION = False


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if path == PROBE and REPLAY_OLD_MONITOR_CONDITION:
        # Reintroduce the precisely reviewed defect in memory only; no file reset.
        source = path.read_text().replace("if not observer_failed:", "if not failed:")
        exec(compile(source, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


class ProbeCleanup(unittest.TestCase):
    def test_job_attachment_failure_retains_monitoring_and_named_termination(self):
        probe = load(PROBE, "probe_test")
        smoke = load(PROBE.with_name("amd_wsl_rocdxg_smoke.py"), "smoke_test")
        real_helper = smoke.load_sibling("amd_bounded_gpu_proof")
        state = {"in_cleanup": False, "observed_during_cleanup": 0, "terminated": 0}
        class Child:
            stdin = io.BytesIO()
            def wait(self, timeout): return 0
        class Job:
            def attach(self, child): raise RuntimeError("CPU fixture: attachment refused")
            def finish(self):
                state["in_cleanup"] = True
                time.sleep(.15)
                state["in_cleanup"] = False
                return {"forced_job_cleanup": False, "owned_tree_exit_verified": True}
        def observe(since):
            if state["in_cleanup"]: state["observed_during_cleanup"] += 1
            return {"returncode": 0, "timed_out": False, "observation": {
                "observer": {"system_log_probe": {"RecordId": 1}}, "events": [],
                "controllers": [{"Name": "AMD Radeon RX 9070 XT", "Status": "OK"}]}}
        def sibling(name):
            if name == "amd_bounded_gpu_proof":
                return types.SimpleNamespace(windows_observation=observe,
                                             healthy_observation=real_helper.healthy_observation)
            if name == "amd_forward_owned_job": return types.SimpleNamespace(OwnedJob=Job)
            raise AssertionError(name)
        def terminate():
            state["terminated"] += 1
            return {"returncode": 0, "command": ["wsl.exe", "--terminate", smoke.DISTRO]}
        with tempfile.TemporaryDirectory(dir=PROBE.parents[2]) as temporary:
            with patch.object(probe, "load", return_value=smoke), \
                    patch.object(smoke, "load_sibling", side_effect=sibling), \
                    patch.object(smoke, "OBSERVATION_INTERVAL", .01), \
                    patch.object(smoke, "terminate_owned_distro", side_effect=terminate), \
                    patch.object(probe.subprocess, "Popen", return_value=Child()):
                self.assertEqual(probe.host(Path(temporary) / "result", True), 1)
        self.assertEqual(state["terminated"], 1)
        self.assertGreater(state["observed_during_cleanup"], 0)


if __name__ == "__main__": unittest.main()
