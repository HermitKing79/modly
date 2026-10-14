"""Focused CPU acceptance; never starts WSL or imports GPU libraries."""
import ctypes
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("amd_wsl_teardown_probe.py")


class TeardownContracts(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), "Missing bounded teardown observer")
        spec = importlib.util.spec_from_file_location("teardown_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_job_pid_list_requires_complete_bounded_unique_membership(self):
        probe = self.load()
        row = probe.JobPids()
        row.assigned = row.count = 2
        row.pids[0], row.pids[1] = 123, 456
        self.assertEqual(probe.valid_pids(row), [123, 456])
        for assigned, count, ids in ((2, 1, [123]), (17, 17, []),
                                      (2, 2, [123, 123]), (1, 1, [0])):
            row = probe.JobPids()
            row.assigned, row.count = assigned, count
            for index, value in enumerate(ids): row.pids[index] = value
            with self.assertRaises(RuntimeError): probe.valid_pids(row)

    def test_grace_is_bounded_monitored_and_never_terminates(self):
        probe = self.load()
        now = [0.0]
        ticks = []
        def sleep(seconds): now[0] += seconds
        result = probe.grace(lambda: now[0] >= .2, lambda: ticks.append(now[0]),
                             .3, clock=lambda: now[0], sleep=sleep)
        self.assertTrue(result["empty"])
        self.assertGreaterEqual(len(ticks), 3)
        now[0] = 0
        result = probe.grace(lambda: False, lambda: None, .3,
                             clock=lambda: now[0], sleep=sleep)
        self.assertFalse(result["empty"])
        self.assertAlmostEqual(result["seconds"], .3)
        def unhealthy(): raise RuntimeError("observer unhealthy")
        with self.assertRaisesRegex(RuntimeError, "observer unhealthy"):
            probe.grace(lambda: True, unhealthy, .3)

    def test_native_process_identity_uses_real_current_process_handle(self):
        probe = self.load()
        owned = probe.smoke.load_sibling("amd_forward_owned_job").OwnedJob()
        observed = probe.JobObservation(owned)
        try:
            # Empty private Job cannot accidentally expose the surrounding Job.
            self.assertEqual(observed.capture("empty"), [])
            self.assertEqual(ctypes.sizeof(probe.JobPids), 136)
            self.assertEqual(ctypes.sizeof(probe.ProcessEntry), 568)
            self.assertEqual(observed.parents().get(__import__('os').getpid(), {})["name"].lower(), "python.exe")
            handle = observed.api.OpenProcess(0x101000, False, __import__('os').getpid())
            self.assertTrue(handle)
            try:
                status = observed.status(handle)
                self.assertFalse(status["exited"])
                self.assertIsNone(status["exit_code"])
                self.assertIsNone(status["exit_filetime_100ns"])
                self.assertGreater(status["created_filetime_100ns"], 0)
            finally: observed.require(observed.api.CloseHandle(handle))
        finally:
            observed.close()
            self.assertFalse(owned.finish()["forced_job_cleanup"])

    def test_confirmation_metadata_failure_does_not_interrupt_genuine_job_finish(self):
        probe = self.load()
        state = {"finished": False, "closed": False}
        class CPUJob:
            def attach(self, child): pass
            def active(self): return 0
            def finish(self, timeout=5.):
                self.assert_empty = self.active() == 0
                state["finished"] = True
                return {"forced_job_cleanup": False}
        class CPUObservation:
            identities = {}; handles = {}
            def __init__(self, job): pass
            def capture(self, phase):
                if phase == "production-active-query": raise RuntimeError("CPU metadata unavailable")
                return []
            def close(self): state["closed"] = True
        def supervisor(output):
            job = probe.smoke.load_sibling("amd_forward_owned_job").OwnedJob()
            job.attach(None)
            self.assertFalse(job.finish()["forced_job_cleanup"])
            raise RuntimeError("CPU confirmation containment verified; no guest launched")
        with patch.object(probe, "JobObservation", CPUObservation), \
                patch.object(probe.smoke, "load_sibling", return_value=SimpleNamespace(OwnedJob=CPUJob)), \
                patch.object(probe.smoke, "supervise", side_effect=supervisor):
            with self.assertRaisesRegex(RuntimeError, "CPU confirmation containment verified"):
                probe.confirm(probe.OUTPUT.with_name("unused-cpu-fixture"))
        self.assertTrue(state["finished"])
        self.assertTrue(state["closed"])


if __name__ == "__main__": unittest.main()
