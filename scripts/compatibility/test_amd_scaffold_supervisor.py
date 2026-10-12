import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location("scaffold_proof", Path(__file__).with_name("amd_worker_scaffold_proof.py"))
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class SupervisorTests(unittest.TestCase):
    def test_classification_requires_clean_supervision_and_complete_ipc(self):
        ipc = {"routes": [{"node": "generate"}, {"node": "refine"}], "hard_stop_restart": True,
               "negative_tests": [{"case": "missing-checkpoint", "rejected": True},
                                  {"case": "incompatible-runtime", "rejected": True}]}
        healthy = {"exit": 0, "timeout": False, "tree_cleanup_verified": True}
        self.assertEqual(proof.classify_result(healthy, True, ipc), "PASS")
        for broken in ({**healthy, "timeout": True}, {**healthy, "tree_cleanup_verified": False},
                       {**healthy, "exit": 1}):
            self.assertEqual(proof.classify_result(broken, True, ipc), "PARTIAL")
        self.assertEqual(proof.classify_result(healthy, False, ipc), "PARTIAL")
        self.assertEqual(proof.classify_result(healthy, True, None), "PARTIAL")
        self.assertEqual(proof.classify_result(healthy, True, {**ipc, "routes": []}), "PARTIAL")

    def test_timeout_reaps_host_and_descendant(self):
        script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        result = proof.run_host([sys.executable, "-B", "-c", script], os.environ.copy(), 1.0)
        self.assertTrue(result["timeout"])
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertIsNotNone(result["exit"])
        pid = int(result["stdout"].strip())
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x100000, False, pid)
        if handle:
            try:
                self.assertEqual(kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0), 0)
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))
        else:
            self.assertEqual(ctypes.get_last_error(), 87, "Could not verify child termination")


if __name__ == "__main__":
    unittest.main()
