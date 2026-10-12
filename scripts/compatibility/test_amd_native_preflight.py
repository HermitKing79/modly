"""Regression checks for the preflight guard; no native/GPU imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

AUDIT = Path(__file__).with_name("amd_trellis2_audit.py")
SDK = Path("D:/AI/Trellis2AMD/venv/Scripts/rocm-sdk.exe")


class GuardTests(unittest.TestCase):
    def probe(self, command, timeout=None, shell=False):
        # Supports the original guard to demonstrate the discovery regression.
        code = f'''
import importlib.util, inspect, json, subprocess, sys
spec = importlib.util.spec_from_file_location("audit", {str(AUDIT)!r})
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
if "sdk_executable" in inspect.signature(audit.block_external_actions).parameters:
    audit.block_external_actions(sdk_executable={str(SDK)!r})
else:
    audit.block_external_actions()
try:
    subprocess.check_output({command!r}, stderr=subprocess.PIPE,
                            timeout={timeout!r}, shell={shell!r})
    result = {{"state": "executed", "returncode": 0}}
except subprocess.CalledProcessError as exc:
    result = {{"state": "executed", "returncode": exc.returncode,
               "stderr": exc.stderr.decode(errors="replace")}}
except Exception as exc:
    result = {{"state": "blocked", "error": str(exc)}}
print(json.dumps(result))
'''
        proc = subprocess.run([sys.executable, "-B", "-I", "-c", code],
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_verified_discovery_reaches_read_only_missing_package_diagnostic(self):
        result = self.probe(["rocm-sdk", "path", "--root"])
        self.assertEqual(result["state"], "executed", result)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("rocm_sdk_devel", result["stderr"])

    def test_other_commands_and_shells_cannot_execute(self):
        for command, shell in ((["rocm-sdk", "init"], False),
                               (["rocm-sdk", "path", "--root", "--extra"], False),
                               (["pip", "install", "anything"], False),
                               (["cmd.exe", "/c", "ver"], False),
                               (["rocm-sdk", "path", "--root"], True)):
            with self.subTest(command=command, shell=shell):
                self.assertEqual(self.probe(command, shell=shell)["state"], "blocked")

    def test_discovery_rejects_invalid_timeout_and_untrusted_executable(self):
        self.assertEqual(self.probe(["rocm-sdk", "path", "--root"], timeout=0)["state"], "blocked")
        self.assertEqual(self.probe(["C:/untrusted/rocm-sdk.exe", "path", "--root"])["state"], "blocked")

    def test_hung_worker_is_terminated_with_bounded_diagnostic(self):
        spec = importlib.util.spec_from_file_location(
            "preflight", AUDIT.with_name("amd_native_import_preflight.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_child([sys.executable, "-B", "-I", "-c",
                                   "import time; time.sleep(60)"], 0.2)
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["returncode"])
        self.assertLess(result["elapsed_seconds"], 5)

    def test_file_probe_can_write_only_inside_workspace_scratch(self):
        with tempfile.TemporaryDirectory(prefix="guard-test-", dir=AUDIT.parents[2]) as scratch:
            code = f'''
import importlib.util, json, pathlib
spec = importlib.util.spec_from_file_location("audit", {str(AUDIT)!r})
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.block_external_actions(sdk_executable={str(SDK)!r}, scratch_dir={scratch!r})
inside = pathlib.Path({scratch!r}) / "probe"
inside.write_text("temporary")
inside.unlink()
try:
    (pathlib.Path({scratch!r}).parent / "forbidden-probe").write_text("forbidden")
except RuntimeError:
    print("outside write blocked")
else:
    raise AssertionError("write escaped scratch")
'''
            result = subprocess.run([sys.executable, "-B", "-I", "-c", code],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("outside write blocked", result.stdout)


if __name__ == "__main__":
    unittest.main()
