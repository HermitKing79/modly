"""Bounded, passive Windows observer check. Never imports or executes GPU code."""
import datetime
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).with_name("amd_bounded_gpu_proof.py")


class ObserverTests(unittest.TestCase):
    def test_stock_module_can_read_real_system_log_and_filtered_window_twice(self):
        spec = importlib.util.spec_from_file_location("gpu_proof", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evidence = SCRIPT.parents[2] / "docs/compatibility/amd-bounded-gpu-proof-evidence"
        for index in range(2):
            since = datetime.datetime.now(datetime.timezone.utc).isoformat()
            capture = module.windows_observation(since)
            with (evidence / f"observer-fixed-check-{index + 1}.json").open("x", encoding="utf-8") as saved:
                saved.write(json.dumps(capture, indent=2) + "\n")
            self.assertTrue(module.healthy_observation(capture), capture)
            observer = capture["observation"]["observer"]
            self.assertEqual(observer["edition"], "Desktop")
            self.assertIn("WindowsPowerShell", observer["diagnostics_module"])
            self.assertGreater(observer["system_log_probe"]["RecordId"], 0)


if __name__ == "__main__":
    unittest.main()
