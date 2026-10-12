"""Supervisor safety checks. System Python only; no GPU imports or execution."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("amd_bounded_gpu_proof.py")


class BoundedProofTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location("bounded_gpu", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_gpu_is_visible_and_all_cache_paths_stay_in_scratch(self):
        module = self.load()
        scratch = SCRIPT.parents[2] / "test-scratch"
        env = module.scratch_environment(scratch)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["FLEX_GEMM_USE_AUTOTUNE_CACHE"], "0")
        self.assertEqual(env["FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE"], "0")
        for key in module.CACHE_KEYS:
            Path(env[key]).relative_to(scratch)

    def test_boundary_gate_refuses_missing_or_partial_initialization_evidence(self):
        module = self.load()
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[2]) as temporary:
            evidence = Path(temporary) / "results.json"
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                module.require_gate_one(evidence)
            evidence.write_text('{"classification":"PARTIAL","gate":1}')
            with self.assertRaises(RuntimeError):
                module.require_gate_one(evidence)
            evidence.write_text('{"classification":"PASS","gate":1,"workers":[]}')
            with self.assertRaises(RuntimeError):
                module.require_gate_one(evidence)
            evidence.write_text('''{"classification":"PASS","gate":1,"inventory_unchanged":true,
                "observer_healthy":false,"workers":[
                {"module":"flex_gemm.kernels.cuda","status":"PASS","worker_returncode":0,"gpu_initialized_after":true},
                {"module":"o_voxel._C","status":"PASS","worker_returncode":0,"gpu_initialized_after":true}]}''')
            with self.assertRaises(RuntimeError):
                module.require_gate_one(evidence)

    def test_native_worker_timeout_is_bounded_and_preserves_output(self):
        module = self.load()
        capture = module.preflight.run_child([
            sys.executable, "-B", "-I", "-c",
            "import time; print('before native call', flush=True); time.sleep(60)"], .5)
        self.assertTrue(capture["timed_out"])
        self.assertIn("before native call", capture["stdout"])
        self.assertLess(capture["elapsed_seconds"], 5)

    def test_missing_observation_or_driver_events_cannot_pass_the_gate(self):
        module = self.load()
        self.assertTrue(hasattr(module, "healthy_observation"), "Missing fail-closed observer validation")
        healthy = {"returncode": 0, "timed_out": False,
                   "observation": {"controllers": [{"Name": "AMD Radeon RX 9070 XT", "Status": "OK"}],
                                   "observer": {"system_log_probe": {"RecordId": 1}},
                                   "events": []}}
        self.assertTrue(module.healthy_observation(healthy))
        for bad in ({"returncode": None, "timed_out": True},
                    {"returncode": 1, "timed_out": False},
                    {"returncode": 0, "timed_out": False},
                    {"returncode": 0, "timed_out": False, "observation": {"event_error": "denied"}},
                    {"returncode": 0, "timed_out": False, "observation": {"events": [{"Id": 4101}]}},
                    {"returncode": 0, "timed_out": False, "observation": {"events": [], "controllers": []}}):
            with self.subTest(bad=bad):
                self.assertFalse(module.healthy_observation(bad))

    def test_transient_directory_size_is_ignored_but_file_changes_remain_visible(self):
        module = self.load()
        self.assertTrue(hasattr(module, "normalize_metadata_rows"), "Missing directory-size normalization")
        # Hand-derived directory/file modes. Only directory allocation size varies.
        cold = [["package", 0, 123, 16895], ["package/module.py", 8, 456, 33206]]
        warm = [["package", 4096, 123, 16895], ["package/module.py", 8, 456, 33206]]
        self.assertEqual(module.normalize_metadata_rows(cold), module.normalize_metadata_rows(warm))
        changed_file = [["package", 4096, 123, 16895], ["package/module.py", 9, 456, 33206]]
        changed_timestamp = [["package", 4096, 124, 16895], ["package/module.py", 8, 456, 33206]]
        self.assertNotEqual(module.normalize_metadata_rows(cold), module.normalize_metadata_rows(changed_file))
        self.assertNotEqual(module.normalize_metadata_rows(cold), module.normalize_metadata_rows(changed_timestamp))

    def test_boundary_result_must_match_one_triangle_three_boundary_edges(self):
        module = self.load()
        module.validate_boundary_result(1, [0, 0, 0], [[0, 1], [0, 2], [1, 2]])
        for count, ids, edges in ((0, [0, 0, 0], [[0, 1], [0, 2], [1, 2]]),
                                  (1, [0, 0], [[0, 1], [0, 2], [1, 2]]),
                                  (1, [0, 1, 0], [[0, 1], [0, 2], [1, 2]]),
                                  (1, [0, 0, 0], [[0, 1], [0, 1], [1, 2]])):
            with self.subTest(count=count, ids=ids, edges=edges):
                with self.assertRaises(RuntimeError):
                    module.validate_boundary_result(count, ids, edges)


if __name__ == "__main__":
    unittest.main()
