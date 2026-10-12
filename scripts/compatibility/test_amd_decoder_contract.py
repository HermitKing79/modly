"""Pure contract tests; no checkpoint load, torch import or GPU activity."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("decoder_contract", Path(__file__).with_name("amd_decoder_contract.py"))
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)
WORKER_SPEC = importlib.util.spec_from_file_location("decoder_proof", Path(__file__).with_name("amd_sparse_decoder_loading.py"))
proof = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(proof)


class DecoderContractTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"weight": {"shape": [2, 3], "dtype": "F16"}, "bias": {"shape": [2], "dtype": "F32"}}
        self.actual = {key: {**value, "device": "cpu", "is_meta": False} for key, value in self.expected.items()}

    def test_exact_keys_shapes_and_dtypes_required(self):
        contract.validate_tensors(self.expected, self.actual, "cpu")
        for bad in ({"weight": self.actual["weight"]}, {**self.actual, "extra": self.actual["bias"]},
                    {**self.actual, "weight": {**self.actual["weight"], "shape": [3, 2]}},
                    {**self.actual, "weight": {**self.actual["weight"], "dtype": "F32"}}):
            with self.subTest(actual=bad), self.assertRaises(ValueError):
                contract.validate_tensors(self.expected, bad, "cpu")

    def test_meta_or_wrong_device_fails(self):
        for changes in ({"device": "cuda:0"}, {"is_meta": True}):
            with self.assertRaises(ValueError):
                contract.validate_tensors(self.expected, {**self.actual, "weight": {**self.actual["weight"], **changes}}, "cpu")

    def test_identity_matches_archived_config_header_and_source(self):
        archived = {"config": {"name": "SparseStructureDecoder", "args": {"latent_channels": 8}},
                    "file_bytes": 24, "mtime_ns": 17, "header_bytes": 8, "header_sha256": "abc",
                    "tensor_count": 2, "parameter_count": 8, "tensors": self.expected,
                    "source_module_sha256": "def", "metadata_only_fixture": False}
        current = dict(archived)
        contract.validate_identity(archived, current)
        for key, value in (("config", {}), ("tensors", {}), ("header_sha256", "wrong"),
                           ("file_bytes", 99), ("source_module_sha256", "wrong"), ("metadata_only_fixture", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_identity(archived, {**current, key: value})

    def test_release_requires_no_live_tensor_or_reservations(self):
        good = {"allocated_bytes": 0, "reserved_bytes": 0, "model_collected": True}
        contract.validate_release(good)
        for change in ({"allocated_bytes": 1}, {"reserved_bytes": 1}, {"model_collected": False}):
            with self.assertRaises(ValueError):
                contract.validate_release({**good, **change})

    def test_pass_requires_gates_observers_clean_exit_and_preservation(self):
        record = {"status": "PASS", "gate_one": {"status": "PASS"}, "gate_two": {"status": "PASS"},
                  "release": {"allocated_bytes": 0, "reserved_bytes": 0, "model_collected": True}}
        self.assertEqual(contract.classify(record, 0, False, True, True), "PASS")
        for code, timeout, observer, preserved in ((1, False, True, True), (0, True, True, True),
                                                  (0, False, False, True), (0, False, True, False)):
            self.assertEqual(contract.classify(record, code, timeout, observer, preserved), "PARTIAL")
        self.assertEqual(contract.classify({**record, "gate_two": {"status": "NOT ATTEMPTED"}}, 0, False, True, True), "PARTIAL")
        self.assertEqual(contract.classify({**record, "release": {"allocated_bytes": 1, "reserved_bytes": 0, "model_collected": True}}, 0, False, True, True), "PARTIAL")

    def test_only_transfer_or_metadata_gpu_operations_allowed(self):
        for name in ("aten._to_copy.default", "aten.copy_.default", "aten.detach.default"):
            self.assertTrue(contract.gpu_operation_allowed(name))
        for name in ("aten.convolution.default", "aten.mm.default", "aten.add.Tensor", "aten.normal_.default"):
            self.assertFalse(contract.gpu_operation_allowed(name))

    def test_gate_two_rejects_incomplete_cpu_evidence_or_changed_checkpoint(self):
        expected = {"tensors": self.expected, "tensor_data_bytes": 20}
        gate = {"status": "PASS", "strict": True, "assign": True, "gpu_initialized": False,
                "missing_keys": [], "unexpected_keys": [], "parameter_bytes": 20, "tensors": self.actual}
        contract.authorize_placement(gate, expected, "sha", "sha")
        for changes in ({"gpu_initialized": True}, {"strict": False}, {"missing_keys": ["bias"]},
                        {"unexpected_keys": ["extra"]}, {"parameter_bytes": 19}, {"tensors": {}}):
            with self.assertRaises(ValueError):
                contract.authorize_placement({**gate, **changes}, expected, "sha", "sha")
        with self.assertRaises(ValueError):
            contract.authorize_placement(gate, expected, "sha", "changed")

    def test_unconditional_cleanup_reaps_worker_after_any_error(self):
        class Child:
            pid = 123
            returncode = None
            def poll(self):
                return self.returncode
            def kill(self):
                self.returncode = -1
            def wait(self, timeout):
                if self.returncode is None:
                    raise proof.subprocess.TimeoutExpired("mock", timeout)
                return self.returncode
        child, process = Child(), {"error": "OSError: evidence write failed"}
        with patch.object(proof.gpu_proof.preflight, "run_child", side_effect=OSError("mock taskkill unavailable")):
            proof.reap_owned_child(child, process)
        self.assertEqual(child.returncode, -1)
        self.assertTrue(process["forced_cleanup"])
        self.assertTrue(process["worker_exit_verified"])
        self.assertFalse(process.get("cleanup_unverified", False))

    def test_cleanup_failure_cannot_be_reported_as_verified(self):
        class Child:
            pid = 123
            returncode = None
            def poll(self):
                return None
            def kill(self):
                raise OSError("mock kill failure")
            def wait(self, timeout):
                raise proof.subprocess.TimeoutExpired("mock", timeout)
        process = {}
        with patch.object(proof.gpu_proof.preflight, "run_child", side_effect=OSError("mock taskkill unavailable")):
            proof.reap_owned_child(Child(), process)
        self.assertTrue(process["cleanup_unverified"])
        self.assertFalse(process["worker_exit_verified"])


if __name__ == "__main__":
    unittest.main()
