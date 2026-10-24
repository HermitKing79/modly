"""Fresh RAM-ready decode routing tests; no WSL, Job, model or marker launch."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'triton', 'trellis2',
                'flex_gemm', 'transformers', 'cumesh', 'o_voxel', 'nvdiffrast'}:
            raise AssertionError('RAM routing tests forbid neural/native import: ' + fullname)


sys.meta_path.insert(0, NoNeuralImports())
import amd_generation_decode as historical
import amd_zombie_decode as zombie
ORIGINAL_HOST_INPUTS = historical.shape_inputs
ORIGINAL_ZOMBIE = {name: getattr(zombie, name) for name in
    ('EVIDENCE', 'RUN_ID', 'PRIVATE', 'PUBLIC', 'source_inputs', 'make_runtime')}
try:
    import amd_zombie_decode_resume as subject
except ModuleNotFoundError as exc:
    if exc.name != 'amd_zombie_decode_resume':
        raise
    subject = None

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT.parent / 'artifacts/amd-zombie-ram-readiness-91b7455'
PUBLIC = ROOT / 'docs/compatibility/amd-zombie-ram-readiness-3-evidence'
FLOOR = 22 * 1024**3
SOURCE_SHA = '8c6c3a73bb1e58691105f99dd27353ec8fa650e17f0a1045a5b9e45bb86a1e78'
TEST_COMMIT = '1' * 40


class ResumeExists(unittest.TestCase):
    def test_fresh_ram_ready_adapter_exists(self):
        self.assertIsNotNone(subject, 'Fresh RAM-ready private decode adapter is required')


@unittest.skipIf(subject is None, 'Fresh RAM adapter not implemented yet')
class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One actual unchanged saved-input qualification for all routing tests.
        cls.actual_bundle = zombie.source_inputs(zombie.shape.EVIDENCE)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        scratch = PRIVATE / 'test-scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='cpu-resume-', dir=scratch)))

    def runtime(self):
        with patch.object(zombie, 'source_inputs', return_value=self.actual_bundle):
            return subject.make_runtime(TEST_COMMIT)

    def fake_runtime(self, available):
        return SimpleNamespace(evidence=self.scratch / 'decode', commit=TEST_COMMIT,
            contract=SimpleNamespace(sha256='a' * 64),
            host_admission=Mock(return_value={'available_ram_bytes': available,
                'required_ram_bytes': 12 * 1024**3, 'D_free_bytes': 100 * 1024**3,
                'required_D_free_bytes': 50 * 1024**3, 'admitted': True}))

    def test_clone_keeps_old_globals_and_uses_fresh_paths_generic_caps_and_source(self):
        runtime = self.runtime()
        for name, original in ORIGINAL_ZOMBIE.items():
            self.assertIs(getattr(zombie, name), original, name)
        self.assertIs(historical.shape_inputs, ORIGINAL_HOST_INPUTS)
        self.assertEqual(subject.EVIDENCE, historical.APPROVED_ROOT / 'zombie-ram-91b7455/decode')
        self.assertEqual(subject.RUN_ID, 'run-zombie-geometry-ram-91b7455-v1')
        self.assertEqual(subject.PRIVATE, PRIVATE)
        self.assertEqual(subject.PUBLIC, PUBLIC)
        self.assertEqual(runtime.evidence, subject.EVIDENCE)
        self.assertEqual(str(runtime.contract.output_directory), '/home/modly/' + subject.RUN_ID + '/geometry')
        self.assertEqual(runtime.contract.initial_count, 1000)
        self.assertEqual(runtime.contract.source_job_sha256, zombie.SOURCE_JOB_SHA)
        self.assertEqual(dict(runtime.contract.input_sha256), {name: row[1] for name, row in zombie.INPUTS.items()})
        self.assertEqual(runtime.contract.maximum_final_rows, 1842968)
        self.assertEqual(runtime.contract.maximum_intermediate_rows, 1000000)
        self.assertTrue(runtime.contract.large_convolution_tiling)
        self.assertIsNone(runtime.contract.final_support_profile)
        private_static = type(runtime).__init__.__globals__['STATIC_FILES']
        for name in ('scripts/compatibility/amd_zombie_decode_resume.py',
                     'scripts/compatibility/test_amd_zombie_decode_resume.py',
                     'docs/compatibility/amd-zombie-ram-readiness-3-evidence/readiness-summary.json'):
            self.assertIn(name, private_static)
            self.assertNotIn(name, historical.STATIC_FILES)
        self.assertFalse(runtime.evidence.exists())

    def test_command_go_transport_and_original_deadlines_are_retained_without_execution(self):
        runtime = self.runtime()
        command = runtime.worker_command(historical.guest_path(historical.GUEST_ENTRY),
                                        historical.guest_path(runtime.evidence / 'run/guest'))
        self.assertEqual(command[:6], ['wsl.exe', '-d', 'Modly-AMD-Ubuntu-24.04', '-u', 'modly', '--exec'])
        for token in ('/opt/modly-wsl/venv/bin/python', '/usr/bin/timeout', '--signal=TERM',
                      '--kill-after=5s', '450s', runtime.contract.sha256):
            self.assertIn(token, command)
        self.assertIn(historical.guest_path(runtime.evidence / 'run-contract.json'), command)
        supervisor = runtime.supervisor()
        self.assertEqual((supervisor.NATIVE_DEADLINE, supervisor.CPU_DEADLINE,
                          supervisor.GPU_DEADLINE, supervisor.HOST_DEADLINE), (60., 120., 240., 480.))
        private_globals = type(runtime).__init__.__globals__
        self.assertEqual(private_globals['guest_bootstrap'].__code__.co_code, historical.guest_bootstrap.__code__.co_code)
        self.assertEqual(runtime.guest.base.await_go.__code__.co_code,
                         __import__('amd_wsl_rocdxg_smoke').await_go.__code__.co_code)
        self.assertFalse(runtime.evidence.exists())

    def test_actual_three_sample_readiness_hashes_and_semantics_are_required(self):
        actual = subject.verify_readiness()
        self.assertEqual(actual, historical.read_json(PUBLIC / 'readiness-summary.json'))
        private_path = PRIVATE / 'memory-readiness-private.json'
        raw = historical.read_json(private_path)
        self.assertEqual(actual['private_memory_receipt_sha256'], hashlib.sha256(private_path.read_bytes()).hexdigest())
        self.assertEqual(raw['classification'], 'READY_FOR_FURTHER_ADMISSION')
        self.assertEqual(raw['source_sha256'], SOURCE_SHA)
        self.assertEqual(len(raw['samples']), 3)
        for sample in raw['samples']:
            self.assertEqual(sample['required_available_bytes'], FLOOR)
            self.assertGreaterEqual(sample['available_physical_bytes'], FLOOR)
            self.assertIs(sample['target_met'], True)
        original_digest = historical.digest
        with patch.object(historical, 'digest', side_effect=lambda path: '0' * 64
                if Path(path) == private_path else original_digest(path)):
            with self.assertRaises((ValueError, RuntimeError)):
                subject.verify_readiness()

    def test_rehashed_or_claimed_ready_rows_cannot_bypass_three_sample_ram_or_no_neural_semantics(self):
        private_path = PRIVATE / 'memory-readiness-private.json'
        original_read = historical.read_json
        baseline = original_read(private_path)
        mutations = (
            lambda row: row.update(classification='NOT_READY'),
            lambda row: row['samples'][0].update(available_physical_bytes=FLOOR - 1, target_met=True),
            lambda row: row['samples'].pop(),
            lambda row: row.update(source_sha256='0' * 64),
            lambda row: row.update(model_loaded=True),
            lambda row: row.update(neural_forwards=1),
        )
        for mutate in mutations:
            altered = copy.deepcopy(baseline)
            mutate(altered)
            with self.subTest(mutate=mutate), patch.object(historical, 'read_json',
                    side_effect=lambda path, *a, **kw: altered if Path(path) == private_path else original_read(path, *a, **kw)):
                with self.assertRaises((ValueError, RuntimeError)):
                    subject.verify_readiness()

    def test_wrong_output_is_rejected_before_readiness_and_host_admission(self):
        runtime = self.fake_runtime(FLOOR)
        reserve = Mock()
        with patch.object(subject, 'verify_readiness') as readiness:
            with self.assertRaises(ValueError):
                subject.reserve_fresh(runtime, reserve, runtime.evidence / 'other')
        readiness.assert_not_called()
        runtime.host_admission.assert_not_called()
        reserve.assert_not_called()
        self.assertFalse(runtime.evidence.exists())

    def test_final_ram_failure_retains_only_private_gate_record_without_reserving(self):
        runtime = self.fake_runtime(FLOOR - 1)
        reserve = Mock()
        with patch.object(subject, 'PRIVATE', self.scratch), patch.object(subject, 'verify_readiness', return_value={}):
            with self.assertRaises(RuntimeError):
                subject.reserve_fresh(runtime, reserve, runtime.evidence / 'run')
        runtime.host_admission.assert_called_once_with()
        reserve.assert_not_called()
        self.assertTrue((self.scratch / 'final-pre-marker-admission.json').is_file())
        self.assertFalse(runtime.evidence.exists())
        self.assertFalse((runtime.evidence / 'decode-attempt-started.json').exists())

    def test_final_ram_success_records_exclusively_before_one_original_reserve_call(self):
        runtime = self.fake_runtime(FLOOR)
        output = runtime.evidence / 'run'
        calls = []
        def reserve(value):
            self.assertEqual(value, output)
            self.assertTrue((self.scratch / 'final-pre-marker-admission.json').is_file())
            calls.append(value)
        with patch.object(subject, 'PRIVATE', self.scratch), patch.object(subject, 'verify_readiness', return_value={}):
            subject.reserve_fresh(runtime, reserve, output)
            self.assertEqual(calls, [output])
            with self.assertRaises(FileExistsError):
                subject.reserve_fresh(runtime, reserve, output)
        self.assertEqual(calls, [output])
        self.assertFalse(runtime.evidence.exists())
        self.assertFalse((runtime.evidence / 'decode-attempt-started.json').exists())


if __name__ == '__main__':
    unittest.main()
