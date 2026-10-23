"""Negative CPU acceptance checks using labelled historical JSON fixtures only.

These tests never produce or authenticate a new neural result. They copy old
records into temporary fixture directories to test refusal of altered evidence.
"""
import copy
from contextlib import ExitStack
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'triton', 'transformers', 'trellis2', 'flex_gemm',
            'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime',
        }:
            raise AssertionError('CPU acceptance imported a model/GPU package: ' + fullname)


sys.meta_path.insert(0, NoModels())
import amd_zombie_occupancy as subject

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT.parent / 'artifacts/amd-zombie-occupancy-c570c2e'
SYNTHETIC_COMMIT = '2' * 40
HISTORICAL = {
    'sampling': ROOT / 'docs/compatibility/amd-sparse-structure-sampling-evidence/run',
    'decoder': ROOT / 'docs/compatibility/amd-sparse-occupancy-evidence/run',
}


class ZombieAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='_acceptance-cpu-', dir=CONTROL))
        self.evidence = Path(directory) / 'synthetic-historical-json-only'
        self.stack.enter_context(patch.object(subject, 'EVIDENCE', self.evidence))
        self.trusted = {}
        self.baseline = {}
        for stage, historical in HISTORICAL.items():
            result = json.loads((historical / 'results.json').read_bytes())
            identity = json.loads((historical / 'checkpoint-before.json').read_bytes())
            gate = json.loads((historical / 'cpu-gate.json').read_bytes())
            process = result['process']
            process['harness_commit'] = SYNTHETIC_COMMIT
            process['command'] = [str(subject.sampling.forward.local.VENV / 'Scripts/python.exe'), '-B', '-I',
                str(Path(subject.__file__).resolve()), '--worker', '--output',
                str(self.evidence / stage / 'run/scratch')]
            hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in subject.CLOSURE}
            self.baseline[stage] = {
                'results.json': result,
                'checkpoint-before.json': copy.deepcopy(identity),
                'checkpoint-after.json': copy.deepcopy(identity),
                'cpu-gate.json': gate,
                'committed-harness.json': {'commit': SYNTHETIC_COMMIT, 'verified_file_sha256': hashes},
            }
            self.trusted['sampling_identity' if stage == 'sampling' else 'decoder_identity'] = identity
            self.restore(stage)
        self.stack.enter_context(patch.object(subject, 'prepared_contract', return_value=self.trusted))

    def restore(self, stage, values=None):
        values = self.baseline[stage] if values is None else values
        run = self.evidence / stage / 'run'
        run.mkdir(parents=True, exist_ok=True)
        for name, value in values.items():
            (run / name).write_text(json.dumps(value), encoding='utf-8')

    def assert_refused(self, stage, mutate):
        values = copy.deepcopy(self.baseline[stage])
        mutate(values)
        self.restore(stage, values)
        with self.assertRaises(ValueError):
            subject.stage_record(stage, SYNTHETIC_COMMIT)
        self.restore(stage)

    def test_consistent_rewritten_model_identity_cannot_replace_the_trusted_cpu_contract(self):
        for stage in HISTORICAL:
            with self.subTest(stage=stage, case='positive labelled historical fixture'):
                self.assertEqual(subject.stage_record(stage, SYNTHETIC_COMMIT), self.baseline[stage]['results.json'])
            for key in ('checkpoint_sha256', 'config_sha256', 'header_sha256'):
                def alter(values, key=key):
                    for name in ('checkpoint-before.json', 'checkpoint-after.json'):
                        values[name][key] = '0' * 64
                    if stage == 'decoder':
                        values['results.json']['worker']['checkpoint_identity'][key] = '0' * 64
                    if key == 'checkpoint_sha256':
                        values['cpu-gate.json']['checkpoint_sha256'] = '0' * 64
                with self.subTest(stage=stage, key=key):
                    self.assert_refused(stage, alter)

    def test_foreign_runtime_or_changed_native_memory_budget_is_rejected(self):
        for stage in HISTORICAL:
            for key, value in (('executable', str(ROOT / 'foreign-python.exe')), ('torch', '0.0'),
                    ('hip', '0.0'), ('cuda', 'unexpected CUDA build'), ('torch_path', str(ROOT / 'foreign-torch.py'))):
                with self.subTest(stage=stage, runtime=key):
                    self.assert_refused(stage, lambda values, key=key, value=value:
                        values['results.json']['worker']['runtime'].__setitem__(key, value))
            with self.subTest(stage=stage, memory='altered allocator ceiling'):
                self.assert_refused(stage, lambda values:
                    values['results.json']['worker']['budget'].__setitem__('allocator_limit_bytes', 13 * 1024**3))
            with self.subTest(stage=stage, memory='insufficient initial free device bytes'):
                self.assert_refused(stage, lambda values:
                    values['results.json']['worker']['before_placement'].__setitem__('available_bytes', 13 * 1024**3 - 1))

    def test_changed_source_closure_or_forced_settlement_cannot_pass(self):
        for stage in HISTORICAL:
            name = subject.CLOSURE[0]
            with self.subTest(stage=stage, source='wrong source bytes'):
                self.assert_refused(stage, lambda values:
                    values['committed-harness.json']['verified_file_sha256'].__setitem__(name, '0' * 64))
            with self.subTest(stage=stage, source='missing source dependency'):
                self.assert_refused(stage, lambda values:
                    values['committed-harness.json']['verified_file_sha256'].pop(name))
            for key in ('forced_cleanup', 'cleanup_unverified', 'job_close_error'):
                with self.subTest(stage=stage, process=key):
                    self.assert_refused(stage, lambda values, key=key:
                        values['results.json']['process'].__setitem__(key, True))
            with self.subTest(stage=stage, process='forced Job cleanup'):
                self.assert_refused(stage, lambda values:
                    values['results.json']['process']['job_cleanup'].__setitem__('forced_job_cleanup', True))

    def test_actual_cpu_ready_gate_must_match_strict_cpu_loading(self):
        for stage in HISTORICAL:
            for field, value in (('event', 'fabricated-ready'), ('checkpoint_sha256', '0' * 64)):
                with self.subTest(stage=stage, gate=field):
                    self.assert_refused(stage, lambda values, field=field, value=value:
                        values['cpu-gate.json'].__setitem__(field, value))
            for field, value in (('strict', False), ('assign', False), ('gpu_initialized', True)):
                with self.subTest(stage=stage, load=field):
                    self.assert_refused(stage, lambda values, field=field, value=value:
                        values['cpu-gate.json']['gate_one'].__setitem__(field, value))
                def alter_both(values, field=field, value=value):
                    values['cpu-gate.json']['gate_one'][field] = value
                    values['results.json']['worker']['cpu_load'][field] = value
                with self.subTest(stage=stage, paired_load=field):
                    self.assert_refused(stage, alter_both)

    def test_owned_input_authenticates_attachment_before_configuration_and_keeps_real_gpu_line(self):
        for authorization in ('EXECUTE_ONE_12_STEP_SPARSE_SAMPLING_SEQUENCE\n', 'EXECUTE_ONE_DECODER_FORWARD\n'):
            events = []

            class RecordedInput(io.StringIO):
                def readline(self, *args):
                    value = super().readline(*args)
                    events.append(('actual-parent-read', value))
                    return value

            stream = RecordedInput('OWNED_JOB_ATTACHED\n' + authorization)
            owned = subject.OwnedInput(stream)
            self.assertEqual(events, [('actual-parent-read', 'OWNED_JOB_ATTACHED\n')])
            events.append(('CPU-configuration-now-contained', None))
            self.assertEqual(owned.readline(), 'OWNED_JOB_ATTACHED\n')
            self.assertEqual(len(events), 2, 'The real attachment line must be replayed once without consuming GPU authorization')
            self.assertEqual(owned.readline(), authorization)
            self.assertEqual(events[-1], ('actual-parent-read', authorization))
            self.assertEqual(owned.readline(), '')
        for invalid in ('', 'FAKE\n', 'OWNED_JOB_ATTACHED', 'EXECUTE_ONE_DECODER_FORWARD\n'):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, RuntimeError)):
                    subject.OwnedInput(io.StringIO(invalid))
        self.assertFalse(any(name in sys.modules for name in ('torch', 'transformers', 'triton', 'trellis2')))


if __name__ == '__main__':
    unittest.main()
