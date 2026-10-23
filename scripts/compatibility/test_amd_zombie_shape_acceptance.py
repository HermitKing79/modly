"""CPU rejection tests using explicitly labelled historical Shape-SLat fixtures.

Temporary copies exercise acceptance only. They are never new neural receipts,
never overwrite authentic roots, and never import or execute models or GPUs.
"""
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
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'triton', 'transformers',
                'trellis2', 'flex_gemm', 'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime'}:
            raise AssertionError('CPU acceptance forbids neural import: ' + fullname)


sys.meta_path.insert(0, NoModels())
import numpy as np
import amd_zombie_shape as subject

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT.parent / 'artifacts/amd-zombie-shape-0d15dd9'
HISTORICAL = ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence'
SYNTHETIC_COMMIT = '2' * 40


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class ZombieShapeAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        CONTROL.mkdir(parents=True, exist_ok=True)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='_historical-json-only-', dir=CONTROL))
        self.evidence = Path(temporary) / 'labelled-synthetic-acceptance-fixture'
        self.run = self.evidence / 'run'
        self.scratch = self.run / 'scratch'
        self.scratch.mkdir(parents=True)
        self.stack.enter_context(patch.object(subject, 'EVIDENCE', self.evidence))
        closure = ['scripts/compatibility/test_amd_zombie_shape_acceptance.py']
        self.stack.enter_context(patch.object(subject, 'CLOSURE', closure))
        result = json.loads((HISTORICAL / 'run/results.json').read_bytes())
        worker = result['worker']
        source_runtime = subject.Runtime()
        trusted = json.loads((HISTORICAL / 'run/checkpoint-before.json').read_bytes())
        coords = np.load(HISTORICAL / 'shape-latent-coordinates.npy', allow_pickle=False)
        self.coords = coords
        feature = worker['provenance']['conditioning']
        self.lineage = dict(schema=subject.generation.SCHEMA, token_count=len(coords),
            input_contract_sha256='b' * 64, coords_payload_sha256=sha(coords.tobytes()),
            feature_file_sha256=feature['file_sha256'], occupancy_provenance_sha256='d' * 64,
            resource_qualification=subject.generation.resource_qualification(len(coords)))
        bundle = dict(lineage={k: v for k, v in self.lineage.items() if k != 'resource_qualification'},
            feature=feature, coords=coords, logits_sha256=worker['provenance']['occupancy_logits_sha256'],
            mask_sha256=worker['provenance']['occupancy_mask_sha256'])
        self.normalization = json.loads(subject.historical.PINS.read_bytes())['normalization']
        self.runtime = SimpleNamespace(local=source_runtime.local, bundle=bundle,
            verify_inputs=lambda: (feature, None, coords),
            verify_sampler=lambda: {'normalization': self.normalization}, sources=source_runtime.sources)
        self.stack.enter_context(patch.object(subject, 'Runtime', return_value=self.runtime))
        self.stack.enter_context(patch.object(subject, 'prepared_contract', return_value={'identity': trusted}))
        worker['generation_inputs'] = copy.deepcopy(self.lineage)
        worker['sampling'].update(minimum_required_free_vram_bytes=subject.generation.LIVE_FREE_RESERVE,
                                  minimum_observed_free_vram_bytes=subject.generation.LIVE_FREE_RESERVE)
        result['process']['harness_commit'] = SYNTHETIC_COMMIT
        result['process']['command'] = [str(source_runtime.local.VENV / 'Scripts/python.exe'), '-B', '-I',
            str(Path(subject.__file__).resolve()), '--worker', '--output', str(self.scratch)]
        for name, row in worker['artifacts'].items():
            target = self.scratch / name
            target.write_bytes((HISTORICAL / name).read_bytes())
            row['path'] = str(target)
        self.values = {'results.json': result,
            'checkpoint-before.json': copy.deepcopy(trusted), 'checkpoint-after.json': copy.deepcopy(trusted),
            'cpu-gate.json': json.loads((HISTORICAL / 'run/cpu-gate.json').read_bytes()),
            'committed-harness.json': dict(commit=SYNTHETIC_COMMIT,
                verified_file_sha256={name: sha((ROOT / name).read_bytes()) for name in closure})}
        self.refresh_sidecar(self.values)
        self.baseline = copy.deepcopy(self.values)
        self.write(self.values)

    def refresh_sidecar(self, values):
        worker = values['results.json']['worker']
        sequence = copy.deepcopy(worker['sampling'])
        sequence['models_sha256'] = sha(json.dumps(sequence.pop('models'), sort_keys=True).encode())
        provenance = dict(harness_commit=SYNTHETIC_COMMIT, checkpoint=worker['provenance'],
            input=worker['input'], parameters=subject.historical.policy.PARAMS,
            sigma_min=subject.historical.policy.SIGMA_MIN, seed=subject.historical.policy.SEED,
            sampler_sources=self.runtime.sources(), normalization=self.normalization, sequence=sequence,
            raw_output=worker['output'], generation_inputs=self.lineage)
        blob = json.dumps(provenance, indent=2).encode()
        path = self.scratch / 'shape-latent-provenance.json'
        path.write_bytes(blob)
        worker['artifacts'][path.name].update(file_bytes=len(blob), sha256=sha(blob))

    def write(self, values):
        for name, value in values.items():
            (self.run / name).write_text(json.dumps(value), encoding='utf-8')

    def assert_refused(self, mutate, *, matching_sidecar=False):
        values = copy.deepcopy(self.baseline)
        mutate(values)
        if matching_sidecar:
            self.refresh_sidecar(values)
        self.write(values)
        with self.assertRaises(ValueError):
            subject.verify_saved(SYNTHETIC_COMMIT)

    def test_labelled_historical_positive_fixture_reads_all_five_actual_files(self):
        result = subject.verify_saved(SYNTHETIC_COMMIT)
        self.assertEqual(result['classification'], 'PASS')
        self.assertEqual(result['readback']['token_count'], len(self.coords))
        self.assertEqual(len(result['readback']['files']), 5)

    def test_exact_f32_denormalization_rejects_rehashed_signed_zero_change(self):
        scratch = self.evidence / 'labelled-synthetic-signed-zero'
        scratch.mkdir()
        raw = np.zeros((1, 32), dtype=np.float32)
        coords = self.coords[:1].copy()
        normalization = {'mean': [0.0] * 32, 'std': [1.0] * 32}
        provenance = {'synthetic_cpu_float_fixture': True}
        rows = subject.artifact.write(scratch, raw, raw.copy(), coords, provenance,
                                      noise=raw.copy(), maximum_tokens=4430)
        altered = raw.copy()
        altered[0, 0] = np.float32(-0.0)
        path = scratch / 'shape-latent-denormalized.npy'
        with path.open('wb') as stream:
            np.save(stream, altered, allow_pickle=False)
        blob = path.read_bytes()
        rows[path.name].update(file_bytes=len(blob), sha256=sha(blob), payload_sha256=sha(altered.tobytes()))
        self.assertTrue(np.array_equal(raw, altered), 'This case specifically differs only in F32 bits')
        with self.assertRaises(ValueError):
            subject.verify_arrays(rows, scratch, coords, normalization, provenance)

    def test_rehashed_checkpoint_semantics_and_extra_fields_cannot_replace_original_contract(self):
        for key, value in [('pooling', 'altered occupancy pooling'),
                           ('sparse_attention', 'altered attention backend'),
                           ('sparse_convolution', 'altered convolution backend'),
                           ('unreviewed_checkpoint_field', True)]:
            with self.subTest(key=key):
                self.assert_refused(lambda values, key=key, value=value:
                    values['results.json']['worker']['provenance'].__setitem__(key, value), matching_sidecar=True)

    def test_rehashed_input_schema_layout_or_device_changes_are_rejected(self):
        for name, field, value in [('x', 'dtype', 'torch.float64'), ('x', 'shape', [1, 32]),
                                  ('coords', 'device', 'cuda:0'), ('cond', 'shape', [1, 1, 1024]),
                                  ('neg_cond', 'dtype', 'torch.bfloat16'), ('batch_map', 'shape', [1]),
                                  ('seqlen', 'device', 'cuda:0')]:
            with self.subTest(name=name, field=field):
                self.assert_refused(lambda values, name=name, field=field, value=value:
                    values['results.json']['worker']['input'][name].__setitem__(field, value), matching_sidecar=True)
        with self.subTest(extra='unreviewed input'):
            self.assert_refused(lambda values:
                values['results.json']['worker']['input'].__setitem__('foreign', {'sha256': 'a' * 64}), matching_sidecar=True)

    def test_original_negative_conditioning_batch_map_and_seqlen_hashes_are_required(self):
        for name in ('neg_cond', 'batch_map', 'seqlen'):
            with self.subTest(name=name):
                self.assert_refused(lambda values, name=name:
                    values['results.json']['worker']['input'][name].__setitem__('sha256', '0' * 64), matching_sidecar=True)

    def test_cpu_gate_must_be_strict_even_when_worker_and_gate_are_changed_together(self):
        for key, value in [('strict', False), ('assign', False), ('gpu_initialized', True),
                           ('missing_keys', ['missing']), ('unexpected_keys', ['foreign']),
                           ('payload_entries', 639), ('reversible_widening_count', 219)]:
            for both in (False, True):
                def alter(values, key=key, value=value, both=both):
                    values['cpu-gate.json']['gate_one'][key] = value
                    if both:
                        values['results.json']['worker']['cpu_load'][key] = value
                with self.subTest(key=key, paired=both):
                    self.assert_refused(alter)

    def test_trusted_preparation_rejects_consistently_changed_checkpoint_identities(self):
        for key in ('checkpoint_sha256', 'config_sha256', 'header_sha256'):
            def alter(values, key=key):
                for name in ('checkpoint-before.json', 'checkpoint-after.json'):
                    values[name][key] = '0' * 64
                if key == 'checkpoint_sha256':
                    values['cpu-gate.json']['checkpoint_sha256'] = '0' * 64
            with self.subTest(key=key):
                self.assert_refused(alter)

    def test_runtime_source_closure_memory_deadlines_and_natural_settlement_remain_required(self):
        for key, value in [('executable', str(ROOT / 'foreign-python.exe')), ('torch', '0.0'),
                           ('hip', '0.0'), ('cuda', 'foreign CUDA build'), ('git_version', 'foreign')]:
            with self.subTest(runtime=key):
                self.assert_refused(lambda values, key=key, value=value:
                    values['results.json']['worker']['runtime'].__setitem__(key, value))
        for key in ('forced_cleanup', 'cleanup_unverified', 'job_cleanup_error', 'job_close_error'):
            with self.subTest(process=key):
                self.assert_refused(lambda values, key=key:
                    values['results.json']['process'].__setitem__(key, True))
        for mapping, key, value in [('budget', 'allocator_limit_bytes', 13 * 1024**3),
                ('before_placement', 'available_bytes', 13 * 1024**3 - 1),
                ('ram_before_strict_load', 'available_bytes', 1),
                ('sampling', 'minimum_observed_free_vram_bytes', subject.generation.LIVE_FREE_RESERVE - 1)]:
            with self.subTest(mapping=mapping, field=key):
                self.assert_refused(lambda values, mapping=mapping, key=key, value=value:
                    values['results.json']['worker'][mapping].__setitem__(key, value))
        with self.subTest(deadline='changed GPU deadline'):
            self.assert_refused(lambda values:
                values['results.json']['process']['stage_timeouts_seconds'].__setitem__('forward_release', 241))
        with self.subTest(source='missing committed dependency'):
            self.assert_refused(lambda values: values['committed-harness.json']['verified_file_sha256'].clear())
        with self.subTest(job='forced settlement'):
            self.assert_refused(lambda values:
                values['results.json']['process']['job_cleanup'].__setitem__('forced_job_cleanup', True))


if __name__ == '__main__':
    unittest.main()
