"""CPU-only bounds and evidence regressions for per-run shape sampling."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from test_amd_shape_sampling_result import candidate


class GenerationShapeTests(unittest.TestCase):
    def adapter(self):
        return importlib.import_module('amd_generation_shape')

    def test_dynamic_count_is_bounded_by_the_measured_4430_input(self):
        adapter = self.adapter()
        for value in (1, 3, 3743, 4430):
            self.assertEqual(adapter.validate_count(value), value)
        for value in (0, -1, 4431, 32768, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.validate_count(value)

    def test_pooling_uses_positive_cells_and_exact_two_cell_max_pool(self):
        adapter = self.adapter()
        logits = np.full((1, 1, 64, 64, 64), -1, dtype=np.float32)
        logits[0, 0, 1, 1, 1] = 1
        logits[0, 0, 2, 4, 6] = 2
        coords = adapter.coordinates(logits, logits[0, 0] > 0)
        np.testing.assert_array_equal(coords, np.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=np.int32))
        with self.assertRaises(ValueError):
            adapter.coordinates(logits, np.zeros((64, 64, 64), dtype=bool))
        with self.assertRaises(ValueError):
            adapter.coordinates(np.ones_like(logits), np.ones((64, 64, 64), dtype=bool))

    def test_fresh_count_classifier_checks_exact_npy_sizes_and_lineage(self):
        adapter = self.adapter()
        row = candidate()
        row['token_count'] = 3
        row['output'].update(shape=[3, 32], count=96)
        for name in ('shape-latent-raw.npy', 'shape-latent-denormalized.npy', 'shape-initial-noise.npy'):
            row['artifacts'][name].update(shape=[3, 32], file_bytes=512)
        row['artifacts']['shape-latent-coordinates.npy'].update(shape=[3, 4], file_bytes=176)
        row['generation_inputs'] = {'schema': 'modly.amd.generation-shape-inputs.v1', 'token_count': 3,
                                    'input_contract_sha256': 'b' * 64, 'coords_payload_sha256': 'a' * 64,
                                    'feature_file_sha256': 'c' * 64, 'occupancy_provenance_sha256': 'd' * 64}
        self.assertEqual(adapter.classify(row, True, True, True), 'PASS')
        for name, field, value in [('shape-initial-noise.npy', 'file_bytes', 479232),
                                   ('shape-latent-coordinates.npy', 'shape', [3743, 4]),
                                   ('shape-latent-raw.npy', 'payload_sha256', 'e' * 64)]:
            bad = copy.deepcopy(row)
            bad['artifacts'][name][field] = value
            self.assertEqual(adapter.classify(bad, True, True, True), 'PARTIAL')
        bad = copy.deepcopy(row)
        bad['generation_inputs']['coords_payload_sha256'] = 'f' * 64
        self.assertEqual(adapter.classify(bad, True, True, True), 'PARTIAL')
        self.assertEqual(adapter.classify(row, False, True, True), 'PARTIAL')

    def test_historical_classifier_default_remains_exact_3743(self):
        import amd_shape_sampling_contract as historical
        self.assertEqual(historical.classify(candidate(), True, True, True), 'PASS')
        row = candidate()
        row['output'].update(shape=[3, 32], count=96)
        self.assertEqual(historical.classify(row, True, True, True), 'PARTIAL')

    def test_context_rejects_historical_and_overlapping_evidence(self):
        adapter = self.adapter()
        root = adapter.APPROVED_ROOT / 'new-image'
        context = adapter.context(root, root / 'shape', '1' * 40)
        self.assertEqual(context['occupancy_evidence'], str(root.resolve()))
        self.assertEqual(context['evidence'], str((root / 'shape').resolve()))
        for occupancy, evidence, commit in [(root, adapter.ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence', '1' * 40),
                                             (root, root, '1' * 40), (root, root / 'shape', 'main')]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                adapter.context(occupancy, evidence, commit)

    def test_pass_release_gate_rejects_cleanup_and_observer_forgery(self):
        adapter = self.adapter()
        good = {'classification': 'PASS', 'observer_healthy': True, 'checkpoint_unchanged': True,
                'worker': {'status': 'PASS', 'stage': 'complete',
                           'release': {'allocated_bytes': 0, 'reserved_bytes': 0, 'model_collected': True,
                                       'owned_tensors_collected': True}},
                'process': {'returncode': 0, 'timed_out': False, 'worker_exit_verified': True,
                            'gpu_authorized': True, 'output_readers_completed': True,
                            'cpu_observer_stopped': True, 'windows_periodic_stopped': True,
                            'diagnostic_journal_stopped': True, 'windows_journal_stopped': True,
                            'job_attached_before_preparation': True,
                            'job_cleanup': {'owned_tree_exit_verified': True, 'forced_job_cleanup': False}}}
        self.assertEqual(adapter.require_released(good), good['worker'])
        for path, value in [(('observer_healthy',), False), (('process', 'forced_cleanup'), True),
                            (('process', 'job_cleanup', 'forced_job_cleanup'), True),
                            (('worker', 'release', 'allocated_bytes'), 1),
                            (('process', 'worker_exit_verified'), 1)]:
            bad = copy.deepcopy(good)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                adapter.require_released(bad)

    def test_fresh_input_rows_reject_wrong_conditioning_coordinates_and_layout(self):
        adapter = self.adapter()
        coords = np.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=np.int32)
        digest = __import__('hashlib').sha256(coords.tobytes()).hexdigest()
        feature = {'output_sha256': 'b' * 64}
        lineage = {'coords_payload_sha256': digest, 'token_count': 2}
        rows = {'coords': {'shape': [2, 4], 'dtype': 'torch.int32', 'device': 'cpu', 'sha256': digest},
                'cond': {'shape': [1, 1029, 1024], 'dtype': 'torch.float32', 'device': 'cpu', 'sha256': 'b' * 64},
                'x': {'shape': [2, 32], 'dtype': 'torch.float32', 'device': 'cpu', 'sha256': 'c' * 64}}
        self.assertEqual(adapter.validate_input_rows(rows, feature, coords, lineage), lineage)
        for key, field, value in [('coords', 'sha256', 'd' * 64), ('cond', 'sha256', 'e' * 64),
                                   ('x', 'shape', [3743, 32]), ('x', 'dtype', 'torch.bfloat16')]:
            bad = copy.deepcopy(rows)
            bad[key][field] = value
            with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                adapter.validate_input_rows(bad, feature, coords, lineage)

    def test_worker_context_is_bound_to_the_exact_owned_scratch(self):
        adapter = self.adapter()
        root = adapter.APPROVED_ROOT / 'new-image'
        value = adapter.context(root, root / 'shape', '1' * 40)
        self.assertEqual(adapter.worker_context(json.dumps(value), root / 'shape/run/scratch'), value)
        with self.assertRaises(ValueError):
            adapter.worker_context(json.dumps(value), root / 'dino/run/scratch')
        with self.assertRaises(ValueError):
            adapter.worker_context(json.dumps({**value, 'unreviewed': True}), root / 'shape/run/scratch')

    def test_sampler_control_uses_job_count_without_increasing_qualified_cap(self):
        from amd_shape_sampling_control import ShapeSamplingControl
        self.assertEqual(ShapeSamplingControl(object(), object()).token_count, 3743)
        self.assertEqual(ShapeSamplingControl(object(), object(), token_count=2).token_count, 2)
        for count in (0, 3744, True):
            with self.assertRaises(ValueError):
                ShapeSamplingControl(object(), object(), token_count=count)
        self.assertEqual(ShapeSamplingControl(object(), object(), token_count=4430, maximum_tokens=4430).token_count, 4430)
        with self.assertRaises(ValueError):
            ShapeSamplingControl(object(), object(), token_count=4431, maximum_tokens=4430)

    def test_measured_4430_noise_and_artifacts_roundtrip_on_cpu_only(self):
        import torch
        from unittest.mock import patch
        import amd_shape_sampling_artifact as artifact
        adapter = self.adapter()
        with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU prohibited')):
            noise = torch.randn(4430, 32, generator=torch.Generator(device='cpu').manual_seed(20261013),
                                dtype=torch.float32, device='cpu').numpy()
            repeat = torch.randn(4430, 32, generator=torch.Generator(device='cpu').manual_seed(20261013),
                                 dtype=torch.float32, device='cpu').numpy()
            np.testing.assert_array_equal(noise, repeat)
            normalized = artifact.denormalize(noise, np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32), maximum_tokens=4430)
            xyz = np.array(np.unravel_index(np.arange(4430), (32, 32, 32))).T.astype(np.int32)
            coords = np.concatenate((np.zeros((4430, 1), dtype=np.int32), xyz), axis=1)
            with tempfile.TemporaryDirectory(prefix='_shape4430-cpu-test-', dir=adapter.ROOT / 'scripts/compatibility') as directory:
                rows = artifact.write(Path(directory), noise, normalized, coords, {'synthetic': True, 'seed': 20261013},
                                      noise=noise, maximum_tokens=4430)
                for name in ('shape-latent-raw.npy', 'shape-latent-denormalized.npy', 'shape-initial-noise.npy'):
                    self.assertEqual(rows[name]['file_bytes'], 567168)
                    np.testing.assert_array_equal(np.load(rows[name]['path'], allow_pickle=False), noise)
                self.assertEqual(rows['shape-latent-coordinates.npy']['file_bytes'], 71008)
                np.testing.assert_array_equal(np.load(rows['shape-latent-coordinates.npy']['path'], allow_pickle=False), coords)
            with self.assertRaises(ValueError):
                artifact.validate(noise)
            with self.assertRaises(ValueError):
                artifact.validate(np.zeros((4431, 32), dtype=np.float32), maximum_tokens=4430)
            self.assertFalse(torch.cuda.is_initialized())

    def test_resource_qualification_keeps_live_caps_and_marks_estimate_unproven(self):
        adapter = self.adapter()
        row = adapter.resource_qualification(4430)
        self.assertEqual(row['token_count'], 4430)
        self.assertEqual(row['historical_token_count'], 3743)
        self.assertEqual(row['allocator_limit_bytes'], 12884901888)
        self.assertEqual(row['minimum_free_vram_bytes'], 13958643712)
        self.assertEqual(row['vram_reserve_bytes'], 1073741824)
        self.assertEqual(row['minimum_live_free_vram_bytes'], 3221225472)
        self.assertEqual(row['self_attention_scores_f32_bytes'], 941995200)
        self.assertEqual((row['cpu_seconds'], row['gpu_seconds'], row['overall_seconds']), (180, 240, 480))
        self.assertGreater(row['whole_peak_quadratic_estimate_bytes'], 4300000000)
        self.assertLess(row['whole_peak_quadratic_estimate_bytes'], 4400000000)
        self.assertFalse(row['estimate_is_guarantee'])
        with self.assertRaises(ValueError):
            adapter.resource_qualification(4431)

    def test_4430_classifier_requires_current_qualification_and_exact_sizes(self):
        adapter = self.adapter()
        row = candidate()
        row['token_count'] = 4430
        row['output'].update(shape=[4430, 32], count=141760)
        for name in ('shape-latent-raw.npy', 'shape-latent-denormalized.npy', 'shape-initial-noise.npy'):
            row['artifacts'][name].update(shape=[4430, 32], file_bytes=567168)
        row['artifacts']['shape-latent-coordinates.npy'].update(shape=[4430, 4], file_bytes=71008)
        row['generation_inputs'] = {'schema': adapter.SCHEMA, 'token_count': 4430,
            'input_contract_sha256': 'b' * 64, 'coords_payload_sha256': 'a' * 64,
            'feature_file_sha256': 'c' * 64, 'occupancy_provenance_sha256': 'd' * 64,
            'resource_qualification': adapter.resource_qualification(4430)}
        row['sampling'].update(minimum_observed_free_vram_bytes=3221225472, minimum_required_free_vram_bytes=3221225472)
        self.assertEqual(adapter.classify(row, True, True, True), 'PASS')
        import amd_shape_sampling_contract as historical
        self.assertEqual(historical.classify(row, True, True, True, token_count=4430), 'PARTIAL')
        bad = copy.deepcopy(row)
        del bad['generation_inputs']['resource_qualification']
        self.assertEqual(adapter.classify(bad, True, True, True), 'PARTIAL')
        bad = copy.deepcopy(row)
        bad['generation_inputs']['resource_qualification']['allocator_limit_bytes'] += 1
        self.assertEqual(adapter.classify(bad, True, True, True), 'PARTIAL')
        bad = copy.deepcopy(row)
        bad['token_count'] = 4431
        self.assertEqual(adapter.classify(bad, True, True, True), 'PARTIAL')

    def test_adapter_live_free_guard_keeps_two_plus_one_gib_reserve(self):
        adapter = self.adapter()
        control = adapter.QualifiedShapeControl(object(), object(), token_count=4430, maximum_tokens=4430)
        memory = dict(allocated_bytes=1, reserved_bytes=1, peak_allocated_bytes=1, peak_reserved_bytes=1,
                      available_bytes=3221225472)
        control.block_memory(memory)
        self.assertEqual(control.report()['minimum_observed_free_vram_bytes'], 3221225472)
        for available in (3221225471, 0, True):
            with self.subTest(available=available), self.assertRaises(RuntimeError):
                control.block_memory({**memory, 'available_bytes': available})

    def test_worker_construction_defers_input_reads_until_owned_handshake(self):
        adapter = self.adapter()
        root = adapter.APPROVED_ROOT / 'new-image'
        runtime = adapter.Runtime(adapter.context(root, root / 'shape', '1' * 40))
        self.assertIsNone(runtime._bundle)
        self.assertEqual(runtime.local.FEATURE, root / 'dino/run/scratch/conditioning.safetensors')

    def test_dynamic_count_runs_genuine_sampler_and_original_guards_on_cpu(self):
        import gc
        import torch
        from unittest.mock import patch
        import test_amd_shape_sampling_runtime as existing
        from amd_shape_sampling_control import ShapeSamplingControl, PHASE_KEY
        from trellis2.modules.sparse import SparseTensor
        from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

        class FreshRope(existing.Rope):
            def forward(self, x):
                if x.get_spatial_cache(PHASE_KEY) is None:
                    torch.ops.aten.to.dtype_layout(self.freqs, device=torch.device('cuda:0'))
                    angles = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64) * .001
                    x.register_spatial_cache(PHASE_KEY, torch.polar(torch.ones_like(angles), angles))
                return x

        model = existing.Fixture()
        for block in model.blocks:
            block.self_attn.rope = FreshRope()
        control = ShapeSamplingControl(torch, FlowEulerGuidanceIntervalSampler, device='cpu', token_count=2)
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.int32)
        noise = torch.randn(2, 32, generator=torch.Generator().manual_seed(20261013))
        sparse = SparseTensor(noise, coords)
        sparse.register_spatial_cache('layout', [slice(0, 2)])
        sparse.register_spatial_cache('seqlen', torch.tensor([2], dtype=torch.int64))
        inputs = dict(cond=torch.ones(1, 1029, 1024) * .25, neg_cond=torch.zeros(1, 1029, 1024))
        trace = existing.diagnostics.Trace(lambda line: None, limit=192)
        hooks, memory = existing.hooks(model, control, trace)
        hooks['policy'] = control.copy_policy(2)
        hooks['policy'].phase = 'forward'
        hooks['policy'].rope = model.blocks[0].self_attn.rope.freqs
        with torch.inference_mode(), existing.CPUSink(), hooks['BoundedDispatch'](), \
                patch.object(torch.nn.Module, '__call__', hooks['module_call']), patch.object(torch, 'tensor', hooks['timestep_tensor']):
            result = control.run(model, sparse, inputs, trace, lambda: None, memory)
        self.assertEqual(control.budget.calls, 21)
        self.assertEqual([row['forwards'] for row in control.steps], [2] * 9 + [1] * 3)
        self.assertEqual(list(result.samples.feats.shape), [2, 32])
        self.assertIs(result.samples.coords, coords)
        self.assertTrue(torch.isfinite(result.samples.feats).all())
        self.assertFalse(torch.cuda.is_initialized())
        result.samples.clear_spatial_cache()
        control.drop_references()
        del result, model, sparse, inputs
        gc.collect()

    def test_input_bundle_reopens_synthetic_chain_and_rejects_forged_natural_exit(self):
        adapter = self.adapter()
        # The fixture uses exclusive files and genuine artifact readers. Its
        # receipts explicitly describe synthetic CPU data, never model output.
        from test_amd_generation_occupancy import GenerationOccupancyTests
        helper = GenerationOccupancyTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.fixture_sequence()
        for stage in ('dino', 'sampling', 'decoder'):
            path = helper.evidence / stage / 'run/results.json'
            row = json.loads(path.read_text())
            row['worker'].update(status='PASS', stage='complete')
            row['worker']['release'].update(model_collected=True, owned_tensors_collected=True,
                input_collected=True, output_collected=True, rope_cache_entries_after=0)
            row['process'].update(timed_out=False, gpu_authorized=True, output_readers_completed=True,
                cpu_observer_stopped=True, windows_periodic_stopped=True, diagnostic_journal_stopped=True,
                windows_journal_stopped=True, job_attached_before_preparation=True)
            path.write_text(json.dumps(row))
        bundle = adapter.input_bundle(helper.evidence)
        self.assertEqual(bundle['lineage']['token_count'], 64)
        np.testing.assert_array_equal(bundle['coords'][[0, -1]], np.array([[0, 0, 0, 0], [0, 3, 3, 3]], dtype=np.int32))
        path = helper.evidence / 'sampling/run/results.json'
        row = json.loads(path.read_text())
        row['process']['forced_cleanup'] = True
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            adapter.input_bundle(helper.evidence)

    def test_profile_retains_qualified_budgets_and_only_commits_static_fresh_inputs(self):
        from unittest.mock import patch
        from test_amd_generation_occupancy import GenerationOccupancyTests
        adapter = self.adapter()
        helper = GenerationOccupancyTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.prepare()
        with patch.object(adapter, 'APPROVED_ROOT', helper.base.parent):
            runtime = adapter.Runtime(adapter.context(helper.evidence, helper.evidence / 'shape', '1' * 40))
            profile = runtime.profile()
        self.assertEqual(profile.worker_script, Path(adapter.__file__).resolve())
        self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds), (180, 240, 480))
        self.assertEqual((profile.maximum_sequences, profile.maximum_steps, profile.maximum_forwards), (1, 12, 21))
        self.assertIn((helper.evidence / 'input-contract.json').relative_to(adapter.ROOT).as_posix(), profile.committed_files)
        fresh_prefix = helper.evidence.relative_to(adapter.ROOT).as_posix() + '/'
        self.assertFalse(any(name.startswith(fresh_prefix) and '/run/' in name for name in profile.committed_files))
        self.assertIsNone(runtime._bundle)

    def test_duplicate_json_identity_fields_are_rejected(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory(prefix='_shape-json-test-', dir=adapter.ROOT / 'scripts/compatibility') as directory:
            path = Path(directory) / 'receipt.json'
            path.write_text('{"classification":"PARTIAL","classification":"PASS"}')
            with self.assertRaises(ValueError):
                adapter._json(path)


if __name__ == '__main__':
    unittest.main()
