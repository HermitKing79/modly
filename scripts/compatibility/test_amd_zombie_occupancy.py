"""CPU-only lineage/routing tests; generated latents are synthetic test data."""
import ast
import copy
from contextlib import ExitStack
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'transformers', 'trellis2', 'flex_gemm',
                'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime'}:
            raise AssertionError('CPU continuation imported a model/GPU package: ' + fullname)


sys.meta_path.insert(0, NoModels())
try:
    import amd_zombie_occupancy as subject
except ModuleNotFoundError as error:
    if error.name != 'amd_zombie_occupancy':
        raise
    subject = None

import amd_feature_checkpoint as features
import amd_sparse_sampling as sampling
import amd_sparse_latent_checkpoint as latents
import amd_sparse_occupancy as occupancy

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT.parent / 'artifacts/amd-zombie-occupancy-c570c2e'
OLD = ROOT / 'artifacts/amd-zombie-dinov3-b8afdd5/dino'
OLD_COMMIT = '93269412602a338acd9fbde41798901ddb5f707d'
NEW_COMMIT = '2' * 40
FEATURE_SHA = 'e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60'


class ZombieOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'The saved-zombie sparse continuation adapter is missing')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        # configure deliberately rebinds these process-local routing attributes.
        for owner, name in ((sampling.forward.local, 'FEATURE'), (sampling.forward.local, 'FEATURE_SHA'),
                (sampling.forward.local, 'verify_feature'), (sampling.forward.local, 'EVIDENCE'),
                (sampling.forward, 'verify_preflight'), (occupancy.data, 'INPUT'),
                (occupancy.data, 'INPUT_SHA'), (occupancy.data, 'read_input'),
                (occupancy, 'EVIDENCE'), (occupancy.base, 'EVIDENCE')):
            self.stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        self.feature = json.loads((OLD / 'host-readback.json').read_text())

    def synthetic_latent(self):
        """A real bounded container with synthetic CPU values, never neural evidence."""
        import numpy as np
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='_latent-cpu-', dir=CONTROL))
        scratch = Path(directory) / 'sampling/run/scratch'
        scratch.mkdir(parents=True)
        identity = dict(checkpoint_sha256='1' * 64, config_sha256='3' * 64)
        runtime = {'scope': 'synthetic CPU provenance fixture; no model execution'}
        model_source = sampling.forward.local.SOURCE / 'trellis2/models/sparse_structure_flow.py'
        provenance = dict(component='SparseStructureFlowModel', model_checkpoint_sha256=identity['checkpoint_sha256'],
            model_config_sha256=identity['config_sha256'], model_source_sha256=hashlib.sha256(model_source.read_bytes()).hexdigest(),
            sampler_sources=json.dumps(sampling.sources(), sort_keys=True),
            params=json.dumps(dict(steps=12, rescale_t=5.0, guidance_strength=7.5, guidance_rescale=.7,
                guidance_interval=[.6, 1.0], dino_lock=0.0), sort_keys=True),
            seed='20261013', sigma_min='1e-05', noise_sha256='a' * 64,
            feature_file_sha256=FEATURE_SHA, feature_provenance=json.dumps(self.feature['provenance'], sort_keys=True),
            harness_commit=NEW_COMMIT, runtime=json.dumps(runtime, sort_keys=True))
        array = np.arange(32768, dtype='<f4').reshape((1, 8, 16, 16, 16)) / np.float32(32768)
        row = latents.write(scratch / 'sparse-structure-latent.safetensors', scratch, array, provenance)
        return row, array, identity, runtime

    def validate_provenance(self, row, identity, runtime):
        return subject.validate_latent_provenance(row['provenance'], self.feature, NEW_COMMIT, identity, runtime)

    def test_real_saved_zombie_feature_reopens_with_original_execution_identity(self):
        row = subject.saved_features()
        self.assertEqual(row, self.feature)
        self.assertEqual(row['file_sha256'], FEATURE_SHA)
        self.assertEqual(row['provenance']['harness_commit'], OLD_COMMIT)
        self.assertEqual(row['provenance']['input_png_sha256'],
            'db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e')
        self.assertEqual((row['shape'], row['dtype'], row['tensor_bytes']), ([1, 1029, 1024], 'F32', 4214784))
        path = OLD / 'run/scratch/conditioning.safetensors'
        expected = {key: self.feature['provenance'][key] for key in features.FIELDS}
        self.assertEqual(features.verify(path, path.parent, expected, self.feature['output_sha256']), row)
        self.assertFalse(any(name in sys.modules for name in ('torch', 'transformers', 'trellis2', 'triton')))

    def test_stage_bindings_preserve_two_real_execution_commits(self):
        self.assertEqual(subject.EVIDENCE, ROOT / 'artifacts/amd-zombie-occupancy-c570c2e')
        self.assertEqual(subject.DINO_COMMIT, OLD_COMMIT)
        self.assertEqual(subject.FEATURE_SHA, FEATURE_SHA)
        self.assertEqual(subject.STAGES, ('sampling', 'decoder'))
        row, _, identity, runtime = self.synthetic_latent()
        self.assertIsNone(self.validate_provenance(row, identity, runtime))
        self.assertEqual(row['provenance']['harness_commit'], NEW_COMMIT)
        self.assertEqual(json.loads(row['provenance']['feature_provenance'])['harness_commit'], OLD_COMMIT)

    def test_sampling_configuration_retains_exact_original_budget_and_authorisation(self):
        before = sampling.profile()
        profile = subject.configure('sampling', NEW_COMMIT)
        self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds), (180, 240, 480))
        self.assertEqual((profile.maximum_sequences, profile.maximum_steps, profile.maximum_forwards), (1, 12, 22))
        self.assertEqual(profile.authorize, b'EXECUTE_ONE_12_STEP_SPARSE_SAMPLING_SEQUENCE\n')
        self.assertEqual(profile.kind, 'sparse-sampling')
        self.assertIs(profile.classify, before.classify)
        self.assertIs(profile.authorize_placement, before.authorize_placement)
        self.assertEqual(profile.evidence, subject.EVIDENCE / 'sampling')
        self.assertEqual(profile.worker_script, Path(subject.__file__).resolve())
        self.assertEqual(sampling.policy.MEMORY_LIMIT, 12 * 1024**3)
        self.assertEqual(sampling.policy.PARAMS, dict(steps=12, rescale_t=5.0, guidance_strength=7.5,
            guidance_rescale=.7, guidance_interval=(.6, 1.0), dino_lock=0.0))
        self.assertEqual((sampling.policy.SEED, sampling.policy.SIGMA_MIN), (20261013, 1e-5))

    def test_sampling_routes_original_feature_reader_without_rewriting_dino_commit(self):
        subject.configure('sampling', NEW_COMMIT)
        local = sampling.forward.local
        self.assertEqual(local.FEATURE, OLD / 'run/scratch/conditioning.safetensors')
        self.assertEqual(local.FEATURE_SHA, FEATURE_SHA)
        self.assertEqual(local.EVIDENCE, subject.EVIDENCE / 'sampling')
        self.assertEqual(local.verify_feature(), self.feature)
        self.assertEqual(local.verify_feature()['provenance']['harness_commit'], OLD_COMMIT)
        self.assertEqual(os.environ['MODLY_SAMPLING_HARNESS_COMMIT'], NEW_COMMIT)

    def test_private_feature_and_future_latent_paths_stay_outside_committed_source_closure(self):
        original_sampling_files = set(sampling.profile().committed_files)
        original_decoder_files = set(occupancy.profile().committed_files)
        sample = subject.configure('sampling', NEW_COMMIT)
        row, *_ = self.synthetic_latent()
        with patch.object(subject, 'saved_sampling', return_value=row):
            decoder = subject.configure('decoder', NEW_COMMIT)
        for profile, required in ((sample, original_sampling_files), (decoder, original_decoder_files)):
            self.assertTrue(required.issubset(profile.committed_files))
            for name in profile.committed_files:
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn('artifacts/', name.replace('\\', '/'))
            self.assertIn('scripts/compatibility/amd_zombie_occupancy.py', profile.committed_files)
            self.assertIn('scripts/compatibility/test_amd_zombie_occupancy.py', profile.committed_files)

    def test_decoder_cannot_be_configured_when_saved_sampling_did_not_pass(self):
        before = occupancy.data.INPUT
        with patch.object(subject, 'saved_sampling', side_effect=ValueError('Synthetic sampling failure')):
            with self.assertRaises(ValueError):
                subject.configure('decoder', NEW_COMMIT)
        self.assertEqual(occupancy.data.INPUT, before)

    def test_decoder_reads_verified_fresh_latent_and_preserves_original_host(self):
        import numpy as np
        before = occupancy.profile()
        row, array, *_ = self.synthetic_latent()
        with patch.object(subject, 'saved_sampling', return_value=row):
            profile = subject.configure('decoder', NEW_COMMIT)
        restored, reference = occupancy.data.read_input()
        self.assertTrue(np.array_equal(restored, array))
        self.assertEqual(reference, row)
        self.assertEqual(occupancy.data.INPUT, Path(row['path']))
        self.assertEqual(occupancy.data.INPUT_SHA, row['file_sha256'])
        self.assertEqual(profile.evidence, subject.EVIDENCE / 'decoder')
        self.assertEqual(occupancy.EVIDENCE, profile.evidence)
        self.assertEqual(occupancy.base.EVIDENCE, profile.evidence)
        self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds), (120, 180, 360))
        self.assertEqual(profile.authorize, b'EXECUTE_ONE_DECODER_FORWARD\n')
        self.assertEqual(profile.kind, 'occupancy')
        self.assertIs(profile.classify, before.classify)
        self.assertEqual(os.environ['MODLY_OCCUPANCY_HARNESS_COMMIT'], NEW_COMMIT)

    def test_unsupported_dino_or_downstream_stages_are_rejected_before_routing(self):
        before_feature = sampling.forward.local.FEATURE
        before_input = occupancy.data.INPUT
        for stage in ('dino', 'shape', 'texture', 'glb', '', None):
            with self.subTest(stage=stage):
                with self.assertRaises(ValueError):
                    subject.configure(stage, NEW_COMMIT)
        self.assertEqual(sampling.forward.local.FEATURE, before_feature)
        self.assertEqual(occupancy.data.INPUT, before_input)

    def test_original_latent_container_roundtrip_is_exclusive_and_finite(self):
        row, array, identity, runtime = self.synthetic_latent()
        path = Path(row['path'])
        core = {key: row['provenance'][key] for key in latents.FIELDS}
        self.assertEqual(latents.verify(path, path.parent, core, row['output_sha256']), row)
        self.assertEqual((row['shape'], row['dtype'], row['tensor_bytes']), ([1, 8, 16, 16, 16], 'F32', 131072))
        self.assertIsNone(self.validate_provenance(row, identity, runtime))
        with self.assertRaises(FileExistsError):
            latents.write(path, path.parent, array, core)
        import numpy as np
        for value in (array.astype('<f2'), np.full(array.shape, np.nan, dtype='<f4'), array[:, :, :, :, :15]):
            with self.assertRaises(ValueError):
                latents.array_hash(value)

    def test_latent_reader_rejects_payload_corruption(self):
        row, *_ = self.synthetic_latent()
        path = Path(row['path'])
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(blob)
        core = {key: row['provenance'][key] for key in latents.FIELDS}
        with self.assertRaises(ValueError):
            latents.verify(path, path.parent, core, row['output_sha256'])

    def test_latent_lineage_rejects_forged_current_or_historical_execution_commit(self):
        row, _, identity, runtime = self.synthetic_latent()
        changed = copy.deepcopy(row)
        changed['provenance']['harness_commit'] = OLD_COMMIT
        with self.assertRaises(ValueError):
            self.validate_provenance(changed, identity, runtime)
        changed = copy.deepcopy(row)
        nested = json.loads(changed['provenance']['feature_provenance'])
        nested['harness_commit'] = NEW_COMMIT
        changed['provenance']['feature_provenance'] = json.dumps(nested, sort_keys=True)
        with self.assertRaises(ValueError):
            self.validate_provenance(changed, identity, runtime)

    def test_latent_lineage_rejects_substituted_feature_model_sources_or_runtime(self):
        row, _, identity, runtime = self.synthetic_latent()
        for field, value in (('feature_file_sha256', '0' * 64), ('feature_provenance', '{}'),
                ('model_checkpoint_sha256', '0' * 64), ('model_config_sha256', '0' * 64),
                ('model_source_sha256', '0' * 64), ('sampler_sources', '{}'), ('runtime', '{}')):
            changed = copy.deepcopy(row)
            changed['provenance'][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.validate_provenance(changed, identity, runtime)

    def test_latent_lineage_rejects_sampler_setting_or_noise_hash_changes(self):
        row, _, identity, runtime = self.synthetic_latent()
        mutations = [('seed', '1'), ('sigma_min', '0.0'), ('noise_sha256', 'invalid')]
        for key, value in (('steps', 13), ('guidance_strength', 1.0), ('guidance_rescale', 0.0),
                ('guidance_interval', [0.0, 1.0]), ('rescale_t', 1.0), ('dino_lock', 1.0)):
            params = json.loads(row['provenance']['params'])
            params[key] = value
            mutations.append(('params', json.dumps(params)))
        for field, value in mutations:
            changed = copy.deepcopy(row)
            changed['provenance'][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.validate_provenance(changed, identity, runtime)

    def test_saved_stage_readers_require_actual_new_results_without_creating_them(self):
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='_absent-cpu-', dir=CONTROL))
        evidence = Path(directory) / 'future-run'
        with patch.object(subject, 'EVIDENCE', evidence):
            for reader in (subject.saved_sampling, subject.saved_occupancy):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        reader(NEW_COMMIT)
                    self.assertFalse(evidence.exists())

    def test_adapter_contains_no_external_main_or_three_stage_dispatch(self):
        tree = ast.parse(Path(subject.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = node.func
                self.assertFalse(isinstance(call, ast.Attribute) and call.attr in ('main', 'run_stages'))
                self.assertFalse(isinstance(call, ast.Name) and call.id == 'run_stages')


if __name__ == '__main__':
    unittest.main()
