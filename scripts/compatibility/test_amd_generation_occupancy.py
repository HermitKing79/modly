"""Focused CPU tests; every generated tensor below is a synthetic test fixture."""
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'trellis2', 'flex_gemm',
                'transformers', 'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime'}:
            raise AssertionError('CPU adapter imported a model/GPU package: ' + fullname)


sys.meta_path.insert(0, NoModels())
import amd_generation_occupancy as subject


class GenerationOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix='_generation-cpu-test-', dir=subject.ROOT / 'docs/compatibility'))
        self.base = Path(directory) / 'new-image'
        self.stack.enter_context(patch.object(subject, 'BASE', self.base))
        self.stack.enter_context(patch.dict(os.environ))
        self.image = subject.ROOT / ('docs/compatibility/amd-object-conditioned-occupancy-evidence/'
            'image-selection/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp')
        self.evidence = self.base / 'attempt'

    def prepare(self):
        row = subject.prepare(self.image, self.evidence)
        subject.configure(self.image, self.evidence)
        return row

    def test_prepare_actual_rgba_source_preserves_pixels_and_cpu_crop(self):
        row = self.prepare()
        image, restored = subject.selected_image()
        self.assertEqual(image.size, (1018, 1018))
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.getchannel('A').getextrema(), (0, 255))
        self.assertEqual(restored, row)
        self.assertEqual(row['source_sha256'], 'a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293')
        self.assertEqual((subject.ROOT / row['source_path']).read_bytes(), self.image.read_bytes())
        self.assertFalse(Path(row['source_path']).is_absolute())
        png = io.BytesIO(); image.save(png, format='PNG')
        self.assertEqual(hashlib.sha256(png.getvalue()).hexdigest(), row['input_png_sha256'])
        crop = subject.original.preprocess_cpu(image)
        self.assertEqual(hashlib.sha256(crop.tobytes()).hexdigest(), row['crop_pixel_sha256'])
        self.assertFalse(row['weights_loaded'])
        self.assertFalse(row['gpu_initialized'])
        self.assertFalse(any(name in sys.modules for name in ('torch', 'trellis2', 'transformers')))

    def test_prepare_refuses_existing_or_foreign_evidence_before_writing(self):
        self.prepare()
        before = (self.evidence / 'input-contract.json').read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            subject.prepare(self.image, self.evidence)
        self.assertEqual((self.evidence / 'input-contract.json').read_bytes(), before)
        foreign = self.base.parent / 'foreign'
        with self.assertRaises(ValueError):subject.prepare(self.image, foreign)
        self.assertFalse(foreign.exists())

    def test_selected_image_rejects_tampered_pixels_dimensions_and_contract(self):
        row = self.prepare()
        path = subject.ROOT / row['source_path']; blob = path.read_bytes()
        path.write_bytes(blob + b'x')
        with self.assertRaises(ValueError):subject.selected_image()
        path.write_bytes(blob)
        contract = self.evidence / 'input-contract.json'
        row['source_size'] = [1042, 1042]
        contract.write_text(json.dumps(row), encoding='utf-8')
        with self.assertRaises(ValueError):subject.selected_image()

    def test_worker_environment_binds_exact_image_evidence_and_contract(self):
        row = self.prepare()
        self.assertEqual(subject.configure_worker(), row)
        os.environ[subject.ENV_SOURCE_SHA] = '0' * 64
        with self.assertRaises(ValueError):subject.configure_worker()
        del os.environ[subject.ENV_EVIDENCE]
        with self.assertRaises(ValueError):subject.configure_worker()

    def test_profiles_route_new_worker_and_retain_original_stage_guards(self):
        self.prepare()
        original = subject.original
        row = {'path': str(self.evidence / 'dino/run/scratch/conditioning.safetensors'),
               'file_sha256': '0' * 64, 'verified': True}
        for owner, name in [(original.dino.local, 'fixture'),
                (original.sampling.forward.local, 'FEATURE'),
                (original.sampling.forward.local, 'FEATURE_SHA'),
                (original.sampling.forward.local, 'verify_feature'),
                (original.sampling.forward.local, 'EVIDENCE'),
                (original.sampling.forward, 'verify_preflight'),
                (original.occupancy.data, 'INPUT'), (original.occupancy.data, 'INPUT_SHA'),
                (original.occupancy.data, 'read_input'), (original.occupancy, 'EVIDENCE'),
                (original.occupancy.base, 'EVIDENCE')]:
            self.stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        self.stack.enter_context(patch.object(original, 'feature_reference', return_value=row))
        self.stack.enter_context(patch.object(original, 'latent_reference', return_value={
            **row, 'path': str(self.evidence / 'sampling/run/scratch/sparse-structure-latent.safetensors')}))
        for stage in ('dino', 'sampling', 'decoder'):
            profile = subject.prepare_profile(stage)
            self.assertEqual(profile.evidence, self.evidence / stage)
            self.assertEqual(profile.worker_script, Path(subject.__file__).resolve())
            self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds),
                {'dino': (120, 180, 360), 'sampling': (180, 240, 480), 'decoder': (120, 180, 360)}[stage])
            self.assertTrue(all(name in profile.committed_files for name in subject.STATIC_FILES))
            self.assertFalse(any('/run/scratch/' in name for name in profile.committed_files))
            if stage == 'sampling':
                self.assertEqual((profile.maximum_sequences, profile.maximum_steps, profile.maximum_forwards), (1, 12, 22))
                self.assertEqual(profile.authorize, original.sampling.AUTHORIZE)

    def fixture_sequence(self):
        import numpy as np
        original = subject.original
        row = self.prepare(); commit = '1' * 40
        stages = {name: self.evidence / name / 'run/scratch' for name in ('dino', 'sampling', 'decoder')}
        for path in stages.values():path.mkdir(parents=True)
        fp = {key: '0' * 64 for key in original.features.FIELDS}
        fp.update(component='DINOv3ViTModel', harness_commit=commit, runtime='synthetic CPU test fixture',
            checkpoint_sha256=json.loads(original.dino.local.PINS.read_text())['checkpoint_sha256'],
            input_pixel_sha256=row['pixel_sha256'], input_png_sha256=row['input_png_sha256'])
        feature = original.features.write(stages['dino'] / 'conditioning.safetensors', stages['dino'],
            np.zeros((1, 1029, 1024), dtype='<f4'), fp)
        lp = {key: '0' * 64 for key in original.latents.FIELDS}
        lp.update(component='SparseStructureFlowModel', harness_commit=commit, runtime='synthetic CPU test fixture',
            seed='20261013', sigma_min='1e-05', params=json.dumps(original.sampling.policy.PARAMS),
            sampler_sources='{}', feature_file_sha256=feature['file_sha256'], feature_provenance=json.dumps(feature['provenance']))
        latent = original.latents.write(stages['sampling'] / 'sparse-structure-latent.safetensors', stages['sampling'],
            np.zeros((1, 8, 16, 16, 16), dtype='<f4'), lp)
        logits = np.full((1, 1, 64, 64, 64), -1, dtype='<f4'); logits[0, 0, :8, :8, :8] = 1
        decoder_sha = '1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a'
        occupancy = original.occupancy.data.write_output(stages['decoder'], logits,
            {'input': latent, 'decoder_checkpoint_sha256': decoder_sha, 'harness_commit': commit})
        workers = {'dino': {'feature_checkpoint': feature}, 'sampling': {'latent_checkpoint': latent},
            'decoder': {'occupancy_artifacts': occupancy, 'checkpoint_identity': {'checkpoint_sha256': decoder_sha}}}
        for stage, worker in workers.items():
            worker['release'] = {'allocated_bytes': 0, 'reserved_bytes': 0}
            result = {'classification': 'PASS', 'observer_healthy': True, 'checkpoint_unchanged': True,
                'process': {'returncode': 0, 'worker_exit_verified': True,
                    'job_cleanup': {'owned_tree_exit_verified': True, 'forced_job_cleanup': False}}, 'worker': worker}
            (stages[stage].parent / 'results.json').write_text(json.dumps(result))
        (self.evidence / 'sequence-attempt-started.json').write_text(json.dumps({'harness_commit': commit}))
        return feature, latent, occupancy

    def test_verified_sequence_preserves_root_artifacts_and_rejects_changed_native_file(self):
        feature, latent, occupancy = self.fixture_sequence()
        verified = subject.verify_sequence(self.evidence)
        self.assertEqual(verified['feature'], feature)
        self.assertEqual(verified['latent'], latent)
        self.assertFalse(verified['full_generation'])
        receipt = subject.preserve_root_artifacts(verified)
        self.assertEqual(set(receipt['files']), {'conditioning.safetensors', 'occupancy-logits.npy', 'occupancy-mask.npy'})
        for row in receipt['files'].values():
            self.assertEqual(Path(row['path']).read_bytes(), Path(row['source_path']).read_bytes())
        with self.assertRaises(FileExistsError):subject.preserve_root_artifacts(verified)
        path = Path(occupancy['files']['occupancy-mask.npy']['path'])
        blob = bytearray(path.read_bytes()); blob[-1] ^= 1; path.write_bytes(blob)
        with self.assertRaises(ValueError):subject.verify_sequence(self.evidence)

    def test_cli_prepare_only_never_enters_original_execution(self):
        self.stack.enter_context(patch.object(subject.original, 'main', side_effect=AssertionError('GPU entry')))
        self.assertEqual(subject.main(['--image', str(self.image), '--evidence', str(self.evidence), '--prepare-only']), 0)
        self.assertTrue((self.evidence / 'input-contract.json').is_file())
        self.assertFalse((self.evidence / 'sequence-attempt-started.json').exists())

    def test_configured_adapter_omits_optional_preview_without_mesh_work(self):
        self.prepare()
        with patch.object(subject.original.occupancy.data, 'surface', side_effect=AssertionError('mesh work')):
            receipt = subject.original.occupancy.data.write_preview(self.evidence, object())
        self.assertEqual(receipt, {'omitted': True,
            'reason': 'repeatable pipeline consumes native occupancy; no diagnostic voxel geometry requested'})
        self.assertIs(subject.original.occupancy.data.write_preview, subject.omit_preview)
        self.assertFalse((self.evidence / 'voxel-preview.obj').exists())
        self.assertFalse((self.evidence / 'voxel-preview.png').exists())

    def test_cli_forwards_only_original_arguments_and_keeps_static_inputs_committable(self):
        row = self.prepare(); observed = []
        def original_entry():
            observed.append(sys.argv[1:])
            return 1  # Exercise routing only; no model entry or successful-result claim.
        self.stack.enter_context(patch.object(subject.original, 'main', side_effect=original_entry))
        self.assertEqual(subject.main(['--image', str(self.image), '--evidence', str(self.evidence),
            '--harness-commit', '1' * 40]), 1)
        scratch = self.evidence / 'dino/run/scratch'
        self.assertEqual(subject.main(['--worker', '--output', str(scratch)]), 1)
        self.assertEqual(observed, [['--harness-commit', '1' * 40], ['--worker', '--output', str(scratch)]])
        required = subject.required_files(self.evidence)
        self.assertIn(row['source_path'], required)
        self.assertIn((self.evidence / 'input-contract.json').relative_to(subject.ROOT).as_posix(), required)
        self.assertFalse(any('/run/' in name for name in required))
        self.assertFalse(any(Path(name).is_absolute() for name in required))


if __name__ == '__main__':unittest.main()
