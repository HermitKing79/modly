"""CPU-only admission tests for one genuine zombie DINO compatibility pass."""
import copy
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'triton', 'trellis2', 'transformers', 'flex_gemm',
            'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime',
        }:
            raise AssertionError('CPU admission imported a model/GPU package: ' + fullname)


sys.meta_path.insert(0, NoModels())
try:
    import amd_zombie_dinov3 as subject
except ModuleNotFoundError as error:
    if error.name != 'amd_zombie_dinov3':
        raise
    subject = None

import amd_object_occupancy as original
import amd_blaslt_teardown as teardown

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT.parent / 'artifacts/amd-preview-dinov3-7d11b36/cpu-input'
PRIVATE = ROOT / 'artifacts/amd-zombie-dinov3-b8afdd5'
CONTROL = ROOT.parent / 'artifacts/amd-zombie-dinov3-b8afdd5'
SOURCE_SHA = 'db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e'
PIXEL_SHA = 'e958193cad52cb3b15efd69410c3727853df61fd35310468d1b4540284ea2fb5'


class ZombieDinoTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'The standalone zombie DINO adapter is missing')
        self.blob = (ARCHIVE / 'source-image.png').read_bytes()
        self.prior_result = json.loads((ARCHIVE / 'cpu-result.json').read_text())
        self.prior_contract = json.loads((ARCHIVE / 'input-contract.json').read_text())

    def validate(self, blob=None, result=None, contract=None):
        return subject.validate_image(
            self.blob if blob is None else blob,
            self.prior_result if result is None else result,
            self.prior_contract if contract is None else contract,
        )

    def test_real_snapshot_uses_original_alpha_preprocessing_and_exact_png_bytes(self):
        image, row = self.validate()
        self.assertEqual(len(self.blob), 642414)
        self.assertEqual(hashlib.sha256(self.blob).hexdigest(), SOURCE_SHA)
        self.assertEqual((image.mode, image.size), ('RGBA', (1216, 1294)))
        self.assertEqual(image.getchannel('A').getextrema(), (0, 255))
        self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), PIXEL_SHA)
        expected = {
            'source_sha256': SOURCE_SHA,
            'pixel_sha256': PIXEL_SHA,
            'source_size': [1216, 1294],
            'source_mode': 'RGBA',
            'input_png_sha256': SOURCE_SHA,
            'crop_size': [980, 980],
            'crop_mode': 'RGB',
            'weights_loaded': False,
            'gpu_initialized': False,
        }
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(row[key], value)
        # This is the verified original upstream method, independent of the adapter.
        crop = original.preprocess_cpu(image)
        self.assertEqual(hashlib.sha256(crop.tobytes()).hexdigest(), row['crop_pixel_sha256'])
        self.assertFalse(any(name in sys.modules for name in ('torch', 'transformers', 'triton')))

    def test_bindings_select_exact_private_archive_and_fresh_dino_leaf(self):
        self.assertEqual(subject.SOURCE, ARCHIVE / 'source-image.png')
        self.assertEqual(subject.SOURCE_SHA, SOURCE_SHA)
        self.assertEqual(subject.PIXEL_SHA, PIXEL_SHA)
        self.assertEqual(subject.EVIDENCE, PRIVATE / 'dino')

    def test_changed_bytes_never_get_admitted_as_the_selected_zombie(self):
        changed = bytearray(self.blob)
        changed[-1] ^= 1
        for blob in (bytes(changed), self.blob + b'x', b'', b'not an image'):
            with self.subTest(size=len(blob)):
                with self.assertRaises(ValueError):
                    self.validate(blob=blob)

    def test_replacement_mode_or_dimensions_cannot_substitute_for_pinned_input(self):
        from PIL import Image
        for image in (Image.new('RGB', (1216, 1294)), Image.new('RGBA', (64, 64))):
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            with self.subTest(mode=image.mode, size=image.size):
                with self.assertRaises(ValueError):
                    self.validate(blob=stream.getvalue())

    def test_conflicting_prior_byte_pixel_dimension_or_parameter_pins_block(self):
        changes = (
            ('image_sha256', '0' * 64),
            ('contract.input_sha256', '0' * 64),
            ('contract.pixel_sha256', '0' * 64),
            ('contract.source_size', [64, 64]),
            ('contract.source_mode', 'RGB'),
            ('canonical_parameters.resolution', 1024),
            ('canonical_parameters.image_mode', 'rembg'),
        )
        for field, value in changes:
            result = copy.deepcopy(self.prior_result)
            owner = result
            keys = field.split('.')
            for key in keys[:-1]:
                owner = owner[key]
            owner[keys[-1]] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.validate(result=result)
        contract = copy.deepcopy(self.prior_contract)
        contract['original_cpu_input_contract']['pixel_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.validate(contract=contract)

    def test_plain_path_rejects_unc_aliases_and_hardlinked_files(self):
        with tempfile.TemporaryDirectory(prefix='_zombie-path-', dir=CONTROL) as directory:
            base = Path(directory)
            path = base / 'input.png'
            path.write_bytes(b'CPU path fixture')
            self.assertEqual(subject.plain_path(path, existing=True), path.resolve())
            (base / 'alias').mkdir()
            with self.assertRaises(ValueError):
                subject.plain_path(base / 'alias/../input.png', existing=True)
            with self.assertRaises(ValueError):
                subject.plain_path(r'\\server\share\input.png')
            link = base / 'hardlink.png'
            os.link(path, link)
            for candidate in (path, link):
                with self.subTest(candidate=candidate.name):
                    with self.assertRaises(ValueError):
                        subject.plain_path(candidate, existing=True)

    def test_plain_path_rejects_reparse_ancestor_without_following_it(self):
        with tempfile.TemporaryDirectory(prefix='_zombie-reparse-', dir=CONTROL) as directory:
            base = Path(directory)
            ancestor = base / 'junction'
            ancestor.mkdir()
            path = ancestor / 'input.png'
            path.write_bytes(b'CPU path fixture')
            real_lstat = Path.lstat

            def marked_lstat(candidate, *args, **kwargs):
                actual = real_lstat(candidate, *args, **kwargs)
                if candidate == ancestor:
                    from types import SimpleNamespace
                    return SimpleNamespace(st_mode=actual.st_mode, st_nlink=actual.st_nlink,
                        st_file_attributes=getattr(actual, 'st_file_attributes', 0) | 0x400)
                return actual

            with patch.object(Path, 'lstat', marked_lstat):
                with self.assertRaises(ValueError):
                    subject.plain_path(path, existing=True)

    def test_image_factory_copies_exact_snapshot_exclusively(self):
        with tempfile.TemporaryDirectory(prefix='_zombie-input-', dir=CONTROL) as directory:
            evidence = Path(directory) / 'dino'
            scratch = evidence / 'run/scratch'
            scratch.mkdir(parents=True)
            target = scratch / 'input.png'
            with patch.object(subject, 'EVIDENCE', evidence):
                image = subject.image_factory(target)
                self.assertEqual(target.read_bytes(), self.blob)
                self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), PIXEL_SHA)
                with self.assertRaises(FileExistsError):
                    subject.image_factory(target)
                self.assertEqual(target.read_bytes(), self.blob)

    def test_image_factory_rejects_any_other_destination_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='_zombie-route-', dir=CONTROL) as directory:
            evidence = Path(directory) / 'dino'
            scratch = evidence / 'run/scratch'
            scratch.mkdir(parents=True)
            with patch.object(subject, 'EVIDENCE', evidence):
                for target in (scratch / 'other.png', evidence / 'input.png', Path(directory) / 'input.png'):
                    with self.subTest(target=target):
                        with self.assertRaises(ValueError):
                            subject.image_factory(target)
                        self.assertFalse(target.exists())

    def test_profile_retains_original_owned_host_deadlines_authorization_and_defaults(self):
        before = teardown.profile()
        original_fixture = original.dino.local.fixture
        profile = subject.profile()
        self.assertEqual(profile.evidence, subject.EVIDENCE)
        self.assertEqual(profile.worker_script, Path(subject.__file__).resolve())
        self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds), (120, 180, 360))
        self.assertEqual(profile.authorize, b'EXECUTE_ONE_DINOV3_CONDITIONING\n')
        self.assertIs(profile.identity, before.identity)
        self.assertIs(profile.classify, before.classify)
        self.assertIsNone(getattr(profile, 'kind', None))
        self.assertEqual(getattr(profile, 'maximum_forwards', 1), 1)
        self.assertIn('scripts/compatibility/amd_zombie_dinov3.py', profile.committed_files)
        self.assertIn('scripts/compatibility/test_amd_zombie_dinov3.py', profile.committed_files)
        self.assertTrue(set(before.committed_files).issubset(profile.committed_files))
        for name in profile.committed_files:
            self.assertFalse(Path(name).is_absolute())
            self.assertNotIn('artifacts/', name.replace('\\', '/'))
            if name not in before.committed_files:
                self.assertNotIn('/run/', name.replace('\\', '/'))
        after = teardown.profile()
        self.assertEqual(after.evidence, before.evidence)
        self.assertEqual(after.committed_files, before.committed_files)
        self.assertEqual(after.authorize, before.authorize)
        self.assertIs(original.dino.local.fixture, original_fixture)

    def test_installed_header_identity_covers_length_prefix_and_padded_json(self):
        from services.amd_model_metadata import safetensors_metadata
        path = original.dino.local.MODEL / 'model.safetensors'
        with path.open('rb') as stream:
            prefix = stream.read(8)
            count = struct.unpack('<Q', prefix)[0]
            raw = stream.read(count)
        self.assertEqual(count, 41400)
        full = hashlib.sha256(prefix + raw).hexdigest()
        self.assertEqual(full, '3495b172735ff88c94b28a3257614f51764abfd7879b1498ec71abf855f259d3')
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), full)
        self.assertEqual(safetensors_metadata(path)['header_sha256'], full)

    def synthetic_saved_fixture(self):
        """Real 4 MiB CPU feature file plus historical host record; never a neural result."""
        import numpy as np
        import amd_feature_checkpoint as features
        directory = tempfile.TemporaryDirectory(prefix='_zombie-synthetic-', dir=CONTROL)
        self.addCleanup(directory.cleanup)
        evidence = Path(directory.name) / 'dino'
        scratch = evidence / 'run/scratch'
        scratch.mkdir(parents=True)
        binding = patch.object(subject, 'EVIDENCE', evidence)
        binding.start()
        self.addCleanup(binding.stop)
        commit = '1' * 40
        pins = json.loads(original.dino.local.PINS.read_text())
        provenance = dict(component='DINOv3ViTModel', checkpoint_sha256=pins['checkpoint_sha256'],
            config_sha256=hashlib.sha256((original.dino.local.MODEL / 'config.json').read_bytes()).hexdigest(),
            input_pixel_sha256=PIXEL_SHA, input_png_sha256=SOURCE_SHA, harness_commit=commit,
            extractor_source_sha256=pins['sources'][str(original.dino.local.SOURCE / 'trellis2/modules/image_feature_extractor.py')],
            pipeline_source_sha256=pins['sources'][str(original.dino.local.SOURCE / 'trellis2/pipelines/trellis2_image_to_3d.py')],
            runtime='Python 3.12.10; PyTorch 2.13.0+rocm10.0.0; HIP 7.15.26333')
        array = np.arange(1053696, dtype='<f4').reshape((1, 1029, 1024)) / np.float32(1053696)
        artifact = features.write(scratch / 'conditioning.safetensors', scratch, array, provenance)
        result = json.loads((ROOT / 'docs/compatibility/amd-blaslt-teardown-evidence/run/results.json').read_text())
        result['process']['harness_commit'] = commit
        result['process']['command'] = [str(original.dino.local.VENV / 'Scripts/python.exe'), '-B', '-I',
            str(Path(subject.__file__).resolve()), '--worker', '--output', str(scratch)]
        result['worker']['input'] = dict(mode='RGBA', size=[1216, 1294], pixel_sha256=PIXEL_SHA,
            png_sha256=SOURCE_SHA, crop_mode='RGB', crop_size=[980, 980], resolution=512,
            alpha_only=True, rmbg_model=None)
        result['worker']['feature_checkpoint'] = artifact
        result['worker']['output'].update(sha256=artifact['output_sha256'], minimum=float(array.min()),
            maximum=float(array.max()), mean=float(array.mean()), stddev=float(array.std()),
            representative_values=array.reshape(-1)[:8].tolist())
        return result, commit, artifact, array, provenance

    def test_synthetic_cpu_feature_roundtrip_and_saved_acceptance_use_original_reader(self):
        result, commit, artifact, array, provenance = self.synthetic_saved_fixture()
        import amd_feature_checkpoint as features
        restored = features.verify(artifact['path'], Path(artifact['path']).parent, provenance, artifact['output_sha256'])
        self.assertEqual(restored, artifact)
        self.assertEqual(restored['tensor_bytes'], 4214784)
        self.assertEqual(subject.validate_saved(result, commit), artifact)
        with self.assertRaises(FileExistsError):
            features.write(artifact['path'], Path(artifact['path']).parent, array, provenance)
        self.assertFalse(any(name in sys.modules for name in ('torch', 'transformers', 'triton')))

    def test_saved_acceptance_rejects_forced_cleanup_or_failed_monitoring(self):
        result, commit, *_ = self.synthetic_saved_fixture()
        mutations = (
            ('process.job_cleanup.forced_job_cleanup', True),
            ('process.job_cleanup.active_before_cleanup', 1),
            ('process.job_cleanup.owned_tree_exit_verified', False),
            ('process.worker_exit_verified', False),
            ('process.job_attached_before_preparation', False),
            ('process.windows_periodic_stopped', False),
            ('process.timed_out', True),
            ('observer_healthy', False),
            ('checkpoint_unchanged', False),
            ('worker.release.allocated_bytes', 1),
            ('worker.release.reserved_bytes', 1),
        )
        for field, value in mutations:
            changed = copy.deepcopy(result)
            owner = changed
            parts = field.split('.')
            for part in parts[:-1]:
                owner = owner[part]
            owner[parts[-1]] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    subject.validate_saved(changed, commit)

    def test_saved_acceptance_requires_all_genuine_operation_counters(self):
        result, commit, *_ = self.synthetic_saved_fixture()
        for key in ('load_calls', 'conditioning_calls', 'transfer_calls', 'normalization_copies'):
            changed = copy.deepcopy(result)
            del changed['worker'][key]
            with self.subTest(counter=key):
                with self.assertRaises(ValueError):
                    subject.validate_saved(changed, commit)
        changed = copy.deepcopy(result)
        del changed['worker']['calls']['layer.23']
        with self.assertRaises(ValueError):
            subject.validate_saved(changed, commit)

    def test_saved_acceptance_rejects_another_image_or_worker_claimed_nonfinite_output(self):
        result, commit, *_ = self.synthetic_saved_fixture()
        for field, value in (('pixel_sha256', '0' * 64), ('png_sha256', '0' * 64),
                ('size', [64, 64]), ('crop_size', [46, 46]), ('alpha_only', False)):
            changed = copy.deepcopy(result)
            changed['worker']['input'][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    subject.validate_saved(changed, commit)
        changed = copy.deepcopy(result)
        changed['worker']['output']['all_finite'] = False
        with self.assertRaises(ValueError):
            subject.validate_saved(changed, commit)

    def test_saved_acceptance_rejects_changed_file_hash_or_payload(self):
        result, commit, artifact, *_ = self.synthetic_saved_fixture()
        changed = copy.deepcopy(result)
        changed['worker']['feature_checkpoint']['file_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            subject.validate_saved(changed, commit)
        path = Path(artifact['path'])
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(blob)
        with self.assertRaises(ValueError):
            subject.validate_saved(result, commit)

    def test_saved_acceptance_derives_provenance_instead_of_trusting_rewritten_metadata(self):
        result, commit, artifact, array, provenance = self.synthetic_saved_fixture()
        from safetensors.numpy import save
        import amd_feature_checkpoint as features
        metadata = features.metadata({**provenance, 'input_pixel_sha256': '0' * 64}, artifact['output_sha256'])
        blob = save({'cond': array}, metadata=metadata)
        Path(artifact['path']).write_bytes(blob)
        result['worker']['feature_checkpoint'].update(file_sha256=hashlib.sha256(blob).hexdigest(),
            file_bytes=len(blob), provenance=metadata)
        with self.assertRaises(ValueError):
            subject.validate_saved(result, commit)

    def test_saved_acceptance_independently_rejects_nonfinite_tensor_with_consistent_hashes(self):
        result, commit, artifact, array, provenance = self.synthetic_saved_fixture()
        from safetensors.numpy import save
        import amd_feature_checkpoint as features
        array[0, 0, 0] = float('nan')
        payload_sha = hashlib.sha256(array.tobytes()).hexdigest()
        metadata = features.metadata(provenance, payload_sha)
        blob = save({'cond': array}, metadata=metadata)
        Path(artifact['path']).write_bytes(blob)
        result['worker']['output']['sha256'] = payload_sha
        result['worker']['feature_checkpoint'].update(file_sha256=hashlib.sha256(blob).hexdigest(),
            file_bytes=len(blob), output_sha256=payload_sha, provenance=metadata)
        with self.assertRaises(ValueError):
            subject.validate_saved(result, commit)

    def test_saved_acceptance_rejects_foreign_process_command(self):
        result, commit, *_ = self.synthetic_saved_fixture()
        replacements = (
            (0, str(ROOT / 'foreign-python.exe')),
            (3, str(ROOT / 'scripts/compatibility/amd_object_occupancy.py')),
            (6, str(ROOT / 'artifacts/foreign/run/scratch')),
        )
        for index, value in replacements:
            changed = copy.deepcopy(result)
            changed['process']['command'][index] = value
            with self.subTest(argument=index):
                with self.assertRaises(ValueError):
                    subject.validate_saved(changed, commit)
        changed = copy.deepcopy(result)
        del changed['process']['command']
        with self.assertRaises(ValueError):
            subject.validate_saved(changed, commit)

    def test_saved_acceptance_rejects_changed_original_deadline_contract(self):
        result, commit, *_ = self.synthetic_saved_fixture()
        for key, value in (('cpu', 25), ('forward_release', 181), ('overall', 180)):
            changed = copy.deepcopy(result)
            changed['process']['stage_timeouts_seconds'][key] = value
            with self.subTest(deadline=key):
                with self.assertRaises(ValueError):
                    subject.validate_saved(changed, commit)
        changed = copy.deepcopy(result)
        del changed['process']['stage_timeouts_seconds']
        with self.assertRaises(ValueError):
            subject.validate_saved(changed, commit)


if __name__ == '__main__':
    unittest.main()
