"""Focused CPU-only saved-zombie shape adapter regressions."""
import copy
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROHIBITED = {'torch', 'torchvision', 'transformers', 'triton', 'trellis2',
              'flex_gemm', 'cumesh', 'nvdiffrast', 'onnxruntime', 'rembg'}


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in PROHIBITED:
            raise AssertionError('Focused zombie shape CPU test forbids neural import: ' + fullname)


IMPORT_GUARD = NoNeuralImports()
sys.meta_path.insert(0, IMPORT_GUARD)
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import amd_shape_sampling_artifact as original_artifact


class ZombieShapeTests(unittest.TestCase):
    def temporary_directory(self):
        root = Path(__file__).resolve().parents[2].parent / 'artifacts/amd-zombie-shape-0d15dd9/test-scratch'
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def adapter(self):
        try:
            return importlib.import_module('amd_zombie_shape')
        except ModuleNotFoundError as exc:
            if exc.name == 'amd_zombie_shape':
                self.fail('Missing strict saved-zombie shape adapter amd_zombie_shape')
            raise

    def fixture(self, root):
        scratch = Path(root) / 'scratch'
        scratch.mkdir()
        raw = np.ascontiguousarray(np.linspace(-2.5, 3.0, 96, dtype=np.float32).reshape(3, 32))
        coords = np.array([[0, 0, 1, 2], [0, 4, 5, 6], [0, 31, 30, 29]], dtype=np.int32)
        normalization = {'mean': np.linspace(-.75, .625, 32, dtype=np.float32).tolist(),
                         'std': np.linspace(.125, 2.5, 32, dtype=np.float32).tolist()}
        denormalized = np.ascontiguousarray(raw * np.array(normalization['std'], dtype=np.float32)[None, :]
                                           + np.array(normalization['mean'], dtype=np.float32)[None, :])
        noise = np.ascontiguousarray(np.linspace(1.0, -1.0, 96, dtype=np.float32).reshape(3, 32))
        provenance = {'fixture': True, 'token_count': 3, 'seed': 20261013,
                      'input_identity': {'coords_payload_sha256': hashlib.sha256(coords.tobytes()).hexdigest()}}
        rows = original_artifact.write(scratch, raw, denormalized, coords, provenance,
                                       noise=noise, maximum_tokens=4430)
        return rows, scratch, coords, normalization, provenance

    def refresh_npy_row(self, rows, scratch, name, array):
        path = scratch / name
        with path.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        blob = path.read_bytes()
        rows[name].update(file_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                          payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                          dtype=str(array.dtype), shape=list(array.shape))

    def verify(self, fixture):
        return self.adapter().verify_arrays(*fixture)

    def test_pooling_preserves_positive_support_and_lexicographic_original_order(self):
        logits = np.full((1, 1, 64, 64, 64), -1, dtype=np.float32)
        for d, h, w in ((0, 0, 0), (1, 1, 1), (2, 4, 6), (63, 63, 63)):
            logits[0, 0, d, h, w] = .25
        pooled, coords = self.adapter().derive_support(logits, logits[0, 0] > 0)
        expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 31, 31, 31]], dtype=np.int32)
        np.testing.assert_array_equal(coords, expected)
        self.assertEqual(pooled.dtype, np.bool_)
        self.assertEqual(pooled.shape, (32, 32, 32))
        self.assertEqual(int(pooled.sum()), 3)

    def test_support_rejects_mask_mismatch_empty_and_4431_without_truncation(self):
        adapter = self.adapter()
        logits = np.full((1, 1, 64, 64, 64), -1, dtype=np.float32)
        with self.assertRaises(ValueError):
            adapter.derive_support(logits, np.ones((64, 64, 64), dtype=bool))
        with self.assertRaises(ValueError):
            adapter.derive_support(logits, logits[0, 0] > 0)
        xyz = np.array(np.unravel_index(np.arange(4431), (32, 32, 32))).T * 2
        logits[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1
        with self.assertRaises(ValueError):
            adapter.derive_support(logits, logits[0, 0] > 0)
        logits[0, 0, *xyz[-1]] = -1
        self.assertEqual(len(adapter.derive_support(logits, logits[0, 0] > 0)[1]), 4430)

    def test_new_execution_commit_rejects_malformed_and_prior_stage_identities(self):
        adapter = self.adapter()
        adapter.valid_commit('1' * 40)
        for value in ('e6b6330ba2f9e51abfb74c726729e0be4ade7cac',
                      '93269412602a338acd9fbde41798901ddb5f707d', 'main', 'A' * 40,
                      '1' * 39, '1' * 41, None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.valid_commit(value)

    def test_runtime_constructor_reads_no_input_before_owned_job_attachment(self):
        adapter = self.adapter()
        with mock.patch.object(Path, 'open', side_effect=AssertionError('No filesystem reads before handshake')):
            runtime = adapter.Runtime()
        self.assertIsNotNone(runtime)

    def test_profile_retains_genuine_supervisor_deadlines_and_single_sequence(self):
        adapter = self.adapter()
        profile = adapter.Runtime().profile()
        self.assertEqual(Path(profile.worker_script).name, 'amd_zombie_shape.py')
        self.assertEqual((profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds), (180, 240, 480))
        self.assertEqual((profile.maximum_forwards, profile.maximum_steps, profile.maximum_sequences), (21, 12, 1))
        self.assertEqual(set(profile.committed_files), set(adapter.CLOSURE))
        self.assertIn('scripts/compatibility/amd_zombie_shape.py', profile.committed_files)
        self.assertIn('scripts/compatibility/test_amd_zombie_shape.py', profile.committed_files)

    def test_all_five_original_writer_outputs_reopen_with_exact_f32_denormalization(self):
        with self.temporary_directory() as directory:
            fixture = self.fixture(directory)
            result = self.verify(fixture)
            self.assertIsInstance(result, dict)
            self.assertFalse(PROHIBITED.intersection(sys.modules))

    def test_changed_saved_bytes_are_rejected_even_when_record_remains_pass(self):
        with self.temporary_directory() as directory:
            fixture = self.fixture(directory)
            path = fixture[1] / 'shape-latent-raw.npy'
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                self.verify(fixture)

    def test_rehashed_nonfinite_and_wrong_precision_arrays_are_rejected(self):
        for name, kind in (('shape-latent-raw.npy', 'nonfinite'),
                           ('shape-latent-denormalized.npy', 'nonfinite'),
                           ('shape-initial-noise.npy', 'nonfinite'),
                           ('shape-latent-raw.npy', 'precision')):
            with self.subTest(name=name, kind=kind), self.temporary_directory() as directory:
                fixture = self.fixture(directory)
                rows, scratch = fixture[:2]
                array = np.load(scratch / name, allow_pickle=False)
                if kind == 'nonfinite':
                    array[0, 0] = np.nan
                else:
                    array = array.astype(np.float64)
                self.refresh_npy_row(rows, scratch, name, array)
                with self.assertRaises(ValueError):
                    self.verify(fixture)

    def test_rehashed_coordinate_change_and_wrong_token_count_are_rejected(self):
        for kind in ('values', 'count'):
            with self.subTest(kind=kind), self.temporary_directory() as directory:
                fixture = self.fixture(directory)
                rows, scratch, coords = fixture[:3]
                altered = coords.copy()
                if kind == 'values':
                    altered[1, 3] += 1
                else:
                    altered = altered[:2]
                self.refresh_npy_row(rows, scratch, 'shape-latent-coordinates.npy', altered)
                with self.assertRaises(ValueError):
                    self.verify(fixture)

    def test_rehashed_finite_wrong_denormalization_is_rejected(self):
        with self.temporary_directory() as directory:
            fixture = self.fixture(directory)
            rows, scratch = fixture[:2]
            altered = np.load(scratch / 'shape-latent-denormalized.npy', allow_pickle=False)
            altered[1, 17] += np.float32(.001)
            self.refresh_npy_row(rows, scratch, 'shape-latent-denormalized.npy', altered)
            with self.assertRaises(ValueError):
                self.verify(fixture)

    def test_rehashed_sidecar_provenance_change_is_rejected(self):
        with self.temporary_directory() as directory:
            fixture = self.fixture(directory)
            rows, scratch, _, _, provenance = fixture
            changed = copy.deepcopy(provenance)
            changed['seed'] += 1
            path = scratch / 'shape-latent-provenance.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            blob = path.read_bytes()
            rows[path.name].update(file_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            with self.assertRaises(ValueError):
                self.verify(fixture)

    def test_foreign_path_and_incomplete_artifact_closure_are_rejected(self):
        for kind in ('foreign', 'missing', 'extra'):
            with self.subTest(kind=kind), self.temporary_directory() as directory:
                fixture = self.fixture(directory)
                rows, scratch = fixture[:2]
                name = 'shape-initial-noise.npy'
                if kind == 'foreign':
                    foreign = Path(directory) / name
                    foreign.write_bytes((scratch / name).read_bytes())
                    rows[name]['path'] = str(foreign)
                elif kind == 'missing':
                    rows.pop(name)
                else:
                    rows['unrelated.npy'] = copy.deepcopy(rows[name])
                with self.assertRaises(ValueError):
                    self.verify(fixture)


if __name__ == '__main__':
    unittest.main()
