"""CPU-only texture-field persistence and independent safetensors readback."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    import amd_texture_field_artifact as artifact
except ModuleNotFoundError:
    artifact = None


class TextureFieldArtifactTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(artifact, 'CPU texture-field writer must exist')
        self.tmp = tempfile.TemporaryDirectory(prefix='texture-field-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw = np.array([[-3, -2, -1, 0, 1, 2], [3, 4, 5, 6, 7, 8]], dtype=np.float32)
        self.features = np.array([[-1, -.5, 0, .5, 1, 1.5], [2, 2.5, 3, 3.5, 4, 4.5]], dtype=np.float32)
        # Preserve genuine arbitrary C2S order instead of sorting coordinates.
        self.coordinates = np.array([[0, 511, 2, 3], [0, 0, 1, 0]], dtype=np.int32)
        self.provenance = {'mode': 'fixture', 'purpose': 'cpu-only-writer-acceptance',
                           'input_sha256': 'a' * 64}

    def write(self, **changes):
        values = dict(root=self.root, raw=self.raw, features=self.features,
                      coordinates=self.coordinates, provenance=self.provenance)
        values.update(changes)
        return artifact.write(**values)

    def test_roundtrip_preserves_unclamped_features_and_channel_statistics(self):
        result = self.write()
        path = self.root / 'texture-field.safetensors'
        self.assertTrue(result['verified'])
        self.assertEqual(result['metadata']['artifact_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['metadata']['payload_bytes'], 128)
        self.assertEqual(result['metadata']['provenance'], self.provenance)
        self.assertEqual(result['metadata']['channels'], ['red', 'green', 'blue', 'metallic', 'roughness', 'alpha'])
        stats = result['metadata']['channel_statistics']
        self.assertEqual(stats['features']['red'], {'min': -1.0, 'max': 2.0, 'mean': .5})
        self.assertEqual(stats['features']['alpha'], {'min': 1.5, 'max': 4.5, 'mean': 3.0})
        self.assertEqual(stats['raw']['red'], {'min': -3.0, 'max': 3.0, 'mean': 0.0})
        self.assertEqual(artifact.verify(path, self.root, result['metadata'])['metadata'], result['metadata'])

    def test_standard_safetensors_reader_recovers_exact_bytes(self):
        from safetensors.numpy import load_file
        result = self.write()
        restored = load_file(result['artifact_path'])
        self.assertEqual(set(restored), {'raw', 'features', 'coordinates'})
        for name, expected in dict(raw=self.raw, features=self.features, coordinates=self.coordinates).items():
            self.assertEqual(restored[name].dtype, expected.dtype)
            self.assertEqual(restored[name].shape, expected.shape)
            self.assertEqual(restored[name].tobytes(), expected.tobytes())

    def test_reader_is_independent_clean_process_without_torch(self):
        result = self.write()
        code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                'import amd_texture_field_artifact as a; '
                'r=a.verify(sys.argv[2],sys.argv[3]); '
                'assert r["verified"] and "torch" not in sys.modules; '
                'print(r["metadata"]["artifact_sha256"])')
        run = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent),
                              result['artifact_path'], str(self.root)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), result['metadata']['artifact_sha256'])

    def test_actual_writer_exits_before_independent_parent_readback(self):
        code = ('import sys,numpy as np; sys.path.insert(0,sys.argv[1]); '
                'import amd_texture_field_artifact as a; '
                'raw=np.array([[-3,-2,-1,0,1,2]],dtype=np.float32); '
                'features=np.array([[-1,-.5,0,.5,1,1.5]],dtype=np.float32); '
                'coords=np.array([[0,511,0,1]],dtype=np.int32); '
                'r=a.write(sys.argv[2],raw,features,coords,{"mode":"fixture"}); '
                'assert r["verified"] and "torch" not in sys.modules; '
                'print(r["metadata"]["artifact_sha256"])')
        run = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent), str(self.root)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        checked = artifact.verify(self.root / 'texture-field.safetensors', self.root)
        self.assertEqual(run.stdout.strip(), checked['metadata']['artifact_sha256'])

    def test_writer_refuses_clamped_or_incorrect_transform_before_creating_files(self):
        for features in (np.clip(self.features, 0, 1), self.features + np.float32(.01)):
            with self.subTest(features=features.tolist()):
                with self.assertRaises(ValueError):
                    self.write(features=features)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_fixture_cannot_claim_production_or_omit_its_mode(self):
        for provenance in ({'mode': 'production'}, {'purpose': 'fixture'}, {'mode': True}, {'mode': 'other'}):
            with self.subTest(provenance=provenance):
                with self.assertRaises(ValueError):
                    self.write(provenance=provenance)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_arrays_never_create_files(self):
        cases = [dict(raw=self.raw.astype(np.float16)), dict(raw=self.raw[:, ::-1]),
                 dict(raw=np.full_like(self.raw, np.nan)), dict(features=np.full_like(self.features, np.inf)),
                 dict(features=self.features[:1]), dict(raw=np.empty((0, 6), np.float32)),
                 dict(coordinates=self.coordinates.astype(np.int64)),
                 dict(coordinates=np.array([[0, 512, 0, 0], [0, 0, 0, 0]], np.int32)),
                 dict(coordinates=np.array([[1, 0, 0, 0], [0, 0, 0, 0]], np.int32)),
                 dict(coordinates=np.zeros((2, 4), np.int32))]
        for change in cases:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    self.write(**change)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_oversized_array_refused_before_reading_payload(self):
        huge = np.broadcast_to(np.zeros((1, 6), np.float32), (1842969, 6))
        with self.assertRaises(ValueError):
            self.write(raw=huge)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_provenance_refused_before_writing(self):
        for provenance in ({'mode': 'fixture', 'x': float('nan')},
                           {'mode': 'fixture', 'checkpoint_sha256': 'bad'},
                           {'mode': 'fixture', 'x': 'x' * 65536},
                           {'mode': 'fixture', 'x': (1, 2)}):
            with self.assertRaises(ValueError):
                self.write(provenance=provenance)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_sidecar_or_artifact_is_never_overwritten(self):
        sidecar = self.root / 'texture-field-metadata.json'
        sidecar.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(sidecar.read_bytes(), b'preserve')
        self.assertFalse((self.root / 'texture-field.safetensors').exists())
        sidecar.unlink()
        result = self.write()
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})
        self.assertTrue(artifact.verify(result['artifact_path'], self.root)['verified'])

    def test_payload_sidecar_and_expected_receipt_tampering_are_detected(self):
        result = self.write()
        path = Path(result['artifact_path'])
        original = path.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 1
        path.write_bytes(changed)
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root)
        path.write_bytes(original)
        expected = dict(result['metadata'], artifact_sha256='0' * 64)
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root, expected)
        sidecar = Path(result['metadata_path'])
        metadata = json.loads(sidecar.read_text())
        metadata['channel_statistics']['features']['red']['min'] = 0
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root)

    def test_boolean_offsets_extra_tensors_and_trailing_bytes_are_rejected(self):
        result = self.write()
        path = Path(result['artifact_path'])
        original = path.read_bytes()
        count = struct.unpack('<Q', original[:8])[0]
        for change in ('boolean', 'extra', 'trailing'):
            header = json.loads(original[8:8 + count])
            if change == 'boolean':
                header['raw']['data_offsets'][0] = False
            if change == 'extra':
                header['unexpected'] = header['raw']
            data = json.dumps(header, separators=(',', ':')).encode()
            data += b' ' * (-len(data) % 8)
            blob = struct.pack('<Q', len(data)) + data + original[8 + count:]
            if change == 'trailing':
                blob += b'\0'
            path.write_bytes(blob)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    artifact.verify(path, self.root)

    def test_outside_git_root_must_exist_be_absolute_and_have_no_alias(self):
        child = self.root / 'child'
        child.mkdir()
        for root in (self.root / 'missing', Path('.'), child / '..'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.write(root=root)
        (self.root / '.git').write_text('gitdir: elsewhere')
        with self.assertRaises(ValueError):
            self.write(root=child)
        self.assertEqual(list(child.iterdir()), [])

    def test_reader_rejects_wrong_root_or_filename(self):
        result = self.write()
        child = self.root / 'child'
        child.mkdir()
        for path, root in ((result['artifact_path'], child), (self.root / 'other.safetensors', self.root)):
            with self.assertRaises(ValueError):
                artifact.verify(path, root)

    def test_hardlinks_cannot_modify_existing_target(self):
        target = self.root / 'target.bin'
        target.write_bytes(b'keep')
        path = self.root / 'texture-field.safetensors'
        try:
            os.link(target, path)
        except OSError as exc:
            self.skipTest('Hardlink creation unavailable: ' + str(exc))
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(target.read_bytes(), b'keep')

    def test_symlinks_cannot_modify_existing_target(self):
        target = self.root / 'target.bin'
        target.write_bytes(b'keep')
        path = self.root / 'texture-field.safetensors'
        try:
            path.symlink_to(target)
        except OSError as exc:
            self.skipTest('Windows symlink privilege unavailable: ' + str(exc))
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(target.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
