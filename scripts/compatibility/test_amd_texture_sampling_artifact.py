"""CPU-only persistence contract; fixtures have the complete approved tensor shape."""
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
import amd_texture_sampling_artifact as artifact


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='texture-artifact-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw = (np.arange(3743 * 32, dtype=np.float32).reshape(3743, 32) / 128)
        self.denormalized = np.ascontiguousarray(self.raw * np.float32(.25) + np.float32(.5))
        self.noise = np.ascontiguousarray(-self.raw - np.float32(1))
        xyz = np.arange(3743, dtype=np.int32)
        self.coords = np.column_stack((np.zeros(3743, np.int32), xyz // 1024,
                                       xyz // 32 % 32, xyz % 32)).astype(np.int32)
        self.provenance = {'component': 'texture-flow', 'fixture': True,
                           'shape_latent_sha256': 'a' * 64,
                           'noise': {'seed': 20261015, 'source': 'fresh-independent-texture-noise'}}

    def write(self, **kwargs):
        args = dict(scratch=self.root, raw=self.raw, denormalized=self.denormalized,
                    coords=self.coords, provenance=self.provenance, noise=self.noise)
        args.update(kwargs)
        return artifact.write(**args)

    def test_full_roundtrip_and_standard_safetensors_layout(self):
        result = self.write()
        path = self.root / 'texture-latent.safetensors'
        blob = path.read_bytes()
        header_bytes = struct.unpack('<Q', blob[:8])[0]
        header = json.loads(blob[8:8 + header_bytes])
        tensors = {'raw': self.raw, 'denormalized': self.denormalized,
                   'coords': self.coords, 'noise': self.noise}
        self.assertEqual(set(header), {'__metadata__', *tensors})
        self.assertLess(len(blob), 1600000)
        for name, expected in tensors.items():
            row = header[name]
            begin, end = row['data_offsets']
            self.assertEqual(row['dtype'], 'I32' if name == 'coords' else 'F32')
            self.assertEqual(row['shape'], list(expected.shape))
            self.assertEqual(blob[8 + header_bytes + begin:8 + header_bytes + end], expected.tobytes())
        self.assertTrue(result['verified'])
        self.assertEqual(result['metadata']['artifact_sha256'], hashlib.sha256(blob).hexdigest())
        self.assertEqual(artifact.verify(path, self.root, result['metadata'])['metadata'], result['metadata'])
        self.assertEqual(result['metadata']['provenance'], self.provenance)

    def test_independent_reader_in_clean_process_without_torch(self):
        result = self.write()
        code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                'import amd_texture_sampling_artifact as a; '
                'r=a.verify(sys.argv[2],sys.argv[3]); '
                'assert r["verified"] and "torch" not in sys.modules; print(r["metadata"]["artifact_sha256"])')
        run = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent),
                              result['artifact_path'], str(self.root)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), result['metadata']['artifact_sha256'])

    def test_actual_probe_writer_exits_before_independent_parent_readback(self):
        run = subprocess.run([sys.executable, str(Path(__file__).with_name('amd_texture_sampling_artifact.py')),
                              '--probe', str(self.root)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertFalse(result['torch_imported'])
        checked = artifact.verify(result['artifact_path'], self.root / 'persistence-probe', result['metadata'])
        self.assertEqual(checked['metadata'], result['metadata'])

    def test_no_overwrite_even_when_only_metadata_already_exists(self):
        sidecar = self.root / 'texture-latent-metadata.json'
        sidecar.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(sidecar.read_bytes(), b'original')
        self.assertFalse((self.root / 'texture-latent.safetensors').exists())

    def test_repeated_write_preserves_both_files(self):
        result = self.write()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)
        self.assertTrue(artifact.verify(result['artifact_path'], self.root)['verified'])

    def test_payload_and_sidecar_tampering_are_rejected(self):
        result = self.write()
        path = Path(result['artifact_path'])
        blob = path.read_bytes()
        damaged = bytearray(blob)
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root)
        path.write_bytes(blob)
        sidecar = Path(result['metadata_path'])
        row = json.loads(sidecar.read_text())
        row['provenance']['fixture'] = False
        sidecar.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root)

    def test_expected_metadata_detects_consistently_replaced_artifact(self):
        result = self.write()
        expected = dict(result['metadata'])
        expected['artifact_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            artifact.verify(result['artifact_path'], self.root, expected)

    def test_boolean_tensor_offsets_cannot_masquerade_as_standard_integers(self):
        result = self.write()
        path = Path(result['artifact_path'])
        blob = path.read_bytes()
        length = struct.unpack('<Q', blob[:8])[0]
        header = json.loads(blob[8:8 + length])
        header['raw']['data_offsets'][0] = False
        changed_header = json.dumps(header, separators=(',', ':')).encode()
        changed_header += b' ' * (-len(changed_header) % 8)
        changed_blob = struct.pack('<Q', len(changed_header)) + changed_header + blob[8 + length:]
        path.write_bytes(changed_blob)
        metadata = result['metadata']
        metadata['artifact_bytes'] = len(changed_blob)
        metadata['artifact_sha256'] = hashlib.sha256(changed_blob).hexdigest()
        Path(result['metadata_path']).write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            artifact.verify(path, self.root)

    def test_hardlinked_output_never_modifies_an_existing_file(self):
        target = self.root / 'preserve.bin'
        target.write_bytes(b'keep')
        os.link(target, self.root / 'texture-latent.safetensors')
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(target.read_bytes(), b'keep')

    def test_reject_invalid_tensors_before_any_file_is_created(self):
        for bad in (self.raw.astype(np.float16), self.raw[:-1], self.raw[:, ::-1],
                    np.full_like(self.raw, np.nan)):
            with self.subTest(dtype=bad.dtype, shape=bad.shape):
                with self.assertRaises(ValueError):
                    self.write(raw=bad)
                self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaises(ValueError):
            self.write(noise=None)
        for bad in (self.coords.astype(np.int64), self.coords[::-1].copy(),
                    np.full_like(self.coords, 32)):
            with self.assertRaises(ValueError):
                self.write(coords=bad)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_reject_invalid_or_oversized_provenance_before_writing(self):
        for bad in ({'x': float('nan')}, {'checkpoint_sha256': 'bad'},
                    {'x': 'x' * 65536}, {'x': (1, 2)}, {}):
            with self.assertRaises(ValueError):
                self.write(provenance=bad)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_root_must_exist_be_absolute_and_be_outside_git(self):
        for bad in (self.root / 'absent', Path('.')):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.write(scratch=bad)
        (self.root / '.git').mkdir()
        with self.assertRaises(ValueError):
            self.write()

    def test_reader_rejects_wrong_root_filename_and_oversized_files(self):
        result = self.write()
        child = self.root / 'other'
        child.mkdir()
        for path, root in ((result['artifact_path'], child),
                           (self.root / 'other.safetensors', self.root)):
            with self.assertRaises(ValueError):
                artifact.verify(path, root)
        with Path(result['artifact_path']).open('ab') as stream:
            stream.write(b'\0' * 100000)
        with self.assertRaises(ValueError):
            artifact.verify(result['artifact_path'], self.root)

    def test_symlink_root_or_output_never_follows_target(self):
        child = self.root / 'target'
        child.mkdir()
        link = self.root / 'link'
        try:
            link.symlink_to(child, target_is_directory=True)
        except OSError as exc:
            self.skipTest('Windows symlink privilege unavailable: ' + str(exc))
        with self.assertRaises(ValueError):
            self.write(scratch=link)
        target = child / 'existing.bin'
        target.write_bytes(b'keep')
        (self.root / 'texture-latent.safetensors').symlink_to(target)
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(target.read_bytes(), b'keep')

    def test_probe_uses_fresh_directory_and_independently_readable_artifact(self):
        result = artifact.persistence_probe(self.root)
        self.assertTrue(result['verified'])
        self.assertFalse(result['torch_imported'])
        self.assertTrue(result['synthetic'])
        probe_root = self.root / 'persistence-probe'
        artifact.verify(result['artifact_path'], probe_root, result['metadata'])
        with self.assertRaises(FileExistsError):
            artifact.persistence_probe(self.root)


if __name__ == '__main__':
    unittest.main()
