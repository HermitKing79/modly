"""Synthetic reader tests; fixture construction does not import an export writer."""
import sys
sys.dont_write_bytecode = True
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest import mock
import numpy as np

MODULE = Path(__file__).with_name('verify_stream.py')
SCRATCH = MODULE.parent.parents[2].parent / 'artifacts/amd-brick-full-sampled-glb-3db265c/unit-scratch'
reader = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('independent_full_brick_reader', MODULE)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)


def fixture():
    vertices = np.array([[-2, -0., 3], [1, -3, -1], [0., 2, 1], [4, 1, -2]], '<f4')
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 2]], '<i4')
    colours = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [-0., 1, .25]], '<f4')
    hashes = {'vertices_SHA256': 'a' * 64, 'faces_SHA256': 'b' * 64,
              'linear_COLOR0_SHA256': 'c' * 64}
    # Literal glTF coordinates keep this fixture independent of reader transforms.
    positions = [[-2, 3, 0.], [1, -1, 3], [0., 1, -2], [4, -2, -1]]
    binary = (struct.pack('<12f', *(v for row in positions for v in row))
              + struct.pack('<12f', *(v for row in colours.tolist() for v in row))
              + struct.pack('<9I', 0, 1, 2, 1, 0, 3, 0, 1, 2))
    doc = {'asset': {'version': '2.0', 'generator': 'independent synthetic fixture'},
           'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'COLOR_0': 1},
                                      'indices': 2, 'material': 0, 'mode': 4}],
                       'extras': hashes.copy()}],
           'materials': [{'alphaMode': 'OPAQUE', 'doubleSided': True,
                          'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1],
                            'metallicFactor': .00011082693761409814,
                            'roughnessFactor': .999995986777544}}],
           'buffers': [{'byteLength': 132}],
           'bufferViews': [
               {'buffer': 0, 'byteOffset': 0, 'byteLength': 48, 'target': 34962},
               {'buffer': 0, 'byteOffset': 48, 'byteLength': 48, 'target': 34962},
               {'buffer': 0, 'byteOffset': 96, 'byteLength': 36, 'target': 34963}],
           'accessors': [
               {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
                'min': [-2, -2, -2], 'max': [4, 3, 3]},
               {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
               {'bufferView': 2, 'componentType': 5125, 'count': 9, 'type': 'SCALAR'}]}
    return doc, binary, vertices, faces, colours, hashes


def container(doc, binary, raw_json=None):
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode() if raw_json is None else raw_json
    encoded += b' ' * (-len(encoded) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary))
            + struct.pack('<II', len(encoded), 0x4e4f534a) + encoded
            + struct.pack('<II', len(binary), 0x004e4942) + binary)


class ReaderAvailability(unittest.TestCase):
    def test_independent_stream_reader_exists(self):
        self.assertIsNotNone(reader, 'Independent chunked reader has not been implemented')


@unittest.skipIf(reader is None, 'Reader implementation pending')
class StreamTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic.glb'

    def run_reader(self, doc=None, binary=None, refs=None, raw_json=None, payload=None, batch_rows=2):
        original_doc, original_binary, *original_refs = fixture()
        self.path.write_bytes(container(original_doc if doc is None else doc,
                                        original_binary if binary is None else binary, raw_json)
                              if payload is None else payload)
        return reader.verify(self.path, *(original_refs if refs is None else refs), batch_rows=batch_rows)

    def test_valid_noncoplanar_mesh_preserves_signed_zero_and_duplicate_triangles(self):
        result = self.run_reader()
        self.assertEqual(result['report']['vertices'], 4)
        self.assertEqual(result['report']['triangles'], 3)
        self.assertTrue(result['report']['original_XYZ_corner_bits_indices_winding_multiplicity_exact'])
        self.assertTrue(result['report']['F32_linear_COLOR0_bits_exact'])
        colour = result['linear_COLOR0']
        self.assertEqual(colour.dtype, np.dtype('<f4'))
        self.assertFalse(colour.flags.writeable)
        np.testing.assert_array_equal(colour.view('<u4'), fixture()[4].view('<u4'))
        # The returned view stays mapped after verify closes its file handle.
        self.assertEqual(float(colour[3, 1]), 1.)
        del result, colour

    def test_flushed_file_path_and_handle_metadata_use_stable_comparisons(self):
        # On Windows path ctime can denote birth time while fstat ctime denotes
        # metadata change time. A delayed, fsynced write exposes that difference.
        payload = container(*fixture()[:2])
        with self.path.open('xb') as handle:
            time.sleep(.02)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        result = reader.verify(self.path, *fixture()[2:], batch_rows=2)
        np.testing.assert_array_equal(result['linear_COLOR0'].view('<u4'), fixture()[4].view('<u4'))
        del result

    def test_metadata_alteration_during_verification_is_rejected(self):
        self.path.write_bytes(container(*fixture()[:2]))
        original_bound = reader._vector_bound
        altered = False

        def bound_with_real_file_change(value):
            # Schedule a real metadata change between the initial and final
            # snapshots, without introducing a timing-dependent race thread.
            nonlocal altered
            result = original_bound(value)
            if not altered:
                before = self.path.stat()
                os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
                altered = True
            return result

        with mock.patch.object(reader, '_vector_bound', side_effect=bound_with_real_file_change):
            with self.assertRaisesRegex(ValueError, 'changed during independent verification'):
                reader.verify(self.path, *fixture()[2:], batch_rows=2)

    def test_changed_position_colour_signed_zero_and_triangle_identity_fail(self):
        changes = [('position', 0, '<f', -1.), ('position zero sign', 8, '<I', 0x80000000),
                   ('colour', 48, '<f', .125), ('colour zero sign', 84, '<I', 0),
                   ('winding', 96, '<I', 1), ('duplicate lost', 120, '<I', 3),
                   ('out of range', 96, '<I', 4), ('NaN', 0, '<I', 0x7fc00000),
                   ('colour infinity', 48, '<I', 0x7f800000)]
        for name, offset, fmt, value in changes:
            doc, binary, *_ = fixture()
            altered = bytearray(binary)
            struct.pack_into(fmt, altered, offset, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_reader(doc, altered)

    def test_source_array_schema_finite_colours_and_indices_are_required(self):
        changes = [lambda r: r.__setitem__(0, r[0].astype('<f8')),
                   lambda r: r.__setitem__(1, r[1].astype('<u4')),
                   lambda r: r.__setitem__(2, r[2].astype('<f8')),
                   lambda r: r[0].__setitem__((0, 0), np.nan),
                   lambda r: r[2].__setitem__((0, 0), 1.1),
                   lambda r: r[1].__setitem__((0, 0), -1),
                   lambda r: r[1].__setitem__((0, 0), 4),
                   lambda r: r.__setitem__(0, r[0][:, :2])]
        for change in changes:
            refs = list(fixture()[2:]); change(refs)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_reader(refs=refs)

    def test_header_chunk_order_count_alignment_and_exact_length_fail_closed(self):
        doc, binary, *_ = fixture()
        valid = container(doc, binary)
        json_len = struct.unpack_from('<I', valid, 12)[0]
        bin_head = 20 + json_len
        payloads = [b'', valid[:-1], valid + b'junk', b'nope' + valid[4:],
                    valid[:4] + struct.pack('<I', 1) + valid[8:],
                    valid[:8] + struct.pack('<I', len(valid) + 4) + valid[12:],
                    valid[:12] + struct.pack('<II', json_len, 0x004e4942) + valid[20:],
                    valid[:12] + struct.pack('<I', json_len - 1) + valid[16:],
                    valid[:bin_head + 4] + struct.pack('<I', 0x4e4f534a) + valid[bin_head + 8:]]
        extra = valid + struct.pack('<II', 4, 0x004e4942) + b'\0' * 4
        payloads.append(extra[:8] + struct.pack('<I', len(extra)) + extra[12:])
        for payload in payloads:
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                self.run_reader(payload=payload)

    def test_json_duplicates_nonfinite_overflow_and_oversize_rejected(self):
        doc, binary, *_ = fixture()
        raw = json.dumps(doc, separators=(',', ':')).encode()
        for changed in (raw.replace(b'"scene":0', b'"scene":0,"scene":0'),
                        raw.replace(b'"version":"2.0"', b'"version":"2.0","x":NaN'),
                        raw.replace(b'"scene":0', b'"scene":1e400'),
                        b' ' * 65536 + raw):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.run_reader(raw_json=changed)

    def test_dense_types_counts_offsets_bounds_and_boolean_substitutes_rejected(self):
        changes = [lambda d: d['accessors'][2].update(componentType=5123),
                   lambda d: d['accessors'][2].update(count=6),
                   lambda d: d['accessors'][1].update(type='VEC4'),
                   lambda d: d['accessors'][0].update(normalized=True),
                   lambda d: d['accessors'][0].update(sparse={}),
                   lambda d: d['accessors'][1].update(bufferView=0),
                   lambda d: d['accessors'][0].update(byteOffset=4),
                   lambda d: d['accessors'][0].update(min=[-2, -2, -3]),
                   lambda d: d['accessors'][0].update(max=[4.000000000000001, 3, 3]),
                   lambda d: d['accessors'][0].update(min=[False, -2, -2]),
                   lambda d: d['bufferViews'][1].update(byteOffset=44),
                   lambda d: d['bufferViews'][0].update(byteLength=52),
                   lambda d: d['bufferViews'][0].update(byteStride=16),
                   lambda d: d['buffers'][0].update(byteLength=136),
                   lambda d: d.update(scene=False),
                   lambda d: d['accessors'][0].update(count=4.),
                   lambda d: d['bufferViews'][0].update(buffer=False),
                   lambda d: d['meshes'][0]['primitives'][0].update(mode=4.)]
        for change in changes:
            doc = fixture()[0]; change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_reader(doc)

    def test_extra_core_content_transforms_external_data_and_extensions_rejected(self):
        changes = [lambda d: d.update(images=[]), lambda d: d.update(textures=[]),
                   lambda d: d.update(samplers=[]), lambda d: d.update(animations=[]),
                   lambda d: d.update(extensionsUsed=[]),
                   lambda d: d['buffers'][0].update(uri='elsewhere.bin'),
                   lambda d: d['nodes'][0].update(matrix=[1] * 16),
                   lambda d: d['nodes'][0].update(translation=[0, 0, 0]),
                   lambda d: d['meshes'][0]['primitives'][0]['attributes'].update(NORMAL=0),
                   lambda d: d['meshes'][0]['primitives'][0]['attributes'].update(TEXCOORD_0=0),
                   lambda d: d['meshes'][0].update(extensions={}),
                   lambda d: d['scenes'].append({'nodes': [0]}),
                   lambda d: d['meshes'][0]['primitives'].append(copy.deepcopy(d['meshes'][0]['primitives'][0]))]
        for change in changes:
            doc = fixture()[0]; change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_reader(doc)

    def test_material_and_compact_source_hash_extras_are_exact(self):
        changes = [lambda d: d['materials'][0].update(alphaMode='BLEND'),
                   lambda d: d['materials'][0].update(doubleSided=False),
                   lambda d: d['materials'][0]['pbrMetallicRoughness'].update(metallicFactor=0.),
                   lambda d: d['materials'][0]['pbrMetallicRoughness'].update(roughnessFactor=1.),
                   lambda d: d['materials'][0]['pbrMetallicRoughness'].update(baseColorFactor=[True, 1, 1, 1]),
                   lambda d: d['materials'][0]['pbrMetallicRoughness'].update(baseColorTexture={'index': 0}),
                   lambda d: d['meshes'][0]['extras'].update(vertices_SHA256='0' * 64),
                   lambda d: d['meshes'][0]['extras'].update(sourceVertexIDs=[1, 2, 3]),
                   lambda d: d['meshes'][0].pop('extras')]
        for change in changes:
            doc = fixture()[0]; change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_reader(doc)

    def test_invalid_batch_sizes_and_reference_hashes_fail(self):
        for batch in (0, -1, True, 1.5, 32769):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.run_reader(batch_rows=batch)
        for hashes in ({}, {'x': []}, {'x': 'short'}):
            refs = list(fixture()[2:]); refs[3] = hashes
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                self.run_reader(refs=refs)

    def test_sparse_oversized_file_rejected_without_reading_payload(self):
        with self.path.open('wb') as handle:
            handle.truncate(128 * 1024**2 + 1)
        with self.assertRaises(ValueError):
            reader.verify(self.path, *fixture()[2:])

    def test_hardlink_rejected_when_platform_allows_fixture(self):
        self.path.write_bytes(container(*fixture()[:2]))
        alias = self.path.with_name('alias.glb')
        try:
            os.link(self.path, alias)
        except PermissionError:
            self.skipTest('Sandbox denies creating synthetic hardlink fixture')
        with self.assertRaises(ValueError):
            reader.verify(self.path, *fixture()[2:])

    def test_directory_rejected(self):
        with self.assertRaises(ValueError):
            reader.verify(self.path.parent, *fixture()[2:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
