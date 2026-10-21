"""Synthetic GLB transport tests; no real assets or Blender execution."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import io
import json
import struct
import unittest
from pathlib import Path
import numpy as np

MODULE = Path(__file__).with_name('verify_glb.py')
reader = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('independent_colour_glb_reader', MODULE)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)


def npy_digest(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def fixture(index_type=5123):
    vertices = np.array([[0, -0.0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3]], np.int32)
    colour = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [0, 1, .25]], np.float32)
    sv = np.array([11, 31, 21, 41], np.int64)
    sf = np.array([101, 103], np.int32)
    reference = {'vertices': vertices, 'faces': faces, 'linear-COLOR0': colour,
                 'source-vertex-ids': sv, 'source-face-ids': sf,
                 'sourceVertexIDs_SHA256': npy_digest(sv), 'sourceFaceIDs_SHA256': npy_digest(sf)}
    position = vertices[:, [0, 2, 1]].copy()
    position[:, 2] *= -1
    binary = bytearray()
    views = []
    for payload, target in ((position.tobytes(), 34962), (colour.tobytes(), 34962),
                            (faces.astype('<u2' if index_type == 5123 else '<u4').tobytes(), 34963)):
        binary.extend(b'\0' * (-len(binary) % 4))
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(payload), 'target': target})
        binary.extend(payload)
    binary.extend(b'\0' * (-len(binary) % 4))
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'COLOR_0': 1},
                     'indices': 2, 'material': 0, 'mode': 4}], 'extras': {
                     'sourceVertexIDs': sv.tolist(), 'sourceFaceIDs': sf.tolist(),
                     'sourceVertexIDs_SHA256': reference['sourceVertexIDs_SHA256'],
                     'sourceFaceIDs_SHA256': reference['sourceFaceIDs_SHA256']}}],
           'materials': [{'alphaMode': 'OPAQUE', 'doubleSided': True, 'pbrMetallicRoughness': {
               'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': .00011082693761409814,
               'roughnessFactor': .999995986777544}}], 'buffers': [{'byteLength': len(binary)}],
           'bufferViews': views, 'accessors': [
               {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
                'min': position.min(0).tolist(), 'max': position.max(0).tolist()},
               {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
               {'bufferView': 2, 'componentType': index_type, 'count': 6, 'type': 'SCALAR'}]}
    return doc, bytes(binary), reference


def container(doc, binary, raw_json=None):
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode() if raw_json is None else raw_json
    encoded += b' ' * (-len(encoded) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary))
            + struct.pack('<II', len(encoded), 0x4e4f534a) + encoded
            + struct.pack('<II', len(binary), 0x004e4942) + binary)


class ReaderAvailability(unittest.TestCase):
    def test_independent_parser_exists(self):
        self.assertIsNotNone(reader, 'Independent portable GLB parser has not been implemented')


@unittest.skipIf(reader is None, 'Parser implementation pending')
class ContainerTests(unittest.TestCase):
    def check(self, doc=None, binary=None, reference=None):
        original_doc, original_binary, original_reference = fixture()
        return reader.parse_glb(container(original_doc if doc is None else doc,
                                          original_binary if binary is None else binary),
                                original_reference if reference is None else reference)

    def test_dense_u16_and_u32_preserve_all_source_bits(self):
        for component in (5123, 5125):
            doc, binary, reference = fixture(component)
            result = reader.parse_glb(container(doc, binary), reference)
            np.testing.assert_array_equal(result['vertices'].view(np.uint32), reference['vertices'].view(np.uint32))
            np.testing.assert_array_equal(result['faces'], reference['faces'])
            np.testing.assert_array_equal(result['linear-COLOR0'].view(np.uint32), reference['linear-COLOR0'].view(np.uint32))
            self.assertEqual(result['report']['triangles'], 2)
            self.assertTrue(result['report']['source_ID_lists_and_NPY_hashes_exact'])

    def test_header_length_version_truncation_and_size_cap_fail(self):
        doc, binary, ref = fixture()
        valid = container(doc, binary)
        corrupt = [b'', valid[:-1], b'bad!' + valid[4:], valid[:4] + struct.pack('<I', 1) + valid[8:],
                   valid[:8] + struct.pack('<I', len(valid) + 4) + valid[12:], b'x' * (2 * 1024**2 + 1)]
        for payload in corrupt:
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                reader.parse_glb(payload, ref)

    def test_duplicate_json_members_and_extra_chunks_fail(self):
        doc, binary, ref = fixture()
        encoded = json.dumps(doc, separators=(',', ':')).encode().replace(b'"scene":0', b'"scene":0,"scene":0')
        with self.assertRaises(ValueError):
            reader.parse_glb(container(doc, binary, encoded), ref)
        payload = container(doc, binary) + struct.pack('<II', 0, 0x004e4942)
        payload = payload[:8] + struct.pack('<I', len(payload)) + payload[12:]
        with self.assertRaises(ValueError):
            reader.parse_glb(payload, ref)

    def test_external_extensions_textures_normals_uv_and_node_transform_fail(self):
        changes = [lambda d: d['buffers'][0].update(uri='outside.bin'),
                   lambda d: d.update(extensionsUsed=['ANY']), lambda d: d.update(images=[]),
                   lambda d: d.update(textures=[]), lambda d: d.update(samplers=[]),
                   lambda d: d['meshes'][0]['primitives'][0]['attributes'].update(NORMAL=0),
                   lambda d: d['meshes'][0]['primitives'][0]['attributes'].update(TEXCOORD_0=0),
                   lambda d: d['nodes'][0].update(translation=[0, 0, 0])]
        for change in changes:
            doc, binary, ref = fixture(); change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, binary), ref)

    def test_accessor_type_count_normalization_sparse_and_bounds_fail(self):
        changes = [lambda d: d['accessors'][0].update(componentType=5125),
                   lambda d: d['accessors'][1].update(count=3),
                   lambda d: d['accessors'][1].update(normalized=True),
                   lambda d: d['accessors'][0].update(sparse={}),
                   lambda d: d['bufferViews'][0].update(byteOffset=2),
                   lambda d: d['accessors'][0].update(byteOffset=2),
                   lambda d: d['bufferViews'][0].update(byteLength=44),
                   lambda d: d['accessors'][0].update(min=[-1, 0, -1]),
                   lambda d: d['accessors'][1].update(bufferView=0)]
        for change in changes:
            doc, binary, ref = fixture(); change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, binary), ref)

    def test_changed_colour_nonfinite_position_and_reversed_winding_fail(self):
        for kind in ('colour', 'position', 'winding'):
            doc, binary, ref = fixture(); binary = bytearray(binary)
            if kind == 'colour':
                offset = doc['bufferViews'][1]['byteOffset']; binary[offset:offset+4] = struct.pack('<f', .125)
            elif kind == 'position':
                binary[:4] = struct.pack('<f', float('nan'))
            else:
                offset = doc['bufferViews'][2]['byteOffset']; binary[offset:offset+4] = struct.pack('<HH', 1, 0)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, bytes(binary)), ref)

    def test_material_factors_alpha_and_sidedness_fail_closed(self):
        changes = [lambda m: m.update(alphaMode='BLEND'), lambda m: m.update(doubleSided=False),
                   lambda m: m['pbrMetallicRoughness'].update(metallicFactor=0),
                   lambda m: m['pbrMetallicRoughness'].update(roughnessFactor=.8),
                   lambda m: m['pbrMetallicRoughness'].update(baseColorFactor=[.9, 1, 1, 1]),
                   lambda m: m['pbrMetallicRoughness'].update(baseColorTexture={'index': 0})]
        for change in changes:
            doc, binary, ref = fixture(); change(doc['materials'][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, binary), ref)

    def test_schema_integer_and_number_fields_reject_boolean_or_float_substitutes(self):
        changes = [lambda d: d.update(scene=False),
                   lambda d: d['meshes'][0]['primitives'][0].update(mode=4.0),
                   lambda d: d['accessors'][0].update(componentType=5126.0),
                   lambda d: d['bufferViews'][0].update(buffer=False),
                   lambda d: d['materials'][0]['pbrMetallicRoughness'].update(baseColorFactor=[True, 1, 1, 1])]
        for change in changes:
            doc, binary, ref = fixture(); change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, binary), ref)

    def test_original_source_ID_lists_and_file_hashes_are_required(self):
        for key in ('sourceVertexIDs', 'sourceFaceIDs', 'sourceVertexIDs_SHA256', 'sourceFaceIDs_SHA256'):
            doc, binary, ref = fixture()
            if key.endswith('SHA256'):doc['meshes'][0]['extras'][key] = '0' * 64
            else:doc['meshes'][0]['extras'][key][0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                reader.parse_glb(container(doc, binary), ref)


@unittest.skipIf(reader is None, 'Parser implementation pending')
class QueryTests(unittest.TestCase):
    def sample(self, target_scalar=1.):
        colours = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        qf = np.array([0], np.int32)
        bary = np.array([[.5, .5, 0]], np.float64)
        raw = np.array([[.7353569830524495] * 3 + [.00011082693761409814,
                        target_scalar, 1]], np.float32)
        return colours, faces, qf, bary, raw, {'all': np.array([0], np.int64)}

    def test_exported_linear_colours_reproduce_srgb_query_and_opaque_scalars(self):
        report = reader.evaluate_queries(*self.sample())
        self.assertTrue(report['all_gates_pass'])
        self.assertLess(report['cohorts']['all']['RGB']['max_abs'], 1e-7)
        self.assertEqual(report['cohorts']['all']['scalars']['alpha']['rmse'], 0)

    def test_scalar_channel_failure_is_not_hidden_by_accurate_rgb(self):
        report = reader.evaluate_queries(*self.sample(.8))
        self.assertFalse(report['all_gates_pass'])
        self.assertTrue(report['cohorts']['all']['RGB']['pass'])
        self.assertFalse(report['cohorts']['all']['scalars']['roughness']['pass'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
