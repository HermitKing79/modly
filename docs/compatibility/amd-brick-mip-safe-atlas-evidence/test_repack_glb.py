"""CPU-only synthetic containers; never opens or writes the accepted artifact."""
import copy
import io
import json
import struct
import unittest
import numpy as np
from PIL import Image

try:
    from repack_glb import repack_glb
except ImportError:
    repack_glb = None


def png(mode, value):
    image = np.full((1024, 1024, len(mode)), value, np.uint8)
    image[100:200, 100:600] = (value + 41) % 256
    out = io.BytesIO()
    Image.fromarray(image, mode).save(out, format='PNG')
    return out.getvalue()


def encode(doc, binary):
    j = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    j += b' ' * (-len(j) % 4)
    binary = bytes(binary) + b'\0' * (-len(binary) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28 + len(j) + len(binary))
            + struct.pack('<II', len(j), 0x4e4f534a) + j
            + struct.pack('<II', len(binary), 0x004e4942) + binary)


def unpack(payload):
    n = struct.unpack_from('<I', payload, 12)[0]
    return json.loads(payload[20:20+n]), payload[28+n:]


def parts(doc, binary):
    return [binary[v.get('byteOffset', 0):v.get('byteOffset', 0)+v['byteLength']]
            for v in doc['bufferViews']]


def fixture():
    # Two coincident triangles are distinguished by actual old UV island.
    v = np.array([[0, 1, 2], [1, 1, 2], [0, 2, 2]] * 2, np.float32)
    f = np.arange(6, dtype=np.int32).reshape(-1, 3)
    old = np.array([[.125, .125], [.25, .125], [.125, .25],
                    [.625, .625], [.75, .625], [.625, .75]], np.float32)
    new = old + np.array([[.125, .25]] * 3 + [[-.125, -.25]] * 3, np.float32)
    order = [4, 2, 0, 5, 1, 3]
    inverse = np.argsort(order)
    positions = v[order][:, [0, 2, 1]].copy()
    positions[:, 2] *= -1
    uv = old[order].copy()
    uv[:, 1] = 1 - uv[:, 1]
    indices = inverse[[1, 2, 0, 5, 3, 4]].astype('<u2')
    raw = [positions.tobytes(), np.arange(18, dtype='<f4').tobytes(),
           uv.tobytes(), indices.tobytes(), png('RGBA', 12), png('RGB', 34)]
    binary = bytearray()
    views = []
    for i, blob in enumerate(raw):
        binary.extend(b'\0' * (-len(binary) % 4))
        view = dict(buffer=0, byteOffset=len(binary), byteLength=len(blob))
        if i < 4:
            view['target'] = 34963 if i == 3 else 34962
        views.append(view)
        binary.extend(blob)
    binary.extend(b'\0' * (-len(binary) % 4))
    doc = dict(asset=dict(version='2.0'), scene=0, scenes=[dict(nodes=[0])],
               nodes=[dict(mesh=0)], buffers=[dict(byteLength=len(binary))],
               bufferViews=views,
               accessors=[dict(bufferView=i, componentType=5123 if i == 3 else 5126,
                               count=6, type=['VEC3', 'VEC3', 'VEC2', 'SCALAR'][i])
                          for i in range(4)],
               meshes=[dict(primitives=[dict(attributes=dict(POSITION=0, NORMAL=1, TEXCOORD_0=2),
                                             indices=3, material=0)])],
               materials=[dict(doubleSided=True, pbrMetallicRoughness=dict(
                   baseColorTexture=dict(index=0), metallicRoughnessTexture=dict(index=1)))],
               textures=[dict(source=0, sampler=0), dict(source=1, sampler=0)],
               images=[dict(bufferView=4, mimeType='image/png'), dict(bufferView=5, mimeType='image/png')],
               samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)])
    return doc, binary, v, f, old, new, order


class RepackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.mr = png('RGBA', 98), png('RGB', 201)

    def call(self, doc=None, binary=None, **changes):
        d, b, v, f, old, new, _ = fixture()
        values = dict(payload=encode(d if doc is None else doc, b if binary is None else binary),
                      vertices=v, faces=f, old_uv=old, new_uv=new,
                      base_png=self.base, mr_png=self.mr)
        values.update(changes)
        self.assertIsNotNone(repack_glb, 'The narrowly scoped repack helper is missing')
        return repack_glb(**values)

    def test_changes_only_uv_and_png_views_and_reflows_aligned_container(self):
        d, b, v, f, old, new, order = fixture()
        original = encode(d, b)
        output = self.call(payload=original)
        after, data = unpack(output)
        before_parts, after_parts = parts(d, b), parts(after, data)
        self.assertEqual(after_parts[:2] + after_parts[3:4], before_parts[:2] + before_parts[3:4])
        want_uv = new[order].copy()
        want_uv[:, 1] = 1 - want_uv[:, 1]
        self.assertEqual(after_parts[2], want_uv.tobytes())
        self.assertEqual(after_parts[4:], [self.base, self.mr])
        allowed = copy.deepcopy(after)
        allowed['bufferViews'], allowed['buffers'] = d['bufferViews'], d['buffers']
        self.assertEqual(allowed, d)
        self.assertEqual(after['buffers'][0]['byteLength'], len(data))
        self.assertTrue(all(x['byteOffset'] % 4 == 0 for x in after['bufferViews']))
        self.assertEqual(original, encode(d, b))
        np.testing.assert_array_equal(old, fixture()[4])
        np.testing.assert_array_equal(new, fixture()[5])
        self.assertEqual(output, self.call(payload=original))

    def test_rejects_changed_orientation_geometry_and_nonactual_old_uv(self):
        _, _, v, f, old, _, _ = fixture()
        wrong = old.copy()
        wrong[0, 0] = np.nextafter(wrong[0, 0], np.float32(1))
        moved = v.copy()
        moved[0, 0] += .001
        for change in (dict(faces=f[:, [0, 2, 1]]), dict(vertices=moved), dict(old_uv=wrong)):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                self.call(**change)

    def test_rejects_ambiguous_position_old_uv_to_new_uv_mapping(self):
        d, b, v, f, old, new, order = fixture()
        # Share one position/oldUV corner, with otherwise distinct triangles.
        v[4, 2] = 3
        old[3] = old[0]
        position = v[order][:, [0, 2, 1]].copy()
        position[:, 2] *= -1
        b[:len(position.tobytes())] = position.tobytes()
        uv = old[order].copy()
        uv[:, 1] = 1 - uv[:, 1]
        start = d['bufferViews'][2]['byteOffset']
        b[start:start+len(uv.tobytes())] = uv.tobytes()
        with self.assertRaises(ValueError):
            self.call(d, b, vertices=v, old_uv=old, new_uv=new)
        new[3] = new[0]
        output, data = unpack(self.call(d, b, vertices=v, old_uv=old, new_uv=new))
        want = new[order].copy()
        want[:, 1] = 1 - want[:, 1]
        self.assertEqual(parts(output, data)[2], want.tobytes())

    def test_rejects_duplicate_json_members(self):
        d, b, *_ = fixture()
        raw = json.dumps(d).encode()[:-1] + b',"asset":{"version":"2.0"}}'
        raw += b' ' * (-len(raw) % 4)
        payload = (struct.pack('<4sII', b'glTF', 2, 28+len(raw)+len(b))
                   + struct.pack('<II', len(raw), 0x4e4f534a) + raw
                   + struct.pack('<II', len(b), 0x004e4942) + b)
        with self.assertRaises(ValueError):
            self.call(payload=payload)

    def test_rejects_external_extensions_sparse_aliasing_and_unsupported_storage(self):
        mutations = [lambda d: d['images'][0].update(uri='foreign.png'),
                     lambda d: d.update(extensionsUsed=['KHR_draco_mesh_compression']),
                     lambda d: d['nodes'][0].update(extensions={'foreign': {}}),
                     lambda d: d['accessors'][2].update(sparse={}),
                     lambda d: d['accessors'][2].update(bufferView=0),
                     lambda d: d['bufferViews'][2].update(byteOffset=0),
                     lambda d: d['bufferViews'][2].update(byteStride=8),
                     lambda d: d['accessors'][2].update(byteOffset=4),
                     lambda d: d['buffers'][0].update(byteLength=999999),
                     lambda d: d['nodes'][0].update(translation=[1, 0, 0]),
                     lambda d: d['meshes'][0]['primitives'][0].update(targets=[]),
                     lambda d: d['samplers'][0].update(minFilter=9729)]
        for i, mutation in enumerate(mutations):
            d, b, *_ = fixture()
            mutation(d)
            with self.subTest(mutation=i), self.assertRaises(ValueError):
                self.call(d, b)

    def test_rejects_malformed_bytes_invalid_arrays_and_wrong_png(self):
        d, b, v, f, old, new, _ = fixture()
        bad_uv = new.copy()
        bad_uv[0, 0] = np.nan
        for change in (dict(payload=b'foreign'), dict(payload=encode(d, b)[:-1]),
                       dict(payload=bytearray(encode(d, b))), dict(new_uv=new.astype(np.float64)),
                       dict(new_uv=bad_uv), dict(new_uv=new+1), dict(faces=f.astype(np.float64)),
                       dict(faces=np.zeros_like(f)), dict(vertices=v[:-1]),
                       dict(base_png=self.mr), dict(mr_png=b'not a PNG')):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                self.call(**change)


if __name__ == '__main__':
    unittest.main(verbosity=2)
