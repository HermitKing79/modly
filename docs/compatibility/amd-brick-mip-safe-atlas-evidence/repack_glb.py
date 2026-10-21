"""Disposable CPU patcher for the accepted six-view static GLB, not an exporter.

Inputs and return value are in memory; no file I/O is performed by repack_glb.
Vertices and UVs use the existing reader's Blender coordinates and V-up UVs.
"""
import importlib.util
import io
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image

_path = Path(__file__).resolve().parent.parent / 'amd-brick-pbr-mip-evidence/assets.py'
_spec = importlib.util.spec_from_file_location('atlas_repack_readonly_assets', _path)
_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_assets)
require = _assets.require


def repack_glb(payload, vertices, faces, old_uv, new_uv, base_png, mr_png):
    """Replace only dense UV0 bytes and the two PNGs; fail closed on other layouts."""
    try:
        return _repack(payload, vertices, faces, old_uv, new_uv, base_png, mr_png)
    except (KeyError, IndexError, TypeError, AttributeError, OSError, struct.error, OverflowError, RecursionError) as error:
        raise ValueError('Malformed or unsupported known GLB input') from error


def _repack(payload, vertices, faces, old_uv, new_uv, base_png, mr_png):
    for array, width, dtype in ((vertices, 3, np.float32), (faces, 3, np.int32),
                                (old_uv, 2, np.float32), (new_uv, 2, np.float32)):
        require(isinstance(array, np.ndarray) and array.dtype == dtype and array.ndim == 2
                and array.shape[1] == width and np.isfinite(array).all(), 'Finite typed corner arrays required')
    require(0 < len(faces) <= 6000 and vertices.shape == (3*len(faces), 3)
            and old_uv.shape == new_uv.shape == (len(vertices), 2)
            and np.array_equal(np.sort(faces.ravel()), np.arange(len(vertices)))
            and ((new_uv >= 0) & (new_uv <= 1)).all(), 'Complete original seam corners / bounded UVs required')
    glb = _assets.decode_glb(payload)
    n = struct.unpack_from('<I', payload, 12)[0]
    require(struct.unpack_from('<I', payload, 16)[0] == 0x4e4f534a, 'JSON must precede BIN')
    def unique(pairs):
        require(len(pairs) == len(dict(pairs)), 'Duplicate JSON members forbidden')
        return dict(pairs)
    doc, binary = json.loads(payload[20:20+n], object_pairs_hook=unique), payload[28+n:]
    require(set(doc) <= set('asset scene scenes nodes materials meshes textures images accessors bufferViews samplers buffers'.split()), 'Known static container only')
    def core(value):
        if isinstance(value, dict):
            require(not set(value) & {'uri', 'extensions', 'extensionsUsed', 'extensionsRequired', 'sparse'}, 'External/extended/sparse data forbidden')
            for child in value.values(): core(child)
        elif isinstance(value, list):
            for child in value: core(child)
    core(doc)
    require(doc['scene'] == 0 and len(doc['scenes']) == len(doc['nodes']) == 1
            and doc['scenes'][0]['nodes'] == [0] and doc['nodes'][0]['mesh'] == 0
            and set(doc['nodes'][0]) <= {'mesh', 'name'}, 'Untransformed original static node required')
    primitive = doc['meshes'][0]['primitives'][0]
    require(primitive['attributes'] == dict(POSITION=0, NORMAL=1, TEXCOORD_0=2)
            and primitive['indices'] == 3 and primitive['material'] == 0
            and set(primitive) <= {'attributes', 'indices', 'material', 'mode'}, 'Known primitive attributes only')
    require(len(doc['bufferViews']) == 6 and len(doc['accessors']) == 4
            and len(doc['textures']) == len(doc['images']) == 2 and len(doc['samplers']) == 1
            and doc['textures'] == [dict(sampler=0, source=0), dict(sampler=0, source=1)]
            and set(doc['buffers'][0]) == {'byteLength'}, 'Unaliased known views/textures required')
    blobs, end = [], 0
    for i, view in enumerate(doc['bufferViews']):
        start, length = view.get('byteOffset', 0), view['byteLength']
        require(type(start) is type(length) is int and view['buffer'] == 0 and length > 0
                and start == (end+3)//4*4 and start+length <= len(binary)
                and not any(binary[end:start]) and set(view) <= {'buffer', 'byteOffset', 'byteLength', 'target'}, 'Disjoint tightly packed known views required')
        if i < 4:
            accessor = doc['accessors'][i]
            require(accessor['bufferView'] == i and accessor.get('byteOffset', 0) == 0
                    and not accessor.get('normalized') and set(accessor) <= {'bufferView', 'byteOffset', 'componentType', 'count', 'type', 'min', 'max'}
                    and accessor['componentType'] == (5123 if i == 3 else 5126)
                    and accessor['type'] == ['VEC3', 'VEC3', 'VEC2', 'SCALAR'][i]
                    and accessor['count'] == (3*len(faces) if i == 3 else len(glb['vertices']))
                    and length == accessor['count'] * [12, 12, 8, 2][i]
                    and view.get('target') == (34963 if i == 3 else 34962)
                    and (i != 2 or not {'min', 'max'} & set(accessor)), 'Dense nonaliased original accessor required')
        else:
            require(doc['images'][i-4]['bufferView'] == i and 'target' not in view, 'Separate embedded image views required')
        blobs.append(binary[start:start+length]); end = start+length
    require(doc['buffers'][0]['byteLength'] == len(binary) == (end+3)//4*4
            and not any(binary[end:]) and np.isfinite(np.frombuffer(blobs[1], '<f4')).all(), 'Complete finite known buffer required')
    actual, _ = _assets.align_uv(vertices, faces, old_uv, glb)
    require(np.array_equal(actual.view(np.uint32), old_uv.view(np.uint32)), 'Actual aligned old GLB UV bits required')
    mapping = {}
    for position, old, new in zip(vertices, old_uv, new_uv):
        key, value = (*map(float, position), *map(float, old)), tuple(map(float, new))
        require(key not in mapping or mapping[key] == value, 'Ambiguous position/old-UV mapping')
        mapping[key] = value
    keys = [(*map(float, p), *map(float, u)) for p, u in zip(glb['vertices'], glb['uv'])]
    require(set(keys) == set(mapping), 'Complete exact position/old-UV correspondence required')
    uv = np.array([mapping[key] for key in keys], dtype='<f4'); uv[:, 1] = 1-uv[:, 1]
    blobs[2] = uv.tobytes()
    for i, (png, mode) in enumerate(((base_png, 'RGBA'), (mr_png, 'RGB')), 4):
        require(isinstance(png, bytes) and 0 < len(png) <= 1024**2, 'Bounded immutable PNG bytes required')
        with Image.open(io.BytesIO(png)) as image:
            require(image.format == 'PNG' and image.mode == mode and image.size == (1024, 1024), 'Original PNG dimensions/encoding required')
            image.load()
        blobs[i] = png
    output = bytearray()
    for view, blob in zip(doc['bufferViews'], blobs):
        output.extend(b'\0' * (-len(output) % 4))
        view.update(byteOffset=len(output), byteLength=len(blob)); output.extend(blob)
    output.extend(b'\0' * (-len(output) % 4)); doc['buffers'][0]['byteLength'] = len(output)
    j = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode(); j += b' ' * (-len(j) % 4)
    require(28+len(j)+len(output) <= 1024**2, 'Known small-container capacity exceeded')
    return (struct.pack('<4sII', b'glTF', 2, 28+len(j)+len(output))
            + struct.pack('<II', len(j), 0x4e4f534a) + j
            + struct.pack('<II', len(output), 0x004e4942) + output)
