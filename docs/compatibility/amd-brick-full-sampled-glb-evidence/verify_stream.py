"""Independent bounded-memory GLB reader; no writer, source loader or Blender."""
import sys
sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path
import stat
import struct
import numpy as np

MAX_BYTES = 128 * 1024**2
MAX_JSON_BYTES = 65536
MAX_VERTICES = 2000000
MAX_FACES = 4173584
MAX_BATCH_ROWS = 32768
METALLIC = .00011082693761409814
ROUGHNESS = .999995986777544


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, message):
    _need(type(value) is int and value >= 0, message)
    return value


def _keys(value, required, optional=()):
    _need(isinstance(value, dict) and set(required) <= set(value)
          and set(value) <= set(required) | set(optional), 'Unsupported object schema')
    if 'name' in value:
        _need(type(value['name']) is str, 'String name required')


def _single(value):
    _need(isinstance(value, list) and len(value) == 1, 'Exactly one object required')
    return value[0]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'Duplicate JSON member')
        result[key] = value
    return result


def _json_float(value):
    number = float(value)
    _need(math.isfinite(number), 'Nonfinite JSON number')
    return number


def _constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def _plain(metadata):
    _need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
          and not (getattr(metadata, 'st_file_attributes', 0)
                   & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
          'Single-link regular file without symlink/reparse required')


def _stamp(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, getattr(metadata, 'st_birthtime_ns', None),
            metadata.st_ctime_ns)


def _vector_bound(value):
    _need(isinstance(value, list) and len(value) == 3
          and all(type(number) in (int, float) and math.isfinite(number) for number in value),
          'Finite numeric VEC3 bound required')
    return np.asarray(value, dtype=np.float64)


def verify(path, verticesF32, facesI32, coloursF32, source_hashes, batch_rows=32768):
    """Verify every original row against a read-only mapped GLB.

    References must have been independently authenticated by the caller. No asset
    or provenance file is loaded here. The returned linear_COLOR0 array is a
    read-only view of the exported BIN, suitable for separate saved-query replay.
    Only bounded JSON and at most batch_rows source rows are copied into RAM.
    """
    try:
        return _verify(path, verticesF32, facesI32, coloursF32, source_hashes, batch_rows)
    except (KeyError, IndexError, TypeError, AttributeError, struct.error, OverflowError,
            RecursionError, UnicodeError, OSError) as error:
        raise ValueError('Malformed or unsupported independent GLB input') from error


def _verify(path, vertices, faces, colours, source_hashes, batch_rows):
    _need(type(batch_rows) is int and 1 <= batch_rows <= MAX_BATCH_ROWS,
          'Batch rows must be an integer in [1,32768]')
    _need(isinstance(vertices, np.ndarray) and vertices.dtype == np.dtype('<f4')
          and vertices.ndim == 2 and vertices.shape[1] == 3
          and 0 < len(vertices) <= MAX_VERTICES, 'Bounded original F32 XYZ rows required')
    _need(isinstance(faces, np.ndarray) and faces.dtype == np.dtype('<i4')
          and faces.ndim == 2 and faces.shape[1] == 3
          and 0 < len(faces) <= MAX_FACES, 'Bounded original I32 triangle rows required')
    _need(isinstance(colours, np.ndarray) and colours.dtype == np.dtype('<f4')
          and colours.shape == vertices.shape, 'Original F32 linear colour rows required')
    _need(isinstance(source_hashes, dict) and 0 < len(source_hashes) <= 32
          and all(type(key) is str and 0 < len(key) <= 128
                  and type(value) is str and len(value) == 64
                  and all(char in '0123456789abcdef' for char in value)
                  for key, value in source_hashes.items()), 'Compact source SHA256 dictionary required')
    nv, nf = len(vertices), len(faces)
    logical = 24 * nv + 12 * nf
    path = Path(path)
    before = path.lstat()
    _plain(before)
    _need(28 <= before.st_size <= MAX_BYTES, 'GLB must be at most 128 MiB')
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        _plain(opened)
        # Windows lstat/fstat can expose different ctime semantics. Compare all
        # common identity fields here; retain ctime in each API's before/after.
        _need(_stamp(opened)[:-1] == _stamp(before)[:-1], 'File identity changed while opening')
        header = handle.read(20)
        _need(len(header) == 20, 'Complete GLB and JSON chunk headers required')
        magic, version, total, json_length, json_kind = struct.unpack('<4sIIII', header)
        _need((magic, version, total) == (b'glTF', 2, before.st_size), 'Exact GLB2 header/length required')
        _need(json_kind == 0x4e4f534a and 0 < json_length <= MAX_JSON_BYTES
              and json_length % 4 == 0 and 28 + json_length <= total,
              'Bounded aligned JSON first chunk required')
        encoded = handle.read(json_length)
        _need(len(encoded) == json_length, 'Complete JSON chunk required')
        decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_float=_json_float,
                                   parse_constant=_constant)
        text = encoded.decode('utf-8')
        stripped = text.lstrip(' \t\r\n')
        document, end = decoder.raw_decode(stripped)
        _need(all(char == ' ' for char in stripped[end:]), 'JSON chunk padding must be spaces')
        chunk_header = handle.read(8)
        _need(len(chunk_header) == 8, 'Complete BIN header required')
        binary_length, binary_kind = struct.unpack('<II', chunk_header)
        binary_offset = 28 + json_length
        _need(binary_kind == 0x004e4942 and binary_length == logical
              and binary_length % 4 == 0 and binary_offset + binary_length == total,
              'Exactly one complete dense BIN after JSON required')
        _keys(document, ('asset', 'scene', 'scenes', 'nodes', 'meshes', 'materials',
                         'buffers', 'bufferViews', 'accessors'))
        asset = document['asset']
        _keys(asset, ('version',), ('generator', 'copyright', 'minVersion'))
        _need(asset['version'] == '2.0'
              and all(type(value) is str for value in asset.values())
              and asset.get('minVersion', '2.0') == '2.0', 'glTF2 core asset required')
        _need(_integer(document['scene'], 'Integer default scene required') == 0,
              'Scene zero must be default')
        scene = _single(document['scenes'])
        _keys(scene, ('nodes',), ('name',))
        _need(scene['nodes'] == [0] and type(scene['nodes'][0]) is int, 'Scene must contain node zero only')
        node = _single(document['nodes'])
        _keys(node, ('mesh',), ('name',))
        _need(_integer(node['mesh'], 'Integer mesh index required') == 0, 'Identity mesh node zero required')
        mesh = _single(document['meshes'])
        _keys(mesh, ('primitives', 'extras'), ('name',))
        _need(mesh['extras'] == source_hashes, 'Exact compact source hashes required')
        primitive = _single(mesh['primitives'])
        _keys(primitive, ('attributes', 'indices', 'material', 'mode'))
        _keys(primitive['attributes'], ('POSITION', 'COLOR_0'))
        _need(_integer(primitive['mode'], 'Integer primitive mode required') == 4
              and _integer(primitive['material'], 'Integer material required') == 0,
              'One TRIANGLES primitive/material required')
        material = _single(document['materials'])
        _keys(material, ('pbrMetallicRoughness', 'alphaMode', 'doubleSided'), ('name',))
        _need(material['alphaMode'] == 'OPAQUE' and material['doubleSided'] is True,
              'OPAQUE double-sided material required')
        pbr = material['pbrMetallicRoughness']
        _keys(pbr, ('baseColorFactor', 'metallicFactor', 'roughnessFactor'))
        _need(isinstance(pbr['baseColorFactor'], list) and pbr['baseColorFactor'] == [1, 1, 1, 1]
              and all(type(value) in (int, float) for value in pbr['baseColorFactor'])
              and type(pbr['metallicFactor']) in (int, float) and pbr['metallicFactor'] == METALLIC
              and type(pbr['roughnessFactor']) in (int, float) and pbr['roughnessFactor'] == ROUGHNESS,
              'Exact verified unit base/M/R factors required')
        buffer = _single(document['buffers'])
        _keys(buffer, ('byteLength',))
        _need(_integer(buffer['byteLength'], 'Integer embedded buffer length required') == logical,
              'Exact dense embedded buffer length required')
        _need(isinstance(document['bufferViews'], list) and len(document['bufferViews']) == 3
              and isinstance(document['accessors'], list) and len(document['accessors']) == 3,
              'Exactly three dense views/accessors required')
        used_accessors, used_views = set(), set()

        def accessor(index, count, component, kind, offset, length, target, bounds=False):
            _integer(index, 'Integer accessor index required')
            _need(index < 3 and index not in used_accessors, 'Distinct in-range accessor required')
            used_accessors.add(index)
            value = document['accessors'][index]
            _keys(value, ('bufferView', 'componentType', 'count', 'type') + (('min', 'max') if bounds else ()),
                  ('byteOffset', 'normalized'))
            _need(_integer(value['componentType'], 'Integer component type required') == component
                  and _integer(value['count'], 'Integer accessor count required') == count
                  and value['type'] == kind and value.get('normalized', False) is False
                  and _integer(value.get('byteOffset', 0), 'Integer accessor byte offset required') == 0,
                  'Exact dense accessor type/count/offset required')
            view_index = _integer(value['bufferView'], 'Integer buffer view required')
            _need(view_index < 3 and view_index not in used_views, 'Distinct in-range view required')
            used_views.add(view_index)
            view = document['bufferViews'][view_index]
            _keys(view, ('buffer', 'byteLength'), ('byteOffset', 'target'))
            _need(_integer(view['buffer'], 'Integer buffer required') == 0
                  and _integer(view.get('byteOffset', 0), 'Integer view offset required') == offset
                  and _integer(view['byteLength'], 'Integer view length required') == length
                  and _integer(view.get('target', target), 'Integer view target required') == target,
                  'Exact contiguous aligned dense views required')
            return value

        position_meta = accessor(primitive['attributes']['POSITION'], nv, 5126, 'VEC3', 0, 12 * nv, 34962, True)
        accessor(primitive['attributes']['COLOR_0'], nv, 5126, 'VEC3', 12 * nv, 12 * nv, 34962)
        accessor(primitive['indices'], 3 * nf, 5125, 'SCALAR', 24 * nv, 12 * nf, 34963)
        minimum, maximum = _vector_bound(position_meta['min']), _vector_bound(position_meta['max'])
        binary = np.memmap(handle, dtype=np.uint8, mode='r', offset=binary_offset, shape=(binary_length,))
        try:
            positions = np.ndarray((nv, 3), dtype='<f4', buffer=binary, offset=0)
            exported_colour = np.ndarray((nv, 3), dtype='<f4', buffer=binary, offset=12 * nv)
            indices = np.ndarray((nf, 3), dtype='<u4', buffer=binary, offset=24 * nv)
            source_minimum = np.full(3, np.inf, dtype=np.float64)
            source_maximum = np.full(3, -np.inf, dtype=np.float64)
            for begin in range(0, nv, batch_rows):
                end = min(begin + batch_rows, nv)
                original, linear = vertices[begin:end], colours[begin:end]
                actual_positions, actual_colour = positions[begin:end], exported_colour[begin:end]
                _need(np.isfinite(original).all() and np.isfinite(actual_positions).all()
                      and np.isfinite(linear).all() and np.isfinite(actual_colour).all()
                      and (linear >= 0).all() and (linear <= 1).all(), 'Finite positions/unit linear colours required')
                expected = np.empty((end - begin, 3), dtype='<f4')
                expected[:, 0], expected[:, 1], expected[:, 2] = original[:, 0], original[:, 2], -original[:, 1]
                _need(np.array_equal(actual_positions.view('<u4'), expected.view('<u4')),
                      'Original XYZ bits or (x,z,-y) transform changed')
                _need(np.array_equal(actual_colour.view('<u4'), linear.view('<u4')),
                      'Saved F32 linear COLOR_0 bits changed')
                source_minimum = np.minimum(source_minimum, expected.min(axis=0))
                source_maximum = np.maximum(source_maximum, expected.max(axis=0))
            for begin in range(0, nf, batch_rows):
                original = faces[begin:min(begin + batch_rows, nf)]
                actual = indices[begin:min(begin + batch_rows, nf)]
                _need((original >= 0).all() and (original < nv).all() and (actual < nv).all(),
                      'In-range original/exported triangle indices required')
                _need(np.array_equal(actual, original), 'Original ordered triangle indices/winding/multiplicity changed')
            _need(np.array_equal(minimum, source_minimum) and np.array_equal(maximum, source_maximum),
                  'Exact POSITION bounds from original transformed data required')
            after = path.lstat()
            _plain(after)
            after_opened = os.fstat(handle.fileno())
            _plain(after_opened)
            _need(_stamp(before) == _stamp(after) and _stamp(opened) == _stamp(after_opened),
                  'GLB changed during independent verification')
            report = {'vertices': nv, 'triangles': nf, 'GLB_bytes': total,
                      'POSITION_dtype': 'float32', 'COLOR_0_dtype': 'float32', 'index_component_type': 5125,
                      'original_XYZ_corner_bits_indices_winding_multiplicity_exact': True,
                      'F32_linear_COLOR0_bits_exact': True, 'compact_source_NPY_hashes_exact': True,
                      'POSITION_bounds_exact': True, 'POSITION_min': source_minimum.tolist(),
                      'POSITION_max': source_maximum.tolist(), 'batch_rows': batch_rows,
                      'vertex_batches': (nv + batch_rows - 1) // batch_rows,
                      'triangle_batches': (nf + batch_rows - 1) // batch_rows,
                      'axis_conversion': 'source Blender XYZ→glTF (x,z,-y), determinant+1; inverse (gx,-gz,gy)',
                      'metallicFactor': METALLIC, 'roughnessFactor': ROUGHNESS,
                      'alphaMode': 'OPAQUE', 'doubleSided': True,
                      'no_UV_NORMAL_textures_images_samplers_external_URI_or_extensions': True,
                      'read_only_mapped_BIN': True, 'all_original_rows_compared': True}
            return {'report': report, 'linear_COLOR0': exported_colour}
        except BaseException:
            binary._mmap.close()
            raise
