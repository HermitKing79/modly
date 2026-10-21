"""Independent bounded GLB reader and saved-query proof; no writer or Blender."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-foliage-colour-glb-cfe852b'
S = ROOT.parent / 'artifacts/amd-bounded-foliage-64d05aa/run'
D = ROOT.parent / 'artifacts/amd-foliage-export-7f1b167/run'
METALLIC = .00011082693761409814
ROUGHNESS = .999995986777544
MAX_BYTES = 2 * 1024**2


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, message):
    _need(type(value) is int and value >= 0, message)
    return value


def _unique(pairs):
    value = {}
    for key, child in pairs:
        _need(key not in value, 'Duplicate JSON member')
        value[key] = child
    return value


def parse_glb(payload, reference):
    """Parse only this core static POSITION/COLOR_0/indices contract.

    All reference arrays must come from an independent original-input admission.
    Source-ID digest fields are SHA256 of the original NPY file bytes, not JSON
    lists or raw integer buffers. Returned positions are original Blender XYZ.
    No caller-provided writer/parser helpers are imported or executed.
    """
    try:
        return _parse(payload, reference)
    except (KeyError, IndexError, TypeError, AttributeError, struct.error,
            OverflowError, RecursionError, UnicodeError) as error:
        raise ValueError('Malformed or unsupported independent GLB input') from error


def _parse(payload, reference):
    import numpy as np
    _need(isinstance(payload, bytes) and 28 <= len(payload) <= MAX_BYTES, 'Bounded immutable GLB bytes required')
    _need(struct.unpack_from('<4sII', payload) == (b'glTF', 2, len(payload)), 'Exact GLB2 header/length required')
    chunks = []
    offset = 12
    while offset < len(payload):
        _need(offset + 8 <= len(payload), 'Complete chunk header required')
        length, kind = struct.unpack_from('<II', payload, offset)
        _need(length > 0 and length % 4 == 0 and offset + 8 + length <= len(payload), 'Complete aligned chunk required')
        chunks.append((kind, payload[offset + 8:offset + 8 + length]))
        offset += 8 + length
    _need([kind for kind, _ in chunks] == [0x4e4f534a, 0x004e4942], 'Exactly one JSON then one BIN chunk required')
    document = json.loads(chunks[0][1].decode('utf-8'), object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
    binary = chunks[1][1]
    _need(isinstance(document, dict) and set(document) <= {
        'asset', 'scene', 'scenes', 'nodes', 'meshes', 'materials', 'buffers', 'bufferViews', 'accessors'}, 'Known core static document only')
    pending = [document]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            _need(not set(item) & {'uri', 'extensions', 'extensionsUsed', 'extensionsRequired', 'sparse'}, 'External/extended/sparse content forbidden')
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    _need(document['asset']['version'] == '2.0', 'glTF2 asset required')
    _need(type(document['scene']) is int and document['scene'] == 0 and len(document['scenes']) == len(document['nodes']) == 1
          and document['scenes'][0]['nodes'] == [0]
          and all(type(index) is int for index in document['scenes'][0]['nodes'])
          and set(document['scenes'][0]) <= {'nodes', 'name'}, 'One default scene/node required')
    node = document['nodes'][0]
    _need(type(node['mesh']) is int and node['mesh'] == 0 and set(node) <= {'mesh', 'name'}, 'Identity mesh node without transforms required')
    _need(len(document['meshes']) == len(document['materials']) == len(document['buffers']) == 1, 'One mesh/material/buffer required')
    mesh = document['meshes'][0]
    _need(set(mesh) <= {'primitives', 'extras', 'name'} and len(mesh['primitives']) == 1, 'One static primitive required')
    primitive = mesh['primitives'][0]
    _need(set(primitive) <= {'attributes', 'indices', 'material', 'mode'}
          and type(primitive['mode']) is int and primitive['mode'] == 4
          and type(primitive['material']) is int and primitive['material'] == 0
          and set(primitive['attributes']) == {'POSITION', 'COLOR_0'}, 'TRIANGLES with POSITION/COLOR_0 only required')
    material = document['materials'][0]
    _need(set(material) <= {'pbrMetallicRoughness', 'alphaMode', 'doubleSided', 'name'}
          and material['alphaMode'] == 'OPAQUE' and material['doubleSided'] is True, 'Opaque two-sided core material required')
    pbr = material['pbrMetallicRoughness']
    _need(set(pbr) == {'baseColorFactor', 'metallicFactor', 'roughnessFactor'}
          and pbr['baseColorFactor'] == [1, 1, 1, 1]
          and all(type(value) in (int, float) for value in pbr['baseColorFactor'])
          and type(pbr['metallicFactor']) in (int, float) and pbr['metallicFactor'] == METALLIC
          and type(pbr['roughnessFactor']) in (int, float) and pbr['roughnessFactor'] == ROUGHNESS, 'Exact verified unit base/M/R factors required')
    _need(len(document['bufferViews']) == len(document['accessors']) == 3, 'Exactly three dense array views/accessors required')
    _need(set(document['buffers'][0]) == {'byteLength'}, 'One embedded buffer without URI required')
    logical = _integer(document['buffers'][0]['byteLength'], 'Integer buffer length required')
    _need(0 < logical <= len(binary) <= logical + 3 and not any(binary[logical:]), 'Complete bounded BIN/padding required')
    expected_v, expected_f, expected_c = (reference[name] for name in ('vertices', 'faces', 'linear-COLOR0'))
    sv, sf = reference['source-vertex-ids'], reference['source-face-ids']
    _need(isinstance(expected_v, np.ndarray) and expected_v.dtype == np.float32 and expected_v.ndim == 2 and expected_v.shape[1] == 3
          and 0 < len(expected_v) <= 18000 and np.isfinite(expected_v).all(), 'Authenticated bounded F32 positions required')
    _need(isinstance(expected_f, np.ndarray) and expected_f.dtype == np.int32 and expected_f.ndim == 2 and expected_f.shape[1] == 3
          and 0 < len(expected_f) <= 6000 and expected_f.min() >= 0 and expected_f.max() < len(expected_v), 'Authenticated bounded original triangles required')
    _need(isinstance(expected_c, np.ndarray) and expected_c.dtype == np.float32 and expected_c.shape == expected_v.shape
          and np.isfinite(expected_c).all() and (expected_c >= 0).all() and (expected_c <= 1).all(), 'Authenticated unit F32 linear colours required')
    _need(isinstance(sv, np.ndarray) and sv.dtype == np.int64 and sv.shape == (len(expected_v),)
          and isinstance(sf, np.ndarray) and sf.dtype == np.int32 and sf.shape == (len(expected_f),), 'Authenticated original ID arrays required')
    extras = mesh['extras']
    _need(set(extras) == {'sourceVertexIDs', 'sourceFaceIDs', 'sourceVertexIDs_SHA256', 'sourceFaceIDs_SHA256'}, 'Exact source correspondence extras required')
    for key, expected in [('sourceVertexIDs', sv), ('sourceFaceIDs', sf)]:
        actual = extras[key]
        _need(isinstance(actual, list) and len(actual) == len(expected) and all(type(value) is int for value in actual)
              and actual == expected.tolist(), 'Original ordered source ID list differs: ' + key)
        _need(extras[key + '_SHA256'] == reference[key + '_SHA256'], 'Original NPY source ID digest differs: ' + key)
    used_accessors, used_views, view_ranges = set(), set(), []

    def accessor(index, width, components, count, target):
        _integer(index, 'Integer accessor index required')
        _need(index < 3 and index not in used_accessors, 'Distinct in-range accessor required')
        used_accessors.add(index)
        data = document['accessors'][index]
        _need(set(data) <= {'bufferView', 'byteOffset', 'componentType', 'count', 'type', 'min', 'max', 'normalized'}
              and type(data['componentType']) is int and data['componentType'] in components and data['type'] == ('SCALAR' if width == 1 else 'VEC3')
              and type(data['count']) is int and data['count'] == count and data.get('normalized', False) is False,
              'Exact dense accessor type/count required')
        view_index = _integer(data['bufferView'], 'Integer buffer view required')
        _need(view_index < 3 and view_index not in used_views, 'Distinct in-range buffer view required')
        used_views.add(view_index)
        view = document['bufferViews'][view_index]
        _need(set(view) <= {'buffer', 'byteOffset', 'byteLength', 'byteStride', 'target'}
              and type(view['buffer']) is int and view['buffer'] == 0
              and type(view.get('target', target)) is int and view.get('target', target) == target, 'One correct typed buffer view required')
        begin = _integer(view.get('byteOffset', 0), 'Integer view offset required')
        length = _integer(view['byteLength'], 'Integer view length required')
        relative = _integer(data.get('byteOffset', 0), 'Integer accessor offset required')
        dtype = np.dtype({5126: '<f4', 5123: '<u2', 5125: '<u4'}[data['componentType']])
        item_width = width * dtype.itemsize
        stride = _integer(view.get('byteStride', item_width), 'Integer byte stride required')
        _need(begin % 4 == 0 and relative % dtype.itemsize == 0 and (begin + relative) % dtype.itemsize == 0
              and stride == item_width and length > 0 and begin + length <= logical
              and relative + count * item_width <= length, 'Aligned in-view dense accessor bounds required')
        view_ranges.append((begin, begin + length))
        value = np.frombuffer(binary, dtype, count=count * width, offset=begin + relative).reshape(count, width).copy()
        _need(np.isfinite(value).all(), 'Finite accessor values required')
        if target == 34962 and index == primitive['attributes']['POSITION']:
            _need(np.array_equal(np.asarray(data['min'], dtype=np.float64), value.min(0).astype(float))
                  and np.array_equal(np.asarray(data['max'], dtype=np.float64), value.max(0).astype(float)), 'Exact POSITION min/max bounds required')
        return value

    position = accessor(primitive['attributes']['POSITION'], 3, (5126,), len(expected_v), 34962)
    colours = accessor(primitive['attributes']['COLOR_0'], 3, (5126,), len(expected_v), 34962)
    indices = accessor(primitive['indices'], 1, (5123, 5125), expected_f.size, 34963).reshape(-1, 3)
    for (begin, end), (other_begin, _) in zip(sorted(view_ranges), sorted(view_ranges)[1:]):
        _need(end <= other_begin, 'Overlapping buffer views forbidden')
    _need(indices.max() < len(position), 'In-range triangle indices required')
    faces = indices.astype(np.int32)
    original = np.empty_like(position)
    original[:, 0], original[:, 1], original[:, 2] = position[:, 0], -position[:, 2], position[:, 1]
    _need(np.array_equal(original.view(np.uint32), expected_v.view(np.uint32)), 'Original XYZ position bits or axis conversion changed')
    _need(np.array_equal(faces, expected_f), 'Original ordered triangle indices/winding/multiplicity changed')
    _need(np.array_equal(colours.view(np.uint32), expected_c.view(np.uint32)), 'Saved F32 linear COLOR_0 bits changed')
    return {'vertices': original, 'faces': faces, 'linear-COLOR0': colours, 'report': {
        'vertices': len(position), 'triangles': len(faces), 'POSITION_dtype': 'float32', 'COLOR_0_dtype': 'float32',
        'index_component_type': document['accessors'][primitive['indices']]['componentType'],
        'original_XYZ_corner_bits_indices_winding_multiplicity_exact': True,
        'F32_linear_COLOR0_bits_exact': True, 'source_ID_lists_and_NPY_hashes_exact': True,
        'axis_conversion': 'source Blender XYZ→glTF (x,z,-y), determinant+1; inverse (gx,-gz,gy)',
        'metallicFactor': METALLIC, 'roughnessFactor': ROUGHNESS, 'alphaMode': 'OPAQUE', 'doubleSided': True,
        'no_UV_NORMAL_textures_images_samplers_external_URI_or_extensions': True}}


def evaluate_queries(colours, faces, query_faces, bary, raw, cohorts, prior_predictions=None):
    """Recheck existing saved queries only, including actual OPAQUE alpha=1."""
    import numpy as np
    _need(colours.dtype == np.float32 and colours.ndim == 2 and colours.shape[1] == 3, 'F32 exported linear colours required')
    _need(faces.dtype == np.int32 and query_faces.dtype == np.int32 and bary.dtype == np.float64
          and query_faces.ndim == 1 and len(query_faces) <= 65536 and bary.shape == (len(query_faces), 3)
          and raw.dtype == np.float32 and raw.shape == (len(query_faces), 6), 'Saved query schemas required')
    _need(len(query_faces) > 0 and query_faces.min() >= 0 and query_faces.max() < len(faces)
          and np.isfinite(bary).all() and (bary >= 0).all() and np.allclose(bary.sum(1), 1, rtol=0, atol=1e-12)
          and np.isfinite(raw).all(), 'Finite in-range saved barycentric queries required')
    linear = np.einsum('qi,qij->qj', bary, colours[faces[query_faces]].astype(float))
    linear = np.clip(linear, 0, 1)
    predicted = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    prior_error = None
    if prior_predictions is not None:
        _need(prior_predictions.dtype == np.float64 and prior_predictions.shape == predicted.shape, 'Saved prediction schema required')
        prior_error = float(np.abs(predicted - prior_predictions).max())
        _need(prior_error <= 2e-15, 'Exported colours changed prior saved RGB predictions')
    target = np.clip(raw.astype(float), 0, 1)

    def statistics(error):
        return {'rmse': float(np.sqrt(np.mean(error * error))),
                'p95_abs': float(np.percentile(np.abs(error), 95, method='linear')),
                'max_abs': float(np.abs(error).max())}

    result = {}
    for name, index in cohorts.items():
        _need(isinstance(index, np.ndarray) and index.dtype.kind in 'iu' and index.ndim == 1 and len(index)
              and index.min() >= 0 and index.max() < len(raw), 'Required bounded nonempty cohort')
        rgb = statistics(predicted[index] - target[index, :3])
        rgb['pass'] = rgb['rmse'] <= .02 and rgb['p95_abs'] <= .05 and rgb['max_abs'] <= .15
        scalar = {}
        for channel, (label, constant) in enumerate(zip(('metallic', 'roughness', 'alpha'), (METALLIC, ROUGHNESS, 1.))):
            scalar[label] = statistics(constant - target[index, channel + 3])
            scalar[label]['pass'] = scalar[label]['rmse'] <= .02
        result[name] = {'count': len(index), 'RGB': rgb, 'scalars': scalar,
                        'pass': rgb['pass'] and all(item['pass'] for item in scalar.values())}
    return {'cohorts': result, 'all_gates_pass': all(item['pass'] for item in result.values()),
            'prior_prediction_max_error': prior_error, 'actual_scalar_factors': {
                'metallic': METALLIC, 'roughness': ROUGHNESS, 'opaque_alpha': 1.},
            'scope': 'Existing58951 saved samples only; no new spatial source sampling, no viewer or renderer equivalence.'}


def main():
    _need(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned binary GO required')
    began = time.monotonic()
    admission_path = ROOT / 'docs/compatibility/amd-foliage-export-decision-evidence/admission.py'
    _need(hashlib.sha256(admission_path.read_bytes()).hexdigest() == 'eabc7e916a19fce3a7ba3a30c002b2f68b945426e9fab0623b9d074e1e5589ab', 'Pinned independent original admission')
    spec = importlib.util.spec_from_file_location('glb_verify_original_admission', admission_path)
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    auth = admission.load_auth()
    auth.P = P
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    out = P / 'verify'
    try:
        data, identities, watched = admission.load_selection(np)
        _need(admission.digest(admission.plain(D / 'result.json')) == '4bc3e9add407742a78897e4cfb07dbfb059b72deab7d5d33cc5e9cddb4b904c1', 'Exact qualified original decision result')
        decision = json.loads((D / 'result.json').read_text())
        _need(decision['classification'] == 'PASS_DECISION' and decision['vertex_colour_candidate'] == 'PASS'
              and decision['zero_support'] == 0 and decision['total_queries'] == 58951, 'Genuine qualified prior saved queries required')
        for name, row in decision['files'].items():
            _need(Path(name).name == name, 'Single decision basename')
            path = admission.plain(D / name)
            _need(path.stat().st_size == row['bytes'] and admission.digest(path) == row['sha256'], 'Exact qualified query data: ' + name)
            watched.append(path)
        linear = np.load(D / 'linear-COLOR0.npy', allow_pickle=False)
        reference = {name: data[name] for name in ('vertices', 'faces', 'source-vertex-ids', 'source-face-ids')}
        reference['linear-COLOR0'] = linear
        reference.update(sourceVertexIDs_SHA256=admission.digest(S / 'source-vertex-ids.npy'),
                         sourceFaceIDs_SHA256=admission.digest(S / 'source-face-ids.npy'))
        export = P / 'export'
        for name in ('vertices', 'faces', 'linear-COLOR0', 'source-vertex-ids', 'source-face-ids'):
            path = admission.plain(export / (name + '.npy'))
            value = np.load(path, allow_pickle=False)
            expected = reference[name]
            _need(value.dtype == expected.dtype and value.shape == expected.shape and value.tobytes() == expected.tobytes(), 'Export reference copy differs from authenticated original: ' + name)
            watched.append(path)
        path = admission.plain(export / 'foliage-learned-colour.glb')
        _need(path.stat().st_size <= MAX_BYTES, 'Bounded GLB file required')
        watched += [path, admission_path, D / 'result.json', Path(__file__).resolve()]
        watched = list(dict.fromkeys(watched))
        before = {str(item): (item.stat().st_size, item.stat().st_mtime_ns, admission.digest(item)) for item in watched}
        payload = path.read_bytes()
        parsed = parse_glb(payload, reference)
        qf, bary, raw, predictions = [np.load(D / (name + '.npy'), allow_pickle=False)
                                      for name in ('query-faces', 'barycentrics', 'source-spatial', 'vertex-RGB')]
        _need(qf.shape == (58951,) and bary.shape == (58951, 3), 'Full original saved query population')
        seam_index = np.arange(44420)
        cohorts = {'centroids': np.arange(4531), 'area_weighted': np.arange(4531, 14531),
                   'required_surface': np.arange(14531), 'seam_edge': seam_index[seam_index % 20 < 10] + 14531,
                   'seam_near': seam_index[seam_index % 20 >= 10] + 14531}
        measurements = evaluate_queries(parsed['linear-COLOR0'], parsed['faces'], qf, bary, raw, cohorts, predictions)
        _need(before == {str(item): (item.stat().st_size, item.stat().st_mtime_ns, admission.digest(item)) for item in watched}, 'All original/export sources unchanged')
        result = {'classification': 'PASS' if measurements['all_gates_pass'] else 'BLOCKED',
                  'stage': 'fresh_independent_GLB_bytes_and_saved_queries',
                  'GLB_SHA256': hashlib.sha256(payload).hexdigest(), 'GLB_bytes': len(payload),
                  'independent_fresh_process': True, 'container': parsed['report'], 'measurements': measurements,
                  'original_input_identities': identities, 'all_watched_sources_unchanged': True,
                  'no_new_sampling_UV_render_bake_or_export': True, 'elapsed_seconds': time.monotonic() - began}
        admission.save(out / 'result.json', result)
        print('INDEPENDENT_COLOUR_GLB_' + result['classification'], flush=True)
        return 0 if result['classification'] == 'PASS' else 1
    except BaseException as error:
        if not (out / 'result.json').exists():
            admission.save(out / 'result.json', {'classification': 'BLOCKED', 'stage': 'fresh_independent_GLB_verification',
                                                'error': type(error).__name__ + ': ' + str(error),
                                                'elapsed_seconds': time.monotonic() - began})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
