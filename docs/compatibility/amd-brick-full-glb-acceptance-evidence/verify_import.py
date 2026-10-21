"""Independent complete CPU readback proof; no Blender import or repair."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import time
import traceback


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_arrays(vertices, faces, colours, actual_vertices, actual_faces,
                  actual_colour_bytes, oracle_colour_bytes, batch_rows=32768):
    """Compare every sorted referenced vertex, oriented triangle and corner byte.

    The caller authenticates the original arrays and native setter oracle.
    Quantization measurements decode IEC sRGB for display; they do not claim
    that byte storage preserves the original linear float32 values.
    """
    import numpy as np
    _need(type(batch_rows) is int and 1 <= batch_rows <= 32768, 'Bounded integer batch size required')
    _need(isinstance(vertices, np.ndarray) and vertices.dtype == np.dtype('<f4')
          and vertices.ndim == 2 and vertices.shape[1] == 3 and 0 < len(vertices) <= 2000000,
          'Original bounded F32 XYZ table required')
    _need(isinstance(faces, np.ndarray) and faces.dtype == np.dtype('<i4')
          and faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= 4173584,
          'Original bounded I32 triangles required')
    _need(isinstance(colours, np.ndarray) and colours.dtype == np.dtype('<f4')
          and colours.shape == vertices.shape, 'Original full F32 linear RGB table required')
    n, f = len(vertices), len(faces)
    _need(isinstance(oracle_colour_bytes, np.ndarray) and oracle_colour_bytes.dtype == np.dtype('u1')
          and oracle_colour_bytes.shape == (n, 4), 'Full original native oracle U8 RGBA table required')
    _need(isinstance(actual_faces, np.ndarray) and actual_faces.dtype == np.dtype('<i4')
          and actual_faces.shape == faces.shape, 'Actual complete I32 triangles required')
    _need(isinstance(actual_colour_bytes, np.ndarray) and actual_colour_bytes.dtype == np.dtype('u1')
          and actual_colour_bytes.shape == (3 * f, 4), 'Actual CORNER U8 RGBA table required')
    used = np.zeros(n, dtype=np.bool_)
    for start in range(0, f, batch_rows):
        part = faces[start:start + batch_rows]
        _need((part >= 0).all() and (part < n).all(), 'Original triangle index out of range')
        used[part.reshape(-1)] = True
    source_ids = np.flatnonzero(used)
    count = len(source_ids)
    _need(isinstance(actual_vertices, np.ndarray) and actual_vertices.dtype == np.dtype('<f4')
          and actual_vertices.shape == (count, 3), 'Actual F32 sorted referenced-vertex count required')
    rank = np.full(n, -1, dtype='<i4')
    rank[source_ids] = np.arange(count, dtype='<i4')
    for start in range(0, n, batch_rows):
        v, c, oracle = vertices[start:start + batch_rows], colours[start:start + batch_rows], oracle_colour_bytes[start:start + batch_rows]
        _need(np.isfinite(v).all() and np.isfinite(c).all() and (c >= 0).all() and (c <= 1).all(),
              'Finite original positions and unit linear RGB required')
        _need((oracle[:, 3] == 255).all(), 'Native oracle alpha must be exactly 255')
    for start in range(0, count, batch_rows):
        actual = actual_vertices[start:start + batch_rows]
        expected = vertices[source_ids[start:start + batch_rows]]
        _need(np.isfinite(actual).all() and np.array_equal(actual.view('<u4'), expected.view('<u4')),
              'Sorted original referenced XYZ float32 bits changed')
    sums = {'linear_RGB': 0., 'encoded_sRGB_RGB': 0.}
    maxima = {'linear_RGB': 0., 'encoded_sRGB_RGB': 0.}
    scalar_count = 0
    for start in range(0, f, batch_rows):
        original_faces = faces[start:start + batch_rows]
        actual = actual_faces[start:start + batch_rows]
        _need((actual >= 0).all() and (actual < count).all()
              and np.array_equal(actual, rank[original_faces]),
              'Original ordered triangle indices/winding/multiplicity changed')
        original_ids = original_faces.reshape(-1)
        exported = actual_colour_bytes[3 * start:3 * (start + len(original_faces))]
        _need(np.array_equal(exported, oracle_colour_bytes[original_ids]),
              'Actual corner RGBA bytes differ from original native setter oracle')
        target = colours[original_ids].astype(np.float64)
        encoded = exported[:, :3].astype(np.float64) / 255.
        decoded = np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)
        target_encoded = np.where(target <= .0031308, 12.92 * target, 1.055 * target ** (1 / 2.4) - .055)
        for label, error in (('linear_RGB', decoded - target), ('encoded_sRGB_RGB', encoded - target_encoded)):
            sums[label] += float(np.sum(error * error, dtype=np.float64))
            maxima[label] = max(maxima[label], float(np.max(np.abs(error))))
        scalar_count += target.size
    encoded_ceiling = .5 / 255 + 2e-6
    _need(maxima['encoded_sRGB_RGB'] <= encoded_ceiling,
          'Encoded RGB quantization exceeds the declared half-byte ceiling')
    measurements = {name: {'max_abs': maxima[name], 'rmse': float(np.sqrt(sums[name] / scalar_count))}
                    for name in sums}
    measurements.update(scope='Every imported corner RGB against original F32; IEC sRGB decoding for display only; not a native getter or F32-exact color claim',
                        corner_count=3 * f, RGB_scalar_count=scalar_count,
                        encoded_RGB_max_abs_ceiling=encoded_ceiling, encoded_RGB_ceiling_pass=True)
    return {'original_vertices': n, 'referenced_vertices': count, 'unreferenced_vertices_omitted': n - count,
            'triangles': f, 'loops': 3 * f,
            'ordered_original_geometry_and_triangle_multiplicity_exact': True,
            'source_vertex_correspondence': 'Sorted original referenced IDs; no welding or geometric deduplication',
            'all_corner_RGBA_bytes_equal_native_oracle': True,
            'all_corner_alpha_bytes_255': True,
            'referenced_membership_SHA256': hashlib.sha256(used.tobytes()).hexdigest(),
            'quantization_display_only': measurements, 'batch_rows': batch_rows}


def main():
    _need(sys.stdin.buffer.readline(64) == b'GO\n', 'Owned exact binary GO required')
    E = Path(__file__).resolve().parent
    sys.path.insert(0, str(E))
    from inputs import P, MESH, COLOURS, authenticate, load_auth, plain, digest, require, save
    auth = load_auth()
    auth.P = P
    sys.addaudithook(auth.runtime_guard)
    began = time.monotonic()
    out, imported = P / 'verify', P / 'import'
    stage = 'authenticated_source_and_import_manifest'
    try:
        import numpy as np
        _, identities, watched, _ = authenticate()
        receipt_path = plain(imported / 'result.json')
        _need(receipt_path.stat().st_size <= 1024**2, 'Bounded import receipt required')
        receipt = json.loads(receipt_path.read_text())
        _need(receipt['classification'] == 'PASS' and receipt['fresh_GLTF_imports'] == 1
              and receipt['render_count'] == 0 and receipt['geometry_repaired'] is False
              and receipt['colours_replaced'] is False and receipt['sources_unchanged'] is True,
              'One genuine unmodified import and unchanged sources required')
        native = receipt['native_oracle']
        _need(native['data_type'] == 'BYTE_COLOR' and native['domain'] == 'POINT'
              and native['RNA_type'] == 'ByteColorAttributeValue'
              and native['source_rgb'] == 'exact original F32 scene-linear COLOR_0'
              and type(native['rows']) is int and native['rows'] == 1957891
              and type(native['batch_rows']) is int and native['batch_rows'] == 32768
              and type(native['RGBA_alpha']) in (int, float) and native['RGBA_alpha'] == 1
              and native['same_RNA_type_as_imported'] is True,
              'Exact original-input native BYTE_COLOR setter evidence required')
        material = receipt['material']
        _need(material['double_sided'] is True and material['opaque_shader_alpha'] == 1
              and material['no_texture_images'] is True and material['effective_emission_zero'] is True,
              'Actual two-sided opaque untextured non-emissive material required')
        for label, expected in (('Metallic', .00011082693761409814), ('Roughness', .999995986777544), ('Alpha', 1.)):
            socket = material['Principled_inputs'][label]
            _need(socket['linked'] is False and type(socket['default']) in (int, float)
                  and socket['default'] == float(np.float32(expected)), 'Actual material scalar differs: ' + label)
        nodes = {node['name']: node['type'] for node in material['nodes']}
        _need(sorted(nodes.values()) == sorted(('ShaderNodeBsdfPrincipled', 'ShaderNodeOutputMaterial', 'ShaderNodeVertexColor'))
              and len(material['nodes']) == len(nodes) == 3, 'Actual material node topology differs')
        links = {(nodes[row['source']], row['source_socket'], nodes[row['destination']], row['destination_socket'])
                 for row in material['links']}
        _need(len(material['links']) == 2 and links == {
            ('ShaderNodeVertexColor', 'Color', 'ShaderNodeBsdfPrincipled', 'Base Color'),
            ('ShaderNodeBsdfPrincipled', 'BSDF', 'ShaderNodeOutputMaterial', 'Surface')},
            'Actual vertex-color material links differ')
        _need(material['colour_node_layer_name'] == material['attribute_name']
              or (material['colour_node_layer_name'] == '' and material['rendered_colour_index'] == 0),
              'Actual rendered color layer does not resolve')
        names = ('actual-vertices.npy', 'actual-faces.npy', 'actual-colour-bytes.npy', 'oracle-colour-bytes.npy')
        _need(set(receipt['files']) == set(names) | {'inspected.blend'}, 'Exact actual readback/scene manifest required')
        actual = {}
        for name in (*names, 'inspected.blend'):
            path = plain(imported / name)
            row = receipt['files'][name]
            _need(type(row['bytes']) is int and row['bytes'] > 0 and row['bytes'] == path.stat().st_size
                  and type(row['sha256']) is str and len(row['sha256']) == 64
                  and all(c in '0123456789abcdef' for c in row['sha256'])
                  and digest(path) == row['sha256'], 'Actual import file identity differs: ' + name)
            watched.append(path)
            if name in names:
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                _need(list(value.shape) == row['shape'] and type(row['dtype']) is str
                      and value.dtype == np.dtype(row['dtype']) and value.flags.c_contiguous
                      and not value.flags.writeable, 'Read-only dense actual NPY schema differs: ' + name)
                actual[name] = value
        watched += [receipt_path, Path(__file__).resolve()]
        watched = list(dict.fromkeys(watched))
        before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns, digest(path)) for path in watched}
        vertices = np.load(plain(MESH / 'vertices.npy'), mmap_mode='r', allow_pickle=False)
        faces = np.load(plain(MESH / 'faces.npy'), mmap_mode='r', allow_pickle=False)
        colours = np.load(plain(COLOURS), mmap_mode='r', allow_pickle=False)
        _need(vertices.shape == colours.shape == (1957891, 3) and faces.shape == (4173584, 3),
              'Complete exact original source counts required')
        stage = 'every_referenced_vertex_triangle_and_corner_byte'
        report = verify_arrays(vertices, faces, colours, *(actual[name] for name in names))
        _need(report['referenced_vertices'] == 1952142 and report['unreferenced_vertices_omitted'] == 5749
              and report['triangles'] == 4173584, 'Only the approved original unused vertices may be omitted')
        _need(receipt['original_compacted_vertices'] == 1952142 and receipt['original_triangles'] == 4173584
              and receipt['omitted_only_original_unused_vertices'] == 5749
              and receipt['all_oriented_triangles_exact'] is True
              and receipt['all_corner_colour_bytes_native_exact'] is True, 'Actual import receipt disagrees with independent full readback')
        _need(before == {str(path): (path.stat().st_size, path.stat().st_mtime_ns, digest(path)) for path in watched},
              'Authenticated sources or actual imported outputs changed')
        _need(not any(name == 'bpy' or name.startswith(('torch', 'triton', 'flex_gemm')) for name in sys.modules),
              'Independent verification must remain CPU-only without Blender/model runtime')
        result = {'classification': 'PASS', 'stage': 'independent_entire_import_readback',
                  'import_result_SHA256': digest(receipt_path), 'native_oracle_evidence': native,
                  'actual_file_SHA256': {name: receipt['files'][name]['sha256'] for name in receipt['files']},
                  'report': report, 'material_evidence_verified': True, 'original_input_identities': identities,
                  'sources_and_import_outputs_unchanged': True, 'geometry_repaired': False,
                  'colours_replaced': False, 'Blender_imports': 0, 'Blender_renders': 0,
                  'elapsed_seconds': time.monotonic() - began}
        save(out / 'result.json', result)
        print('INDEPENDENT_FULL_IMPORT_PASS', flush=True)
        return 0
    except BaseException as error:
        save(out / 'failure.json', {'classification': 'NO-GO', 'stage': stage, 'error': str(error),
                                   'traceback': traceback.format_exc(), 'retry': False,
                                   'elapsed_seconds': time.monotonic() - began})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
