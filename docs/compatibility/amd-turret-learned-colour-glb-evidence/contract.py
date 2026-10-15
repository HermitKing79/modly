"""CPU-only, exact triangle/learned-colour equivalence. No model dependencies."""
import hashlib
import json
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
ART = ROOT.parent / 'artifacts'
OUT = ART / 'amd-turret-coloured-glb-357e541'
SCENE = ART / 'amd-turret-learned-pbr-3485263/learned-pbr.blend'
MESH = ART / 'amd-turret-cpu-mesh-ad07f96/turret-full.ply'
FIELD = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-owner-067f668/field/texture-field.safetensors')
NEURAL = ART / 'amd-wsl-turret-writer-ready-4870856/turret-neural-output.safetensors'
SOURCES = {SCENE: '555ef22321040a806b8d21cad65087eb0b424d46e6d9ed8a2a11c6a49cc79339',
           MESH: '89549a36c6acc6f884174a2097ad7e346980d565c999117ea29f89cda9baad1a',
           FIELD: '7b24c946765d8e5de9718dbc358ce35141a56d836aa54fe7f8945cb1acbf7a80',
           NEURAL: 'dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22',
           ART / 'amd-turret-learned-colour-fa3a98c/learned-colour.blend':
           'b5c91db6b4c9717da987ab7c6d65ba495f78b0fcb87cd10e0ad3944dc1e17632'}
N, F = 1842968, 4012360
XYZ_SHA = 'ad9f926cd1168ede80979f359a949d818f92aa3871cc950df17231cdb83ef559'
FACE_SHA = '4e2de7427def4b8077dcc41bdcccd657032aadafbf860ef6edbd98e5b69e4725'
RGB_SHA = '45db29e954f49c90235a86b7477cc1990c6fe3c1e2e6a3e021bb9cd9f557d514'

def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def preserve():
    result = []
    for p, expected in SOURCES.items():
        actual = digest(p)
        assert actual == expected, str(p)
        result.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': actual})
    return result

def original():
    xyz = np.load(MESH.parent / 'vertices.npy', mmap_mode='r', allow_pickle=False)
    faces = np.load(MESH.parent / 'faces.npy', mmap_mode='r', allow_pickle=False)
    assert xyz.shape == (N, 3) and xyz.dtype == np.float32 and ahash(xyz) == XYZ_SHA
    assert faces.shape == (F, 3) and faces.dtype == np.int32 and ahash(faces) == FACE_SHA
    with FIELD.open('rb') as f:
        length = int.from_bytes(f.read(8), 'little')
        assert 0 < length <= 65536
        header = json.loads(f.read(length))
    row = header['features']
    assert row['dtype'] == 'F32' and row['shape'] == [N, 6]
    a, b = row['data_offsets']
    assert b-a == N*6*4 and b <= FIELD.stat().st_size-8-length
    features = np.memmap(FIELD, mode='r', offset=8+length+a, dtype='<f4', shape=(N, 6))
    assert np.isfinite(features).all()
    rgb = np.rint(np.clip(features[:, :3], 0, 1)*np.float32(255)).astype('u1')
    assert ahash(rgb) == RGB_SHA
    return xyz, faces, rgb

def fixture():
    with np.load(OUT/'fixture-source.npz', allow_pickle=False) as data:
        return data['xyz'], data['faces'], data['rgb']

def keys(xyz, rgb):
    """Exact float32 world positions + display RGB; +/- zero are equivalent."""
    assert xyz.shape == rgb.shape and xyz.shape[1] == 3 and np.isfinite(xyz).all()
    record = np.empty(len(xyz), dtype=[('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    record['xyz'] = xyz
    record['xyz'][record['xyz'] == 0] = 0  # canonical positive zero
    record['rgb'] = rgb
    return record.view('V15').reshape(-1)

def canonical_triangles(ids):
    """Preserve winding and multiplicity; allow only cyclic corner rotations."""
    best = np.asarray(ids, dtype='<i4').copy()
    for shift in (1, 2):
        candidate = np.roll(ids, -shift, axis=1)
        choose = ((candidate[:, 0] < best[:, 0]) |
                  ((candidate[:, 0] == best[:, 0]) & (candidate[:, 1] < best[:, 1])) |
                  ((candidate[:, 0] == best[:, 0]) & (candidate[:, 1] == best[:, 1]) &
                   (candidate[:, 2] < best[:, 2])))
        best[choose] = candidate[choose]
    return np.sort(best.view('V12').reshape(-1))

def compare(xyz, faces, rgb, target_xyz, target_faces, target_rgb, domain,
            source_linear=None, target_linear=None):
    """Match every triangle corner, including colour, with the original data."""
    assert target_faces.shape == faces.shape and domain in ('POINT', 'CORNER')
    assert target_faces.min() >= 0 and target_faces.max() < len(target_xyz)
    unique, first = np.unique(keys(xyz, rgb), return_index=True)
    source_ids = np.searchsorted(unique, keys(xyz, rgb)).astype('<i4')
    actual_ids = np.empty_like(faces)
    max_linear_error = 0.0
    for start in range(0, len(faces), 65536):
        stop = min(start+65536, len(faces))
        indices = target_faces[start:stop].reshape(-1)
        colours = target_rgb[indices] if domain == 'POINT' else target_rgb[start*3:stop*3]
        wanted = keys(target_xyz[indices], colours)
        found = np.searchsorted(unique, wanted)
        assert np.all(found < len(unique)), 'Foreign position or RGB'
        assert np.array_equal(unique[found], wanted), 'Position or learned RGB changed'
        if source_linear is not None:
            assert target_linear is not None
            actual_linear = target_linear[indices] if domain == 'POINT' else target_linear[start*3:stop*3]
            expected_linear = source_linear[first[found]]
            error = float(np.max(np.abs(actual_linear-expected_linear)))
            assert np.isfinite(actual_linear).all() and error <= 1e-7, 'Linear RGB changed'
            max_linear_error = max(max_linear_error, error)
        actual_ids[start:stop] = found.reshape(-1, 3)
    expected = canonical_triangles(source_ids[faces])
    actual = canonical_triangles(actual_ids)
    assert np.array_equal(actual, expected), 'Triangle coverage, winding or RGB changed'
    return {'all_triangle_corners_exact_position_and_RGB': True,
            'triangle_multiset_including_winding_and_multiplicity_exact': True,
            'canonical_triangle_sha256': ahash(expected), 'triangles': len(faces),
            'source_vertices': len(xyz), 'imported_vertices': len(target_xyz),
            'source_unique_position_RGB_keys': len(unique), 'import_colour_domain': domain,
            'maximum_linear_RGB_error': max_linear_error if source_linear is not None else None}
