"""CPU-only, <=6000-triangle face-fan separation; no coordinate welding.

origin[new_vertex] is an input vertex row. POINT RGBA, input dtypes, face
order, corner order and duplicate multiplicity survive unchanged. Unused
vertices are omitted. join_allowed follows edges(...)[0], the LOCAL sorted
edge keys; callers must block local edges whose GLOBAL incidence is not two.
"""
import numpy as np


def _faces(faces, n):
    faces = np.asarray(faces)
    if not (0 < n < 2**32 and faces.ndim == 2 and faces.shape[1] == 3
            and 0 < len(faces) <= 6000 and faces.dtype.kind in 'iu'):
        raise ValueError('Expected 1..6000 indexed triangles and a positive vertex count')
    if faces.min() < 0 or faces.max() >= n or np.any(faces == np.roll(faces, 1, axis=1)):
        raise ValueError('Invalid or repeated triangle vertex index')
    return faces


def _xyz(xyz):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind != 'f' or not np.isfinite(xyz).all():
        raise ValueError('Expected finite floating point XYZ rows')
    return xyz


def edges(faces, n):
    """Return sorted uint64 keys, occurrence counts, and F-by-3 edge keys."""
    faces = _faces(faces, n).astype(np.uint64, copy=False)
    following = faces[:, (1, 2, 0)]
    per_face = np.minimum(faces, following) * np.uint64(n) + np.maximum(faces, following)
    unique, counts = np.unique(per_face, return_counts=True)
    return unique, counts, per_face


class _DSU:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = int(self.parent[i])
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)

    def roots(self):
        return np.array([self.find(i) for i in range(len(self.parent))])


def separate(xyz, faces, rgba, join_allowed=None):
    xyz = _xyz(xyz)
    faces = _faces(faces, len(xyz))
    rgba = np.asarray(rgba)
    if rgba.shape != (len(xyz), 4) or rgba.dtype.kind != 'f' or not np.isfinite(rgba).all():
        raise ValueError('Expected finite POINT floating point RGBA rows')
    keys, counts, per_face = edges(faces, len(xyz))
    allowed = np.ones(len(keys), dtype=bool) if join_allowed is None else np.asarray(join_allowed)
    if allowed.shape != keys.shape or allowed.dtype != np.dtype(bool):
        raise ValueError('join_allowed must be boolean and align with sorted local edge keys')
    order = np.argsort(per_face.ravel(), kind='stable')
    starts = np.cumsum(counts) - counts
    dsu = _DSU(faces.size)
    for edge in np.flatnonzero((counts == 2) & allowed):
        a, b = order[starts[edge]:starts[edge] + 2]
        fa, ca, fb, cb = int(a // 3), int(a % 3), int(b // 3), int(b % 3)
        for corner in (ca, (ca + 1) % 3):
            other = cb if faces[fb, cb] == faces[fa, corner] else (cb + 1) % 3
            dsu.join(3 * fa + corner, 3 * fb + other)
    _, first, inverse = np.unique(dsu.roots(), return_index=True, return_inverse=True)
    origin = faces.ravel()[first].astype(np.int64)
    out_faces = inverse.reshape(faces.shape).astype(np.int32)
    out_xyz, out_rgba = xyz[origin].copy(), rgba[origin].copy()
    assert np.array_equal(origin[out_faces], faces)
    assert out_xyz[out_faces].tobytes() == xyz[faces].tobytes()
    assert out_rgba[out_faces].tobytes() == rgba[faces].tobytes()
    assert np.all(edges(out_faces, len(origin))[1] <= 2)
    return out_xyz, out_faces, out_rgba, origin


def stats(xyz, faces):
    """Return JSON-safe metrics plus int32 face-component IDs; areas align to IDs.

    Components use shared indexed edges, including edges of incidence >2.
    Bounds cover referenced surface vertices. Coincident copies remain distinct.
    """
    xyz = _xyz(xyz)
    faces = _faces(faces, len(xyz))
    keys, counts, per_face = edges(faces, len(xyz))
    order = np.argsort(per_face.ravel(), kind='stable')
    starts = np.cumsum(counts) - counts
    dsu = _DSU(len(faces))
    for edge in np.flatnonzero(counts > 1):
        touching = order[starts[edge]:starts[edge] + counts[edge]] // 3
        for face in touching[1:]:
            dsu.join(int(touching[0]), int(face))
    _, ids = np.unique(dsu.roots(), return_inverse=True)
    ids = ids.astype(np.int32)
    triangles = xyz[faces].astype(np.float64)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * .5
    component_areas = np.bincount(ids, weights=areas)
    boundary = keys[counts == 1]
    lengths = np.linalg.norm(xyz[(boundary // len(xyz)).astype(np.int64)].astype(np.float64)
                             - xyz[(boundary % len(xyz)).astype(np.int64)], axis=1)
    total = float(component_areas.sum())
    return {'vertices': len(xyz), 'triangles': len(faces),
            'nonmanifold_gt2_edges': int(np.count_nonzero(counts > 2)),
            'boundary_edges': len(boundary), 'boundary_length': float(lengths.sum()),
            'components': len(component_areas), 'component_areas': component_areas.tolist(),
            'top4_area_fraction': float(np.sort(component_areas)[-4:].sum() / total) if total else 0.,
            'zero_area_triangles': int(np.count_nonzero(areas == 0)),
            'bounds': {'min': triangles.min(axis=(0, 1)).tolist(),
                       'max': triangles.max(axis=(0, 1)).tolist()}}, ids
