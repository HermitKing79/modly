"""Pure CPU NumPy disk-chart cuts; no asset loading or UV/native execution."""

import numpy as np


MAX_FACES = 6000


class _DisjointSet:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first == second:
            return False
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        return True


def _source_ids(value, size, name):
    if not isinstance(value, np.ndarray) or value.dtype.kind != "i":
        raise TypeError(f"{name} must be a signed integer ndarray")
    if value.ndim != 1 or len(value) != size:
        raise ValueError(f"{name} has the wrong shape")
    if value.size and (value.min() < 0 or np.unique(value).size != value.size):
        raise ValueError(f"{name} must contain unique nonnegative source IDs")


def _disk_report(chart_id, face_indices, seam_faces, source_face_ids):
    triangles = seam_faces[face_indices]
    vertices = set(map(int, triangles.reshape(-1)))
    edges = {}
    for triangle in triangles:
        for corner in range(3):
            a, b = int(triangle[corner]), int(triangle[(corner + 1) % 3])
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append((a, b))
    boundary = {}
    for (a, b), occurrences in edges.items():
        if len(occurrences) == 1:
            boundary.setdefault(a, []).append(b)
            boundary.setdefault(b, []).append(a)
        elif len(occurrences) != 2 or occurrences[0] != occurrences[1][::-1]:
            raise RuntimeError("cut chart contains a nonmanifold or inconsistently oriented edge")
    if any(len(neighbors) != 2 for neighbors in boundary.values()):
        raise RuntimeError("cut chart boundary is not a union of simple cycles")
    unseen, boundary_loops = set(boundary), 0
    while unseen:
        boundary_loops += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in boundary[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    count = len(face_indices)
    boundary_edges = sum(len(neighbors) for neighbors in boundary.values()) // 2
    if (len(vertices) != count + 2 or len(edges) != 2 * count + 1 or
            boundary_edges != count + 2 or boundary_loops != 1):
        raise RuntimeError("triangle-tree disk invariants failed")
    return {
        "chart_id": chart_id,
        "minimum_source_face_id": int(source_face_ids[face_indices].min()),
        "face_count": count,
        "vertex_count": len(vertices),
        "edge_count": len(edges),
        "boundary_edge_count": boundary_edges,
        "boundary_loop_count": boundary_loops,
        "euler_characteristic": len(vertices) - len(edges) + count,
    }


def forest(vertices, faces, source_face_ids, source_vertex_ids):
    """Cut an oriented triangle selection into deterministic disk charts.

    Inputs: float32 [V,3] positions, int32 [F,3] triangles (1 <= F <= 6000),
    unique signed integer source face IDs [F] and source vertex IDs [V].
    Geometry must be finite and triangles must have nonzero area.

    Only original edges with exactly two opposite-oriented occurrences are
    eligible for joining. Kruskal keeps a maximum squared-Euclidean-length
    dual forest; ties use lexicographic original endpoint IDs, then source
    face IDs. Every other original edge is explicitly cut. Corner endpoints
    are glued only across retained edges, including within the same chart;
    source vertex identity alone never welds corners.

    Output seam vertices follow earliest selected face/corner index. Face and
    corner order is unchanged. ``source_corner_indices`` records that earliest
    flattened index; source vertex maps refer to local and global source IDs.
    Chart IDs are ordered by minimum original source face ID. Returned source
    edge arrays are lexicographically sorted. No inputs are modified.
    """
    if not isinstance(vertices, np.ndarray) or vertices.dtype != np.dtype(np.float32):
        raise TypeError("vertices must be a float32 ndarray")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must have shape [V,3] and finite coordinates")
    if not isinstance(faces, np.ndarray) or faces.dtype != np.dtype(np.int32):
        raise TypeError("faces must be an int32 ndarray")
    if faces.ndim != 2 or faces.shape[1] != 3 or not 1 <= len(faces) <= MAX_FACES:
        raise ValueError("faces must have shape [F,3] with 1 <= F <= 6000")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contains an out-of-range local vertex ID")
    _source_ids(source_face_ids, len(faces), "source_face_ids")
    _source_ids(source_vertex_ids, len(vertices), "source_vertex_ids")
    corners = vertices[faces].astype(np.float64)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    if np.any(np.all(cross == 0, axis=1)):
        raise ValueError("faces contains a zero-area triangle")

    occurrences = {}
    for face_index, triangle in enumerate(faces):
        for corner in range(3):
            a = int(source_vertex_ids[triangle[corner]])
            b = int(source_vertex_ids[triangle[(corner + 1) % 3]])
            key = (min(a, b), max(a, b))
            occurrences.setdefault(key, []).append((face_index, corner, a, b))
    eligible = []
    boundary_count = nonmanifold_count = inconsistent_count = 0
    for key, uses in occurrences.items():
        if len(uses) == 1:
            boundary_count += 1
        elif len(uses) > 2:
            nonmanifold_count += 1
        elif uses[0][2:] != uses[1][2:][::-1]:
            inconsistent_count += 1
        else:
            f0, corner, _, _ = uses[0]
            f1 = uses[1][0]
            difference = corners[f0, corner] - corners[f0, (corner + 1) % 3]
            length_squared = float(difference @ difference)
            face_pair = tuple(sorted((int(source_face_ids[f0]), int(source_face_ids[f1]))))
            eligible.append((-length_squared, *key, *face_pair, key))
    eligible.sort()
    face_sets, corner_sets = _DisjointSet(len(faces)), _DisjointSet(3 * len(faces))
    retained = set()
    for *_, key in eligible:
        first, second = occurrences[key]
        f0, c0, _, _ = first
        f1, c1, _, _ = second
        if face_sets.union(f0, f1):
            retained.add(key)
            corner_sets.union(3 * f0 + c0, 3 * f1 + (c1 + 1) % 3)
            corner_sets.union(3 * f0 + (c0 + 1) % 3, 3 * f1 + c1)

    chart_members = {}
    for face_index in range(len(faces)):
        chart_members.setdefault(face_sets.find(face_index), []).append(face_index)
    ordered_charts = sorted(chart_members.values(), key=lambda group: int(source_face_ids[group].min()))
    face_chart_ids = np.empty(len(faces), dtype=np.int32)
    for chart_id, group in enumerate(ordered_charts):
        face_chart_ids[group] = chart_id

    corner_roots = np.fromiter((corner_sets.find(i) for i in range(3 * len(faces))),
                               dtype=np.int32, count=3 * len(faces))
    minimum_corner = np.full(3 * len(faces), 3 * len(faces), dtype=np.int64)
    np.minimum.at(minimum_corner, corner_roots, np.arange(3 * len(faces), dtype=np.int64))
    source_corner_indices = np.unique(minimum_corner[corner_roots])
    seam_faces = np.searchsorted(source_corner_indices, minimum_corner[corner_roots]).astype(np.int32)
    seam_faces = seam_faces.reshape(-1, 3)
    source_local_vertex_ids = faces.reshape(-1)[source_corner_indices].copy()
    seam_vertices = vertices[source_local_vertex_ids].copy()
    if not np.array_equal(seam_vertices[seam_faces].view(np.uint32), vertices[faces].view(np.uint32)):
        raise RuntimeError("oriented source corner coordinate bits changed")
    charts = [_disk_report(i, group, seam_faces, source_face_ids)
              for i, group in enumerate(ordered_charts)]
    source_edges = sorted(occurrences)
    retained_edges = sorted(retained)
    cut_edges = [key for key in source_edges if key not in retained]
    return {
        "seam_vertices": seam_vertices,
        "seam_faces": seam_faces,
        "source_vertex_ids": source_vertex_ids[source_local_vertex_ids].copy(),
        "source_local_vertex_ids": source_local_vertex_ids,
        "source_corner_indices": source_corner_indices,
        "source_face_ids": source_face_ids.copy(),
        "face_chart_ids": face_chart_ids,
        "source_edges": np.asarray(source_edges, dtype=np.int64).reshape(-1, 2),
        "source_edge_incidence": np.asarray([len(occurrences[key]) for key in source_edges], dtype=np.int32),
        "retained_source_edges": np.asarray(retained_edges, dtype=np.int64).reshape(-1, 2),
        "retained_source_face_pairs": np.asarray(
            [sorted(int(source_face_ids[use[0]]) for use in occurrences[key]) for key in retained_edges],
            dtype=np.int64,
        ).reshape(-1, 2),
        "cut_source_edges": np.asarray(cut_edges, dtype=np.int64).reshape(-1, 2),
        "report": {
            "face_count": len(faces),
            "chart_count": len(charts),
            "seam_vertex_count": len(seam_vertices),
            "original_boundary_edge_count": boundary_count,
            "nonmanifold_edge_count": nonmanifold_count,
            "orientation_inconsistent_edge_count": inconsistent_count,
            "eligible_edge_count": len(eligible),
            "retained_edge_count": len(retained),
            "cycle_cut_edge_count": len(eligible) - len(retained),
            "seam_edge_count": len(cut_edges),
            "original_oriented_corner_bits_preserved": True,
            "charts": charts,
        },
    }
