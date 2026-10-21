"""Bounded, CPU NumPy diagnostics; no asset loading or geometry changes.

Original vertex IDs are only used for undirected edge keys. Global source
faces, when supplied as an ndarray or memmap, are scanned in bounded chunks.
Coverage is a union of projected triangle footprints at pixel centres and
does not establish depth visibility or semantic segmentation.
"""

import numpy as np


ORIGINAL_VERTEX_COUNT = 1957891
MAX_SELECTED_FACES = 6000
MAX_GLOBAL_FACE_CHUNK = 262144
ROOT_ROI = (346, 393, 393, 436)
PROXY_WINDOWS = {"leaf": (350, 399, 378, 425), "vine": (380, 394, 392, 434)}


def _component_sizes(labels, face_ids, face_count):
    """Join faces sharing a label, then return descending component sizes."""
    parent = np.arange(face_count, dtype=np.int32)
    sizes = np.ones(face_count, dtype=np.int32)

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    order = np.argsort(labels, kind="stable")
    adjacent = np.flatnonzero(labels[order][1:] == labels[order][:-1])
    for index in adjacent:
        left, right = find(face_ids[order[index]]), find(face_ids[order[index + 1]])
        if left != right:
            if sizes[left] < sizes[right]:
                left, right = right, left
            parent[right] = left
            sizes[left] += sizes[right]
    return sorted((int(sizes[i]) for i in range(face_count) if parent[i] == i), reverse=True)


def _global_incidence(selected_keys, original_faces, chunk_size):
    counts = np.zeros(selected_keys.size, dtype=np.int64)
    for start in range(0, original_faces.shape[0], chunk_size):
        chunk = original_faces[start:start + chunk_size]
        if chunk.size and (chunk.min() < 0 or chunk.max() >= ORIGINAL_VERTEX_COUNT):
            raise ValueError("original_faces contains an out-of-range original vertex ID")
        if not selected_keys.size:
            continue
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a = chunk[:, first].astype(np.int64, copy=False)
            b = chunk[:, second].astype(np.int64, copy=False)
            keys = np.minimum(a, b) * ORIGINAL_VERTEX_COUNT + np.maximum(a, b)
            positions = np.searchsorted(selected_keys, keys)
            in_range = positions < selected_keys.size
            positions, keys = positions[in_range], keys[in_range]
            matches = positions[selected_keys[positions] == keys]
            counts += np.bincount(matches, minlength=selected_keys.size)
    return counts


def analyze_topology(selected_faces_local, original_vertex_ids, original_faces=None,
                     *, global_face_chunk_size=MAX_GLOBAL_FACE_CHUNK):
    """Report triangle topology without mutating or remapping input arrays.

    ``selected_faces_local`` must be int32 [F,3], F <= 6000; the unique int64
    ``original_vertex_ids`` maps local vertices to unchanged source IDs.
    Keys are min(IDa,IDb)*1957891+max(IDa,IDb), in ascending key order.
    Local incidence counts face-edge occurrences. Face connectivity is
    reported separately for shared edges and shared vertices.

    Global incidence allocates only chunk-sized arrays and selected-key counts,
    never an edge table for the entire original mesh. Chunk requests above
    262144 faces are capped. None denotes unavailable global boundary evidence;
    a zero source incidence is reported separately as unmatched evidence.
    All return values are JSON-compatible.
    """
    faces, ids = selected_faces_local, original_vertex_ids
    if not isinstance(faces, np.ndarray) or faces.dtype != np.dtype(np.int32):
        raise TypeError("selected_faces_local must be an int32 ndarray")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) > MAX_SELECTED_FACES:
        raise ValueError("selected_faces_local must have shape [F,3] with F <= 6000")
    if not isinstance(ids, np.ndarray) or ids.dtype != np.dtype(np.int64):
        raise TypeError("original_vertex_ids must be an int64 ndarray")
    if ids.ndim != 1:
        raise ValueError("original_vertex_ids must be one-dimensional")
    if ids.size and (ids.min() < 0 or ids.max() >= ORIGINAL_VERTEX_COUNT):
        raise ValueError("original_vertex_ids contains an out-of-range ID")
    if np.unique(ids).size != ids.size:
        raise ValueError("original_vertex_ids must be unique")
    if faces.size and (faces.min() < 0 or faces.max() >= ids.size):
        raise ValueError("selected_faces_local contains an out-of-range local vertex")
    if faces.size and np.any((faces[:, 0] == faces[:, 1]) |
                             (faces[:, 1] == faces[:, 2]) |
                             (faces[:, 2] == faces[:, 0])):
        raise ValueError("selected_faces_local contains a repeated-vertex triangle")
    if (isinstance(global_face_chunk_size, (bool, np.bool_)) or
            not isinstance(global_face_chunk_size, (int, np.integer))):
        raise TypeError("global_face_chunk_size must be a positive integer")
    if global_face_chunk_size <= 0:
        raise ValueError("global_face_chunk_size must be positive")
    if original_faces is not None:
        if not isinstance(original_faces, np.ndarray) or original_faces.dtype.kind not in "iu":
            raise TypeError("original_faces must be an integer ndarray or memmap")
        if original_faces.ndim != 2 or original_faces.shape[1] != 3:
            raise ValueError("original_faces must have shape [F,3]")

    mapped = ids[faces]
    first, second = mapped, mapped[:, [1, 2, 0]]
    edge_keys = (np.minimum(first, second) * ORIGINAL_VERTEX_COUNT +
                 np.maximum(first, second)).reshape(-1)
    keys, local_counts = np.unique(edge_keys, return_counts=True)
    face_ids = np.repeat(np.arange(len(faces), dtype=np.int32), 3)
    edge_components = _component_sizes(edge_keys, face_ids, len(faces))
    vertex_components = _component_sizes(faces.reshape(-1), face_ids, len(faces))
    global_counts = None
    if original_faces is not None:
        global_counts = _global_incidence(
            keys, original_faces, min(int(global_face_chunk_size), MAX_GLOBAL_FACE_CHUNK)
        )
    boundary = local_counts == 1
    return {
        "face_count": int(len(faces)),
        "used_vertex_count": int(np.unique(faces).size),
        "face_edge_connected_components": len(edge_components),
        "face_edge_component_sizes": edge_components,
        "face_vertex_connected_components": len(vertex_components),
        "face_vertex_component_sizes": vertex_components,
        "edge_count": int(keys.size),
        "edge_incidence_1": int(np.count_nonzero(boundary)),
        "edge_incidence_2": int(np.count_nonzero(local_counts == 2)),
        "edge_incidence_gt2": int(np.count_nonzero(local_counts > 2)),
        "selected_original_edge_keys": keys.tolist(),
        "selected_edge_local_incidence": local_counts.tolist(),
        "selected_edge_global_incidence": None if global_counts is None else global_counts.tolist(),
        "artificial_crop_boundary_edges": None if global_counts is None else
            int(np.count_nonzero(boundary & (global_counts > 1))),
        "natural_boundary_edges": None if global_counts is None else
            int(np.count_nonzero(boundary & (global_counts == 1))),
        "unmatched_boundary_edges": None if global_counts is None else
            int(np.count_nonzero(boundary & (global_counts == 0))),
    }


def _window(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 4 or
            any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                for x in value)):
        raise TypeError(f"{name} must contain four integer pixel bounds")
    x0, y0, x1, y1 = map(int, value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must have positive width and height")
    return x0, y0, x1, y1


def coverage_proxies(projected_xy, roi=ROOT_ROI, proxies=None):
    """Count union pixel-centre hits in a bounded image ROI and proxy windows.

    Bounds use [x0,y0,x1,y1) integer pixels. The root ROI is at most 47x43;
    proxy windows must lie within it. Degenerate triangles have zero area.
    A triangle edge through a pixel centre includes that pixel. Projection
    coordinates must be finite float64 [F,3,2], F <= 6000.
    """
    if not isinstance(projected_xy, np.ndarray) or projected_xy.dtype != np.dtype(np.float64):
        raise TypeError("projected_xy must be a float64 ndarray")
    if (projected_xy.ndim != 3 or projected_xy.shape[1:] != (3, 2) or
            len(projected_xy) > MAX_SELECTED_FACES):
        raise ValueError("projected_xy must have shape [F,3,2] with F <= 6000")
    if not np.isfinite(projected_xy).all():
        raise ValueError("projected_xy must contain only finite coordinates")
    x0, y0, x1, y1 = _window(roi, "roi")
    if x1 - x0 > 47 or y1 - y0 > 43:
        raise ValueError("roi must be bounded to at most 47x43 pixels")
    proxies = PROXY_WINDOWS if proxies is None else proxies
    if not isinstance(proxies, dict) or any(not isinstance(name, str) for name in proxies):
        raise TypeError("proxies must map string names to integer pixel windows")
    windows = {name: _window(value, f"proxy {name}") for name, value in proxies.items()}
    if any(a < x0 or b < y0 or c > x1 or d > y1 for a, b, c, d in windows.values()):
        raise ValueError("proxy windows must lie within roi")

    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx, yy = xx + 0.5, yy + 0.5
    covered = np.zeros(xx.shape, dtype=bool)
    for triangle in projected_xy:
        a, b, c = triangle
        # Bounding rejection also keeps irrelevant triangles out of arithmetic.
        if (triangle[:, 0].max() < x0 + 0.5 or triangle[:, 0].min() > x1 - 0.5 or
                triangle[:, 1].max() < y0 + 0.5 or triangle[:, 1].min() > y1 - 0.5):
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0:
            continue
        positive, negative = np.ones(xx.shape, dtype=bool), np.ones(xx.shape, dtype=bool)
        for first, second in ((a, b), (b, c), (c, a)):
            side = ((second[0] - first[0]) * (yy - first[1]) -
                    (second[1] - first[1]) * (xx - first[0]))
            positive &= side >= 0
            negative &= side <= 0
        covered |= positive | negative
    counts = {name: int(np.count_nonzero(covered[b-y0:d-y0, a-x0:c-x0]))
              for name, (a, b, c, d) in windows.items()}
    return {
        "covered_pixel_count": int(np.count_nonzero(covered)),
        "roi_pixel_count": int(covered.size),
        "proxy_covered_pixel_counts": counts,
        "proxies": {name: {"covered_pixels": counts[name], "pixel_count": (c-a) * (d-b)}
                    for name, (a, b, c, d) in windows.items()},
        "all_proxies_have_coverage": bool(counts) and all(count > 0 for count in counts.values()),
        "interpretation": "manually nominated image windows; selected projected footprints only; "
                          "no depth visibility or semantic segmentation claim",
    }
