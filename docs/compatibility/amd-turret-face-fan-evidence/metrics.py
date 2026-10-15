"""Bounded, sheet-aware CPU comparison; run inside Blender (no scene mutation).

compare((xyz, faces, rgba), (xyz, faces, rgba), source_face_sheets,
        target_face_sheets) returns JSON-safe dictionaries. Coordinates/colors
are vertex attributes; face sheet labels must survive decimation unchanged.
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _mesh(data, labels):
    xyz, faces, rgba = (np.asarray(a) for a in data)
    labels = np.asarray(labels)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2 or
            faces.shape[1] != 3 or rgba.shape != (len(xyz), 4) or
            labels.shape != (len(faces),)):
        raise ValueError("Expected xyz[N,3], faces[F,3], rgba[N,4], face sheets[F]")
    if len(faces) > 6000 or len(xyz) > 18000:
        raise ValueError("Bounded comparison permits at most 6000 faces / 18000 vertices")
    if not np.issubdtype(faces.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Face indices and sheet labels must be integers")
    if not np.isfinite(xyz).all() or not np.isfinite(rgba).all():
        raise ValueError("Non-finite position/color")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("Face index outside vertex array")
    xyz, rgba = xyz.astype(np.float64), rgba.astype(np.float64)
    faces, labels = faces.astype(np.int64), labels.astype(np.int64)
    tri = xyz[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    return dict(xyz=xyz, faces=faces, rgba=rgba, sheets=labels, tri=tri, area=area)


def _areas(mesh):
    return {int(s): float(mesh["area"][mesh["sheets"] == s].sum())
            for s in np.unique(mesh["sheets"])}


def _trees(mesh):
    trees = {}
    for sheet in np.unique(mesh["sheets"]):
        ids = np.flatnonzero((mesh["sheets"] == sheet) & (mesh["area"] > 0))
        if len(ids):
            tri = mesh["tri"][ids]
            polygons = np.arange(len(ids) * 3).reshape(-1, 3)
            tree = BVHTree.FromPolygons(tri.reshape(-1, 3).tolist(), polygons.tolist(), all_triangles=True)
            trees[int(sheet)] = (tree, tri, mesh["rgba"][mesh["faces"][ids]])
    return trees


def _samples(mesh):
    f, labels = mesh["faces"], mesh["sheets"]
    pairs = np.unique(np.column_stack((f.ravel(), np.repeat(labels, 3))), axis=0)
    # Reserve half of the vertex budget for sheet boundaries when available.
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    tagged = np.column_stack((np.sort(edges, axis=1), np.tile(labels, 3)))
    unique, counts = np.unique(tagged, axis=0, return_counts=True)
    boundary = {(int(v), int(e[2])) for e in unique[counts == 1] for v in e[:2]}
    boundary_ids = np.array([i for i, p in enumerate(pairs) if tuple(p) in boundary], dtype=np.int64)
    rng = np.random.default_rng(0)
    if len(pairs) > 512:
        chosen = rng.choice(boundary_ids, min(256, len(boundary_ids)), replace=False)
        remaining = np.setdiff1d(np.arange(len(pairs)), chosen)
        chosen = np.sort(np.concatenate((chosen, rng.choice(remaining, 512 - len(chosen), replace=False))))
        pairs = pairs[chosen]
    k, total = min(512, int(np.count_nonzero(mesh["area"]))), float(mesh["area"].sum())
    # Deduplicate strata that select the same large face: still at most 512 centroids.
    face_ids = (np.unique(np.searchsorted(np.cumsum(mesh["area"]),
                (np.arange(k) + .5) * total / k)) if k and total else np.empty(0, dtype=np.int64))
    points = np.concatenate((mesh["xyz"][pairs[:, 0]], mesh["tri"][face_ids].mean(axis=1)))
    colors = np.concatenate((mesh["rgba"][pairs[:, 0]], mesh["rgba"][f[face_ids]].mean(axis=1)))
    sheets = np.concatenate((pairs[:, 1], labels[face_ids]))
    return points, colors, sheets, dict(vertices=len(pairs), centroids=len(face_ids))


def _summary(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(rms=float(np.sqrt(np.mean(a * a))) if a.size else None,
                p95=float(np.percentile(a, 95)) if a.size else None,
                max=float(a.max()) if a.size else None)


def _barycentric(point, tri):
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    weights = np.array([np.dot(np.cross(tri[(i + 1) % 3] - point,
                                       tri[(i + 2) % 3] - point), normal) for i in range(3)])
    weights = np.clip(weights / np.dot(normal, normal), 0, 1)
    return weights / weights.sum()


def _direction(mesh, destination, diagonal):
    points, colors, sheets, sample_counts = _samples(mesh)
    sheet_area = _areas(mesh)
    missing = sorted(set(sheet_area) - set(destination))
    missing_area = sum(sheet_area[s] for s in missing)
    distances, differences, failed = [], [], 0
    for point, color, sheet in zip(points, colors, sheets):
        if int(sheet) not in destination:
            continue
        tree, tri, rgba = destination[int(sheet)]
        nearest, _, index, distance = tree.find_nearest(Vector(point))
        if nearest is None or index is None:
            failed += 1
            continue
        weights = _barycentric(np.asarray(nearest), tri[index])
        distances.append(float(distance) / diagonal)
        differences.append(color - weights @ rgba[index])
    delta = np.asarray(differences, dtype=np.float64).reshape(-1, 4)
    matched = dict(geometry_normalized=_summary(distances),
                   rgb_channel_error=_summary(np.sqrt(np.mean(delta[:, :3] ** 2, axis=1))),
                   alpha_error=_summary(np.abs(delta[:, 3])),
                   rgba_rmse_by_channel=np.sqrt(np.mean(delta ** 2, axis=0)).tolist() if len(delta) else None)
    complete = not missing and not failed
    # Partial-sheet measurements cannot stand in for the complete surface.
    overall = matched if complete else dict(geometry_normalized=_summary([]),
        rgb_channel_error=_summary([]), alpha_error=_summary([]), rgba_rmse_by_channel=None)
    return dict(complete=complete, samples=sample_counts, matched_samples=len(distances),
                failed_queries=failed, missing_sheet_ids=missing, lost_sheet_area=missing_area,
                lost_sheet_area_fraction=missing_area / sum(sheet_area.values()) if sum(sheet_area.values()) else 0.,
                **overall, matched_sheets_only=matched)


def _raster(xyz, faces, axes, center, span):
    projected = (xyz[:, axes] - center) * (256. / span) + 128.
    mask = np.zeros((256, 256), dtype=bool)
    for tri in projected[faces]:
        a, b, c = tri
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) < 1e-10:
            continue
        lo = np.maximum(np.ceil(tri.min(axis=0) - .5).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(axis=0) - .5).astype(int), 255)
        if np.any(hi < lo):
            continue
        x = np.arange(lo[0], hi[0] + 1)[None, :] + .5
        y = np.arange(lo[1], hi[1] + 1)[:, None] + .5
        inside = np.ones((len(y), x.shape[1]), dtype=bool)
        for p, q in ((a, b), (b, c), (c, a)):
            edge = (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
            inside &= edge >= -1e-9 if cross > 0 else edge <= 1e-9
        mask[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] |= inside
    referenced = np.unique(faces)
    clipped = int(np.count_nonzero(np.any((projected[referenced] < 0) | (projected[referenced] > 256), axis=1)))
    return mask, clipped


def _silhouettes(source, target, diagonal):
    results = {}
    source_points = source["tri"].reshape(-1, 3)
    for name, axes in (("x", [1, 2]), ("y", [0, 2]), ("z", [0, 1])):
        points = source_points[:, axes]
        low, high = points.min(axis=0), points.max(axis=0)
        center, span = (low + high) * .5, max(float((high - low).max()) * 1.1, diagonal * 1e-9)
        src, src_clip = _raster(source["xyz"], source["faces"], axes, center, span)
        dst, dst_clip = _raster(target["xyz"], target["faces"], axes, center, span)
        a, b = int(src.sum()), int(dst.sum())
        intersection, union = int((src & dst).sum()), int((src | dst).sum())
        results[name] = dict(informative=bool(a), reason=None if a else "source projects to no occupied pixel centers",
            source_pixels=a, target_pixels=b, intersection_pixels=intersection, union_pixels=union,
            iou=intersection / union if a and union else None,
            lost_fraction=float((src & ~dst).sum()) / a if a else None,
            gained_fraction=float((dst & ~src).sum()) / a if a else None,
            source_clipped_vertices=src_clip, target_clipped_vertices=dst_clip,
            original_bounds_center=center.tolist(), original_bounds_span=span)
    return results


def compare(source_tuple, target_tuple, source_sheet_ids, target_sheet_ids):
    """Compare <=6000-face meshes, keeping source sheet identity in both directions.

    Geometry statistics are distances / source AABB diagonal. RGB error is the
    per-sample channel RMSE in the supplied color space; alpha is separate.
    A missing destination sheet makes complete-surface error statistics null.
    """
    source, target = _mesh(source_tuple, source_sheet_ids), _mesh(target_tuple, target_sheet_ids)
    if not len(source["faces"]):
        raise ValueError("Source must contain triangles")
    points = source["tri"].reshape(-1, 3)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if diagonal <= 0:
        raise ValueError("Source diagonal must be positive")
    forward = _direction(source, _trees(target), diagonal)
    backward = _direction(target, _trees(source), diagonal)
    complete = forward["complete"] and backward["complete"]
    worst = {key: max(forward["geometry_normalized"][key], backward["geometry_normalized"][key])
             if complete and forward["geometry_normalized"][key] is not None and
             backward["geometry_normalized"][key] is not None else None for key in ("rms", "p95", "max")}
    return dict(source_diagonal=diagonal, source_to_target=forward, target_to_source=backward,
        worst_direction_geometry_normalized=worst, source_sheet_areas=_areas(source), target_sheet_areas=_areas(target),
        source_degenerate_faces=int(np.count_nonzero(source["area"] == 0)),
        target_degenerate_faces=int(np.count_nonzero(target["area"] == 0)),
        silhouettes=_silhouettes(source, target, diagonal),
        sampling_limits=["Fixed seed 0; at most 512 (vertex, sheet) occurrences and 512 unique area-stratified face centroids per direction.",
            "Finite samples can miss narrow holes, small fragments and unsampled maxima; these are not Hausdorff bounds.",
            "Centroid strata selecting one face are deduplicated; reported errors are unweighted sample statistics.",
            "Nearest matches are restricted to the same face sheet; ambiguities within one sheet remain possible.",
            "Silhouettes use three axis views, 256x256 pixel centers and a fixed source frame with 10% padding; clipping and empty views are explicit.",
            "Color errors use supplied channel values without color-space conversion; isolated unreferenced vertices are ignored."])
