"""Bounded CPU evidence for the authenticated square orthographic brick view.

This is geometric center-ray sampling, not a replacement image renderer. It does
not predict area-light visibility, shadows, AgX, denoising, or pixel coverage.
Original vertex/face order and winding are retained. Coordinates are used in
their original world order; the authenticated render used an identity object
transform. PNG coordinates have their origin at the top left, so integer pixel
(column, row) is sampled at (column + .5, row + .5).

Only square ORTHO cameras with an explicitly supplied basis are accepted: the
available metadata does not establish a nonsquare sensor-fit convention. The
actual +X basis is right +Y, up +Z, camera outward +X. Camera-space positive
depth is distance along the inward ray, and the closest positive depth wins.

Typical use: project once, build_triangle_bounds once, then sample 20--40 rays.
The bounds cache is four float32 numbers per original face (about 64 MiB for
4,173,584 faces), with outward rounding. Triangle arithmetic is float64 in
bounded chunks. Inputs are never written and the module performs no file I/O.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OrthographicProjection:
    screen: np.ndarray
    depth: np.ndarray
    outward: np.ndarray
    width: int
    height: int
    clip: tuple


@dataclass(frozen=True)
class TriangleBounds:
    minimum: np.ndarray
    maximum: np.ndarray


def project(vertices, camera, width, height, *, chunk_size=131072):
    """Project original world vertices into PNG boundary coordinates + depth.

    ``camera`` is one view record from render-metadata.json. ``screen`` values
    refer to image boundaries: (0,0) is the upper left corner, not pixel center.
    Exposure does not enter this geometric operation.
    """
    if camera.get("camera_type") != "ORTHO":
        raise ValueError("Only the authenticated ORTHO projection is supported")
    if width != height or width <= 0 or int(width) != width:
        raise ValueError("Only positive square image dimensions have a verified camera contract")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Original vertices must have shape [N,3]")
    location = np.asarray(camera["camera_location"], dtype=np.float64)
    right = np.asarray(camera["camera_right"], dtype=np.float64)
    up = np.asarray(camera["camera_up"], dtype=np.float64)
    axis = np.asarray(camera["axis_direction"], dtype=np.float64)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (location, right, up, axis)):
        raise ValueError("Camera location and basis must be finite three-vectors")
    outward = np.cross(right, up)
    if not (np.isclose(np.linalg.norm(right), 1., atol=1e-7, rtol=0.) and
            np.isclose(np.linalg.norm(up), 1., atol=1e-7, rtol=0.) and
            np.isclose(np.dot(right, up), 0., atol=1e-7, rtol=0.) and
            np.allclose(outward, axis / np.linalg.norm(axis), atol=1e-7, rtol=0.)):
        raise ValueError("Camera basis must be orthonormal and agree with outward axis_direction")
    scale = float(camera["ortho_scale"])
    clip = tuple(float(x) for x in camera["clip"])
    if not math.isfinite(scale) or scale <= 0 or len(clip) != 2 or not 0 < clip[0] < clip[1]:
        raise ValueError("Invalid orthographic scale or clip distances")
    if not np.isfinite(clip).all():
        raise ValueError("Clip distances must be finite")
    screen = np.empty((len(vertices), 2), dtype=np.float64)
    depth = np.empty(len(vertices), dtype=np.float64)
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        relative = np.asarray(vertices[start:stop], dtype=np.float64) - location
        if not np.isfinite(relative).all():
            raise ValueError("Original vertices contain nonfinite coordinates")
        screen[start:stop, 0] = width * (.5 + relative @ right / scale)
        screen[start:stop, 1] = height * (.5 - relative @ up / scale)
        depth[start:stop] = -(relative @ outward)
    return OrthographicProjection(screen, depth, outward, int(width), int(height), clip)


def build_triangle_bounds(faces, projected, *, chunk_size=131072):
    """Build a reusable conservative 2D bounding box for every original face."""
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("Original faces must have integer shape [F,3]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    lower = np.empty((len(faces), 2), dtype=np.float32)
    upper = np.empty_like(lower)
    for start in range(0, len(faces), chunk_size):
        stop = min(start + chunk_size, len(faces))
        face = faces[start:stop]
        if len(face) and (face.min() < 0 or face.max() >= len(projected.depth)):
            raise ValueError("Face index outside the original vertex array")
        xy = projected.screen[face]
        lower[start:stop] = np.nextafter(xy.min(axis=1).astype(np.float32), -np.inf)
        upper[start:stop] = np.nextafter(xy.max(axis=1).astype(np.float32), np.inf)
    return TriangleBounds(lower, upper)


def edge_counts(pairs, edge_incidence):
    """Lookup original incidence using sorted ``(uint64 keys, counts, N)``.

    The topology key is min(vertex_a, vertex_b) * original_vertex_count + max.
    A missing key is returned as zero; it is never treated as a boundary edge.
    No dictionary containing every original edge is constructed.
    """
    keys, counts, vertex_count = edge_incidence
    keys, counts = np.asarray(keys), np.asarray(counts)
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("Edge pairs must have integer shape [E,2]")
    if keys.ndim != 1 or counts.shape != keys.shape or vertex_count <= 0:
        raise ValueError("Invalid original edge incidence arrays")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= vertex_count):
        raise ValueError("Edge vertex outside original vertex count")
    query = (pairs.min(axis=1).astype(np.uint64) * np.uint64(vertex_count) +
             pairs.max(axis=1).astype(np.uint64))
    positions = np.searchsorted(keys, query)
    result = np.zeros(len(pairs), dtype=np.uint32)
    in_range = positions < len(keys)
    matching = np.flatnonzero(in_range)
    matching = matching[keys[positions[matching]] == query[matching]]
    result[matching] = counts[positions[matching]]
    return result


def _record(point, sample):
    return {
        "pixel": point.tolist(), "sample_xy": sample.tolist(), "hit": False,
        "empty_ray": True, "face_id": None, "depth": None, "barycentric": None,
        "position": None, "raw_rgb": None, "linear_rgb": None, "raw_features": None,
        "normal": None, "normal_cos_camera": None, "abs_normal_cos_camera": None,
        "candidate_count": 0, "intersection_count": 0, "clipped_intersection_count": 0,
        "projected_degenerate_candidate_count": 0, "nearest_depth_tie_face_ids": [],
        "nearest_depth_tie_count": 0, "face_edge_incidence_opposite_vertex": None,
        "nearest_edge_vertices": None, "nearest_edge_incidence": None,
        "nearest_edge_distance_px": None, "ambiguities": [],
    }


def sample_rays(vertices, faces, projected, pixels, features=None, *, pixel_centers=True,
                bounds=None, chunk_size=131072, edge_incidence=None,
                barycentric_tolerance=1e-9, depth_tolerance=1e-9, max_queries=256):
    """Trace a few opaque center rays against *all* original triangles.

    ``pixels`` are integer PNG column,row pairs by default. For precise edge
    midpoints or other explicit subpixel coordinates, set pixel_centers=False.
    Orthographic screen barycentrics are also world-space barycentrics. Both
    windings intersect; the reported normal cosine retains original winding.

    ``features`` must be the verified full-row original [N,C>=3] field. Raw RGB
    is barycentric interpolation of columns 0:3. ``linear_rgb`` instead first
    clamps and raises each vertex RGB to 2.2, then interpolates, matching the
    authenticated renderer's per-vertex native gamma22 conversion. Neither is
    a prediction of the shaded, exposed PNG color.

    A miss establishes no accepted opaque triangle on this exact center ray.
    Degenerate projected candidates and clip exclusions remain explicit; pixel
    antialiasing, transparency, and illumination are outside this conclusion.
    A hit on a triangle edge or at tied depth remains explicitly ambiguous.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("pixels must be finite [Q,2] column,row pairs")
    if len(points) > max_queries or chunk_size < 1:
        raise ValueError("Query or chunk budget exceeded")
    if barycentric_tolerance < 0 or depth_tolerance < 0:
        raise ValueError("Tolerances must be nonnegative")
    if len(vertices) != len(projected.depth):
        raise ValueError("Projection must preserve all original vertex rows")
    if pixel_centers and (not np.equal(points, np.floor(points)).all() or
                          (points < 0).any() or
                          (points[:, 0] >= projected.width).any() or
                          (points[:, 1] >= projected.height).any()):
        raise ValueError("Default pixel queries must be integer indices within the PNG")
    samples = points + .5 if pixel_centers else points.copy()
    if features is not None:
        features = np.asarray(features)
        if features.ndim != 2 or len(features) != len(vertices) or features.shape[1] < 3:
            raise ValueError("features must be verified full-row [N,C>=3] values")
    if bounds is None:
        bounds = build_triangle_bounds(faces, projected, chunk_size=chunk_size)
    if bounds.minimum.shape != (len(faces), 2) or bounds.maximum.shape != bounds.minimum.shape:
        raise ValueError("Bounds must preserve all original face rows")
    records = []
    for point, sample in zip(points, samples):
        record = _record(point, sample)
        candidate = np.flatnonzero((bounds.minimum[:, 0] <= sample[0] + 1e-8) &
                                   (bounds.maximum[:, 0] >= sample[0] - 1e-8) &
                                   (bounds.minimum[:, 1] <= sample[1] + 1e-8) &
                                   (bounds.maximum[:, 1] >= sample[1] - 1e-8))
        record["candidate_count"] = int(len(candidate))
        hit_ids, hit_depths, hit_bary = [], [], []
        for start in range(0, len(candidate), chunk_size):
            ids = candidate[start:start + chunk_size]
            tri = projected.screen[faces[ids]]
            x0, y0 = tri[:, 0, 0], tri[:, 0, 1]
            x1, y1 = tri[:, 1, 0], tri[:, 1, 1]
            x2, y2 = tri[:, 2, 0], tri[:, 2, 1]
            product_a, product_b = (y1 - y2) * (x0 - x2), (x2 - x1) * (y0 - y2)
            denominator = product_a + product_b
            threshold = 32 * np.finfo(np.float64).eps * np.maximum(
                np.abs(product_a) + np.abs(product_b), np.finfo(np.float64).tiny)
            usable = np.abs(denominator) > threshold
            record["projected_degenerate_candidate_count"] += int((~usable).sum())
            if not usable.any():
                continue
            ids = ids[usable]
            x0, y0, x1, y1, x2, y2, denominator = (
                a[usable] for a in (x0, y0, x1, y1, x2, y2, denominator))
            b0 = ((y1 - y2) * (sample[0] - x2) + (x2 - x1) * (sample[1] - y2)) / denominator
            b1 = ((y2 - y0) * (sample[0] - x2) + (x0 - x2) * (sample[1] - y2)) / denominator
            bary = np.column_stack((b0, b1, 1. - b0 - b1))
            inside = (bary >= -barycentric_tolerance).all(axis=1)
            ids, bary = ids[inside], bary[inside]
            depth = np.einsum("ij,ij->i", bary, projected.depth[faces[ids]])
            clipped = (depth < projected.clip[0]) | (depth > projected.clip[1])
            record["clipped_intersection_count"] += int(clipped.sum())
            if (~clipped).any():
                hit_ids.append(ids[~clipped])
                hit_depths.append(depth[~clipped])
                hit_bary.append(bary[~clipped])
        if record["projected_degenerate_candidate_count"]:
            record["ambiguities"].append("projected_degenerate_candidates")
        if record["clipped_intersection_count"]:
            record["ambiguities"].append("intersections_outside_clip")
        if not hit_ids:
            records.append(record)
            continue
        hit_ids, hit_depths, hit_bary = (np.concatenate(a) for a in (hit_ids, hit_depths, hit_bary))
        winner = int(np.argmin(hit_depths))  # IDs are ascending; exact ties choose the first original face.
        face_id, depth, bary = int(hit_ids[winner]), float(hit_depths[winner]), hit_bary[winner]
        tied = hit_ids[np.abs(hit_depths - depth) <= depth_tolerance]
        face = faces[face_id]
        world_tri = np.asarray(vertices[face], dtype=np.float64)
        normal = np.cross(world_tri[1] - world_tri[0], world_tri[2] - world_tri[0])
        normal /= np.linalg.norm(normal)
        cosine = float(np.dot(normal, projected.outward))
        record.update(hit=True, empty_ray=False, face_id=face_id, depth=depth,
                      barycentric=bary.tolist(), position=(bary @ world_tri).tolist(),
                      normal=normal.tolist(), normal_cos_camera=cosine,
                      abs_normal_cos_camera=abs(cosine), intersection_count=int(len(hit_ids)),
                      nearest_depth_tie_face_ids=tied[:16].tolist(), nearest_depth_tie_count=int(len(tied)))
        if len(tied) > 1:
            record["ambiguities"].append("coincident_nearest_hits")
        if features is not None:
            values = np.asarray(features[face], dtype=np.float64)
            record["raw_features"] = (bary @ values).tolist()
            record["raw_rgb"] = (bary @ values[:, :3]).tolist()
            vertex_linear = np.clip(features[face, :3], 0., 1.).astype(np.float32) ** np.float32(2.2)
            record["linear_rgb"] = (bary @ vertex_linear.astype(np.float64)).tolist()
        tri = projected.screen[face]
        pairs = face[np.array([[1, 2], [2, 0], [0, 1]])]
        edge_a, edge_b = tri[1] - tri[0], tri[2] - tri[0]
        twice_area = abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        lengths = np.linalg.norm(tri[[1, 2, 0]] - tri[[2, 0, 1]], axis=1)
        distance = np.abs(bary) * twice_area / lengths
        nearest_edge = int(np.argmin(distance))
        record["nearest_edge_vertices"] = sorted(int(x) for x in pairs[nearest_edge])
        record["nearest_edge_distance_px"] = float(distance[nearest_edge])
        if np.min(np.abs(bary)) <= barycentric_tolerance:
            record["ambiguities"].append("on_triangle_edge")
        if edge_incidence is not None:
            incidence = edge_counts(pairs, edge_incidence)
            record["face_edge_incidence_opposite_vertex"] = incidence.tolist()
            record["nearest_edge_incidence"] = int(incidence[nearest_edge])
            if (incidence == 0).any():
                record["ambiguities"].append("missing_original_edge_incidence")
        records.append(record)
    return records


def visible_edge_samples(vertices, faces, projected, edge_pairs, incidence, window, *,
                         max_edges=16, bounds=None, depth_tolerance=1e-7):
    """Depth-test a small spatially spread pool of original topology edges.

    Input edges may be boundary or nonmanifold; only counts 1 or >2 qualify.
    Pick longer edges in a 4x4 screen grid inside [left,top,right,bottom), with
    at most max_edges*3 midpoint rays. Return both visible and occluded probes
    so a caller can audit exclusions. The caller should draw only ``visible``
    rows; a midpoint proves visibility at that point, not along the full edge.
    Thus an honest overlay uses a marker or a very short segment at the probe.
    """
    pairs, incidence = np.asarray(edge_pairs), np.asarray(incidence)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or incidence.shape != (len(pairs),):
        raise ValueError("Supply original edge pairs [E,2] and their incidence counts")
    if max_edges < 1 or max_edges * 3 > 256 or depth_tolerance < 0:
        raise ValueError("Invalid bounded edge sampling budget")
    left, top, right, bottom = (float(v) for v in window)
    if not left < right or not top < bottom:
        raise ValueError("Invalid screen window")
    xy = projected.screen[pairs]
    midpoint = xy.mean(axis=1)
    lengths = np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1)
    valid = ((incidence == 1) | (incidence > 2)) & (lengths > 1e-8)
    valid &= ((midpoint[:, 0] >= left) & (midpoint[:, 0] < right) &
              (midpoint[:, 1] >= top) & (midpoint[:, 1] < bottom))
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    cells = (np.minimum(3, ((midpoint[ids, 0] - left) * 4 / (right - left)).astype(int)) +
             4 * np.minimum(3, ((midpoint[ids, 1] - top) * 4 / (bottom - top)).astype(int)))
    # Round robin among screen cells, longest first within each cell.
    buckets = [list(ids[cells == c][np.lexsort((ids[cells == c], -lengths[ids[cells == c]]))])
               for c in range(16)]
    selected = []
    for rank in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if rank < len(bucket):
                selected.append(bucket[rank])
                if len(selected) == max_edges * 3:
                    break
        if len(selected) == max_edges * 3:
            break
    selected = np.asarray(selected, dtype=np.int64)
    rays = sample_rays(vertices, faces, projected, midpoint[selected], pixel_centers=False,
                       bounds=bounds)
    result = []
    visible_count = 0
    for edge_id, ray in zip(selected, rays):
        edge_depth = float(projected.depth[pairs[edge_id]].mean())
        occlusion_depth = None if not ray["hit"] else edge_depth - ray["depth"]
        visible = bool(ray["hit"] and abs(occlusion_depth) <= depth_tolerance)
        if visible and visible_count >= max_edges:
            continue
        visible_count += int(visible)
        result.append({"edge_input_id": int(edge_id), "vertices": pairs[edge_id].tolist(),
                       "incidence": int(incidence[edge_id]), "screen_endpoints": xy[edge_id].tolist(),
                       "midpoint_xy": midpoint[edge_id].tolist(), "edge_depth": edge_depth,
                       "visible": visible, "occlusion_depth": occlusion_depth,
                       "ray_face_id": ray["face_id"], "ray_depth": ray["depth"],
                       "ray_ambiguities": ray["ambiguities"]})
    return result
