"""Bounded CPU YZ islands for the unchanged original brick patch.

This is a deterministic seam cut and rectangle packer, not a general unwrap.
No source triangle is edited, removed, reordered, or merged. Insufficient
capacity, projected degeneracy, or missing triangle samples is a hard failure.
Raster arrays use V-up rows and texel-center sampling, as in patch_math.py.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np


MAX_FACES = 6000
MAX_ISLANDS = 256
MAX_CLIP_TESTS = 250000
MAX_RASTER_TEXELS = 64 * 1024 * 1024  # Bounding-box samples per raster phase.
MAX_SIZE = 1024
MIN_DENSITY = 4096.0  # Eight texels per original 1/512 world-space voxel.


class LayoutError(ValueError):
    pass


@dataclass
class Layout:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    source_vertex_ids: np.ndarray
    source_face_ids: np.ndarray
    face_island_ids: np.ndarray
    island_rects: np.ndarray  # [x0,y0,x1,y1], half-open, INCLUDING gutters.
    face_coverage: np.ndarray
    raster_ids: np.ndarray
    barycentrics: np.ndarray
    report: dict


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _twice_area(triangles):
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _intersection(a, b):
    """Same convex half-plane clipping as the old analytic injectivity check."""
    if _cross(b[1] - b[0], b[2] - b[0]) < 0:
        b = b[[0, 2, 1]]
    poly = list(a.copy())
    for k in range(3):
        edge, origin = b[(k + 1) % 3] - b[k], b[k]
        result = []
        for u, z in zip(poly, poly[1:] + poly[:1]):
            du, dz = _cross(edge, u - origin), _cross(edge, z - origin)
            if du >= 0:
                result.append(u)
            if (du >= 0) != (dz >= 0):
                result.append(u + (z - u) * (du / (du - dz)))
        poly = result
        if not poly:
            break
    if len(poly) < 3:
        return 0.0, np.empty((0, 2), np.float64)
    q = np.asarray(poly, np.float64)
    # Translation avoids cancellation for tiny intersections far from zero.
    local = q - q[0]
    area = abs(np.sum(local[:, 0] * np.roll(local[:, 1], -1)
                      - local[:, 1] * np.roll(local[:, 0], -1))) / 2
    return float(area), q


def first_overlap(uv, faces, area_tolerance=1e-12,
                  max_clip_tests=MAX_CLIP_TESTS, stats=None):
    """Return first lexicographic pair, area and polygon, or None.

    Coordinates are used as supplied: normalized-UV input yields normalized
    area. Opposite winding is handled for diagnostics. Contacts have zero area.
    Bounding-box culling is vectorized and clipping has a fixed work ceiling.
    """
    uv, faces = np.asarray(uv), np.asarray(faces)
    if (uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or not 0 < len(faces) <= MAX_FACES
            or faces.min() < 0 or faces.max() >= len(uv)
            or not np.isfinite(area_tolerance) or area_tolerance < 0
            or not 1 <= max_clip_tests <= MAX_CLIP_TESTS):
        raise LayoutError('Invalid bounded overlap input')
    triangles = uv[faces].astype(np.float64)
    low, high = triangles.min(1), triangles.max(1)
    tested = 0
    for i, a in enumerate(triangles):
        candidate = np.flatnonzero((np.minimum(high[i], high[i + 1:])
                                   - np.maximum(low[i], low[i + 1:]) > 0).all(1)) + i + 1
        for j in candidate:
            tested += 1
            if tested > max_clip_tests:
                raise LayoutError('Analytic clipping work bound exceeded')
            area, poly = _intersection(a, triangles[j])
            if area > area_tolerance:
                if stats is not None:
                    stats['analytic_bbox_candidate_pairs'] = tested
                return dict(face_i=int(i), face_j=int(j), intersection_area=area,
                            intersection_polygon=poly.tolist())
    if stats is not None:
        stats['analytic_bbox_candidate_pairs'] = tested
    return None


def _inputs(vertices, faces, source_face_ids, source_vertex_ids, size,
            gutter, min_density, min_face_texels, max_faces, max_islands, max_clip_tests):
    if (not isinstance(vertices, np.ndarray) or vertices.dtype != np.float32
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or not isinstance(faces, np.ndarray) or faces.dtype != np.int32
            or faces.ndim != 2 or faces.shape[1] != 3):
        raise LayoutError('F32 vertices / I32 original triangles required')
    if not isinstance(max_faces, int) or not 1 <= max_faces <= MAX_FACES or not 0 < len(faces) <= max_faces:
        raise LayoutError('Bounded original face count exceeded')
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise LayoutError('Invalid original triangle positions or indices')
    sf = np.asarray(source_face_ids)
    sv = np.arange(len(vertices), dtype=np.int64) if source_vertex_ids is None else np.asarray(source_vertex_ids)
    if sf.shape != (len(faces),) or sf.dtype.kind not in 'iu' or (sf < 0).any():
        raise LayoutError('Complete nonnegative integer source face IDs required')
    if sv.shape != (len(vertices),) or sv.dtype.kind not in 'iu' or (sv < 0).any():
        raise LayoutError('Complete nonnegative integer source vertex IDs required')
    if not isinstance(size, int) or not 16 <= size <= MAX_SIZE:
        raise LayoutError('Atlas dimension must remain bounded by 1024 texels')
    if not isinstance(gutter, int) or not 4 <= gutter < size / 4:
        raise LayoutError('Each island requires at least 4 integer gutter texels per side')
    if not np.isfinite(min_density) or min_density < MIN_DENSITY:
        raise LayoutError('Source-space density floor must be at least 4096 texels/world-unit')
    if not isinstance(min_face_texels, int) or min_face_texels < 1:
        raise LayoutError('Every original face requires at least one covered texel')
    if not isinstance(max_islands, int) or not 1 <= max_islands <= MAX_ISLANDS:
        raise LayoutError('Bounded island count required')
    if not isinstance(max_clip_tests, int) or not 1 <= max_clip_tests <= MAX_CLIP_TESTS:
        raise LayoutError('Bounded clipping work required')
    return sf, sv


def _islands(projected, faces, max_islands, max_clip_tests):
    """Greedy edge-connected islands, split on every positive-area collision."""
    areas = _twice_area(projected)
    if (abs(areas) <= 1e-24).any():
        raise LayoutError('Source triangle has degenerate YZ projection')
    signs = np.where(areas > 0, 1, -1)
    low, high = projected.min(1), projected.max(1)
    edge_islands, edge_faces, face_edges = {}, {}, []
    for i, face in enumerate(faces):
        edges = [tuple(sorted((int(face[k]), int(face[(k + 1) % 3])))) for k in range(3)]
        face_edges.append(edges)
        for edge in edges:
            edge_faces.setdefault(edge, []).append(i)
    # Traverse source-edge connectivity before cutting UVs. Source storage order
    # can place a distant face before its bridge, needlessly fragmenting a chart.
    # Every edge list is visited once, including nonmanifold source edges.
    order, visited, seen_edges = [], set(), set()
    for seed in range(len(faces)):
        if seed in visited:
            continue
        pending = deque([seed])
        visited.add(seed)
        while pending:
            i = pending.popleft()
            order.append(i)
            neighbours = set()
            for edge in face_edges[i]:
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    neighbours.update(edge_faces[edge])
            for j in sorted(neighbours - visited):
                visited.add(j)
                pending.append(j)
    islands, face_islands = [], np.empty(len(faces), np.int32)
    tested = 0
    for i in order:
        edges = face_edges[i]
        adjacent = sorted(set(j for edge in edges for j in edge_islands.get(edge, ())))
        chosen = None
        for island_id in adjacent:
            members = islands[island_id]
            if signs[members[0]] != signs[i]:
                continue
            candidate = np.asarray(members, np.int32)
            candidate = candidate[(np.minimum(high[i], high[candidate])
                                   - np.maximum(low[i], low[candidate]) > 0).all(1)]
            collision = False
            for j in candidate:
                tested += 1
                if tested > max_clip_tests:
                    raise LayoutError('Island clipping work bound exceeded')
                area, _ = _intersection(projected[i], projected[j])
                span = np.maximum(high[i], high[j]) - np.minimum(low[i], low[j])
                tolerance = max(1e-24, float(np.prod(span)) * np.finfo(float).eps * 32)
                if area > tolerance:
                    collision = True
                    break
            if not collision:
                chosen = island_id
                break
        if chosen is None:
            if len(islands) >= max_islands:
                raise LayoutError('Bounded island count exceeded')
            chosen = len(islands)
            islands.append([])
        islands[chosen].append(i)
        face_islands[i] = chosen
        for edge in edges:
            edge_islands.setdefault(edge, set()).add(chosen)
    return islands, face_islands, signs, tested


def _pack(extents, density, size, gutter):
    # Tiny fixed slack protects the requested gutter after F32 UV conversion.
    with np.errstate(over='ignore'):
        scaled = extents * density + 1e-4
    if not np.isfinite(scaled).all() or (scaled > size - 2 * gutter).any():
        return None
    dims = np.ceil(scaled).astype(np.int64) + 2 * gutter
    order = sorted(range(len(dims)), key=lambda i: (-int(dims[i, 1]), -int(dims[i, 0]), i))
    shelves, rects = [], np.empty((len(dims), 4), np.int32)
    bottom = 0
    for i in order:
        width, height = map(int, dims[i])
        shelf = next((s for s in shelves if height <= s[2] and s[0] + width <= size), None)
        if shelf is None:
            if bottom + height > size:
                return None
            shelf = [0, bottom, height]
            shelves.append(shelf)
            bottom += height
        x, y = shelf[:2]
        rects[i] = [x, y, x + width, y + height]
        shelf[0] += width
    return rects


def _rasterize(uv, faces, size):
    ids = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    tested = 0
    for i, face in enumerate(faces):
        tri = uv[face].astype(np.float64) * size
        low = np.maximum(0, np.ceil(tri.min(0) - .5).astype(int))
        high = np.minimum(size - 1, np.floor(tri.max(0) - .5).astype(int))
        if np.any(high < low):
            continue
        tested += int(np.prod(high - low + 1))
        if tested > MAX_RASTER_TEXELS:
            raise LayoutError('Bounded raster work exceeded')
        x, y = np.meshgrid(np.arange(low[0], high[0] + 1) + .5,
                           np.arange(low[1], high[1] + 1) + .5)
        a, b = tri[1] - tri[0], tri[2] - tri[0]
        det = _cross(a, b)
        if det <= 1e-12:
            raise LayoutError('Positive nondegenerate UV triangle required')
        rx, ry = x - tri[0, 0], y - tri[0, 1]
        w1, w2 = (rx * b[1] - ry * b[0]) / det, (a[0] * ry - a[1] * rx) / det
        weights = np.stack((1 - w1 - w2, w1, w2), -1)
        hit = weights.min(-1) >= -1e-10
        yy, xx = np.nonzero(hit)
        yy, xx = yy + low[1], xx + low[0]
        weights = weights[hit]
        old = ids[yy, xx]
        if ((old >= 0) & (weights.min(-1) > 1e-7)
                & (bary[yy, xx].min(-1) > 1e-7)).any():
            raise LayoutError('Interior UV raster overlap')
        empty = old < 0
        ids[yy[empty], xx[empty]] = i
        bary[yy[empty], xx[empty]] = weights[empty]
    return ids, bary


def validate_layout(result, original_vertices, original_faces, max_clip_tests=MAX_CLIP_TESTS,
                    *, source_face_ids=None, source_vertex_ids=None):
    """Independent final geometry, area, overlap, rectangle and raster checks."""
    size, gutter = result.report['texture_size'][0], result.report['min_gutter_texels']
    expected = original_vertices[original_faces].reshape(-1, 3)
    if (result.vertices.dtype != np.float32 or result.vertices.shape != expected.shape
            or not np.array_equal(result.vertices.view(np.uint32), expected.view(np.uint32))
            or result.faces.dtype != np.int32
            or not np.array_equal(result.faces, np.arange(len(expected)).reshape(-1, 3))):
        raise LayoutError('Original triangle corner bits/order/winding/multiplicity changed')
    if (result.source_face_ids.shape != (len(original_faces),)
            or result.source_face_ids.dtype.kind not in 'iu'
            or (result.source_face_ids < 0).any()
            or (source_face_ids is not None
                and not np.array_equal(result.source_face_ids, source_face_ids))):
        raise LayoutError('Original source face ID correspondence changed')
    if (result.source_vertex_ids.shape != (len(expected),)
            or result.source_vertex_ids.dtype.kind not in 'iu'
            or (result.source_vertex_ids < 0).any()
            or (source_vertex_ids is not None and not np.array_equal(
                result.source_vertex_ids, np.asarray(source_vertex_ids)[original_faces].ravel()))):
        raise LayoutError('Original source vertex ID correspondence changed')
    if result.uv.dtype != np.float32 or result.uv.shape != (len(expected), 2) or not np.isfinite(result.uv).all():
        raise LayoutError('Finite F32 UV corners required')
    if ((result.uv < 0) | (result.uv > 1)).any():
        raise LayoutError('UV bounds exceed fixed atlas')
    tri = result.uv[result.faces].astype(np.float64)
    area = _twice_area(tri) * size * size / 2
    if (area <= 0).any():
        raise LayoutError('Nonpositive UV area')
    # Measure the final F32 UV Jacobian in an orthonormal basis of each exact
    # original 3D triangle. Its smaller singular value is the actual least
    # resolved source direction, including UV conversion/packing roundoff.
    xyz = original_vertices[original_faces].astype(np.float64)
    edge, other = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    edge_length = np.linalg.norm(edge, axis=1)
    tangent = edge / edge_length[:, None]
    normal = np.cross(edge, other)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    across = np.cross(normal, tangent)
    along_component = np.sum(other * tangent, axis=1)
    across_component = np.sum(other * across, axis=1)
    uv_along = (tri[:, 1] - tri[:, 0]) * size / edge_length[:, None]
    uv_across = ((tri[:, 2] - tri[:, 0]) * size
                 - uv_along * along_component[:, None]) / across_component[:, None]
    jacobian = np.stack((uv_along, uv_across), axis=-1)
    tangent_density = np.linalg.svd(jacobian, compute_uv=False)[:, -1]
    if (not np.isfinite(tangent_density).all()
            or (tangent_density < result.report['min_density_texels_per_world_unit']).any()):
        raise LayoutError('Actual UV-to-source 3D tangent density floor failed')
    stats = {}
    overlap = first_overlap(result.uv, result.faces, max_clip_tests=max_clip_tests, stats=stats)
    if overlap is not None:
        raise LayoutError('Analytic positive-area UV overlap: ' + repr(overlap))
    rects = result.island_rects
    if (rects.dtype != np.int32 or rects.ndim != 2 or rects.shape[1] != 4
            or (rects[:, :2] < 0).any() or (rects[:, 2:] > size).any()
            or (rects[:, 2:] <= rects[:, :2]).any()
            or result.face_island_ids.shape != (len(original_faces),)
            or result.face_island_ids.min() < 0 or result.face_island_ids.max() >= len(rects)):
        raise LayoutError('Invalid bounded island rectangles')
    content_low, content_high = [], []
    for island_id, rect in enumerate(rects):
        points = tri[result.face_island_ids == island_id].reshape(-1, 2) * size
        if len(points) == 0:
            raise LayoutError('Empty UV island')
        if (points.min(0) < rect[:2] + gutter - 1e-5).any() or (points.max(0) > rect[2:] - gutter + 1e-5).any():
            raise LayoutError('UV island lacks reserved gutter')
        content_low.append(rect[:2] + gutter)
        content_high.append(rect[2:] - gutter)
    separation = None
    for i, rect in enumerate(rects):
        for j in range(i + 1, len(rects)):
            if (np.minimum(rect[2:], rects[j, 2:]) - np.maximum(rect[:2], rects[j, :2]) > 0).all():
                raise LayoutError('UV island outer rectangles overlap')
            gap = np.maximum(np.asarray(content_low[i]) - content_high[j],
                             np.asarray(content_low[j]) - content_high[i])
            distance = int(gap.max())
            separation = distance if separation is None else min(separation, distance)
            if distance < 2 * gutter:
                raise LayoutError('UV island content boxes lack empty gutter separation')
    ids, bary = _rasterize(result.uv, result.faces, size)
    counts = np.bincount(ids[ids >= 0], minlength=len(original_faces))
    if (counts < result.report['min_face_texels_required']).any():
        raise LayoutError('Every original face must have the required covered texel centers')
    if (not np.array_equal(ids, result.raster_ids) or not np.array_equal(bary, result.barycentrics)
            or not np.array_equal(counts, result.face_coverage)):
        raise LayoutError('UV raster provenance differs from revalidation')
    edge_lengths = np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2) * size
    return dict(positive_area_overlap_count=0, analytic_bbox_candidate_pairs=stats['analytic_bbox_candidate_pairs'],
                UV_area_tolerance=1e-12, original_triangle_corners_bitwise_preserved=True,
                triangle_area_texels=area.tolist(), minimum_triangle_area_texels=float(area.min()),
                minimum_3d_tangent_texels_per_world_unit=float(tangent_density.min()),
                triangle_minimum_3d_tangent_texels_per_world_unit=tangent_density.tolist(),
                minimum_triangle_altitude_texels=float(np.min(2 * area / edge_lengths.max(1))),
                face_covered_texels=counts.tolist(), minimum_face_covered_texels=int(counts.min()),
                covered_texels=int((ids >= 0).sum()), outer_rectangles_disjoint=True,
                minimum_island_rectangle_separation_texels=separation)


def layout(vertices, faces, source_face_ids, source_vertex_ids=None, *, size=1024,
           gutter=4, min_density=MIN_DENSITY, min_face_texels=1,
           max_faces=MAX_FACES, max_islands=MAX_ISLANDS, max_clip_tests=MAX_CLIP_TESTS):
    """Cut/pack a bounded original patch, preserving all original corner bits.

    Rectangles include four or more gutter texels on each side. The fixed
    density floor is never relaxed; sixteen deterministic binary-search steps
    increase density only when every rectangle still fits. Validation failures
    stop this candidate rather than retry another layout or discard triangles.
    """
    sf, sv = _inputs(vertices, faces, source_face_ids, source_vertex_ids, size, gutter,
                     min_density, min_face_texels, max_faces, max_islands, max_clip_tests)
    projected = vertices[faces, 1:].astype(np.float64)
    islands, face_islands, signs, tested = _islands(projected, faces, max_islands, max_clip_tests)
    xyz = vertices[faces].astype(np.float64)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    cosines = abs(normals[:, 0]) / np.linalg.norm(normals, axis=1)
    # Projection compresses the most sloped tangent by |normal_x| / |normal|.
    # A shared YZ density must cover that compression to meet a 3D spacing floor.
    projected_floor = float(min_density / cosines.min()) * (1 + 1e-5)
    local = projected.copy()
    local[:, :, 0] *= signs[:, None]  # UV-only reflection; geometry winding remains unchanged.
    bounds = [(local[members].reshape(-1, 2).min(0), local[members].reshape(-1, 2).max(0))
              for members in islands]
    extents = np.asarray([hi - lo for lo, hi in bounds])
    rects = _pack(extents, projected_floor, size, gutter)
    if rects is None:
        raise LayoutError('Fixed 1024 atlas cannot fit the source-space density floor')
    low_density = projected_floor
    high_density = float((size - 2 * gutter) / extents.max())
    for _ in range(16):
        middle = (low_density + high_density) / 2
        candidate = _pack(extents, middle, size, gutter)
        if candidate is None:
            high_density = middle
        else:
            low_density, rects = middle, candidate
    corners = np.empty_like(local)
    for island_id, members in enumerate(islands):
        corners[members] = ((local[members] - bounds[island_id][0]) * low_density
                            + rects[island_id, :2] + gutter) / size
    uv = corners.reshape(-1, 2).astype(np.float32)
    out_faces = np.arange(len(faces) * 3, dtype=np.int32).reshape(-1, 3)
    ids, bary = _rasterize(uv, out_faces, size)
    counts = np.bincount(ids[ids >= 0], minlength=len(faces))
    report = dict(method='edge-connected YZ projection, UV reflection and collision seams; deterministic shelf packing',
                  texture_size=[size, size], min_gutter_texels=gutter,
                  min_density_texels_per_world_unit=float(min_density),
                  texels_per_world_unit=low_density,
                  island_texels_per_world_unit=[low_density] * len(islands),
                  density_metric='uniform YZ-plane texels/world-unit; 3D tangent compression separately bounded',
                  ideal_minimum_3d_tangent_texels_per_world_unit=float((low_density * cosines).min()),
                  minimum_absolute_normal_x_cosine=float(cosines.min()),
                  min_face_texels_required=min_face_texels, selected_original_face_count=len(faces),
                  island_count=len(islands), island_construction_clip_tests=tested,
                  maximum_face_count=max_faces, maximum_island_count=max_islands,
                  maximum_clip_tests_per_phase=max_clip_tests, density_search_steps=16,
                  maximum_raster_bbox_texels_per_phase=MAX_RASTER_TEXELS,
                  face_processing_order='source-edge breadth-first; smallest original local face seed first',
                  island_rectangles='half-open [x0,y0,x1,y1], includes gutter on every side; indexed by face_island_ids',
                  source_vertex_ids='one original vertex ID per duplicated corner; equal positions never merged')
    result = Layout(vertices[faces].reshape(-1, 3).copy(), out_faces, uv,
                    sv[faces].reshape(-1).copy(), sf.copy(), face_islands,
                    rects, counts, ids, bary, report)
    report.update(validate_layout(result, vertices, faces, max_clip_tests=max_clip_tests,
                                  source_face_ids=sf, source_vertex_ids=sv))
    return result
