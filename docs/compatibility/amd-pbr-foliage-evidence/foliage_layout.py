"""One bounded CPU chart candidate; no asset I/O, search, baking, or GPU work.

Axes are deterministic argmax(abs(face normal)); UV-only reflections preserve
the exact original triangle corner positions, winding, order and multiplicity.
GO certifies this candidate. BLOCKED never claims an optimal packing theorem.
"""
import hashlib
import math
from pathlib import Path
import sys
import types
import numpy as np


def _pinned(name, relative, digest):
    path = Path(__file__).resolve().parent.parent / relative
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('Pinned CPU primitive changed: ' + str(path))
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


seam_uv = _pinned('_foliage_seam_uv', 'amd-brick-seam-uv-bake-evidence/uv_layout.py',
                  'faf43221c182ea0a473423bfdfb1ff90a8fd8b17650b6cbab9d9a325ca5dfaa7')
footprint = _pinned('_foliage_footprint', 'amd-brick-mip-safe-atlas-evidence/footprint.py',
                    'ddc49d066465f07a8e561bd85194e9d2768f149e98cc054f4311af3a5a40bdd1')
LayoutError = seam_uv.LayoutError
DENSITY = 4096 * math.sqrt(3) * (1 + 1e-6)
SIZE = 1024


def _verify(uv, out_faces, xyz, island_ids, rects):
    tri = uv[out_faces].astype(np.float64)
    areas = seam_uv._twice_area(tri) * SIZE**2 / 2
    if (areas <= 0).any():
        raise LayoutError('Nonpositive UV triangle area')
    edge, other = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    lengths = np.linalg.norm(edge, axis=1)
    along = edge / lengths[:, None]
    normal = np.cross(edge, other)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    across = np.cross(normal, along)
    u = (tri[:, 1] - tri[:, 0]) * SIZE / lengths[:, None]
    v = ((tri[:, 2] - tri[:, 0]) * SIZE - u * np.sum(other * along, axis=1)[:, None])
    v /= np.sum(other * across, axis=1)[:, None]
    density = np.linalg.svd(np.stack((u, v), axis=-1), compute_uv=False)[:, -1]
    if not np.isfinite(density).all() or (density < 4096).any():
        raise LayoutError('Actual 3D tangent density is below 4096')
    stats = {}
    if seam_uv.first_overlap(uv * SIZE, out_faces, area_tolerance=1e-12, stats=stats):
        raise LayoutError('Analytic positive-area UV overlap')
    support = footprint.validate_isolation(tri, island_ids, rects)
    ids, bary = seam_uv._rasterize(uv, out_faces, SIZE)
    counts = np.bincount(ids[ids >= 0], minlength=len(xyz))
    if (counts < 1).any():
        raise LayoutError('Every original face requires a covered texel center')
    return ids, bary, counts, dict(minimum_3d_tangent_density=float(density.min()),
        minimum_triangle_area_texels=float(areas.min()), covered_texels=int((ids >= 0).sum()),
        minimum_face_covered_texels=int(counts.min()), analytic_overlap_stats=stats, support=support)


def build_layout(vertices, faces, source_face_ids, source_vertex_ids):
    """Return {classification, report, [layout, portable_uv_f32]} for <=6000 faces.

    Invalid inputs raise LayoutError. Valid but unsuitable candidates return
    BLOCKED. Each axis group reuses pinned greedy edge-connected collision seams;
    one descending-height/width first-fit shelf pass packs mip-aligned rectangles.
    Content minima are exactly 32 pixels beyond rectangle minima. The prescribed
    rectangle extent encloses every positive filter ancestor through LOD 5.
    """
    if source_vertex_ids is None:
        raise LayoutError('Complete source vertex IDs required')
    sf, sv = seam_uv._inputs(vertices, faces, source_face_ids, source_vertex_ids,
                            SIZE, 32, 4096, 1, 6000, 256, 250000)
    xyz = vertices[faces].astype(np.float64)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    magnitudes = np.linalg.norm(normals, axis=1)
    if (magnitudes == 0).any():
        raise LayoutError('Degenerate original triangle')
    axes = abs(normals).argmax(1)
    local, face_islands = np.empty((len(faces), 3, 2), np.float64), np.empty(len(faces), np.int32)
    islands, tests = [], 0
    report = dict(selected_original_face_count=len(faces), texture_size=[SIZE, SIZE],
        projected_density_texels_per_world_unit=DENSITY, density_search_steps=0,
        density_floor_texels_per_world_unit=4096, face_projection_axes=axes.tolist(),
        axis_face_counts=np.bincount(axes, minlength=3).tolist(), maximum_face_count=6000,
        maximum_island_count=256, maximum_clip_tests_per_phase=250000,
        maximum_raster_bbox_texels_per_phase=seam_uv.MAX_RASTER_TEXELS,
        packing='single descending-height/width first-fit shelf pass; no rotations',
        packing_optimality_claimed=False, UV_overlap_area_tolerance_square_texels=1e-12,
        minimum_dominant_normal_cosine=float((abs(normals).max(1) / magnitudes).min()))

    def blocked(reason):
        report['reason'] = reason
        return dict(classification='BLOCKED', report=report)

    try:
        for axis in range(3):
            indices = np.flatnonzero(axes == axis)
            if not len(indices):
                continue
            projection = xyz[indices][:, :, [a for a in range(3) if a != axis]]
            charts, chart_ids, signs, tested = seam_uv._islands(
                projection, faces[indices], 6000, 250000 - tests)
            local[indices] = projection
            local[indices, :, 0] *= signs[:, None]
            face_islands[indices] = chart_ids + len(islands)
            islands.extend(indices[members] for members in charts)
            tests += tested
    except LayoutError as exc:
        return blocked(str(exc))
    bounds = np.array([[local[members].min((0, 1)), local[members].max((0, 1))]
                       for members in islands])
    spans = (bounds[:, 1] - bounds[:, 0]) * DENSITY
    dims = 32 * (np.ceil(spans / 32 + .5 + 1e-5) + 1)
    report.update(island_count=len(islands), island_construction_clip_tests=tests,
        island_projected_bounds_world=bounds.tolist(), island_content_spans_texels=spans.tolist(),
        island_rectangle_dimensions_texels=dims.tolist(), rectangle_area_texels=float(np.prod(dims, 1).sum()),
        atlas_area_texels=SIZE**2, triangle_area_world_sum=float(magnitudes.sum() / 2),
        projected_triangle_area_texels_sum=float((abs(normals).max(1) * DENSITY**2).sum() / 2),
        rectangle_dimension_exceeds_atlas=bool((dims > SIZE).any()),
        area_lower_bound_exceeds_atlas=bool(np.prod(dims, 1).sum() > SIZE**2))
    if len(islands) > 256 or (dims > SIZE).any() or report['area_lower_bound_exceeds_atlas']:
        return blocked('Bounded island count or prescribed rectangle capacity exceeded')
    dims = dims.astype(np.int32)
    rects, shelves, bottom = np.empty((len(islands), 4), np.int32), [], 0
    for i in sorted(range(len(islands)), key=lambda j: (-int(dims[j, 1]), -int(dims[j, 0]), j)):
        width, height = map(int, dims[i])
        shelf = next((s for s in shelves if height <= s[2] and s[0] + width <= SIZE), None)
        if shelf is None:
            if bottom + height > SIZE:
                return blocked('This fixed shelf candidate does not fit; optimal packing is unproved')
            shelf = [0, bottom, height]
            shelves.append(shelf)
            bottom += height
        x, y = shelf[:2]
        rects[i] = [x, y, x + width, y + height]
        shelf[0] += width
    uv = np.empty_like(local)
    for i, members in enumerate(islands):
        uv[members] = ((local[members] - bounds[i, 0]) * DENSITY + rects[i, :2] + 32) / SIZE
    uv = uv.reshape(-1, 2)
    out_faces = np.arange(faces.size, dtype=np.int32).reshape(-1, 3)
    try:
        ids, bary, counts, checks = _verify(uv, out_faces, xyz, face_islands, rects)
    except (LayoutError, ValueError) as exc:
        return blocked(str(exc))
    out = seam_uv.Layout(vertices[faces].reshape(-1, 3).copy(), out_faces, uv,
                         sv[faces].ravel().copy(), sf.copy(), face_islands, rects, counts, ids, bary, report)
    exact = (np.array_equal(out.vertices.view(np.uint32), vertices[faces].reshape(-1, 3).view(np.uint32))
             and np.array_equal(out.faces, np.arange(faces.size).reshape(-1, 3))
             and np.array_equal(out.source_face_ids, sf)
             and np.array_equal(out.source_vertex_ids, sv[faces].ravel()))
    if not exact:
        raise LayoutError('Original corner bits, winding, order, multiplicity or source ID mapping changed')
    report.update(original_triangle_corners_bitwise_preserved=True, exact_source_ID_mapping=True,
                  positive_area_overlap_count=0, minimum_3d_tangent_density_f64=checks.pop('minimum_3d_tangent_density'),
                  f64_checks=checks, island_rectangles=rects.tolist())
    result = dict(classification='GO', report=report, layout=out)
    portable = uv.astype(np.float32)
    try:
        _, _, _, checks32 = _verify(portable, out_faces, xyz, face_islands, rects)
        report.update(portable_f32_status='GO', portable_f32_support=checks32['support'], portable_f32_checks=checks32)
        result['portable_uv_f32'] = portable
    except (LayoutError, ValueError) as exc:
        report.update(portable_f32_status='BLOCKED', portable_f32_reason=str(exc))
    return result
