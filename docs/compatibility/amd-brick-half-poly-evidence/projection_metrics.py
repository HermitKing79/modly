"""Bounded NumPy pixel-centre geometry metrics; no I/O or rendering runtime.

The caller authenticates cameras/arrays and saves returned masks/depths under its
owned private paths. Background depth is zero. Both windings are opaque; exact
depth ties choose the lowest original face-row index. No antialiasing is claimed.
"""
import sys
sys.dont_write_bytecode = True
import math
import numpy as np

MAX_EVENTS = 16777216
MAX_CHUNK_EVENTS = 262144


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _vertices(value):
    _need(isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3
          and value.dtype.kind == 'f' and value.dtype.itemsize in (4, 8)
          and 0 < len(value) <= 2000000, 'Bounded floating XYZ rows required')


def project(vertices, camera, size=640):
    """Use the authenticated square ORTHO basis/frame without reframing.

    camera requires camera_type, camera_location, camera_right, camera_up,
    axis_direction, ortho_scale and clip. Production uses size=640; smaller
    square fixture grids are supported within the same arithmetic contract.
    Any nonfinite point or clipping rejects the whole measurement.
    """
    _vertices(vertices)
    _need(type(size) is int and 0 < size <= 640, 'Square integer dimension in 1..640 required')
    _need(isinstance(camera, dict) and set(('camera_type', 'camera_location', 'camera_right', 'camera_up',
          'axis_direction', 'ortho_scale', 'clip')) <= set(camera)
          and camera['camera_type'] == 'ORTHO', 'Explicit authenticated ORTHO camera required')
    location, right, up, axis = [np.asarray(camera[key], dtype=np.float64)
                                for key in ('camera_location', 'camera_right', 'camera_up', 'axis_direction')]
    _need(all(value.shape == (3,) and np.isfinite(value).all() for value in (location, right, up, axis)),
          'Finite camera three-vectors required')
    outward = np.cross(right, up)
    axis_length = float(np.linalg.norm(axis))
    _need(axis_length > 0 and np.isclose(np.linalg.norm(right), 1, rtol=0, atol=1e-7)
          and np.isclose(np.linalg.norm(up), 1, rtol=0, atol=1e-7)
          and np.isclose(np.dot(right, up), 0, rtol=0, atol=1e-7)
          and np.allclose(outward, axis / axis_length, rtol=0, atol=1e-7),
          'Orthonormal camera basis must agree with the outward direction')
    scale = camera['ortho_scale']
    clip = np.asarray(camera['clip'], dtype=np.float64)
    _need(type(scale) in (int, float) and math.isfinite(scale) and scale > 0
          and clip.shape == (2,) and np.isfinite(clip).all() and 0 < clip[0] < clip[1],
          'Finite positive orthographic frame and clip distances required')
    screen = np.empty((len(vertices), 2), dtype=np.float64)
    depth = np.empty(len(vertices), dtype=np.float64)
    for first in range(0, len(vertices), 32768):
        last = min(first + 32768, len(vertices))
        block = vertices[first:last].astype(np.float64)
        _need(np.isfinite(block).all(), 'Nonfinite source XYZ rejected')
        relative = block - location
        with np.errstate(over='ignore', invalid='ignore'):
            screen[first:last, 0] = size * (.5 + relative @ right / scale)
            screen[first:last, 1] = size * (.5 - relative @ up / scale)
            depth[first:last] = -(relative @ outward)
        xy, z = screen[first:last], depth[first:last]
        _need(np.isfinite(xy).all() and np.isfinite(z).all() and (xy >= 0).all() and (xy <= size).all()
              and (z >= clip[0]).all() and (z <= clip[1]).all(), 'Projection clipping or nonfinite coordinates rejected')
    return {'screen': screen, 'depth': depth, 'size': size, 'clip': clip.tolist()}


def raster(vertices, faces, projected, foliage_faces=None, *, max_events=MAX_EVENTS,
           chunk_events=MAX_CHUNK_EVENTS):
    """Raster all triangles using only their pixel-centre bounding-box events.

    Full work is pre-counted before rasterization. At most 262144 events are
    expanded together. 3D zero-area triangles reject; valid 3D triangles with
    exactly zero projected area are counted as edge-on and cover no centres.
    projected must come from project for these same vertex rows and camera.
    """
    _vertices(vertices)
    _need(type(max_events) is int and 1 <= max_events <= MAX_EVENTS
          and type(chunk_events) is int and 1 <= chunk_events <= MAX_CHUNK_EVENTS,
          'Fixed event/chunk work budgets exceeded')
    _need(isinstance(faces, np.ndarray) and faces.dtype.kind in 'iu' and faces.dtype.itemsize in (4, 8)
          and faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= 4173584,
          'Bounded integer triangle rows required')
    size, screen, vertex_depth = projected['size'], projected['screen'], projected['depth']
    clip = projected['clip']
    _need(type(size) is int and 0 < size <= 640 and isinstance(screen, np.ndarray)
          and screen.dtype == np.float64 and screen.shape == (len(vertices), 2)
          and isinstance(vertex_depth, np.ndarray) and vertex_depth.dtype == np.float64
          and vertex_depth.shape == (len(vertices),) and len(clip) == 2
          and all(type(v) in (int, float) and math.isfinite(v) for v in clip) and 0 < clip[0] < clip[1],
          'Exact bounded projection schema required')
    for first in range(0, len(vertices), 32768):
        xy, z = screen[first:first + 32768], vertex_depth[first:first + 32768]
        _need(np.isfinite(xy).all() and np.isfinite(z).all() and (xy >= 0).all() and (xy <= size).all()
              and (z >= clip[0]).all() and (z <= clip[1]).all(), 'Clipped/nonfinite projected rows rejected')
    foliage_flags = None
    if foliage_faces is not None:
        _need(isinstance(foliage_faces, np.ndarray) and foliage_faces.dtype.kind in 'iu'
              and foliage_faces.ndim == 1 and 0 < len(foliage_faces) <= 4531
              and (foliage_faces >= 0).all() and (foliage_faces < len(faces)).all()
              and len(np.unique(foliage_faces)) == len(foliage_faces), 'Unique original foliage face IDs required')
        foliage_flags = np.zeros(len(faces), dtype=np.bool_)
        foliage_flags[foliage_faces] = True
    lower = np.empty((len(faces), 2), dtype=np.int32)
    upper = np.empty_like(lower)
    counts = np.empty(len(faces), dtype=np.int64)
    total, edge_on = 0, 0
    for first in range(0, len(faces), 32768):
        last = min(first + 32768, len(faces))
        indices = faces[first:last]
        _need((indices >= 0).all() and (indices < len(vertices)).all(), 'Triangle index outside original vertex rows')
        xyz = vertices[indices].astype(np.float64)
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        _need(np.isfinite(xyz).all() and np.isfinite(cross).all() and np.any(cross != 0, axis=1).all(),
              'Nonfinite or zero-area 3D triangle rejected')
        xy = screen[indices]
        determinant = ((xy[:, 1, 1] - xy[:, 2, 1]) * (xy[:, 0, 0] - xy[:, 2, 0])
                       + (xy[:, 2, 0] - xy[:, 1, 0]) * (xy[:, 0, 1] - xy[:, 2, 1]))
        flat = determinant == 0
        edge_on += int(np.count_nonzero(flat))
        lower[first:last] = np.ceil(xy.min(axis=1) - .5).astype(np.int32)
        upper[first:last] = np.floor(xy.max(axis=1) - .5).astype(np.int32)
        extent = np.maximum(upper[first:last].astype(np.int64) - lower[first:last] + 1, 0)
        count = extent[:, 0] * extent[:, 1]
        count[flat] = 0
        counts[first:last] = count
        total += int(count.sum(dtype=np.int64))
        _need(total <= max_events, 'Pixel-triangle event work budget exceeded before rasterization')
    cumulative = np.cumsum(counts, dtype=np.int64)
    depth = np.full(size * size, np.inf, dtype=np.float64)
    sentinel = np.iinfo(np.int32).max
    front = np.full(size * size, sentinel, dtype=np.int32)
    for first in range(0, total, chunk_events):
        last = min(first + chunk_events, total)
        a = int(np.searchsorted(cumulative, first, side='right'))
        b = int(np.searchsorted(cumulative, last - 1, side='right')) + 1
        local_ids = np.arange(a, b, dtype=np.int64)
        bases = cumulative[a:b] - counts[a:b]
        repetitions = np.minimum(cumulative[a:b], last) - np.maximum(bases, first)
        triangle_ids = np.repeat(local_ids, repetitions)
        offset = np.arange(first, last, dtype=np.int64) - (cumulative[triangle_ids] - counts[triangle_ids])
        width = upper[triangle_ids, 0].astype(np.int64) - lower[triangle_ids, 0] + 1
        px = lower[triangle_ids, 0] + offset % width
        py = lower[triangle_ids, 1] + offset // width
        xy = screen[faces[triangle_ids]]
        x0, y0 = xy[:, 0, 0], xy[:, 0, 1]
        x1, y1 = xy[:, 1, 0], xy[:, 1, 1]
        x2, y2 = xy[:, 2, 0], xy[:, 2, 1]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        _need((denominator != 0).all(), 'Unexpected projected edge-on expansion')
        b0 = ((y1 - y2) * (px + .5 - x2) + (x2 - x1) * (py + .5 - y2)) / denominator
        b1 = ((y2 - y0) * (px + .5 - x2) + (x0 - x2) * (py + .5 - y2)) / denominator
        b2 = 1 - b0 - b1
        inside = ((b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
                  & (b0 <= 1 + 1e-9) & (b1 <= 1 + 1e-9) & (b2 <= 1 + 1e-9))
        if not inside.any():
            continue
        tri = triangle_ids[inside]
        z = vertex_depth[faces[tri]]
        z = b0[inside] * z[:, 0] + b1[inside] * z[:, 1] + b2[inside] * z[:, 2]
        _need(np.isfinite(z).all() and (z >= clip[0]).all() and (z <= clip[1]).all(), 'Interpolated depth clipping rejected')
        pixel = py[inside] * size + px[inside]
        before = depth.copy()
        np.minimum.at(depth, pixel, z)
        front[depth < before] = sentinel
        nearest = z == depth[pixel]
        np.minimum.at(front, pixel[nearest], tri[nearest].astype(np.int32))
    mask = np.isfinite(depth)
    _need((front[mask] >= 0).all() and (front[mask] < len(faces)).all(), 'Incomplete front-face raster result')
    visible_foliage = np.zeros(size * size, dtype=np.bool_)
    if foliage_flags is not None:
        visible_foliage[mask] = foliage_flags[front[mask]]
    depth[~mask] = 0
    front[~mask] = -1
    return {'size': size, 'mask': mask.reshape(size, size), 'depth': depth.reshape(size, size),
            'front_face_ids': front.reshape(size, size), 'visible_foliage': visible_foliage.reshape(size, size),
            'triangle_count': len(faces), 'projected_edge_on_triangles': edge_on,
            'pixel_triangle_events': total, 'event_budget': max_events, 'event_chunk_limit': chunk_events,
            'pixel_center_contract': 'Opaque two-sided centres; barycentric edge tolerance1e-9; exact depth ties lowest face row; no antialiasing'}


def compare(source, candidate, diagonal):
    """Apply the frozen per-view silhouette and depth-aware foliage gates."""
    _need(type(diagonal) in (int, float) and math.isfinite(diagonal) and diagonal > 0,
          'Positive original AABB diagonal required')
    size = source['size']
    _need(type(size) is int and 0 < size <= 640 and candidate['size'] == size, 'Matched bounded square views required')
    for row in (source, candidate):
        mask, depth, leaf = row['mask'], row['depth'], row['visible_foliage']
        _need(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == (size, size)
              and isinstance(depth, np.ndarray) and depth.dtype == np.float64 and depth.shape == mask.shape
              and isinstance(leaf, np.ndarray) and leaf.dtype == np.bool_ and leaf.shape == mask.shape
              and np.isfinite(depth).all() and (depth[mask] > 0).all() and (depth[~mask] == 0).all()
              and not (leaf & ~mask).any(), 'Finite zero-background depth and matching silhouette/foliage masks required')
    a, b = source['mask'], candidate['mask']
    source_pixels, candidate_pixels = int(a.sum()), int(b.sum())
    _need(source_pixels > 0, 'Empty original view cannot establish the required silhouette')
    overlap, union = int((a & b).sum()), int((a | b).sum())
    iou = overlap / union
    leaf = source['visible_foliage']
    required = int(leaf.sum())
    tolerance = .005 * diagonal
    retained = int((leaf & b & (np.abs(candidate['depth'] - source['depth']) <= tolerance)).sum())
    retention = retained / required if required else None
    return {'source_pixels': source_pixels, 'candidate_pixels': candidate_pixels,
            'silhouette_IoU': iou, 'silhouette_IoU_minimum': .98,
            'source_lost_fraction': int((a & ~b).sum()) / source_pixels,
            'candidate_gained_fraction': int((b & ~a).sum()) / source_pixels,
            'foliage_informative': bool(required), 'foliage_required_pixels': required,
            'foliage_retained_pixels': retained, 'foliage_visible_pixel_retention': retention,
            'foliage_retention_minimum': .99, 'foliage_depth_tolerance': tolerance,
            'pass': bool(iou >= .98 and (retention is None or retention >= .99)),
            'scope': 'Frozen orthographic pixel centres only; no antialiasing, continuous silhouette or shaded-image equivalence claim'}
