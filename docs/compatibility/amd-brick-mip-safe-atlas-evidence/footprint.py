"""Exact positive box-mip/bilinear ancestry bounds; no assets or sampling."""
import math
import numpy as np


def axis_support(lo, hi, size=1024, lod_min=0, lod_max=5):
    """Inclusive base-texel support for every UV/LOD in the closed intervals.

    At level L, q=clamp(clamp(u,0,1)*(size>>L)-.5,0,n-1).
    Positive bilinear taps span floor(q_low)..ceil(q_high): an exact integer
    high endpoint does NOT activate its zero-weight next tap. Each positive
    tap inherits its complete dyadic base block. Trilinear positive levels
    are floor(lod_min)..ceil(lod_max); half-LOD adds no other ancestors.
    """
    values = (lo, hi, lod_min, lod_max)
    if (type(size) is not int or not 1 <= size <= 1024 or size & (size - 1)
            or any(isinstance(v, (bool, np.bool_)) or not np.isscalar(v)
                   or not np.isrealobj(v) for v in values)):
        raise ValueError('Real scalar intervals and bounded power-of-two size required')
    try:
        lo, hi, lod_min, lod_max = map(float, values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Finite scalar intervals required') from exc
    if (not all(map(math.isfinite, (lo, hi, lod_min, lod_max))) or lo > hi
            or not 0 <= lod_min <= lod_max <= min(5, size.bit_length() - 1)):
        raise ValueError('Ordered finite UV/available LOD intervals required')
    low, high = size, -1
    for level in range(math.floor(lod_min), math.ceil(lod_max) + 1):
        side, block = size >> level, 1 << level
        first = min(side - 1, max(0., min(1., max(0., lo)) * side - .5))
        last = min(side - 1, max(0., min(1., max(0., hi)) * side - .5))
        low = min(low, math.floor(first) * block)
        high = max(high, (math.ceil(last) + 1) * block - 1)
    return low, high


def validate_isolation(triangle_uv, triangle_islands, rects, size=1024, lod_min=0, lod_max=5):
    """Prove all continuous triangle UV ranges stay inside disjoint own rects.

    Rectangles are integer base-texel half-open [x0,y0,x1,y1]. Triangle axis
    extrema occur at corners. Bounding both axes therefore encloses every
    barycentric surface position, including vertices/edges, without sampling.
    """
    axis_support(0., 1., size, lod_min, lod_max)
    uv, ids, boxes = map(np.asarray, (triangle_uv, triangle_islands, rects))
    if (uv.dtype.kind != 'f' or uv.ndim != 3 or uv.shape[1:] != (3, 2)
            or not 1 <= len(uv) <= 6000 or not np.isfinite(uv).all()
            or (uv < 0).any() or (uv > 1).any()
            or ids.dtype.kind not in 'iu' or ids.shape != (len(uv),)
            or boxes.dtype.kind not in 'iu' or boxes.ndim != 2 or boxes.shape[1] != 4
            or not 1 <= len(boxes) <= 256 or (boxes[:, :2] < 0).any()
            or (boxes[:, 2:] > size).any() or (boxes[:, 2:] <= boxes[:, :2]).any()
            or (ids < 0).any() or (ids >= len(boxes)).any()):
        raise ValueError('Bounded unit triangle UVs, island IDs and integer rectangles required')
    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            if max(box[0], other[0]) < min(box[2], other[2]) and max(box[1], other[1]) < min(box[3], other[3]):
                raise ValueError('Owned padding rectangles overlap')
    bounds = np.tile(np.array([size, size, -1, -1], np.int64), (len(boxes), 1))
    counts = np.zeros(len(boxes), np.int64)
    for index, (triangle, island) in enumerate(zip(uv, ids)):
        x = axis_support(float(triangle[:, 0].min()), float(triangle[:, 0].max()), size, lod_min, lod_max)
        y = axis_support(float(triangle[:, 1].min()), float(triangle[:, 1].max()), size, lod_min, lod_max)
        x0, y0, x1, y1 = map(int, boxes[island])
        if not x0 <= x[0] <= x[1] < x1 or not y0 <= y[0] <= y[1] < y1:
            raise ValueError(f'Positive footprint escapes own rectangle at triangle{index}: x{x}, y{y}')
        bounds[island, :2] = np.minimum(bounds[island, :2], [x[0], y[0]])
        bounds[island, 2:] = np.maximum(bounds[island, 2:], [x[1], y[1]])
        counts[island] += 1
    if (counts == 0).any():
        raise ValueError('Every owned island requires authenticated triangles')
    return dict(classification='GO', size=size, lod_range=[float(lod_min), float(lod_max)],
                triangle_count=len(uv), all_continuous_triangle_UVs_protected=True,
                all_continuous_LODs_protected=True, islands=[dict(id=i, rect=box.tolist(),
                triangles=int(counts[i]), base_texel_bounds_inclusive=bounds[i].tolist(),
                minimum_clearance_texels=int(min(*(bounds[i, :2] - box[:2]), *(box[2:] - 1 - bounds[i, 2:]))))
                for i, box in enumerate(boxes)])
