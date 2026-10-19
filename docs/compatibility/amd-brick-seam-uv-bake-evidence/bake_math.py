"""CPU-only isolated native padding and bounded seam/footprint fixtures."""
import numpy as np


SIZE = 1024
MAX_FACES = 6000
MAX_ISLANDS = 256
MAX_QUERIES = 65536


def isolate_padding(values, raster_ids, face_island_ids, island_rects):
    """Apply the pinned native byte-first Telea padding to each island alone.

    reference stores exact raw covered F32 values; uncovered owned texels use
    the final encoded/scalar padding bytes divided by255. Unowned pixels are0.
    All maps use V-up array rows; this helper does not write images or files.
    """
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32
            or values.shape != (SIZE, SIZE, 6) or not np.isfinite(values).all()):
        raise ValueError('Finite F32[1024,1024,6] sampled values required')
    if (not isinstance(raster_ids, np.ndarray) or raster_ids.dtype != np.int32
            or raster_ids.shape != (SIZE, SIZE)):
        raise ValueError('I32[1024,1024] raster face IDs required')
    if (not isinstance(face_island_ids, np.ndarray) or face_island_ids.dtype != np.int32
            or face_island_ids.ndim != 1 or not 0 < len(face_island_ids) <= MAX_FACES):
        raise ValueError('Bounded I32 face island IDs required')
    if (not isinstance(island_rects, np.ndarray) or island_rects.dtype != np.int32
            or island_rects.ndim != 2 or island_rects.shape[1] != 4
            or not 0 < len(island_rects) <= MAX_ISLANDS
            or (island_rects[:, :2] < 0).any() or (island_rects[:, 2:] > SIZE).any()
            or (island_rects[:, 2:] <= island_rects[:, :2]).any()):
        raise ValueError('Bounded nonempty integer island rectangles required')
    if ((raster_ids < -1).any() or (raster_ids >= len(face_island_ids)).any()):
        raise ValueError('Foreign raster face ID')
    if (face_island_ids.min() < 0 or face_island_ids.max() >= len(island_rects)):
        raise ValueError('Foreign face island ID')
    ownership = np.full((SIZE, SIZE), -1, np.int16)
    for island, (x0, y0, x1, y1) in enumerate(island_rects):
        owned = ownership[y0:y1, x0:x1]
        if (owned >= 0).any():
            raise ValueError('Island padding rectangles overlap')
        owned[:] = island
    covered = raster_ids >= 0
    if not np.array_equal(ownership[covered], face_island_ids[raster_ids[covered]]):
        raise ValueError('Covered face belongs outside its own island rectangle')
    for island, (x0, y0, x1, y1) in enumerate(island_rects):
        if not covered[y0:y1, x0:x1].any():
            raise ValueError('Every island rectangle requires source coverage')
    from common import originals
    old_math = originals()  # Reauthenticates the historical implementation pins.
    base = np.zeros((SIZE, SIZE, 4), np.uint8)
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    original_base, original_mr = np.zeros_like(base), np.zeros_like(mr)
    reference = np.zeros_like(values)
    for island, (x0, y0, x1, y1) in enumerate(island_rects):
        region = np.s_[y0:y1, x0:x1]
        own_mask = covered[region]
        source = values[region]
        padded_base, padded_mr, raw_base, raw_mr = old_math.source_padding(source, own_mask)
        base[region], mr[region] = padded_base, padded_mr
        original_base[region], original_mr[region] = raw_base, raw_mr
        local_reference = np.concatenate((padded_base[:, :, :3], padded_mr[:, :, 2:3],
                                          padded_mr[:, :, 1:2], padded_base[:, :, 3:4]), axis=2)
        local_reference = local_reference.astype(np.float32) / np.float32(255)
        local_reference[own_mask] = source[own_mask]
        reference[region] = local_reference
    if not np.array_equal(reference[covered].view(np.uint32), values[covered].view(np.uint32)):
        raise ValueError('Padding changed raw covered source float bits')
    if (not np.array_equal(base[covered], original_base[covered])
            or not np.array_equal(mr[covered], original_mr[covered])):
        raise ValueError('Padding changed native covered bytes')
    return dict(base_color_rgba=base, metallic_roughness_rgb=mr, reference=reference,
                ownership=ownership, original_base_color_rgba=original_base,
                original_metallic_roughness_rgb=original_mr)


def footprint_owner(uv, expected_island, ownership):
    """Return whether all four clamped bilinear texel centers have one owner.

    Even a zero-weight neighbour must be owned by the expected island. The
    returned mask lets the caller count and reject unsafe quality queries.
    """
    uv = np.asarray(uv)
    expected_island = np.asarray(expected_island)
    if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) > MAX_QUERIES
            or uv.dtype.kind != 'f' or not np.isfinite(uv).all()):
        raise ValueError('Bounded finite floating UV[Q,2] required')
    if (not isinstance(ownership, np.ndarray) or ownership.dtype != np.int16
            or ownership.shape != (SIZE, SIZE) or (ownership < -1).any()
            or (ownership >= MAX_ISLANDS).any()):
        raise ValueError('I16[1024,1024] bounded ownership required')
    if (expected_island.dtype.kind not in 'iu' or expected_island.shape not in ((), (len(uv),))
            or (expected_island < 0).any() or (expected_island >= MAX_ISLANDS).any()):
        raise ValueError('Expected island must be a scalar or one valid ID per UV')
    q = np.clip(uv.astype(np.float64) * SIZE - .5, 0, SIZE - 1)
    p = np.floor(q).astype(np.int32)
    x, y = p.T
    nx, ny = np.minimum(x + 1, SIZE - 1), np.minimum(y + 1, SIZE - 1)
    return ((ownership[y, x] == expected_island) & (ownership[y, nx] == expected_island)
            & (ownership[ny, x] == expected_island) & (ownership[ny, nx] == expected_island))


def seam_queries(corner_source_ids, face_islands, max_queries=MAX_QUERIES):
    """Emit every cross-island source-edge face pair, including nonmanifolds.

    Cohort0 samples canonical low-ID->high-ID edge t=0,.25,.5,.75,1. Cohort1
    blends .005 of the third corner; endpoint t is nudged to .005/.995 so near
    samples have strictly positive barycentrics. pairing indexes cohort0 only.
    Source face order is retained; dictionary traversal follows first encounter.
    """
    ids, islands = np.asarray(corner_source_ids), np.asarray(face_islands)
    if (ids.ndim != 2 or ids.shape[1] != 3 or ids.dtype.kind not in 'iu'
            or not 0 < len(ids) <= MAX_FACES or (ids < 0).any()
            or (ids > np.iinfo(np.int64).max).any()
            or (ids[:, 0] == ids[:, 1]).any() or (ids[:, 1] == ids[:, 2]).any()
            or (ids[:, 2] == ids[:, 0]).any()):
        raise ValueError('Bounded distinct nonnegative integer source corners required')
    if (islands.shape != (len(ids),) or islands.dtype.kind not in 'iu'
            or (islands < 0).any() or (islands >= MAX_ISLANDS).any()):
        raise ValueError('One bounded source island ID per face required')
    if type(max_queries) is not int or not 1 <= max_queries <= MAX_QUERIES:
        raise ValueError('Seam query budget must be within1..65536')
    edges = {}
    for face_id, corners in enumerate(ids):
        for a in range(3):
            b, third = (a + 1) % 3, (a + 2) % 3
            if corners[a] > corners[b]:
                a, b = b, a
            key = (int(corners[a]), int(corners[b]))
            edges.setdefault(key, []).append((face_id, a, b, third))
    pairs = []
    for edge_ids, members in edges.items():
        for i, first in enumerate(members):
            for second in members[i + 1:]:
                if islands[first[0]] != islands[second[0]]:
                    if (len(pairs) + 1) * 20 > max_queries:
                        raise ValueError('Cross-island seam query budget exceeded; no pairs dropped')
                    pairs.append((edge_ids, first, second))
    count = len(pairs) * 20
    query_faces = np.empty(count, np.int32)
    bary = np.zeros((count, 3), np.float64)
    cohort = np.empty(count, np.int8)
    pairing = np.empty((len(pairs) * 5, 2), np.int32)
    edge_source_ids = np.empty((len(pairing), 2), np.int64)
    t_values = np.empty(len(pairing), np.float64)
    cursor = pair_cursor = 0
    for edge_ids, first, second in pairs:
        for group in (0, 1):
            for t in (0., .25, .5, .75, 1.):
                at = t if group == 0 else min(.995, max(.005, t))
                inward = 0. if group == 0 else .005
                start = cursor
                for face_id, a, b, third in (first, second):
                    query_faces[cursor], cohort[cursor] = face_id, group
                    bary[cursor, a] = (1 - inward) * (1 - at)
                    bary[cursor, b] = (1 - inward) * at
                    bary[cursor, third] = inward
                    cursor += 1
                if group == 0:
                    pairing[pair_cursor] = [start, start + 1]
                    edge_source_ids[pair_cursor], t_values[pair_cursor] = edge_ids, t
                    pair_cursor += 1
    return dict(query_faces=query_faces, bary=bary, cohort=cohort, pairing=pairing,
                edge_source_ids=edge_source_ids, t=t_values)
