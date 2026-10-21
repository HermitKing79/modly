"""Pure CPU original-face statistics and a bounded, input-only query plan.

No file access, material sampling, exports, scene construction or geometry
modification occurs here. Region names describe geometric predicates; no
occlusion or complete semantic segmentation is inferred from them.
"""
import sys
sys.dont_write_bytecode = True
import numpy as np

BATCH_ROWS = 32_768
MAX_VERTICES = 2_000_000
MAX_FACES = 4_173_584
MAX_QUERIES = 65_536
DEFAULT_BUDGETS = {
    'global': 1024, 'brick': 781, 'foliage': 512, 'rear_x': 512,
    'backfacing_x': 512, 'boundary': 512, 'nonmanifold': 512,
    'high_variation': 1024,
    **{f'octant_{i}': 512 for i in range(8)},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _interior_barycentrics(uniforms):
    """Guard F64 endpoint rounding, including a possible exact-zero RNG draw."""
    epsilon = np.finfo(np.float64).eps
    uv = np.clip(uniforms, epsilon, 1-epsilon)
    root = np.sqrt(uv[:,:,0])
    result = np.empty((*uv.shape[:2],3), np.float64)
    result[:,:,0] = 1-root
    result[:,:,1] = root*(1-uv[:,:,1])
    result[:,:,2] = root*uv[:,:,1]
    return result


def scan(vertices, faces, encoded):
    """Scan authenticated F32/I32 inputs without modifying or copying them whole.

    The only full edge-occurrence array is UInt64[3*F] (95.53 MiB at
    MAX_FACES). Sorting, unique-edge counting and lookup temporaries, geometry
    statistics and all declared inputs together remain below a conservative
    1.5 GiB allocation budget at the fixed caps. Face-coordinate gathers have
    at most BATCH_ROWS rows. Natural boundary means original global incidence
    one, and nonmanifold means original global incidence greater than two.
    """
    _require(isinstance(vertices, np.ndarray) and vertices.dtype == np.float32
             and vertices.ndim == 2 and vertices.shape[1] == 3
             and 0 < len(vertices) <= MAX_VERTICES, 'Original finite F32[N,3] vertices required')
    _require(isinstance(faces, np.ndarray) and faces.dtype == np.int32
             and faces.ndim == 2 and faces.shape[1] == 3
             and 0 < len(faces) <= MAX_FACES, 'Original I32[F,3] faces required')
    _require(isinstance(encoded, np.ndarray) and encoded.dtype == np.float32
             and encoded.shape == vertices.shape, 'Original encoded F32[N,3] RGB required')
    for start in range(0, len(vertices), BATCH_ROWS):
        _require(np.isfinite(vertices[start:start+BATCH_ROWS]).all(), 'Nonfinite original vertex')
        _require(np.isfinite(encoded[start:start+BATCH_ROWS]).all(), 'Nonfinite original encoded RGB')
    count, vertex_count = len(faces), len(vertices)
    low = vertices.min(axis=0).astype(np.float64)
    high = vertices.max(axis=0).astype(np.float64)
    middle = (low + high) * .5
    areas = np.empty(count, np.float64)
    centers = np.empty((count, 3), np.float32)
    normal_x = np.empty(count, np.float32)
    span = np.empty(count, np.float32)
    octant = np.empty(count, np.uint8)
    keys = np.empty(count * 3, np.uint64)
    for start in range(0, count, BATCH_ROWS):
        end = min(start+BATCH_ROWS, count)
        block = faces[start:end]
        _require(np.all(block >= 0) and np.all(block < vertex_count), 'Original face index outside vertex array')
        positions = vertices[block].astype(np.float64)
        cross = np.cross(positions[:,1]-positions[:,0], positions[:,2]-positions[:,0])
        magnitude = np.linalg.norm(cross, axis=1)
        _require(np.isfinite(magnitude).all() and np.all(magnitude > 0), 'Original zero-area or nonfinite triangle')
        areas[start:end] = magnitude * .5
        centers[start:end] = positions.mean(axis=1).astype(np.float32)
        normal_x[start:end] = (cross[:,0]/magnitude).astype(np.float32)
        colours = np.clip(encoded[block], np.float32(0), np.float32(1))
        span[start:end] = np.ptp(colours, axis=1).max(axis=1)
        side = centers[start:end].astype(np.float64) >= middle
        octant[start:end] = side[:,0] + np.uint8(2)*side[:,1] + np.uint8(4)*side[:,2]
        for column, (a, b) in enumerate(((0,1), (1,2), (2,0))):
            left = np.minimum(block[:,a], block[:,b]).astype(np.uint64)
            right = np.maximum(block[:,a], block[:,b]).astype(np.uint64)
            keys[column*count+start:column*count+end] = left*np.uint64(vertex_count)+right
    keys.sort(kind='quicksort')
    changes = np.empty(len(keys), np.bool_)
    changes[0] = True
    np.not_equal(keys[1:], keys[:-1], out=changes[1:])
    starts = np.flatnonzero(changes)
    unique = keys[starts]
    incidence = np.diff(starts, append=len(keys))
    boundary_keys = unique[incidence == 1]
    nonmanifold_keys = unique[incidence > 2]
    edge_counts = dict(unique=int(len(unique)), boundary=int(len(boundary_keys)),
                       nonmanifold=int(len(nonmanifold_keys)), maximum_incidence=int(incidence.max()))
    del keys, changes, starts, unique, incidence
    boundary = np.zeros(count, np.bool_)
    nonmanifold = np.zeros(count, np.bool_)
    for start in range(0, count, BATCH_ROWS):
        end = min(start+BATCH_ROWS, count)
        block = faces[start:end]
        for a, b in ((0,1), (1,2), (2,0)):
            left = np.minimum(block[:,a], block[:,b]).astype(np.uint64)
            right = np.maximum(block[:,a], block[:,b]).astype(np.uint64)
            face_keys = left*np.uint64(vertex_count)+right
            for marked_keys, mask in ((boundary_keys,boundary), (nonmanifold_keys,nonmanifold)):
                if not len(marked_keys):
                    continue
                insertion = np.searchsorted(marked_keys, face_keys)
                safe = np.minimum(insertion, len(marked_keys)-1)
                mask[start:end] |= (insertion < len(marked_keys)) & (marked_keys[safe] == face_keys)
    return dict(areas=areas, centers=centers, normal_x=normal_x, span=span,
                octant=octant, boundary=boundary, nonmanifold=nonmanifold,
                face_count=count, vertex_count=vertex_count, bounds=[low.tolist(),high.tolist()],
                edge_counts=edge_counts, face_batch_rows=BATCH_ROWS)


def select(statistics, brick_ids, foliage_ids, seed=20261022, *, budgets=None):
    """Return separate contiguous cohorts, each with six queries per face.

    Face pools depend only on original indices, geometry and supplied encoded
    vertex RGB. Original face IDs are ascending in each pool. Area-weighted
    PCG64 choice without replacement uses SeedSequence([seed,cohort_index,0]);
    underfilled populations are retained whole without a draw. Selected IDs
    are sorted ascending before query generation. The complete brick list is
    mandatory: it is never thinned to meet a budget.

    Per face query order is corner0, corner1, corner2, centroid, interior0,
    interior1. Interior uniforms are drawn as [selected_face,2_points,2_values]
    in C order from independent PCG64 SeedSequence([seed,cohort_index,1]). For
    (u,v), barycentric weights are (1-sqrt(u),sqrt(u)*(1-v),sqrt(u)*v).
    Uniforms are clipped to [F64 epsilon,1-epsilon] to guard endpoint rounding.
    This draws strictly interior points without changing mesh coordinates.
    """
    _require(type(seed) is int and seed >= 0, 'Nonnegative integer seed required')
    configured = DEFAULT_BUDGETS.copy() if budgets is None else dict(budgets)
    _require(set(configured) == set(DEFAULT_BUDGETS), 'Exact declared cohort budget names required')
    _require(all(type(value) is int and value >= 0 for value in configured.values()),
             'Nonnegative integer face budgets required')
    _require(sum(configured.values())*6 <= MAX_QUERIES, 'Declared face budgets exceed hard spatial-query cap')
    names = tuple(DEFAULT_BUDGETS)
    count = statistics['face_count']
    areas = statistics['areas']
    _require(areas.shape == (count,) and np.isfinite(areas).all() and np.all(areas > 0),
             'Positive finite original triangle areas required')
    all_ids = np.arange(count, dtype=np.int32)
    def original_ids(value):
        _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in 'iu',
                 'Integer original face-ID vector required')
        _require(np.all(value >= 0) and np.all(value < count), 'Semantic original face ID outside mesh')
        ordered = np.sort(value.astype(np.int32, copy=True))
        _require(not len(ordered) or np.all(ordered[1:] != ordered[:-1]), 'Duplicate semantic original face IDs')
        return ordered
    brick = original_ids(brick_ids)
    foliage = original_ids(foliage_ids)
    _require(len(brick) <= configured['brick'], 'Complete original brick cohort exceeds budget; deletion forbidden')
    x = statistics['centers'][:,0]
    q25 = float(np.quantile(x, .25, method='linear'))
    median = float(np.quantile(x, .5, method='linear'))
    high_count = max(1, (count+99)//100)
    variation = np.argsort(-statistics['span'], kind='stable')[:high_count]
    pools = {
        'global': all_ids, 'brick': brick, 'foliage': foliage,
        'rear_x': np.flatnonzero(x <= q25).astype(np.int32),
        'backfacing_x': np.flatnonzero((statistics['normal_x'] < 0) & (x < median)).astype(np.int32),
        'boundary': np.flatnonzero(statistics['boundary']).astype(np.int32),
        'nonmanifold': np.flatnonzero(statistics['nonmanifold']).astype(np.int32),
        'high_variation': np.sort(variation.astype(np.int32)),
        **{f'octant_{i}': np.flatnonzero(statistics['octant'] == i).astype(np.int32) for i in range(8)},
    }
    selected = {}
    for index, name in enumerate(names):
        pool = pools[name]
        take = min(configured[name], len(pool))
        pool_area = float(areas[pool].sum(dtype=np.float64))
        if take == len(pool):
            chosen = pool.copy()
        elif not take:
            chosen = np.empty(0, np.int32)
        else:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,index,0])))
            weights = areas[pool]/pool_area
            chosen = np.sort(rng.choice(pool, size=take, replace=False, p=weights)).astype(np.int32)
        chosen_area = float(areas[chosen].sum(dtype=np.float64))
        selected[name] = dict(selected_face_ids=chosen, budget=configured[name],
                              pool_count=int(len(pool)), selected_count=int(len(chosen)),
                              underfilled=len(pool) < configured[name],
                              pool_area=pool_area, selected_area=chosen_area,
                              area_fraction=chosen_area/pool_area if pool_area else 0.)
    total = sum(row['selected_count'] for row in selected.values())*6
    _require(total <= MAX_QUERIES, 'Actual spatial-query count exceeds hard cap')
    query_face = np.empty(total, np.int32)
    bary = np.empty((total,3), np.float64)
    offset = 0
    for index, name in enumerate(names):
        row = selected[name]
        chosen = row['selected_face_ids']
        size = len(chosen)*6
        row['query_start'], row['query_stop'] = offset, offset+size
        query_face[offset:offset+size] = np.repeat(chosen, 6)
        block = bary[offset:offset+size].reshape(-1,6,3)
        block[:,:3] = np.eye(3)
        block[:,3] = 1/3
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,index,1])))
        uv = rng.random((len(chosen),2,2), dtype=np.float64)
        block[:,4:] = _interior_barycentrics(uv)
        offset += size
    return dict(query_face=query_face, bary=bary, cohorts=selected, total_queries=total,
                seed=seed, cohort_order=list(names), budgets=configured,
                thresholds=dict(rear_x_q25=q25, center_x_median=median,
                                high_variation_pool_count=high_count),
                draw_order=dict(generator='PCG64', numpy=np.__version__,
                    selection_seed='SeedSequence([seed,cohort_index,0])',
                    interior_seed='SeedSequence([seed,cohort_index,1])',
                    pool_order='ascending original face ID',
                    selected_order='ascending original face ID',
                    per_face=['corner0','corner1','corner2','centroid','interior0','interior1'],
                    interior_uniform_shape='[selected_face,2_points,2_values], C order',
                    interior_transform='(1-sqrt(u),sqrt(u)*(1-v),sqrt(u)*v)',
                    interior_endpoint_guard='clip uniforms to [F64 epsilon,1-epsilon]',
                    high_variation='ceil(F/100) highest input span, stable original face ID ties',
                    underfill='retain available pool; no replacement, deletion or refill'))
