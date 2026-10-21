"""Independent saved-query gates; no historical metrics, I/O, Blender or fitting."""
import sys
sys.dont_write_bytecode = True
import math

REGIONS = ('global', 'brick', 'foliage', 'rear_x', 'backfacing_x', 'boundary',
           'nonmanifold', 'high_variation', *(f'octant_{i}' for i in range(8)))
FACTORS = {'metallic': .00011082693761409814, 'roughness': .999995986777544, 'alpha': 1.}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _stats(error, np):
    # Independently implement the linear 95th percentile from order statistics.
    absolute = np.sort(np.abs(error).reshape(-1))
    _need(len(absolute) > 0, 'No empty required metric row')
    rank = .95 * (len(absolute) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    p95 = absolute[lo] + (absolute[hi] - absolute[lo]) * (rank - lo)
    flat = error.reshape(-1)
    return {'rmse': float(np.sqrt(np.sum(flat * flat, dtype=np.float64) / len(flat))),
            'p95_abs': float(p95), 'max_abs': float(absolute[-1])}


def _population(faces, oracle, decoded_linear, query_faces, bary, raw, kinds,
                support, frozen, expected_count, held, np):
    n = expected_count
    _need(isinstance(query_faces, np.ndarray) and query_faces.dtype == np.dtype('<i4')
          and query_faces.shape == (n,) and (query_faces >= 0).all() and (query_faces < len(faces)).all(),
          'Complete bounded saved I32 face-query population required')
    _need(isinstance(bary, np.ndarray) and bary.dtype == np.dtype('<f8') and bary.shape == (n, 3)
          and np.isfinite(bary).all() and (bary >= 0).all() and (bary <= 1).all()
          and np.all(np.abs(bary.sum(axis=1) - 1) <= 1e-12), 'Finite original barycentric queries required')
    _need(isinstance(raw, np.ndarray) and raw.dtype == np.dtype('<f4') and raw.shape == (n, 6)
          and np.isfinite(raw).all(), 'Complete original finite six-channel F32 source references required')
    _need(isinstance(kinds, np.ndarray) and kinds.dtype == np.dtype('i1') and kinds.shape == (n,)
          and ((kinds == 4) | (kinds == 5)).all() if held else
          isinstance(kinds, np.ndarray) and kinds.dtype == np.dtype('i1') and kinds.shape == (n,)
          and ((kinds >= 0) & (kinds <= 5)).all(), 'Exact original saved query categories required')
    _need(isinstance(support, np.ndarray) and support.dtype == np.dtype('<f4') and support.shape == (n,)
          and np.isfinite(support).all() and (support > 0).all(), 'Every saved query must retain positive original support')
    groups = frozen['cohorts']
    _need(isinstance(groups, dict) and set(groups) == set(REGIONS), 'All sixteen original frozen regions required')
    covered = np.zeros(n, dtype=np.bool_)
    selections = {}
    for name in REGIONS:
        first, last = groups[name]['query_start'], groups[name]['query_stop']
        _need(type(first) is int and type(last) is int and 0 <= first < last <= n
              and not covered[first:last].any(), 'Disjoint in-range original frozen cohort ranges required')
        covered[first:last] = True
        indices = np.arange(first, last, dtype=np.int32)
        parts = {'all': indices}
        if held:
            parts.update(interior0=indices[kinds[indices] == 4], interior1=indices[kinds[indices] == 5])
        else:
            parts.update(corners=indices[kinds[indices] < 3], centroids=indices[kinds[indices] == 3],
                         interiors=indices[kinds[indices] >= 4])
        _need(all(len(index) for index in parts.values()), 'Every declared frozen subcohort must be nonempty')
        selections.update({name + ':' + part: index for part, index in parts.items()})
    _need(covered.all() and len(selections) == (48 if held else 64), 'Every original saved query and subcohort is required')
    prediction = np.empty((n, 3), dtype=np.float64)
    for first in range(0, n, 32768):
        last = min(first + 32768, n)
        corner_ids = faces[query_faces[first:last]]
        _need((corner_ids >= 0).all() and (corner_ids < len(oracle)).all(), 'Original corner index outside oracle')
        if decoded_linear is None:
            encoded = oracle[corner_ids, :3].astype(np.float64) / 255.
            linear_corners = np.where(encoded <= .04045, encoded / 12.92,
                                      ((encoded + .055) / 1.055) ** 2.4).astype(np.float32).astype(np.float64)
        else:
            linear_corners = decoded_linear[corner_ids].astype(np.float64)
        linear = np.zeros((last - first, 3), dtype=np.float64)
        for corner in range(3):
            linear += bary[first:last, corner, None] * linear_corners[:, corner, :]
        linear = np.clip(linear, 0, 1)
        prediction[first:last] = np.where(linear <= .0031308, linear * 12.92,
                                          1.055 * linear ** (1 / 2.4) - .055)
    target = np.clip(raw.astype(np.float64), 0, 1)
    rows = {}
    for name, indices in selections.items():
        rgb = _stats(prediction[indices] - target[indices, :3], np)
        rgb['pass'] = rgb['rmse'] <= .02 and rgb['p95_abs'] <= .05 and rgb['max_abs'] <= .15
        scalars = {}
        for channel, (label, factor) in enumerate(FACTORS.items(), 3):
            row = _stats(factor - target[indices, channel], np)
            row['pass'] = row['rmse'] <= .02
            scalars[label] = row
        rows[name] = {'count': len(indices), 'RGB': rgb, 'scalars': scalars,
                      'pass': bool(rgb['pass'] and all(row['pass'] for row in scalars.values()))}
    return {'query_count': n, 'cohorts': rows, 'all_gates_pass': all(row['pass'] for row in rows.values()),
            'predicted_RGB': prediction, 'RGB_limits': {'rmse': .02, 'p95_abs': .05, 'max_abs': .15},
            'scalar_RMSE_limit': .02, 'actual_scalar_factors': dict(FACTORS), 'zero_support': 0}


def verify(data, oracle, decoded_linear=None):
    """Check all 56,910 original plus 2,048 holdout queries and 112 rows.

    Inputs and frozen definitions must already be authenticated by the caller.
    Phase 1 decodes stored bytes by float64 IEC sRGB then rounds linear RGB to
    F32. Optional decoded_linear is original-row F32 RGB captured through the
    native BYTE_COLOR getter, independently mapped by the caller; it replaces
    the IEC decode for phase 2. Neither path invokes a native kernel here.
    Returned predicted_RGB arrays must be saved/removed before JSON encoding.
    """
    import numpy as np
    vertices, faces = data['vertices'], data['faces']
    _need(isinstance(vertices, np.ndarray) and vertices.dtype == np.dtype('<f4')
          and vertices.ndim == 2 and vertices.shape[1] == 3 and 0 < len(vertices) <= 2000000,
          'Original complete F32 vertex-table schema required')
    _need(isinstance(faces, np.ndarray) and faces.dtype == np.dtype('<i4')
          and faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= 4173584,
          'Original complete I32 triangle-table schema required')
    _need(isinstance(oracle, np.ndarray) and oracle.dtype == np.dtype('u1')
          and oracle.shape == (len(vertices), 4), 'Original-row native oracle U8 RGBA schema required')
    if decoded_linear is not None:
        _need(isinstance(decoded_linear, np.ndarray) and decoded_linear.dtype == np.dtype('<f4')
              and decoded_linear.shape == (len(vertices), 3), 'Original-row native getter F32 RGB required')
    for first in range(0, len(oracle), 32768):
        _need((oracle[first:first + 32768, 3] == 255).all(), 'Native oracle OPAQUE alpha must be exactly 255')
        if decoded_linear is not None:
            part = decoded_linear[first:first + 32768]
            _need(np.isfinite(part).all() and (part >= 0).all() and (part <= 1).all(), 'Finite unit native getter RGB required')
    prior, held = data['prior'], data['resampled']
    original = _population(faces, oracle, decoded_linear, prior['query-faces'], prior['barycentrics'],
                           prior['source_spatial'], prior['query-kinds'], prior['raw_weight_sum'],
                           data['frozen'], 56910, False, np)
    holdout = _population(faces, oracle, decoded_linear, held['holdout-query-faces'], held['holdout-barycentrics'],
                          held['holdout_source_spatial'], held['holdout-query-kinds'], held['holdout_weight_sum'],
                          data['holdout_frozen'], 2048, True, np)
    return {'original_queries': original, 'holdout_queries': holdout,
            'all_gates_pass': bool(original['all_gates_pass'] and holdout['all_gates_pass']),
            'required_metric_rows': 112,
            'decode_policy': ('Supplied actual native BYTE_COLOR getter F32 RGB at original source IDs'
                              if decoded_linear is not None else
                              'IEC sRGB float64 decode of stored bytes, rounded to F32; no native-getter bit-equality claim'),
            'no_new_queries_sampling_fit_or_native_kernel': True}
