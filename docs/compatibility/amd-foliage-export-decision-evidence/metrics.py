"""Pure CPU appearance metrics; NumPy only, no asset I/O or model execution."""

import numpy as np


RGB_LIMITS = {"rmse": 0.02, "p95_abs": 0.05, "max_abs": 0.15}
SCALAR_RMSE_LIMIT = 0.02
SCALARS = ("metallic", "roughness", "alpha")
CHANNELS = ("red", "green", "blue", *SCALARS)


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a {np.dtype(dtype)} ndarray")
    if value.ndim != len(shape) or any(b is not None and a != b for a, b in zip(value.shape, shape)):
        raise ValueError(f"{name} has the wrong shape")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def _indices(value, count, name):
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer ndarray")
    if value.ndim != 1 or not value.size or value.min() < 0 or value.max() >= count:
        raise ValueError(f"{name} must be nonempty, one-dimensional and in range")


def _stats(error):
    absolute = np.abs(error)
    return {"rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
            "p95_abs": float(np.percentile(absolute, 95, method="linear")),
            "max_abs": float(absolute.max())}


def channel_ranges(raw):
    """Report per-channel raw/clipped bounds and clipping counts for float32[N,6]."""
    _array(raw, np.float32, (None, 6), "raw")
    if not len(raw):
        raise ValueError("raw must contain at least one sample")
    ranges = {}
    for channel, name in enumerate(CHANNELS):
        original = raw[:, channel]
        clipped = np.clip(original, 0, 1)
        below, above = int(np.count_nonzero(original < 0)), int(np.count_nonzero(original > 1))
        ranges[name] = {"raw": [float(original.min()), float(original.max())],
                        "clipped": [float(clipped.min()), float(clipped.max())],
                        "below_zero_count": below, "above_one_count": above, "clipped_count": below + above}
    return ranges


def color_prediction(encoded_vertices, faces, query_faces, bary):
    """Return (encoded RGB float64[Q,3], portable linear COLOR_0 float32[V,3]).

    Clip encoded vertex RGB, decode IEC 61966-2-1 sRGB, round the linear
    values to float32 COLOR_0, interpolate those values in float64, and encode
    the final colour with the same IEC transfer function. Barycentrics must
    be nonnegative and sum to one within 1e-12. No inputs are modified.
    """
    _array(encoded_vertices, np.float32, (None, 3), "encoded_vertices")
    _array(faces, np.int32, (None, 3), "faces")
    _array(query_faces, np.int32, (None,), "query_faces")
    _array(bary, np.float64, (len(query_faces), 3), "bary")
    if not faces.size or faces.min() < 0 or faces.max() >= len(encoded_vertices):
        raise ValueError("faces must contain in-range vertex indices")
    _indices(query_faces, len(faces), "query_faces")
    if np.any(bary < 0) or np.any(bary > 1) or not np.allclose(bary.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("bary must be nonnegative barycentric weights summing to one")
    encoded = np.clip(encoded_vertices.astype(np.float64), 0, 1)
    linear_color0 = np.where(encoded <= 0.04045, encoded / 12.92,
                             ((encoded + 0.055) / 1.055) ** 2.4).astype(np.float32)
    linear = np.einsum("qi,qij->qj", bary, linear_color0[faces[query_faces]].astype(np.float64))
    linear = np.clip(linear, 0, 1)
    predicted = np.where(linear <= 0.0031308, 12.92 * linear,
                         1.055 * linear ** (1 / 2.4) - 0.055)
    return predicted, linear_color0


def evaluate(predicted, raw, cohorts, fit_indices):
    """JSON-safe errors against clipped raw float32[Q,6] RGB/M/R/alpha.

    All supplied nonempty cohorts are required and use the same constants:
    the float64 mean of clipped scalar targets at ``fit_indices``. Repeated
    indices retain their sampling weights. RGB metrics pool all components;
    each scalar has its own RMSE gate. Scalar p95/max errors are descriptive.
    Per-cohort fitted constants/floors are optimistic diagnostics, never used
    to replace the shared candidate constants. Ranges cover all query targets.
    """
    _array(predicted, np.float64, (None, 3), "predicted")
    if np.any(predicted < 0) or np.any(predicted > 1):
        raise ValueError("predicted encoded RGB must lie in [0,1]")
    _array(raw, np.float32, (len(predicted), 6), "raw")
    _indices(fit_indices, len(raw), "fit_indices")
    if not isinstance(cohorts, dict) or not cohorts or any(not isinstance(n, str) for n in cohorts):
        raise TypeError("cohorts must be a nonempty mapping of names to index arrays")
    for name, indices in cohorts.items():
        _indices(indices, len(raw), f"cohort {name}")
    targets = np.clip(raw.astype(np.float64), 0, 1)
    constants = targets[fit_indices, 3:].mean(axis=0, dtype=np.float64)
    result = {}
    for name, indices in cohorts.items():
        selected = targets[indices]
        rgb = _stats(predicted[indices] - selected[:, :3])
        rgb["pass"] = all(rgb[key] <= limit for key, limit in RGB_LIMITS.items())
        optimistic = selected[:, 3:].mean(axis=0, dtype=np.float64)
        scalars, floors = {}, {}
        for channel, scalar in enumerate(SCALARS):
            stats = _stats(constants[channel] - selected[:, channel + 3])
            stats["pass"] = stats["rmse"] <= SCALAR_RMSE_LIMIT
            scalars[scalar] = stats
            floors[scalar] = float(np.sqrt(np.mean((optimistic[channel] - selected[:, channel + 3]) ** 2)))
        result[name] = {"sample_count": int(indices.size), "RGB": rgb, "scalars": scalars,
                        "optimistic_scalar_constants": dict(zip(SCALARS, map(float, optimistic))),
                        "irreducible_scalar_rmse_floor": floors,
                        "pass": rgb["pass"] and all(value["pass"] for value in scalars.values())}
    rgb_pass = all(value["RGB"]["pass"] for value in result.values())
    scalars_pass = all(value["pass"] for cohort in result.values() for value in cohort["scalars"].values())
    return {"constants": dict(zip(SCALARS, map(float, constants))), "fit_sample_count": int(fit_indices.size),
            "metrics": result, "RGB_all_pass": rgb_pass, "scalars_all_pass": scalars_pass,
            "decision_pass": rgb_pass and scalars_pass, "ranges": channel_ranges(raw),
            "RGB_limits": dict(RGB_LIMITS), "scalar_rmse_limit": SCALAR_RMSE_LIMIT,
            "required_cohorts": list(cohorts), "target_policy": "clip all six raw target channels to [0,1]"}
