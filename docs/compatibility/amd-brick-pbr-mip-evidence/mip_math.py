"""Bounded CPU reference box mipmaps and glTF-style trilinear filtering.

Arrays use V-up rows. These helpers neither read assets nor perform inference.
Mip generation is explicitly a floating linear-light 2x2 box, not a claim about
every glTF client's unspecified mip-generation kernel or storage precision.
"""
import numpy as np

MAX_SIZE, MAX_CHANNELS, MAX_QUERIES = 1024, 8, 65536


def _values(values):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Finite numeric values required') from exc
    if not np.isfinite(result).all():
        raise ValueError('Finite numeric values required')
    return np.clip(result, 0, 1)


def srgb_decode(encoded):
    """IEC sRGB decode after explicit display-only unit-range clipping."""
    values = _values(encoded)
    return np.where(values <= .04045, values / 12.92,
                    ((values + .055) / 1.055) ** 2.4)


def srgb_encode(linear):
    """IEC sRGB encode after explicit display-only unit-range clipping."""
    values = _values(linear)
    return np.where(values <= .0031308, values * 12.92,
                    1.055 * values ** (1 / 2.4) - .055)


def _image(image, strict_f64=False):
    if (not isinstance(image, np.ndarray) or image.dtype.kind != 'f'
            or strict_f64 and image.dtype != np.float64
            or image.ndim != 3 or image.shape[0] != image.shape[1]
            or not 1 <= image.shape[0] <= MAX_SIZE
            or image.shape[0] & (image.shape[0] - 1)
            or not 1 <= image.shape[2] <= MAX_CHANNELS
            or not np.isfinite(image).all()):
        raise ValueError('Finite floating square power-of-two HWC image <=1024 and <=8 channels required')


def pyramid(image, max_level=5):
    """Return independent F64 levels 0..max_level, using linear 2x2 means."""
    _image(image)
    if type(max_level) is not int or not 0 <= max_level <= image.shape[0].bit_length() - 1:
        raise ValueError('Integer mip level must fit the original power-of-two image')
    levels = [image.astype(np.float64, copy=True)]
    for _ in range(max_level):
        side, _, channels = levels[-1].shape
        try:
            with np.errstate(over='raise', invalid='raise'):
                levels.append(levels[-1].reshape(side // 2, 2, side // 2, 2, channels).mean(axis=(1, 3)))
        except FloatingPointError as exc:
            raise ValueError('Mip box arithmetic produced a nonfinite result') from exc
    return tuple(levels)


def _chain(levels):
    if not isinstance(levels, (list, tuple)) or not 1 <= len(levels) <= 11:
        raise ValueError('Nonempty bounded mip chain required')
    for index, image in enumerate(levels):
        _image(image, strict_f64=True)
        expected = levels[0].shape[0] >> index
        if image.shape != (expected, expected, levels[0].shape[2]):
            raise ValueError('Every mip level must halve dimensions and preserve channels')


def _bilinear(image, uv):
    side = image.shape[0]
    q = np.clip(np.clip(uv, 0, 1) * side - .5, 0, side - 1)
    low = np.floor(q).astype(np.int32)
    high = np.minimum(low + 1, side - 1)
    frac = q - low
    x, y = low.T
    nx, ny = high.T
    dx, dy = frac.T
    return (image[y, x] * ((1 - dx) * (1 - dy))[:, None]
            + image[y, nx] * (dx * (1 - dy))[:, None]
            + image[ny, x] * ((1 - dx) * dy)[:, None]
            + image[ny, nx] * (dx * dy)[:, None])


def sample(levels, uv, lod):
    """Clamp-to-edge bilinear sampling at two levels, then linear LOD blending.

    UVs outside [0,1] are intentionally clamped, matching CLAMP_TO_EDGE.
    LODs outside the supplied chain are rejected rather than silently clamped.
    """
    _chain(levels)
    if (not isinstance(uv, np.ndarray) or uv.dtype.kind != 'f' or uv.ndim != 2
            or uv.shape[1] != 2 or not 0 < len(uv) <= MAX_QUERIES or not np.isfinite(uv).all()):
        raise ValueError('Finite floating Qx2 UVs with 1..65536 queries required')
    lod = np.asarray(lod)
    if (lod.dtype.kind not in 'iuf' or lod.shape not in ((), (len(uv),))
            or not np.isfinite(lod).all() or (lod < 0).any() or (lod > len(levels) - 1).any()):
        raise ValueError('Finite scalar or Q-vector LOD inside the supplied mip chain required')
    lod = np.broadcast_to(lod.astype(np.float64), (len(uv),))
    low = np.floor(lod).astype(np.int32)
    high = np.minimum(low + 1, len(levels) - 1)
    frac = lod - low
    output = np.zeros((len(uv), levels[0].shape[2]), dtype=np.float64)
    for index, image in enumerate(levels):
        weight = (low == index) * (1 - frac) + (high == index) * frac
        selected = weight > 0
        if selected.any():
            output[selected] += _bilinear(image, uv[selected].astype(np.float64)) * weight[selected, None]
    return output


def conditioned(numerator_levels, weight_levels, uv, lod):
    """Normalize accumulated numerator/weight only after complete filtering.

    Membership weights must be in [0,1]. Zero support is a failure, never a
    reason to silently omit a query. Return (normalized_values, Qx1_mass).
    """
    _chain(numerator_levels)
    _chain(weight_levels)
    if (len(numerator_levels) != len(weight_levels)
            or numerator_levels[0].shape[:2] != weight_levels[0].shape[:2]
            or weight_levels[0].shape[2] != 1
            or any((level < 0).any() or (level > 1).any() for level in weight_levels)):
        raise ValueError('Matching one-channel unit-interval membership weight chain required')
    numerator = sample(numerator_levels, uv, lod)
    mass = sample(weight_levels, uv, lod)
    if (mass <= 0).any():
        raise ValueError('Source footprint has zero support; no queries may be discarded')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            values = numerator / mass
    except FloatingPointError as exc:
        raise ValueError('Conditioned footprint arithmetic produced a nonfinite result') from exc
    return values, mass
