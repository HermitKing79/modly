"""Bounded CPU port of the pinned FlexGEMM sparse trilinear forward algorithm.

See source-semantics.json for exact installed sources and arithmetic limitations.
NumPy float32 operations preserve source ordering and missing-support policy;
native HIP/Triton fusion may round differently. No GPU parity claim is made.
Inputs remain untouched. This module imports only dataclasses and NumPy.
"""
from dataclasses import dataclass

import numpy as np


MAX_ROWS = 2_000_000
MAX_QUERIES = 65_536
MAX_TEXTURE_TEXELS = 1_048_576
_F32 = np.float32
_OFFSETS = ((0,0,0), (1,0,0), (0,1,0), (1,1,0),
            (0,0,1), (1,0,1), (0,1,1), (1,1,1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_rows(array):
    return all(np.isfinite(array[start:start+32_768]).all()
               for start in range(0, len(array), 32_768))


def _validate_queries(query):
    _require(isinstance(query, np.ndarray) and query.dtype == np.float32
             and query.ndim == 2 and query.shape[1] == 3,
             "Queries must be original-axis float32[M,3]")
    _require(len(query) <= MAX_QUERIES, "Query count exceeds bounded batch")
    _require(_finite_rows(query), "Queries must be finite")
    _require(not len(query) or np.max(np.abs(query)) <= _F32(1_000_000),
             "Query magnitude exceeds bounded integer conversion")


@dataclass(frozen=True)
class SampleResult:
    values: np.ndarray
    neighbour_rows: np.ndarray
    normalized_weights: np.ndarray
    raw_weight_sum: np.ndarray
    found_neighbour_count: np.ndarray
    positive_weight_neighbour_count: np.ndarray
    zero_support: np.ndarray


class SparseSampler:
    """Exact sparse support lookup with one sorted key array and row permutation.

    Coordinates are I32[N,4] in batch,x,y,z order, batch0 only. Features are
    original finite F32[N,6], with no clamp or second decoder transformation.
    Keep the original feature array immutable while this sampler is in use.
    """

    def __init__(self, coordinates, features, spatial_shape=(512,512,512), *, max_rows=MAX_ROWS):
        _require(type(max_rows) is int and 1 <= max_rows <= MAX_ROWS,
                 "Maximum row bound must be an integer within the proof cap")
        _require(isinstance(coordinates, np.ndarray) and coordinates.dtype == np.int32
                 and coordinates.ndim == 2 and coordinates.shape[1] == 4,
                 "Coordinates must be int32[N,4]")
        _require(0 < len(coordinates) <= max_rows, "Support row count is outside bounded proof")
        _require(isinstance(features, np.ndarray) and features.dtype == np.float32
                 and features.shape == (len(coordinates),6), "Features must be float32[N,6]")
        _require(isinstance(spatial_shape, (tuple,list)) and len(spatial_shape) == 3
                 and all(type(v) is int and 1 <= v <= 512 for v in spatial_shape),
                 "Spatial shape must contain three integer dimensions within1..512")
        self.spatial_shape = tuple(spatial_shape)
        for start in range(0, len(coordinates), 32_768):
            block = coordinates[start:start+32_768]
            _require(np.all(block[:,0] == 0), "Only original batch0 is supported")
            _require(np.all(block[:,1:] >= 0) and np.all(block[:,1:] < self.spatial_shape),
                     "Sparse coordinates are outside spatial bounds")
        _require(_finite_rows(features), "Original features must be finite")
        _, height, depth = self.spatial_shape
        keys = ((coordinates[:,1].astype(np.int64)*height
                 + coordinates[:,2])*depth + coordinates[:,3])
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]
        _require(np.all(self.keys[1:] != self.keys[:-1]), "Duplicate sparse coordinates")
        self.keys.flags.writeable = False
        self.order.flags.writeable = False
        self.features = features.view()
        self.features.flags.writeable = False

    def sample_world(self, positions):
        """Sample original XYZ world positions using native q=(world+0.5)*512."""
        _validate_queries(positions)
        query = (positions + _F32(.5))*_F32(512)
        return self.sample_grid(query)

    def sample_grid(self, query):
        """Sample float32 grid coordinates, exposing support before quantization.

        Partial support renormalizes over found, in-bounds neighbours. Zero
        support returns six zeros, including alpha. A sum below1e-12 retains
        the native denominator floor. Found zero-weight rows are distinguished
        from positive-weight support in the returned diagnostics.
        """
        _validate_queries(query)
        count = len(query)
        base = np.floor(query - _F32(.5)).astype(np.int64)
        rows = np.full((count,8), -1, dtype=np.int64)
        weights = np.zeros((count,8), dtype=np.float32)
        weight_sum = np.zeros(count, dtype=np.float32)
        width, height, depth = self.spatial_shape
        for i, offset in enumerate(_OFFSETS):
            cell = base + offset
            inside = ((cell[:,0] >= 0) & (cell[:,0] < width)
                      & (cell[:,1] >= 0) & (cell[:,1] < height)
                      & (cell[:,2] >= 0) & (cell[:,2] < depth))
            keys = (cell[:,0]*height + cell[:,1])*depth + cell[:,2]
            insertion = np.searchsorted(self.keys, keys)
            safe = np.minimum(insertion, len(self.keys)-1)
            found = inside & (insertion < len(self.keys)) & (self.keys[safe] == keys)
            rows[found,i] = self.order[insertion[found]]
            # Preserve HIP subtraction and x*y*z multiplication order in F32.
            distance = np.abs((query - cell.astype(np.float32)) - _F32(.5))
            axis_weight = _F32(1) - distance
            raw_weight = (axis_weight[:,0]*axis_weight[:,1])*axis_weight[:,2]
            weights[:,i] = np.where(found, raw_weight, _F32(0))
            weight_sum += weights[:,i]
        positive_count = np.count_nonzero(weights > _F32(0), axis=1)
        weights /= np.maximum(weight_sum, _F32(1e-12))[:,None]
        values = np.zeros((count,6), dtype=np.float32)
        for i in range(8):
            # Finite original inputs permit zero-weight missing rows to gather0.
            values += self.features[np.maximum(rows[:,i],0)]*weights[:,i,None]
        return SampleResult(values, rows, weights, weight_sum,
                            np.count_nonzero(rows >= 0, axis=1), positive_count,
                            weight_sum == _F32(0))


def pack_native_textures(values):
    """Native byte packing only: RGBA, (unused0,roughness,metallic), alpha.

    RGB remains encoded, scalars receive no gamma, and alpha is retained even
    when a separately created material chooses OPAQUE. No AO map is inferred.
    """
    _require(isinstance(values, np.ndarray) and values.dtype == np.float32
             and values.ndim in (2,3) and values.shape[-1] == 6,
             "Sampled texture values must be float32[N,6] or float32[H,W,6]")
    _require(values.size//6 <= MAX_TEXTURE_TEXELS, "Texture exceeds bounded proof texels")
    _require(_finite_rows(values), "Sampled texture values must be finite")
    # Match np.clip(sample*255,0,255).astype(uint8), including truncation.
    byte = np.clip(values*_F32(255), _F32(0), _F32(255)).astype(np.uint8)
    return {
        "base_color_rgba": byte[..., [0,1,2,5]],
        "metallic_roughness_rgb": np.stack((np.zeros_like(byte[...,3]),
                                            byte[...,4], byte[...,3]), axis=-1),
        "alpha": byte[...,5],
    }
