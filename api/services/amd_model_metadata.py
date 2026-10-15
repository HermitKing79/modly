"""Bounded local JSON/safetensors metadata; no model or torch imports."""
import hashlib
import json
import math
import struct
from services.amd_runtime import local_path

HEADER_LIMIT = 16 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
               "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F16": 2,
               "BF16": 2, "F32": 4, "F64": 8, "C64": 8, "C128": 16,
               "F8_E4M3": 1, "F8_E5M2": 1}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_metadata(path):
    path = local_path(str(path), "configuration")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Configuration exceeds 1 MiB limit: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def safetensors_metadata(path):
    path = local_path(str(path), "safetensors")
    if path.suffix != ".safetensors":
        raise ValueError("Only explicit safetensors files are supported; no GGUF/pickle fallback")
    info = path.stat()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(HEADER_LIMIT, info.st_size - 8):
            raise ValueError("Invalid or oversized safetensors header")
        raw = stream.read(length)
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError("Safetensors metadata must be string-valued")
    payload = info.st_size - 8 - length
    intervals, parameters, dtypes = [], 0, {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor descriptor: {name}")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(shape) > 16 or \
                any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"Invalid tensor dtype/shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or \
                any(type(value) is not int for value in offsets):
            raise ValueError(f"Invalid tensor offsets: {name}")
        start, end = offsets
        count = math.prod(shape)
        if not 0 <= start <= end <= payload or end - start != count * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor byte count/layout mismatch: {name}")
        intervals.append((start, end, name))
        parameters += count
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Overlapping or gapped tensor layout: {name}")
        cursor = end
    if cursor != payload:
        raise ValueError("Unreferenced/truncated tensor payload")
    return {"path": str(path), "file_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "header_bytes": length, "bytes_read": 8 + length, "tensor_data_bytes": payload,
            "header_sha256": hashlib.sha256(prefix + raw).hexdigest(), "tensor_count": len(tensors),
            "parameter_count": parameters, "dtypes": dtypes,
            "metadata_only_fixture": not tensors or parameters == 0 or payload == 0,
            "metadata": metadata, "tensors": tensors}


