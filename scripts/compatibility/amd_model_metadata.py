"""Bounded, stdlib-only model metadata audit. Never imports model code/tensors."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
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


def local_reference(root, value):
    if not isinstance(value, str) or ":" in value or "\\" in value or value.startswith("/"):
        raise ValueError(f"Nonlocal model reference: {value!r}")
    candidate = local_path(str(root / value), "model reference")
    candidate.relative_to(root)
    return candidate


def flow_shape_checks(config, tensors):
    name, args = config["name"], config["args"]
    if name not in ("SLatFlowModel", "ElasticSLatFlowModel", "SparseStructureFlowModel"):
        return {}
    multiplier = args.get("patch_size", 1) ** 3 if name == "SparseStructureFlowModel" else 1
    expected = {"input_layer.weight": [args["model_channels"], args["in_channels"] * multiplier],
                "out_layer.weight": [args["out_channels"] * multiplier, args["model_channels"]]}
    for key, shape in expected.items():
        if tensors.get(key, {}).get("shape") != shape:
            raise ValueError(f"Configuration/tensor shape mismatch: {key}; expected {shape}")
    return expected


def require_gate_one(result):
    if result.get("essential_assets_verified") is not True:
        raise ValueError("Gate 1 is incomplete; headless imports/model loading are prohibited")


def model_classes(source):
    tree = ast.parse((source / "trellis2/models/__init__.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__attributes" for target in node.targets))


def checkpoint(prefix, source, output, classes):
    config_path = local_path(str(prefix) + ".json", "checkpoint config")
    weights_path = local_path(str(prefix) + ".safetensors", "checkpoint weights")
    missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
    if missing:
        return {"prefix": str(prefix), "status": "MISSING", "missing": missing}
    try:
        config = json_metadata(config_path)
        if not isinstance(config, dict) or config.get("name") not in classes or \
                not isinstance(config.get("args"), dict) or not config["args"]:
            raise ValueError("Checkpoint config requires a known AMD source class and nonempty args")
        module = source / ("trellis2/models/" + classes[config["name"]].replace(".", "/") + ".py")
        module_bytes = module.read_bytes()
        class_tree = ast.parse(module_bytes.decode("utf-8"))
        if not any(isinstance(node, ast.ClassDef) and node.name == config["name"] for node in class_tree.body):
            raise ValueError("Checkpoint class is not declared by its source module")
        tensor_info = safetensors_metadata(weights_path)
        if tensor_info["metadata_only_fixture"]:
            raise ValueError("Metadata-only fixture is not an actual model checkpoint")
        checks = flow_shape_checks(config, tensor_info["tensors"])
        capture = output / (prefix.name + ".metadata.json")
        capture.write_text(json.dumps({"config": config, "source_module": str(module),
                                       "source_module_sha256": hashlib.sha256(module_bytes).hexdigest(),
                                       "core_shape_checks": checks, **tensor_info}, indent=2), encoding="utf-8")
        return {"prefix": str(prefix), "status": "VERIFIED_METADATA", "config": config,
                "core_shape_checks": checks, "metadata_file": capture.name,
                **{key: value for key, value in tensor_info.items() if key not in ("tensors", "metadata")}}
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {"prefix": str(prefix), "status": "INVALID", "error": repr(exc)}
