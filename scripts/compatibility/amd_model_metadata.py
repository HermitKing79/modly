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

from services.amd_model_metadata import (HEADER_LIMIT, DTYPE_BYTES, unique_object, json_metadata, safetensors_metadata)


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
