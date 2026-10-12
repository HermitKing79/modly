"""Gate 1: explicit local TRELLIS.2 assets, metadata only. No pipeline imports."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amd_model_metadata import (ROOT, checkpoint, json_metadata, local_path,
                                local_reference, model_classes, safetensors_metadata)

GENERATE = {"sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_decoder",
            "shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_decoder",
            "tex_slat_flow_model_512", "tex_slat_flow_model_1024"}
REFINE = {"shape_slat_encoder", "tex_slat_decoder", "tex_slat_flow_model_512", "tex_slat_flow_model_1024"}
PACKAGES = ("torch", "torchvision", "rocm", "transformers", "safetensors", "huggingface-hub",
            "numpy", "Pillow", "trimesh", "opencv-python", "scipy", "einops", "timm",
            "utils3d", "easydict", "cumesh", "flex-gemm", "o-voxel", "nvdiffrast", "triton-windows")


def audit_assets(source, models, amd_venv, caches, output):
    source = local_path(str(source), "AMD source")
    models = local_path(str(models), "model root")
    output = local_path(str(output), "audit output")
    output.resolve().relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    classes = model_classes(source)
    root = models / "microsoft/TRELLIS.2-4B"
    result = {"profile": "microsoft/TRELLIS.2-4B standard BF16/FP16; 512 and 1024",
              "source": str(source), "models_root": str(models), "pipelines": {},
              "checkpoints": {}, "blockers": [], "optional": [],
              "scope": "JSON and safetensors headers only; no tensor data read or model code executed"}
    prefixes = {}
    for file, name, roles in (("pipeline.json", "Trellis2ImageTo3DPipeline", GENERATE),
                              ("texturing_pipeline.json", "Trellis2TexturingPipeline", REFINE)):
        config = json_metadata(root / file)
        if config.get("name") != name or not roles <= set(config.get("args", {}).get("models", {})):
            raise ValueError(f"Actual pipeline config missing required class/roles: {file}")
        args = config["args"]
        for key in ("shape_slat_normalization", "tex_slat_normalization"):
            normal = args[key]
            if len(normal["mean"]) != 32 or len(normal["std"]) != 32 or any(value <= 0 for value in normal["std"]):
                raise ValueError(f"Invalid latent normalization: {file}/{key}")
        resolved = {}
        for role, value in args["models"].items():
            prefix = local_reference(root, value) if role != "sparse_structure_decoder" else \
                local_reference(models, "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
            prefixes[str(prefix)] = prefix
            resolved[role] = {"declared": value, "resolved_prefix": str(prefix),
                              "decoder_override": role == "sparse_structure_decoder"}
        result["pipelines"][file] = {"config": config, "resolved_models": resolved,
                                     "sha256": hashlib.sha256((root / file).read_bytes()).hexdigest()}
    # Geometry's later mesh-encoding route injects this role in base.Pipeline.
    prefix = local_reference(root, "ckpts/shape_enc_next_dc_f16c32_fp16")
    prefixes[str(prefix)] = prefix
    for key, prefix in sorted(prefixes.items()):
        record = checkpoint(prefix, source, output, classes)
        result["checkpoints"][key] = record
        if record["status"] != "VERIFIED_METADATA":
            result["blockers"].append(record)
    dino = models / "facebook/dinov3-vitl16-pretrain-lvd1689m"
    try:
        config = json_metadata(dino / "config.json")
        preprocessor = json_metadata(dino / "preprocessor_config.json")
        tensors = safetensors_metadata(dino / "model.safetensors")
        expected = [config["hidden_size"], config["num_channels"], config["patch_size"], config["patch_size"]]
        if config.get("model_type") != "dinov3_vit" or tensors["metadata_only_fixture"] or \
                tensors["tensors"].get("embeddings.patch_embeddings.weight", {}).get("shape") != expected:
            raise ValueError("DINOv3 config/patch embedding metadata mismatch")
        for record in result["checkpoints"].values():
            cond = record.get("config", {}).get("args", {}).get("cond_channels")
            if cond is not None and cond != config["hidden_size"]:
                raise ValueError("DINOv3 hidden size does not match TRELLIS conditioning channels")
        capture = output / "dinov3.metadata.json"
        capture.write_text(json.dumps({"config": config, "preprocessor": preprocessor, **tensors}, indent=2))
        result["conditioning"] = {"status": "VERIFIED_METADATA", "local_path": str(dino),
                                  "patch_embedding_shape": expected, "metadata_file": capture.name,
                                  **{key: value for key, value in tensors.items() if key not in ("tensors", "metadata")}}
    except (ValueError, OSError, KeyError) as exc:
        result["conditioning"] = {"status": "INVALID_OR_MISSING", "path": str(dino), "error": repr(exc)}
        result["blockers"].append(result["conditioning"])

    # The unmodified TexturingPipeline constructs this remote-code model
    # unconditionally. Do not substitute another rembg model or fake success.
    auxiliary_dirs = [models / "briaai/RMBG-2.0", models / "background_removal/RMBG-2.0"]
    searches = []
    for cache in caches:
        cache = local_path(str(cache), "cache root")
        repo = local_path(str(cache / "models--briaai--RMBG-2.0"), "RMBG cache")
        snapshots = local_path(str(repo / "snapshots"), "RMBG snapshots")
        searches.append({"path": str(repo), "exists": repo.is_dir()})
        if snapshots.is_dir():
            auxiliary_dirs.extend(sorted(path for path in snapshots.iterdir() if path.is_dir()))
    candidates = []
    for directory in auxiliary_dirs:
        directory = local_path(str(directory), "RMBG candidate")
        paths = [local_path(str(directory / name), "RMBG asset") for name in
                 ("config.json", "model.safetensors", "model.safetensors.index.json",
                  "pytorch_model.bin", "pytorch_model.bin.index.json")]
        candidates.append({"path": str(directory), "exists": directory.is_dir(),
                           "files_present": [str(path) for path in paths if path.is_file()]})
    result["background_removal"] = {"repository_id": "briaai/RMBG-2.0", "class": "BiRefNet",
                                    "required_by": "Trellis2TexturingPipeline.from_pretrained -> BiRefNet.__init__",
                                    "trust_remote_code": True, "cache_search": searches, "candidates": candidates,
                                    "status": "MISSING" if not any(row["files_present"] for row in candidates) else "PRESENT_UNAUDITED",
                                    "missing_requirement": "Local config.json, supported weight file or shard index with all shards, and every config.auto_map/custom-code dependency"}
    result["blockers"].append(result["background_removal"])
    result["offline_plan"] = {"DinoV3FeatureExtractor.model_name": str(dino),
                               "sparse_structure_decoder": str(models / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"),
                               "BiRefNet.model_name": None,
                               "network_fallback_permitted": False,
                               "note": "Texturing config retains remote model IDs; do not call from_pretrained. No configs were rewritten."}
    recon = root / "reconviagen_pipeline.json"
    if recon.is_file():
        config = json_metadata(recon)
        pairs = []
        for role, value in config["args"]["models"].items():
            if role not in GENERATE:
                prefix = local_reference(root, value)
                pairs.append({"role": role, "prefix": str(prefix),
                              "missing": [str(local_path(str(Path(str(prefix) + suffix)), "optional asset"))
                                          for suffix in (".json", ".safetensors") if not Path(str(prefix) + suffix).is_file()]})
        result["optional"].append({"profile": "ReconViaGen", "selected": False, "extra_checkpoints": pairs})
    result["optional"] += [{"profile": "FP8", "selected": False,
                             "expected_pipeline": str(models / "visualbruno/TRELLIS.2-4B-FP8/pipeline_fp8.json")},
                            {"profile": "Pixal3D/MoGe", "selected": False,
                             "model_roots": [str(models / "TencentARC/Pixal3D-T"), str(models / "Ruicheng/moge-2-vitl/model.pt")]}]
    extra = local_reference(root, "ckpts/tex_enc_next_dc_f16c32_fp16")
    result["optional"].append({"profile": "unused texture encoder", "selected": False,
                               "checkpoint": checkpoint(extra, source, output, classes)})
    site = local_path(str(amd_venv / "Lib/site-packages"), "AMD packages")
    distributions = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                     for dist in metadata.distributions(path=[str(site)])}
    result["dependencies"] = {name: distributions.get(name.lower().replace("_", "-")) for name in PACKAGES}
    result["headless_requirements"] = {"folder_paths": str(source.parents[1] / "folder_paths.py"),
                                       "comfy.utils.ProgressBar": str(source.parents[1] / "comfy/utils.py"),
                                       "status": "Not implemented/imported; Gate 1 closed"}
    sources = ["nodes.py", "nodes.py.before-profiler-fix", "trellis2/pipelines/base.py", "trellis2/models/__init__.py",
               "trellis2/models/structured_latent_flow.py", "trellis2/models/sparse_structure_flow.py",
               "trellis2/pipelines/trellis2_image_to_3d.py", "trellis2/pipelines/trellis2_texturing.py",
               "trellis2/pipelines/rembg/BiRefNet.py", "trellis2/modules/image_feature_extractor.py"]
    result["source_sha256"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in sources}
    result.update({"essential_assets_verified": False, "classification": "PARTIAL",
                   "gate_two": "NOT ATTEMPTED: essential auxiliary asset not verified",
                   "torch_imported": "torch" in sys.modules, "gpu_initialization": "Not queried; no GPU packages imported",
                   "tensor_payloads_read": False, "models_loaded": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "models", "amd-venv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = audit_assets(args.source, args.models, args.amd_venv, args.hf_cache, args.output)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"classification": result["classification"],
                      "checkpoint_statuses": {Path(key).name: value["status"] for key, value in result["checkpoints"].items()},
                      "conditioning": result["conditioning"]["status"], "auxiliary": result["background_removal"]["status"],
                      "gate_two": result["gate_two"]}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
