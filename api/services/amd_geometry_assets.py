"""Resolve the audited local profile, reading JSON and safetensors headers only."""
import ast
import hashlib
import math
from pathlib import Path
from services.amd_runtime import local_path, absolute_file
from services.amd_model_metadata import json_metadata, safetensors_metadata

MODEL_PATHS = {
    "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    "sparse_structure_flow_model": "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16",
    "shape_slat_decoder": "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    "shape_slat_flow_model_512": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_slat_flow_model_1024": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_slat_decoder": "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16",
    "tex_slat_flow_model_512": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex_slat_flow_model_1024": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}
GEOMETRY_ROLES = tuple(name for name in MODEL_PATHS if not name.startswith("tex_"))
MODEL_CLASSES = {role: ("SparseStructureDecoder" if role == "sparse_structure_decoder"
                       else "SparseStructureFlowModel" if role == "sparse_structure_flow_model"
                       else "FlexiDualGridVaeDecoder" if role == "shape_slat_decoder"
                       else "SparseUnetVaeDecoder" if role == "tex_slat_decoder"
                       else "SLatFlowModel") for role in MODEL_PATHS}


def object_metadata(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def source_tree(path):
    try:
        return ast.parse(path.read_text())
    except SyntaxError as exc:
        raise ValueError("Invalid AMD model source syntax: " + str(path)) from exc


def validate_geometry_assets(config, source_root):
    if not isinstance(config, dict) or set(config) != {"models_root", "pipeline_dir", "dino_dir"}:
        raise ValueError("Geometry configuration requires exactly models_root, pipeline_dir and dino_dir")
    paths = {name: local_path(value, name).resolve(strict=True) for name, value in config.items()}
    if any(not value.is_dir() for value in paths.values()):
        raise ValueError("Geometry paths must be existing local directories")
    root, directory, dino = (paths[name] for name in ("models_root", "pipeline_dir", "dino_dir"))
    if directory != root / "microsoft/TRELLIS.2-4B" or dino != root / "facebook/dinov3-vitl16-pretrain-lvd1689m":
        raise ValueError("Use the explicitly audited local TRELLIS.2-4B and DINOv3 directories")
    source = local_path(str(source_root), "AMD source").resolve(strict=True)
    tree = source_tree(source / "trellis2/models/__init__.py")
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__attributes" for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError("AMD source must declare one explicit model class registry")
    classes = object_metadata(ast.literal_eval(declarations[0]), "AMD model registry")
    if not classes or any(not isinstance(key, str) or not isinstance(value, str) for key, value in classes.items()):
        raise ValueError("Invalid AMD model class registry")
    pipeline_path = absolute_file(str(directory / "pipeline.json"), "geometry pipeline JSON")
    pipeline = object_metadata(json_metadata(pipeline_path), "Geometry pipeline")
    args = object_metadata(pipeline.get("args"), "Geometry pipeline arguments")
    models = object_metadata(args.get("models"), "Geometry model map")
    if pipeline.get("name") != "Trellis2ImageTo3DPipeline" or set(models) != set(MODEL_PATHS):
        raise ValueError("Geometry pipeline must declare the audited eight component roles")
    if args.get("default_pipeline_type", "1024_cascade") not in ("512", "1024", "1024_cascade"):
        raise ValueError("Unsupported geometry pipeline profile")
    for name in ("shape_slat_normalization", "tex_slat_normalization"):
        normal = object_metadata(args.get(name), name)
        if any(not isinstance(normal.get(key), list) or len(normal[key]) != 32 for key in ("mean", "std")) \
                or any(type(v) not in (int, float) or not math.isfinite(v) for key in ("mean", "std") for v in normal[key]) \
                or any(v <= 0 for v in normal["std"]):
            raise ValueError("Invalid local latent normalization: " + name)
    for name in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
        sampler = object_metadata(args.get(name), name)
        if sampler.get("name") != "FlowEulerGuidanceIntervalSampler" or not isinstance(sampler.get("args"), dict) or not isinstance(sampler.get("params"), dict):
            raise ValueError("Invalid audited sampler configuration: " + name)
        sampler_args = sampler["args"]
        if set(sampler_args) != {"sigma_min"} or type(sampler_args["sigma_min"]) not in (int, float) or not 0 <= sampler_args["sigma_min"] < 1:
            raise ValueError("Invalid sampler constructor arguments: " + name)
    conditioner = object_metadata(args.get("image_cond_model"), "Image conditioner")
    object_metadata(conditioner.get("args"), "Image conditioner arguments")
    if conditioner.get("name") != "DinoV3FeatureExtractor":
        raise ValueError("Geometry requires the audited DINOv3 image conditioner")
    result = {"pipeline_dir": str(directory), "pipeline_config": str(pipeline_path), "models_root": str(root),
              "pipeline_config_sha256": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
              "config": pipeline, "models": {}, "geometry_roles": list(GEOMETRY_ROLES),
              "weights_loaded": False, "tensor_payload_bytes_read": 0}
    for role, relative in MODEL_PATHS.items():
        declared = args["models"][role]
        expected = relative if role == "sparse_structure_decoder" else "ckpts/" + Path(relative).name
        if declared != expected:
            raise ValueError(f"Nonlocal or unaudited checkpoint reference for {role}: {declared!r}")
        prefix = root / relative
        cfg = absolute_file(str(prefix) + ".json", role + " configuration")
        weights = absolute_file(str(prefix) + ".safetensors", role + " weights")
        description, metadata = object_metadata(json_metadata(cfg), role + " configuration"), safetensors_metadata(weights)
        if description.get("name") != MODEL_CLASSES[role] or MODEL_CLASSES[role] not in classes or not isinstance(description.get("args"), dict) or not description["args"] or metadata["metadata_only_fixture"]:
            raise ValueError("Unknown model configuration or empty checkpoint: " + role)
        module = local_path(str(source / ("trellis2/models/" + classes[description["name"]].replace(".", "/") + ".py")), role + " source").resolve(strict=True)
        module.relative_to(source)
        module_tree = source_tree(module)
        if not any(isinstance(node, ast.ClassDef) and node.name == description["name"] for node in module_tree.body):
            raise ValueError("Checkpoint class is not declared by the local AMD source")
        if description["name"] in ("SparseStructureFlowModel", "SLatFlowModel", "ElasticSLatFlowModel"):
            values = description["args"]
            if any(type(values.get(key)) is not int or values[key] <= 0 for key in ("model_channels", "in_channels", "out_channels")) or type(values.get("patch_size", 1)) is not int or values.get("patch_size", 1) <= 0:
                raise ValueError("Missing/invalid model dimensions: " + role)
            multiplier = values.get("patch_size", 1) ** 3 if description["name"] == "SparseStructureFlowModel" else 1
            shapes = {"input_layer.weight": [values["model_channels"], values["in_channels"] * multiplier],
                      "out_layer.weight": [values["out_channels"] * multiplier, values["model_channels"]]}
            if any(metadata["tensors"].get(key, {}).get("shape") != shape for key, shape in shapes.items()):
                raise ValueError("Core checkpoint shape/configuration mismatch: " + role)
        result["models"][role] = {"prefix": str(prefix), "config_path": str(cfg), "weights": str(weights),
                                  "config": description, "metadata": metadata, "source_module": str(module),
                                  "source_sha256": hashlib.sha256(module.read_bytes()).hexdigest()}
    dino_config = object_metadata(json_metadata(absolute_file(str(dino / "config.json"), "DINO configuration")), "DINO configuration")
    processor = json_metadata(absolute_file(str(dino / "preprocessor_config.json"), "DINO preprocessor"))
    dino_weights = absolute_file(str(dino / "model.safetensors"), "DINO safetensors")
    metadata = safetensors_metadata(dino_weights)
    if (dino_config.get("model_type") != "dinov3_vit" or dino_config.get("architectures") != ["DINOv3ViTModel"]
            or dino_config.get("auto_map") or dino_config.get("trust_remote_code") or metadata["metadata_only_fixture"]
            or metadata["tensors"].get("embeddings.patch_embeddings.weight", {}).get("shape") !=
                [dino_config.get("hidden_size"), dino_config.get("num_channels"), dino_config.get("patch_size"), dino_config.get("patch_size")]
            or not isinstance(processor, dict)):
        raise ValueError("DINOv3 local configuration/weights mismatch or remote code requested")
    if any(row["config"]["args"].get("cond_channels", dino_config["hidden_size"]) != dino_config["hidden_size"] for row in result["models"].values()):
        raise ValueError("DINOv3 conditioning width does not match geometry checkpoints")
    result["dino"] = {"directory": str(dino), "config_path": str(dino / "config.json"), "weights": str(dino_weights),
                      "config": dino_config, "metadata": metadata}
    return result
