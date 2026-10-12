"""Reusable real pipeline structure and deferred offline component requirements."""
import gc
import hashlib
import inspect
from pathlib import Path
from alpha_input import decode_alpha_image, preprocess_alpha_image
from headless import import_pipeline_class
from model_assets import validate_geometry_assets, GEOMETRY_ROLES
from strict_loader import StrictLocalComponent


class GeometryPreparation:
    def __init__(self, config, source_root, guards, pipeline_class=None):
        self.config, self.source_root, self.guards = config, Path(source_root), guards
        self.pipeline_class = pipeline_class
        self.pipeline = None
        self.components = {}

    def prepare(self, image_bytes):
        self.close()
        image = None
        try:
            image, alpha = decode_alpha_image(image_bytes)
            assets = validate_geometry_assets(self.config, self.source_root)
            with self.guards.factory_context():
                if self.pipeline_class is None:
                    self.pipeline_class = import_pipeline_class(self.source_root, assets["models_root"], "generate")
                from trellis2 import models
                models.from_pretrained = self.guards.forbidden
                # Reuse the genuine factory; its pinned base creates None slots.
                pipeline = self.pipeline_class.from_pretrained(assets["pipeline_dir"], keep_models_loaded=False)
                if (any(value is not None for value in pipeline.models.values())
                        or pipeline.image_cond_model is not None or pipeline.rembg_model is not None):
                    raise RuntimeError("Geometry factory unexpectedly populated heavyweight model objects")
                self.pipeline = pipeline
                pipeline._pretrained_args["models"] = {name: row["prefix"] for name, row in assets["models"].items()}
                pipeline._pretrained_args["image_cond_model"]["args"]["model_name"] = assets["dino"]["directory"]
                pipeline._pretrained_args["rembg_model"] = None
                for definition in pipeline.PIXAL3D_IMAGE_COND_CONFIGS.values():
                    definition["model_name"] = assets["dino"]["directory"]
                conditioned = preprocess_alpha_image(pipeline, image)
            requirements = {}
            for name in GEOMETRY_ROLES:
                row = assets["models"][name]
                def create(config, descriptor=row):
                    from trellis2 import models
                    cls = getattr(models, config["name"])
                    if (Path(inspect.getfile(cls)).resolve() != Path(descriptor["source_module"]).resolve()
                            or hashlib.sha256(Path(descriptor["source_module"]).read_bytes()).hexdigest() != descriptor["source_sha256"]):
                        raise RuntimeError("Lazy model class provenance changed")
                    return cls(**config["args"])
                self.components[name] = StrictLocalComponent(row["config_path"], row["weights"], create)
                requirements[name] = self.requirement(row)
            def create_dino(config):
                from transformers import DINOv3ViTConfig, DINOv3ViTModel
                # Config construction has no remote factory/code/download path.
                return DINOv3ViTModel(DINOv3ViTConfig.from_dict(config))
            dino = assets["dino"]
            self.components["dino_backbone"] = StrictLocalComponent(dino["config_path"], dino["weights"], create_dino)
            requirements["dino_backbone"] = self.requirement(dino)
            gpu = self.guards.gpu_diagnostics()
            return {"schema": "modly.amd.geometry-preparation.v1", "mode": "alpha-only", "pipeline_prepared": True,
                    "weights_loaded": any(component.loaded for component in self.components.values()),
                    "asset_generated": False, "glb_generated": False, "ready_for_inference": False,
                    "pipeline": {"class": type(pipeline).__name__, "module": type(pipeline).__module__,
                                 "source": inspect.getfile(type(pipeline)), "factory_reused": True,
                                 "configuration": assets["pipeline_config"], "configuration_sha256": assets["pipeline_config_sha256"],
                                 "checkpoint_map": pipeline._pretrained_args["models"],
                                 "model_slots": {name: None for name in pipeline.models},
                                 "image_cond_model": None, "rembg_model": None,
                                 "dino_directory": dino["directory"], "device": pipeline._device},
                    "components": requirements, "lazy_load_contract": "strict CPU state dictionary, parameters and persistent buffers; no remote factory",
                    "real_load_enabled_in_preparation_worker": False,
                    "alpha_input": alpha, "conditioning_image": {"mode": conditioned.mode, "size": list(conditioned.size),
                        "sha256": hashlib.sha256(conditioned.tobytes()).hexdigest(), "dino_executed": False},
                    "gpu": gpu, "torchvision_version_queries": list(self.guards.version_queries),
                    "stateless_activation_objects": list(self.guards.activations), "tensor_payload_bytes_read": 0}
        except BaseException:
            self.close()
            raise
        finally:
            if image is not None:
                image.close()

    @staticmethod
    def requirement(row):
        metadata = row["metadata"]
        return {"configuration": row["config_path"], "weights": row["weights"], "state": "unloaded",
                "tensor_count": metadata["tensor_count"], "payload_bytes": metadata["tensor_data_bytes"],
                "header_sha256": metadata["header_sha256"], "architecture_schema": "verified on subsequent strict load"}

    def close(self):
        for component in self.components.values():
            if component.loaded:
                component.unload()
        self.components.clear()
        if self.pipeline is not None:
            self.pipeline.models.clear()
        self.pipeline = None
        gc.collect()
