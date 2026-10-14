"""Pinned neural parent decoder: strict original F16 load, then 76 exact widenings.

No native/Torch import occurs until load_cpu(), after the native import gate.
This component does not call a decoder forward, move tensors to a GPU, alter
the shared strict loader, or admit a neural execution/workspace policy.
"""
import gc
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

from services.amd_runtime import local_path
from services.amd_model_metadata import json_metadata
from strict_loader import StrictLocalComponent

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD" if sys.platform == "win32"
              else "/mnt/d/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD")
PARAMETER_PROOF = ROOT / "docs/compatibility/amd-shape-decoder-neural-evidence/cpu-probe-4/results.json"
PARAMETER_MANIFEST_SHA256 = "45fb3288d46a8f0bb4dc4f40293c4c7771971256f6c0383b7b2c94bc4da93359"
SOURCE_SHA256 = {
    "models/sc_vaes/sparse_unet_vae.py": "c9ce51db70bb1374c8bdc858129f6ff1170af1447e5210d434b7ebb289dbd2ce",
    "modules/utils.py": "7e833200485f47e5cb02950f383a3266932cba78fde927cd2f49923220f4058e",
    "modules/norm.py": "72824f6e9a1df993f42cc999a4e0b214400946606d20120b289d74d10c2afd00",
    "modules/sparse/basic.py": "e867e429b9c2fe154000725596b626273fb567025b9c232f3cc79140f00605b7",
    "modules/sparse/linear.py": "05cb619ff9cceb27674fc7bbb5aa2bcba93d1bf328d1f3340bb98c0f1240b413",
    "modules/sparse/conv/conv.py": "a1e9f62fdf70f5443967a81b4ca99c0c863d20b063b44ffe264b0fd68bf3e5e3",
    "modules/sparse/conv/conv_flex_gemm.py": "281fa1c8449732a94d99a24facb437600ab08d8185214d8434008456d56b83d8",
    "modules/sparse/spatial/spatial2channel.py": "f36819339e48a89ed368f45feb0987309de23a4fe1d7d43eb347ab30a0b9f09c",
}
FP32_KEYS = tuple(sorted(
    [f"blocks.{stage}.{block}.norm.{field}"
     for stage, count in enumerate((4, 16, 8, 4)) for block in range(count) for field in ("weight", "bias")]
    + [f"blocks.{stage}.{block}.norm1.{field}"
       for stage, block in enumerate((4, 16, 8, 4)) for field in ("weight", "bias")]
    + [f"{projection}.{field}" for projection in ("from_latent", "output_layer") for field in ("weight", "bias")]
))
AUDITED_CONFIG = {
    "name": "FlexiDualGridVaeDecoder",
    "args": {
        "resolution": 256, "model_channels": [1024, 512, 256, 128, 64], "latent_channels": 32,
        "num_blocks": [4, 16, 8, 4, 0], "block_type": ["SparseConvNeXtBlock3d"] * 5,
        "up_block_type": ["SparseResBlockC2S3d"] * 4, "block_args": [{}, {}, {}, {}, {}],
        "use_fp16": True,
    },
}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _native_import_complete():
    module = sys.modules.get("flex_gemm")
    spec = getattr(module, "__spec__", None)
    return module is not None and spec is not None and not getattr(spec, "_initializing", False)


def _import_parent(source_root):
    if "trellis2" not in sys.modules:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("trellis2.models.sc_vaes.sparse_unet_vae")
    expected = (source_root / "trellis2/models/sc_vaes/sparse_unet_vae.py").resolve(strict=True)
    cls = module.SparseUnetVaeDecoder
    if Path(inspect.getfile(cls)).resolve(strict=True) != expected:
        raise ValueError("Pinned parent constructor source identity differs")
    return cls


def _manifest(model):
    if list(model.named_buffers()):
        raise ValueError("Pinned parent constructor must have zero buffers")
    dtypes = {"torch.float16": "F16", "torch.float32": "F32"}
    return {key: {"shape": list(value.shape), "dtype": dtypes.get(str(value.dtype), str(value.dtype)),
                  "device": value.device.type} for key, value in model.named_parameters()}


def _bytes(value, torch):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


class ShapeDecoderComponent:
    def __init__(self, config_path, weights_path, source_root=None, parameter_manifest=None, *, native_gate=None):
        self.source = local_path(str(SOURCE if source_root is None else source_root), "shape decoder source").resolve(strict=True)
        if not self.source.is_dir():
            raise ValueError("Shape decoder source must be a local directory")
        if parameter_manifest is None:
            parameter_manifest = json.loads(PARAMETER_PROOF.read_text(encoding="utf-8"))["worker"]["parameters"]
        # Freeze the externally supplied metadata and admit only the exact historical contract.
        self.parameter_manifest = json.loads(json.dumps(parameter_manifest))
        if _fingerprint(self.parameter_manifest) != PARAMETER_MANIFEST_SHA256:
            raise ValueError("Pinned decoder parameter manifest differs")
        f32 = {key for key, row in self.parameter_manifest.items() if row["dtype"] == "F32"}
        if f32 != set(FP32_KEYS) or len(self.parameter_manifest) != 292:
            raise ValueError("Pinned decoder widening manifest differs")
        self.native_gate = _native_import_complete if native_gate is None else native_gate
        if not callable(self.native_gate):
            raise ValueError("Shape decoder native import gate must be callable")
        self._failed = False
        self._model = None
        self._component = StrictLocalComponent(config_path, weights_path, self._factory)
        if self._component.config != AUDITED_CONFIG:
            raise ValueError("Pinned shape decoder configuration differs")
        self._validate_checkpoint_header()
        self.diagnostics = {"weights_loaded": False, "device": "cpu", "strict": True,
                            "widened_keys": [], "roundtrip_verified": False, "model_forward_calls": 0}

    @property
    def loaded(self):
        return self._model is not None

    def _validate_checkpoint_header(self):
        tensors = self._component.metadata["tensors"]
        if set(tensors) != set(self.parameter_manifest) or any(
            tensors[key]["dtype"] != "F16" or tensors[key]["shape"] != row["shape"]
            for key, row in self.parameter_manifest.items()
        ):
            raise ValueError("Complete original F16 checkpoint shape/key/dtype contract differs")

    def _verify_source(self):
        for name, expected in SOURCE_SHA256.items():
            path = self.source / "trellis2" / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Pinned decoder source differs: " + name)

    def _factory(self, config):
        import torch
        self._verify_source()
        args = dict(config["args"])
        args.pop("resolution")
        model = _import_parent(self.source)(out_channels=7, **args)
        if _manifest(model) != self.parameter_manifest:
            raise ValueError("Actual parent constructor metadata differs from pinned manifest")
        parameters = dict(model.named_parameters())
        # Only a temporary metadata dtype normalization. No values or forward exist here.
        for key in FP32_KEYS:
            parameters[key].data = parameters[key].data.half()
        return model

    def load_cpu(self):
        if self._failed:
            raise RuntimeError("Shape decoder loading previously failed; create a new validated component")
        if self.loaded:
            return self._model
        model = parameters = parameter = original = roundtrip = None
        try:
            if json_metadata(self._component.config_path) != AUDITED_CONFIG:
                raise ValueError("Pinned shape decoder configuration changed")
            self._verify_source()
            if self.native_gate() is not True:
                raise RuntimeError("Completed genuine native FlexGEMM import is required before constructor import")
            import torch
            # The unchanged shared loader proves every original F16 byte first.
            model = self._component.load_cpu()
            parameters = dict(model.named_parameters())
            original_hashes = {}
            for key in FP32_KEYS:
                parameter = parameters[key]
                original = _bytes(parameter, torch)
                parameter.data = parameter.detach().float()
                roundtrip = _bytes(parameter.detach().half(), torch)
                if original != roundtrip:
                    raise ValueError("F16/F32 byte-exact round-trip failed: " + key)
                original_hashes[key] = hashlib.sha256(original).hexdigest()
            final = _manifest(model)
            expected = {key: dict(row, device="cpu") for key, row in self.parameter_manifest.items()}
            if final != expected or any(value.is_meta or value.device.type != "cpu" for value in parameters.values()):
                raise ValueError("Loaded CPU parent parameter contract differs after exact widening")
            model.eval().requires_grad_(False)
            self._model = model
            self.diagnostics.update(self._component.diagnostics, widened_keys=list(FP32_KEYS),
                                    roundtrip_verified=True, original_f16_widened_sha256=original_hashes,
                                    dtype_counts={"F16": 216, "F32": 76}, original_checkpoint_dtype="F16")
            return model
        except BaseException:
            self._failed = True
            self._model = None
            model = parameters = parameter = original = roundtrip = None
            self._component.unload()
            self.diagnostics["weights_loaded"] = False
            gc.collect()
            raise

    def unload(self):
        self._model = None
        self._component.unload()
        self.diagnostics["weights_loaded"] = False
        gc.collect()
