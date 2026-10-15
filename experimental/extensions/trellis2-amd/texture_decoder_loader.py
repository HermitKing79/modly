"""Genuine guided texture decoder, retaining its configured F16/F32 policy.

Import is stdlib-only. The checkpoint remains 284 original F16 tensors. The
unchanged strict loader validates every loaded payload byte before restoring
the constructor's 76 F32 norm/projection parameters by exact widening. The 208
primitive body parameters stay F16; no BF16 substitution or blanket cast exists.

``meta_factory`` proves the genuine parent schema without importing FlexGEMM,
reading weights, executing a model forward, or initializing a GPU. ``load_cpu``
is a separate boundary requiring the completed genuine native import gate.
"""
import ast
import gc
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from services.amd_runtime import local_path
from services.amd_model_metadata import json_metadata
from strict_loader import StrictLocalComponent

SOURCE = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD" if sys.platform == "win32"
              else "/mnt/d/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD")
CONFIG_SHA256 = "22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1"
HEADER_SHA256 = "ad3ed458569d937eace6ac42d9b6aa83853d654ebd0847791bbf19e2e31ab02b"
CHECKPOINT_PAYLOAD_BYTES = 948432012
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
AUDITED_CONFIG = {
    "name": "SparseUnetVaeDecoder",
    "args": {
        "out_channels": 6, "model_channels": [1024, 512, 256, 128, 64], "latent_channels": 32,
        "num_blocks": [4, 16, 8, 4, 0], "block_type": ["SparseConvNeXtBlock3d"] * 5,
        "up_block_type": ["SparseResBlockC2S3d"] * 4, "block_args": [{}, {}, {}, {}, {}],
        "pred_subdiv": False, "use_fp16": True,
    },
}
FP32_KEYS = tuple(sorted(
    [f"blocks.{stage}.{block}.norm.{field}"
     for stage, count in enumerate((4, 16, 8, 4)) for block in range(count) for field in ("weight", "bias")]
    + [f"blocks.{stage}.{block}.norm1.{field}"
       for stage, block in enumerate((4, 16, 8, 4)) for field in ("weight", "bias")]
    + [f"{projection}.{field}" for projection in ("from_latent", "output_layer") for field in ("weight", "bias")]
))


def schema(device="meta"):
    """Hand-derived from the pinned 32 ConvNeXt, four C2S and two projections."""
    rows = {}
    def add(prefix, shape, dtype):
        rows[prefix] = {"shape": list(shape), "dtype": dtype, "device": device}
    def affine(prefix, output, input_=None, dtype="F16", convolution=False):
        shape = [output] if input_ is None else ([output, 3, 3, 3, input_] if convolution else [output, input_])
        add(prefix + ".weight", shape, dtype)
        add(prefix + ".bias", [output], dtype)
    channels = (1024, 512, 256, 128, 64)
    for stage, count in enumerate((4, 16, 8, 4)):
        width = channels[stage]
        for block in range(count):
            prefix = f"blocks.{stage}.{block}."
            affine(prefix + "conv", width, width, convolution=True)
            affine(prefix + "norm", width, dtype="F32")
            affine(prefix + "mlp.0", width * 4, width)
            affine(prefix + "mlp.2", width, width * 4)
        prefix = f"blocks.{stage}.{count}."
        next_width = channels[stage + 1]
        affine(prefix + "conv1", next_width * 8, width, convolution=True)
        affine(prefix + "conv2", next_width, next_width, convolution=True)
        affine(prefix + "norm1", width, dtype="F32")
    affine("from_latent", 1024, 32, dtype="F32")
    affine("output_layer", 6, 64, dtype="F32")
    return rows


def manifest(model):
    if list(model.named_buffers()):
        raise ValueError("Pinned texture constructor must have zero buffers")
    dtypes = {"torch.float16": "F16", "torch.float32": "F32"}
    return {key: {"shape": list(value.shape), "dtype": dtypes.get(str(value.dtype), str(value.dtype)),
                  "device": value.device.type} for key, value in model.named_parameters()}


def _source(source_root=None):
    root = local_path(str(SOURCE if source_root is None else source_root), "texture decoder source").resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Texture decoder source must be a local directory")
    return root


def _verify_source(source_root):
    for name, expected in SOURCE_SHA256.items():
        path = source_root / "trellis2" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned texture decoder source differs: " + name)


def _import_parent(source_root):
    if "trellis2" not in sys.modules:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("trellis2.models.sc_vaes.sparse_unet_vae")
    expected = (source_root / "trellis2/models/sc_vaes/sparse_unet_vae.py").resolve(strict=True)
    cls = module.SparseUnetVaeDecoder
    if Path(inspect.getfile(cls)).resolve(strict=True) != expected:
        raise ValueError("Pinned texture parent constructor source identity differs")
    return cls


def _native_import_complete():
    module = sys.modules.get("flex_gemm")
    spec = getattr(module, "__spec__", None)
    return module is not None and spec is not None and not getattr(spec, "_initializing", False)


def _bytes(value, torch):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def meta_factory(config=None, source_root=None):
    """CPU-only schema proof using the unchanged genuine backend initializer AST.

    This is the existing shape preflight technique. It temporarily supplies only
    the source ``sparse_conv3d_init`` while constructing on meta. Neither native
    imports nor a sparse convolution forward are supplied. Backend state is
    restored even on failure; this helper grants no execution permission.
    """
    config = AUDITED_CONFIG if config is None else config
    if config != AUDITED_CONFIG:
        raise ValueError("Pinned texture decoder configuration differs")
    root = _source(source_root)
    _verify_source(root)
    if "flex_gemm" in sys.modules:
        raise RuntimeError("CPU metadata proof requires no native FlexGEMM import")
    import torch
    parent = _import_parent(root)
    conv = importlib.import_module("trellis2.modules.sparse.conv.conv")
    initializer = root / "trellis2/modules/sparse/conv/conv_flex_gemm.py"
    functions = [node for node in ast.parse(initializer.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name == "sparse_conv3d_init"]
    if len(functions) != 1:
        raise ValueError("Pinned texture backend constructor is ambiguous")
    namespace = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(initializer), "exec"), namespace)
    def deny(*args, **kwargs):
        raise RuntimeError("CPU metadata proof forbids native/model forward")
    prior_backend = conv.config.CONV
    absent = object()
    prior_impl = conv._backends.get("flex_gemm", absent)
    try:
        conv.config.set_conv_backend("flex_gemm")
        conv._backends["flex_gemm"] = SimpleNamespace(sparse_conv3d_init=namespace["sparse_conv3d_init"],
                                                     sparse_conv3d_forward=deny)
        with torch.device("meta"):
            model = parent(**config["args"])
        if manifest(model) != schema():
            raise ValueError("Actual texture parent constructor metadata differs")
        return model
    finally:
        conv.config.set_conv_backend(prior_backend)
        if prior_impl is absent:
            conv._backends.pop("flex_gemm", None)
        else:
            conv._backends["flex_gemm"] = prior_impl


class TextureDecoderComponent:
    def __init__(self, config_path, weights_path, source_root=None, *, native_gate=None):
        self.source = _source(source_root)
        self.native_gate = _native_import_complete if native_gate is None else native_gate
        if not callable(self.native_gate):
            raise ValueError("Texture decoder native import gate must be callable")
        self._failed = False
        self._model = None
        self._component = StrictLocalComponent(config_path, weights_path, self._factory)
        if self._component.config != AUDITED_CONFIG:
            raise ValueError("Pinned texture decoder configuration differs")
        tensors = self._component.metadata["tensors"]
        expected = schema()
        if set(tensors) != set(expected) or any(
            tensors[key]["dtype"] != "F16" or tensors[key]["shape"] != row["shape"]
            for key, row in expected.items()
        ):
            raise ValueError("Complete original texture F16 checkpoint shape/key/dtype contract differs")
        self.diagnostics = {"weights_loaded": False, "device": "cpu", "strict": True,
                            "widened_keys": [], "roundtrip_verified": False, "model_forward_calls": 0}

    @property
    def loaded(self):
        return self._model is not None

    def _verify_source(self):
        _verify_source(self.source)

    def _factory(self, config):
        self._verify_source()
        if config != AUDITED_CONFIG:
            raise ValueError("Pinned texture decoder configuration differs")
        model = _import_parent(self.source)(**config["args"])
        if manifest(model) != schema():
            raise ValueError("Actual texture parent constructor metadata differs")
        parameters = dict(model.named_parameters())
        # Metadata-only dtype normalization; no values exist before strict load.
        for key in FP32_KEYS:
            parameters[key].data = parameters[key].data.half()
        return model

    def load_cpu(self):
        if self._failed:
            raise RuntimeError("Texture decoder loading previously failed; create a new validated component")
        if self.loaded:
            return self._model
        model = parameters = parameter = original = roundtrip = None
        try:
            if json_metadata(self._component.config_path) != AUDITED_CONFIG:
                raise ValueError("Pinned texture decoder configuration changed")
            self._verify_source()
            if self.native_gate() is not True:
                raise RuntimeError("Completed genuine native FlexGEMM import is required before constructor import")
            import torch
            # Every one of the 284 original F16 payloads is proved first.
            model = self._component.load_cpu()
            parameters = dict(model.named_parameters())
            original_hashes = {}
            for key in FP32_KEYS:
                parameter = parameters[key]
                original = _bytes(parameter, torch)
                original_hashes[key] = hashlib.sha256(original).hexdigest()
                if original_hashes[key] != self._component.diagnostics["payload_sha256"][key]:
                    raise ValueError("Original texture F16 payload proof differs: " + key)
                parameter.data = parameter.detach().float()
                roundtrip = _bytes(parameter.detach().half(), torch)
                if original != roundtrip:
                    raise ValueError("Texture F16/F32 byte-exact round-trip failed: " + key)
            if manifest(model) != schema("cpu") or any(p.is_meta or p.device.type != "cpu" for p in parameters.values()):
                raise ValueError("Loaded CPU texture parameter contract differs after exact widening")
            model.eval().requires_grad_(False)
            self._model = model
            self.diagnostics.update(self._component.diagnostics, widened_keys=list(FP32_KEYS),
                                    roundtrip_verified=True, original_f16_widened_sha256=original_hashes,
                                    dtype_counts={"F16": 208, "F32": 76}, original_checkpoint_dtype="F16",
                                    runtime_policy="original source F16 primitive body; F32 norms and projections")
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
