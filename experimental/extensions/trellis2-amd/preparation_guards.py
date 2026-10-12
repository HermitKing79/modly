"""The validated headless-import boundary, reused for configuration preparation."""
from contextlib import contextmanager
import hashlib
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import sys

SOURCE_FILES = {
    "trellis2/pipelines/base.py": "1081dfc48e112d00653ae45f3ed2a3bbee3d82e3926e701c059978e519edd917",
    "trellis2/pipelines/trellis2_image_to_3d.py": "faade40274e44823e72e1659b8ced96f8815ac9e08b3abb89271077c4394982b",
    "trellis2/models/__init__.py": "07b4379d46eb68cf138bf6d36256ea34a59981906eae854f8e28a39bf4b60683",
}
VERSION_QUERY_FILES = {
    "torchvision/extension.py": "1f9990bf3b40ebe858bd781ee14b3e09863d6c43b45b86e5bc216fc3d09950c9",
    "torchvision/version.py": "539adb2889bdef71816f557c25583d555c6e9515ab1bf2655a049b6f8e9d6ed6",
    "torchvision/_C.pyd": "ee06af8d42b07377a4840a0b87ceceb72f4f174a1b1ee6f3cf429c342870a424",
    "transformers/activations.py": "7b83cb02127676a22e6af6855aa7bd6baaa5db9bb5afcdaa843a271552f27773",
    "torch/nn/modules/activation.py": "5dfca6c85e4b84a4b950abdba92507737bcfaedba3c655e97fef2cf70870a678",
}
ACTIVATIONS = {("transformers.activations", name) for name in
               ("GELUActivation", "NewGELUActivation", "FastGELUActivation", "QuickGELUActivation", "MishActivation", "LinearActivation")} | {("torch.nn.modules.activation", "SiLU")}


class PreparationGuards:
    def __init__(self, source_root):
        import torch
        self.torch = torch
        self.source = Path(source_root)
        self.factory_active = False
        self.activations, self.version_queries = [], []
        self.verify_source()
        site = Path(sys.prefix) / "Lib/site-packages"
        if (metadata.version("torchvision") != "0.28.0+rocm10.0.0" or metadata.version("transformers") != "4.56.0"
                or {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in VERSION_QUERY_FILES} != VERSION_QUERY_FILES):
            raise RuntimeError("Headless dependency definitions changed; preparation refuses unverified imports")
        if importlib.util.find_spec("mmgp") is not None:
            raise RuntimeError("The factory's optional mmgp import is unverified in this preparation runtime")
        original_init = torch.nn.Module.__init__
        def stateless_init(instance, *args, **kwargs):
            identity = (type(instance).__module__, type(instance).__name__)
            if identity not in ACTIVATIONS or args or kwargs or len(self.activations) >= 8:
                return self.forbidden()
            original_init(instance)
            self.activations.append(".".join(identity))
        torch.nn.Module.__init__ = stateless_init
        torch.nn.Module.register_parameter = self.forbidden
        torch.nn.Module.register_buffer = self.forbidden
        torch.nn.Module.__call__ = self.forbidden
        torch.load = self.forbidden
        import safetensors
        import safetensors.torch
        safetensors.safe_open = self.forbidden
        safetensors.torch.load_file = self.forbidden
        import huggingface_hub
        huggingface_hub.hf_hub_download = self.forbidden
        huggingface_hub.snapshot_download = self.forbidden
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self
        class NoTensors(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if (str(func) == "torchvision._cuda_version.default"
                        and str(func._schema) == "torchvision::_cuda_version() -> int _0" and not args and not kwargs):
                    if len(owner.version_queries) >= 2:
                        raise RuntimeError("Repeated unapproved TorchVision version query")
                    result = func(*args, **(kwargs or {}))
                    owner.version_queries.append(result)
                    if result != -1:
                        raise RuntimeError("TorchVision unexpectedly declares CUDA compilation")
                    return result
                raise RuntimeError("Geometry preparation forbids tensor operations: " + str(func))
        self.no_tensors = NoTensors
        def read_guard(event, args):
            if self.factory_active and event == "open" and isinstance(args[0], (str, bytes)):
                import os
                if Path(os.fsdecode(args[0])).suffix.lower() in (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf"):
                    raise RuntimeError("Geometry factory preparation forbids model-file reads")
        sys.addaudithook(read_guard)

    @staticmethod
    def forbidden(*args, **kwargs):
        raise RuntimeError("Geometry preparation prohibits model construction/loading, inference, compilation and launch")

    def verify_source(self):
        if {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest() for name in SOURCE_FILES} != SOURCE_FILES:
            raise RuntimeError("The inspected AMD model/base/geometry factory source changed")

    @contextmanager
    def factory_context(self):
        self.verify_source()
        self.factory_active = True
        try:
            with self.no_tensors():
                import triton
                triton.compile = self.forbidden
                triton.runtime.JITFunction.run = self.forbidden
                triton.runtime.Autotuner.run = self.forbidden
                yield
        finally:
            self.factory_active = False

    def gpu_diagnostics(self):
        torch = self.torch
        initialized = torch.cuda.is_initialized()
        result = {"initialized": initialized, "allocated_bytes": 0, "reserved_bytes": 0}
        if initialized:
            properties = torch.cuda.get_device_properties(0)
            if "RX 9070 XT" not in properties.name or properties.gcnArchName.split(":")[0] != "gfx1201":
                raise RuntimeError("Unexpected GPU identity during preparation")
            result.update(name=properties.name, architecture=properties.gcnArchName,
                          allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
        if result["allocated_bytes"] or result["reserved_bytes"]:
            raise RuntimeError("Geometry preparation unexpectedly allocated GPU tensors")
        return result
