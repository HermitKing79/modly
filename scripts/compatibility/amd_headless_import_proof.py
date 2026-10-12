"""One bounded real import per route, plus CPU alpha preprocessing. No weights.

Metadata GPU initialization is observed; torch tensor operations, model
construction, model-file reads, network, compilation and launches are blocked.
"""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD")
MODELS = Path("D:/AI/Trellis2AMD/ComfyUI/models")
VENV = Path("D:/AI/Trellis2AMD/venv")
MARKER = "HEADLESS_RESULT="
VERSION_QUERY_FILES = {
    "torchvision/extension.py": "1f9990bf3b40ebe858bd781ee14b3e09863d6c43b45b86e5bc216fc3d09950c9",
    "torchvision/version.py": "539adb2889bdef71816f557c25583d555c6e9515ab1bf2655a049b6f8e9d6ed6",
    "torchvision/_C.pyd": "ee06af8d42b07377a4840a0b87ceceb72f4f174a1b1ee6f3cf429c342870a424",
}
STATELESS_SOURCE_FILES = {
    "transformers/activations.py": "7b83cb02127676a22e6af6855aa7bd6baaa5db9bb5afcdaa843a271552f27773",
    "torch/nn/modules/activation.py": "5dfca6c85e4b84a4b950abdba92507737bcfaedba3c655e97fef2cf70870a678",
}
STATELESS_ACTIVATIONS = {
    ("transformers.activations", name) for name in
    ("GELUActivation", "NewGELUActivation", "FastGELUActivation", "QuickGELUActivation", "MishActivation", "LinearActivation")
} | {("torch.nn.modules.activation", "SiLU")}
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "experimental/extensions/trellis2-amd"))


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_workers(healthy, workers):
    return "PASS" if (healthy is True and [w.get("route") for w in workers] == ["generate", "refine"]
                     and all(w.get("status") == "PASS" and w.get("process", {}).get("returncode") == 0
                             and w.get("process", {}).get("timed_out") is False for w in workers)) else "PARTIAL"


def is_version_query(name, schema, args, kwargs):
    return (name == "torchvision._cuda_version.default" and schema == "torchvision::_cuda_version() -> int _0"
            and not args and not kwargs)


def worker(route, scratch):
    record = {"route": route, "status": "FAIL", "executable": sys.executable,
              "python": platform.python_version(), "sdk_discovery": [],
              "weights_loaded": False, "inference_executed": False}
    torch = None
    try:
        scratch = scratch.resolve()
        scratch.relative_to(ROOT)
        gpu = sibling("amd_bounded_gpu_proof")
        env = gpu.scratch_environment(scratch)
        for key in gpu.CACHE_KEYS:
            if key != "FLEX_GEMM_AUTOTUNE_CACHE_PATH":
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        audit = gpu.preflight.load_audit()
        record["sdk_verification"] = audit.block_external_actions(
            sdk_executable=VENV / "Scripts/rocm-sdk.exe",
            discovery_records=record["sdk_discovery"], scratch_dir=scratch)
        os.environ.update(env)  # Real metadata queries, without hiding the GPU.
        def weight_guard(event, args):
            if event.startswith("socket."):
                raise RuntimeError("Headless proof prohibits network: " + event)
            if event == "open" and isinstance(args[0], (str, bytes)):
                if Path(os.fsdecode(args[0])).suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"):
                    raise RuntimeError("Headless proof prohibits model-file access: " + str(args[0]))
        sys.addaudithook(weight_guard)
        import torch
        from services import amd_runtime
        site = Path(sys.prefix) / "Lib/site-packages"
        Path(sys.executable).resolve().relative_to(VENV)
        Path(torch.__file__).resolve().relative_to(site)
        record.update(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
                      torch_path=torch.__file__, gpu_initialized_before=torch.cuda.is_initialized())
        if (record["python"] != "3.12.10" or torch.__version__ != amd_runtime.VERSIONS["torch"]
                or not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized()):
            raise RuntimeError("Unexpected AMD runtime or pre-existing GPU context")
        record["source_identity"] = amd_runtime.source_identity(SOURCE)
        record["source_identity"].pop("upstream_imported", None)
        record["dependencies"] = {}
        for name, expected in amd_runtime.VERSIONS.items():
            dist = metadata.distribution(name)
            Path(dist.locate_file("")).resolve().relative_to(site)
            if dist.version != expected:
                raise RuntimeError("Incompatible AMD package: " + name)
        record["dependencies"] = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
        record["native_provenance"] = [amd_runtime.pe_imports(
            site / (name + ".cp312-win_amd64.pyd"), cpu_utility=name == "cumesh/_xatlas")
            for name in amd_runtime.NATIVES]
        record["torchvision_version_query"] = []
        actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in VERSION_QUERY_FILES}
        if actual != VERSION_QUERY_FILES or metadata.version("torchvision") != "0.28.0+rocm10.0.0":
            raise RuntimeError("TorchVision version-query implementation differs from inspected code")
        record["torchvision_query_source_sha256"] = actual
        actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in STATELESS_SOURCE_FILES}
        if actual != STATELESS_SOURCE_FILES or metadata.version("transformers") != "4.56.0":
            raise RuntimeError("Stateless activation definitions differ from inspected code")
        record["stateless_source_sha256"] = actual
        record["stateless_activation_objects"] = []
        def forbidden(*args, **kwargs):
            raise RuntimeError("Headless proof prohibits model construction/loading, inference, compilation or launch")
        torch.load = forbidden
        original_module_init = torch.nn.Module.__init__
        def stateless_init(self, *args, **kwargs):
            identity = (type(self).__module__, type(self).__name__)
            if identity not in STATELESS_ACTIVATIONS or args or kwargs or len(record["stateless_activation_objects"]) >= 8:
                return forbidden()
            original_module_init(self)
            record["stateless_activation_objects"].append(".".join(identity))
        torch.nn.Module.__init__ = stateless_init
        torch.nn.Module.register_parameter = forbidden
        torch.nn.Module.register_buffer = forbidden
        torch.nn.Module.__call__ = forbidden
        import safetensors
        safetensors.safe_open = forbidden
        import safetensors.torch
        safetensors.torch.load_file = forbidden
        import huggingface_hub
        huggingface_hub.hf_hub_download = forbidden
        huggingface_hub.snapshot_download = forbidden
        from torch.utils._python_dispatch import TorchDispatchMode
        class NoTensors(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if is_version_query(str(func), str(func._schema), args, kwargs):
                    if len(record["torchvision_version_query"]) >= 2:
                        raise RuntimeError("Unexpected repeated TorchVision version query")
                    value = func(*args, **(kwargs or {}))
                    record["torchvision_version_query"].append({"schema": str(func._schema), "result": value})
                    if value != -1:
                        raise RuntimeError("TorchVision unexpectedly declares CUDA compilation")
                    return value
                raise RuntimeError("Headless proof prohibits tensor operations: " + str(func))
        with NoTensors():
            import triton
            triton.compile = forbidden
            triton.runtime.JITFunction.run = forbidden
            triton.runtime.Autotuner.run = forbidden
            from headless import import_pipeline_class
            print("HEADLESS_STAGE=import-" + route, flush=True)
            cls = import_pipeline_class(SOURCE, MODELS, route)
            record.update(class_name=cls.__name__, module=cls.__module__, path=inspect.getfile(cls))
            Path(record["path"]).resolve().relative_to(SOURCE)
            properties = torch.cuda.get_device_properties(0)
            if "RX 9070 XT" not in properties.name:
                raise RuntimeError("GPU identity differs from RX 9070 XT")
            record["gpu"] = {"name": properties.name, "architecture": getattr(properties, "gcnArchName", None),
                             "total_memory_bytes": properties.total_memory}
            record["allocated_bytes"] = torch.cuda.memory_allocated(0)
            record["reserved_bytes"] = torch.cuda.memory_reserved(0)
            if record["allocated_bytes"] or record["reserved_bytes"]:
                raise RuntimeError("Import unexpectedly allocated GPU tensor memory")
            from alpha_input import preprocess_alpha_image
            from PIL import Image
            # Empty model dictionary exercises only the real pipeline constructor;
            # no fake model/extractor or dummy successful inference is supplied.
            pipeline = cls(models={}, rembg_model=None)
            samples = []
            for size in ((16, 12), (1200, 1080)):
                image = Image.new("RGBA", size, (37, 121, 203, 0))
                for y in range(size[1] // 4, 3 * size[1] // 4):
                    for x in range(size[0] // 4, 3 * size[0] // 4):
                        image.putpixel((x, y), (37, 121, 203, 255))
                image.putpixel((size[0] // 4, size[1] // 4), (200, 100, 50, 128))
                baseline = pipeline.preprocess_image(image)
                actual = preprocess_alpha_image(pipeline, image)
                if actual.mode != "RGB" or actual.size != baseline.size or actual.tobytes() != baseline.tobytes():
                    raise RuntimeError("Alpha-only preprocessing differs from real upstream conditioning input")
                samples.append({"input_size": list(size), "output_size": list(actual.size),
                                "output_mode": actual.mode, "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                                "upstream_pixel_equal": True, "rembg_model": None})
            record["alpha_preprocessing"] = samples
            record["model_slots"] = pipeline.models
        source_modules, native_modules = {}, {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            if name.startswith("trellis2"):
                path = Path(path).resolve()
                path.relative_to(SOURCE)
                source_modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if name.startswith(("cumesh", "flex_gemm", "o_voxel", "nvdiffrast", "_nvdiffrast")) and path.endswith(".pyd"):
                path = Path(path).resolve()
                path.relative_to(site)
                native_modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record.update(source_modules=source_modules, native_modules=native_modules,
                      source_nodes_imported="nodes" in sys.modules, status="PASS")
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if torch is not None:
            record["gpu_initialized_after"] = torch.cuda.is_initialized()
        print(MARKER + json.dumps(record), flush=True)
    return 0 if record["status"] == "PASS" else 1


def supervise(output):
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise RuntimeError("No automatic retries or evidence overwrite; select a new reviewed output directory")
    output.mkdir(parents=True, exist_ok=False)
    gpu = sibling("amd_bounded_gpu_proof")
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    observation = gpu.windows_observation(start)
    save("windows-before.json", observation)
    healthy = gpu.healthy_observation(observation)
    workers = []
    for route in ("generate", "refine") if healthy else ():
        scratch = output / ("scratch-" + route)
        scratch.mkdir(exist_ok=False)
        command = [str(VENV / "Scripts/python.exe"), "-B", "-I", str(Path(__file__).resolve()),
                   "--worker", route, "--scratch", str(scratch)]
        capture = gpu.preflight.run_child(command, 30)
        for stream in ("stdout", "stderr"):
            (output / (route + "." + stream + ".txt")).write_bytes(capture.pop(stream).encode("utf-8"))
        lines = (output / (route + ".stdout.txt")).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line[len(MARKER):]) for line in lines if line.startswith(MARKER)]
        record = records[-1] if records else {"route": route, "status": "FAIL", "error": "Worker exited without result"}
        record.update(command=command, process=capture, timeout_seconds=30)
        workers.append(record)
        observation = gpu.windows_observation(start)
        save(route + ".windows-after.json", observation)
        healthy = healthy and gpu.healthy_observation(observation)
        print(route + ": " + record["status"], flush=True)
        if not healthy or record["status"] != "PASS" or capture["returncode"] != 0 or capture["timed_out"]:
            break
    result = {"classification": classify_workers(healthy, workers),
              "observer_healthy": healthy, "workers": workers, "start": start}
    save("results.json", result)
    print(json.dumps({"classification": result["classification"], "observer_healthy": healthy, "workers": len(workers)}))
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("generate", "refine"))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker, args.scratch) if args.worker else supervise(args.output))
