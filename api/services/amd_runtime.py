"""Opt-in, offline contract for the experimental AMD mock worker.

This module has no torch/native imports at import time. Nothing here installs,
discovers a replacement interpreter, loads weights, or selects a fallback device.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import struct
import sys

CONTRACT = "amd-offline-scaffold-v1"
PREPARATION_MODE = "geometry-preparation-alpha-only"
SOURCE_COMMIT = "3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5"
VERSIONS = {"torch": "2.13.0+rocm10.0.0", "rocm": "10.0.0",
            "cumesh": "1.0+rocm10.0", "flex-gemm": "1.0.0+rocm10.0",
            "nvdiffrast": "0.4.0+rocm10.0", "o-voxel": "0.0.1+rocm.10.0",
            "triton-windows": "3.8.0.post29"}
NATIVES = ("cumesh/_C", "cumesh/_cubvh", "cumesh/_xatlas",
           "flex_gemm/kernels/cuda", "o_voxel/_C", "_nvdiffrast_c")


class PreparationValidationError(ValueError):
    """A rejected input/configuration before real import; safe to correct in place."""


def is_amd(manifest):
    contract = manifest.get("runtime_contract")
    if contract is None:
        return False
    if contract != CONTRACT or manifest.get("experimental") is not True:
        raise ValueError(f"Unsupported experimental runtime contract: {contract!r}")
    return True


def local_path(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an explicit absolute local path")
    if value.replace("\\", "/").startswith("//"):
        raise ValueError(f"{label} must be local; network and device paths are prohibited")
    path = Path(os.path.abspath(value))
    if sys.platform == "win32":
        import ctypes
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(path.anchor))
        if drive_type not in (2, 3, 5, 6):
            raise ValueError(f"{label} must be on a local volume; mapped network drives are prohibited")
        # Refuse reparse points before resolving them, including junctions to
        # remote volumes. This is intentionally narrower than general Modly IO.
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                break
            if getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"{label} must be local without reparse points: {current}")
    return path


def absolute_file(value, label):
    path = local_path(value, label).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def runtime_config(manifest):
    if not is_amd(manifest):
        raise ValueError("AMD runtime contract required")
    if os.environ.get("MODLY_ENABLE_EXPERIMENTAL_AMD") != "1":
        raise ValueError("AMD scaffold requires opt-in MODLY_ENABLE_EXPERIMENTAL_AMD=1")
    path = absolute_file(os.environ.get("MODLY_AMD_RUNTIME_CONFIG"), "MODLY_AMD_RUNTIME_CONFIG")
    config = json.loads(path.read_text(encoding="utf-8"))
    preparing = isinstance(config, dict) and config.get("mode") == PREPARATION_MODE
    expected_keys = {"python", "source_root", "mode", "geometry"} if preparing else {"python", "source_root", "models"}
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError("AMD config requires exactly " + ", ".join(sorted(expected_keys)))
    config["python"] = str(absolute_file(config["python"], "AMD Python"))
    source = local_path(config["source_root"], "AMD source_root")
    if not source.is_dir():
        raise ValueError("AMD source_root must be an existing absolute local directory")
    config["source_root"] = str(source.resolve())
    if preparing:
        if not isinstance(config["geometry"], dict) or set(config["geometry"]) != {"models_root", "pipeline_dir", "dino_dir"}:
            raise ValueError("AMD geometry requires explicit models_root, pipeline_dir and dino_dir")
        config["geometry"] = {key: str(local_path(value, key).resolve(strict=True)) for key, value in config["geometry"].items()}
        return config
    if not isinstance(config["models"], dict) or set(config["models"]) != {"generate", "refine"}:
        raise ValueError("AMD models must explicitly declare generate and refine checkpoint lists")
    return config


def validate_models(models, node):
    """Read bounded configuration/header metadata only; never read tensor data."""
    prefixes = models.get(node)
    if node not in ("generate", "refine") or not isinstance(prefixes, list) or not prefixes:
        raise ValueError(f"Offline safetensors checkpoint prefixes required for {node}")
    result = []
    for value in prefixes:
        if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).suffix:
            raise ValueError("Use an absolute local checkpoint prefix without a suffix; GGUF is unsupported")
        prefix = local_path(value, "checkpoint prefix")
        try:
            config = absolute_file(value + ".json", "checkpoint JSON")
            weights = absolute_file(value + ".safetensors", "checkpoint safetensors")
            if config.stat().st_size > 1024 * 1024:
                raise ValueError("Checkpoint JSON exceeds scaffold metadata limit")
            description = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(description, dict) or not isinstance(description.get("name"), str):
                raise ValueError("Checkpoint JSON requires model class name")
            with weights.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= length <= min(1024 * 1024, weights.stat().st_size - 8):
                    raise ValueError("Invalid or oversized safetensors header")
                header = json.loads(stream.read(length))
            if not isinstance(header, dict):
                raise ValueError("Invalid safetensors header object")
        except (OSError, struct.error, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Offline safetensors pair unavailable or invalid at {prefix}: {exc}") from exc
        result.append({"prefix": str(prefix.resolve()), "config": str(config), "weights": str(weights),
                       "class": description["name"], "header_only": True})
    return result


def pe_imports(path, cpu_utility=False):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe + 4)[0] != 0x8664:
        raise ValueError(f"Native binary is not Windows AMD64 PE: {path}")
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    if struct.unpack_from("<H", data, opt)[0] != 0x20B:
        raise ValueError(f"Native binary is not PE32+: {path}")
    table = opt + struct.unpack_from("<H", data, pe + 20)[0]
    def offset(rva):
        for index in range(sections):
            size, address, raw_size, raw = struct.unpack_from("<IIII", data, table + index * 40 + 8)
            if address <= rva < address + max(size, raw_size):
                return raw + rva - address
        raise ValueError("Unmapped native PE address")
    rva = struct.unpack_from("<I", data, opt + 120)[0]
    dlls = []
    if rva:
        entry = offset(rva)
        while any(data[entry:entry + 20]):
            start = offset(struct.unpack_from("<I", data, entry + 12)[0])
            dlls.append(data[start:data.index(b"\0", start)].decode("ascii").lower())
            entry += 20
    digest = hashlib.sha256(data).hexdigest()
    # The already-inventoried xatlas UV utility is a CPU-only component of
    # this AMD wheel, not a CPU inference fallback. Pin that specific binary.
    utility_verified = cpu_utility and digest == "66da37345044e2161f7811f2d9ff67750f8fb3c203c16bc5abdfd6e739111e00"
    if ("amdhip64_7.dll" not in dlls and not utility_verified) or \
            "python312.dll" not in dlls or any(
                token in name for name in dlls for token in
                ("cuda", "nvrtc", "cublas", "cudnn", "cufft", "curand", "cusparse", "cusolver")):
        raise ValueError(f"Native binary lacks verified HIP provenance or imports CUDA: {path}: {dlls}")
    return {"path": str(path), "dlls": dlls, "sha256": digest, "cpu_uv_utility": utility_verified}


def source_identity(root):
    git = root / ".git"
    if not git.is_dir():
        raise ValueError("AMD source must be the explicitly selected existing Git checkout")
    head = (git / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        if (git / ref).is_file():
            head = (git / ref).read_text().strip()
        else:
            head = next((row.split()[0] for row in (git / "packed-refs").read_text().splitlines()
                         if row.endswith(" " + ref)), "")
    if head != SOURCE_COMMIT:
        raise ValueError(f"AMD source commit changed: expected {SOURCE_COMMIT}, got {head}")
    names = ("trellis2/models/__init__.py", "trellis2/pipelines/trellis2_image_to_3d.py",
             "trellis2/pipelines/trellis2_texturing.py")
    return {"root": str(root), "commit": head,
            "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            "upstream_imported": False}


def worker_preflight(config, model_id):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12):
        raise ValueError("AMD scaffold requires Windows Python 3.12; no interpreter fallback")
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise ValueError("AMD interpreter differs from explicit runtime config")
    site = Path(sys.prefix).resolve() / "Lib/site-packages"
    versions = {}
    for name, expected in VERSIONS.items():
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"AMD dependency missing: {name}; installation is disabled") from exc
        Path(dist.locate_file("")).resolve().relative_to(site)
        if dist.version != expected:
            raise ValueError(f"AMD dependency {name}: expected {expected}, found {dist.version}; no repair")
        versions[name] = dist.version
    for name in ("safetensors", "numpy", "Pillow"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"AMD dependency missing: {name}; installation is disabled") from exc
        Path(dist.locate_file("")).resolve().relative_to(site)
        versions[name] = dist.version
    binaries = []
    wrappers = {}
    for name in ("cumesh/__init__.py", "flex_gemm/__init__.py", "o_voxel/__init__.py",
                 "nvdiffrast/torch/__init__.py", "triton/__init__.py"):
        path = (site / name).resolve(strict=True)
        path.relative_to(site)
        wrappers[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in NATIVES:
        path = (site / (name + ".cp312-win_amd64.pyd")).resolve(strict=True)
        path.relative_to(site)
        binaries.append(pe_imports(path, cpu_utility=name == "cumesh/_xatlas"))
    source = source_identity(Path(config["source_root"]))
    import torch
    Path(torch.__file__).resolve().relative_to(site)
    if torch.__version__ != VERSIONS["torch"] or not torch.version.hip or torch.version.cuda is not None:
        raise ValueError("PyTorch must be the selected ROCm build; CUDA/CPU fallback is prohibited")
    if torch.cuda.is_initialized():
        raise ValueError("Scaffold preflight unexpectedly initialized a GPU context")
    # Every tensor operation is forbidden, including CPU allocations. Native
    # imports and GPU discovery are intentionally deferred to another milestone.
    def no_gpu(*args, **kwargs):
        raise RuntimeError("AMD scaffold forbids GPU initialization")
    preparing = config.get("mode") == PREPARATION_MODE
    if not preparing:
        torch.cuda._lazy_init = no_gpu
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoTensors(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise RuntimeError("AMD scaffold forbids tensor operations")
    mode = None if preparing else NoTensors()
    if mode is not None:
        mode.__enter__()
    return {"contract": CONTRACT, "model_id": model_id, "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0], "prefix": sys.prefix,
            "torch": torch.__version__, "torch_path": torch.__file__, "hip": torch.version.hip,
            "cuda": torch.version.cuda, "packages": versions, "native_binaries": binaries,
            "native_wrapper_sha256": wrappers,
            "source": source, "format": "safetensors", "mock_only": not preparing,
            "preparation_only": preparing, "runtime_mode": PREPARATION_MODE if preparing else "mock",
            "gpu_initialized": False, "weights_loaded": False}, mode


def verify_identity(identity, manifest, config):
    if not isinstance(identity, dict):
        raise ValueError("AMD worker identity missing")
    preparing = config.get("mode") == PREPARATION_MODE
    checks = {"contract": CONTRACT, "model_id": manifest["id"], "python": config["python"],
              "torch": VERSIONS["torch"], "cuda": None, "mock_only": not preparing,
              "gpu_initialized": False, "weights_loaded": False, "format": "safetensors"}
    if preparing:
        checks.update(preparation_only=True, runtime_mode=PREPARATION_MODE)
    if any(identity.get(k) != v or k not in identity for k, v in checks.items()) or \
            not identity.get("python_version", "").startswith("3.12.") or not identity.get("hip"):
        raise ValueError("AMD worker identity does not match the requested offline ROCm runtime/node")
