"""Read-only dependency/PE inventory and bounded import or CPU GGUF proof.

No installers, model files, GPU operations, or compiler subprocesses. Invoke
inventory with system Python; imports/gguf-cpu with the existing AMD Python.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def pe_imports(path):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0"
    machine, sections = struct.unpack_from("<HH", data, pe + 4)
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    assert struct.unpack_from("<H", data, opt)[0] == 0x20B
    table = opt + opt_size
    def offset(rva):
        for index in range(sections):
            entry = table + index * 40
            size, address, raw_size, raw = struct.unpack_from("<IIII", data, entry + 8)
            if address <= rva < address + max(size, raw_size):
                return raw + rva - address
        raise ValueError("Unmapped PE address")
    rva, size = struct.unpack_from("<II", data, opt + 112 + 8)
    dlls = []
    if rva:
        entry = offset(rva)
        while any(data[entry:entry + 20]):
            name_rva = struct.unpack_from("<I", data, entry + 12)[0]
            start = offset(name_rva)
            dlls.append(data[start:data.index(b"\0", start)].decode("ascii"))
            entry += 20
    return {"machine": hex(machine), "direct_import_dlls": dlls}


def inventory(args):
    result = {}
    wanted = {"torch", "torchvision", "cumesh", "flex-gemm", "nvdiffrast",
              "o-voxel", "gguf", "sdnq", "transformers", "triton-windows",
              "rocm", "numpy", "utils3d", "pillow", "safetensors", "accelerate"}
    for label, venv in (("modly", args.modly_venv), ("amd", args.amd_venv)):
        site = venv / "Lib" / "site-packages"
        versions = {}
        for dist in metadata.distributions(path=[str(site)]):
            name = dist.metadata["Name"].lower().replace("_", "-")
            if name in wanted:
                versions[name] = {"version": dist.version, "wheel": dist.read_text("WHEEL")}
        binaries = []
        for package in ("cumesh", "flex_gemm", "o_voxel", "nvdiffrast"):
            for path in sorted((site / package).rglob("*.pyd")):
                binaries.append({"path": str(path.relative_to(site)), **pe_imports(path)})
        for path in sorted(site.glob("*nvdiffrast*.pyd")):
            binaries.append({"path": path.name, **pe_imports(path)})
        result[label] = {"pyvenv_cfg": (venv / "pyvenv.cfg").read_text(),
                         "packages": versions, "binaries": binaries}
    paths = [args.extension_source / name for name in ("generator.py", "setup.py", "manifest.json")]
    modly_site = args.modly_venv / "Lib" / "site-packages"
    paths += [modly_site / "trellis2_gguf" / name for name in (
        "utils/gguf_utils.py", "models/__init__.py", "pipelines/trellis2_image_to_3d.py",
        "modules/sparse/config.py", "modules/sparse/attention/full_attn.py")]
    paths += [args.amd_source / name for name in ("nodes.py", "trellis2/models/__init__.py",
        "trellis2/pipelines/trellis2_image_to_3d.py", "trellis2/modules/sparse/config.py",
        "trellis2/modules/sparse/attention/full_attn.py")]
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return result


def block_external_actions():
    # CPython's Windows platform lookup may use a read-only `cmd /c ver` probe.
    # Cache that lookup before forbidding compiler/installer subprocesses.
    platform.system()
    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "socket.connect"):
            raise RuntimeError("Proof forbids external process/network action: " + event)
    sys.addaudithook(audit)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "", "FLEX_GEMM_USE_AUTOTUNE_CACHE": "0",
        "FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE": "0"})


def imports():
    block_external_actions()
    import torch
    assert torch.version.hip and torch.version.cuda is None
    names = ("cumesh._C", "cumesh._cubvh", "cumesh._xatlas",
             "flex_gemm.kernels.cuda", "o_voxel._C", "nvdiffrast.torch", "_nvdiffrast_c")
    records = []
    for name in names:
        module = importlib.import_module(name)
        records.append({"module": name, "path": module.__file__, "import": "PASS"})
    return {"python": platform.python_version(), "torch": torch.__version__,
            "hip": torch.version.hip, "cuda": torch.version.cuda,
            "gpu_initialized": torch.cuda.is_initialized(), "imports": records}


def gguf_cpu(args):
    block_external_actions()
    import numpy as np
    import torch
    # Stage only the existing pure Python GGUF code, never its CUDA site-packages.
    # This explicitly does not reconstruct a complete GGUF runtime environment.
    source = args.modly_venv / "Lib" / "site-packages"
    with tempfile.TemporaryDirectory(prefix="gguf-cpu-proof-", dir=ROOT.parent) as tmp:
        stage = Path(tmp)
        shutil.copytree(source / "gguf", stage / "gguf",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        assert not list(stage.rglob("*.pyd")) and not list(stage.rglob("*.dll"))
        shutil.copyfile(source / "trellis2_gguf/utils/gguf_utils.py", stage / "gguf_utils.py")
        sys.path.insert(0, str(stage))
        import gguf
        spec = importlib.util.spec_from_file_location("proof_gguf_utils", stage / "gguf_utils.py")
        utility = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utility)
        records = []
        for name in ("Q8_0", "Q4_K", "Q5_K", "Q6_K"):
            qtype = getattr(gguf.GGMLQuantizationType, name)
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            # Synthetic one-block data with finite half-float scales, no weights.
            data = np.arange(size, dtype=np.uint8).reshape(1, size)
            if name == "Q8_0": data[0, :2] = [0, 60]
            elif name in ("Q4_K", "Q5_K"): data[0, :4] = [0, 60, 0, 56]
            else: data[0, -2:] = [0, 60]
            tensor = torch.from_numpy(data.copy())
            actual = utility.dequantize(tensor, qtype, (1, block), dtype=torch.float32)
            reference = gguf.quants.dequantize(data, qtype)
            assert actual.device.type == "cpu"
            np.testing.assert_allclose(actual.numpy(), reference, rtol=0, atol=0)
            records.append({"qtype": name, "elements": block, "reference_match": "PASS"})
        return {"python": platform.python_version(), "torch": torch.__version__,
                "gguf_source": "staged existing pure Python gguf 0.19.0",
                "strategy": utility.GGMLOps.__name__, "gpu_initialized": torch.cuda.is_initialized(),
                "synthetic_cpu_blocks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "imports", "gguf-cpu"))
    parser.add_argument("--modly-venv", type=Path)
    parser.add_argument("--amd-venv", type=Path)
    parser.add_argument("--amd-source", type=Path)
    parser.add_argument("--extension-source", type=Path)
    args = parser.parse_args()
    result = inventory(args) if args.mode == "inventory" else imports() if args.mode == "imports" else gguf_cpu(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
