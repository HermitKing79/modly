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
import subprocess
import sys
import tempfile
import time

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


SDK_SOURCE_HASHES = {
    "rocm_sdk/__main__.py": "da3cf689993144403ebd20543c26d2aae20b0298acb81f590df9e78997c9fbf2",
    "rocm_sdk/_devel.py": "608314f7e710bc44e93e3f8b5c729b5072656d15b463ff98123b826d347f1af6",
    "rocm_sdk/_dist_info.py": "de0065098100898af4ffc558f0d54190d89923840bc074c4c194673d0915206d",
    "rocm_sdk/__init__.py": "ed87e36cc153b7811133fb934ff1f018114c394f32a56aa040663ef66c147c3d",
    "triton/backends/amd/driver.py": "048f3915c6942e64c7c84eb47c62f265ad10e6be4584940fe0530f5a630b23f5",
}
SDK_EXE_HASH = "95ba504bd187d810fba48cf25f094629d1e17dc67e384706ffb2621f2b869e4c"


def verify_sdk_discovery(executable):
    """Only this inspected missing-devel state can execute path --root safely.

    get_devel_root can otherwise expand files and rewrite package RECORDs,
    even for an already-expanded developer package. Refuse that entire state.
    """
    executable = Path(executable).resolve()
    if executable != (Path(sys.prefix) / "Scripts/rocm-sdk.exe").resolve():
        raise RuntimeError("SDK executable must belong to this worker's venv")
    site = Path(sys.prefix) / "Lib/site-packages"
    actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest()
              for name in SDK_SOURCE_HASHES}
    if actual != SDK_SOURCE_HASHES or hashlib.sha256(executable.read_bytes()).hexdigest() != SDK_EXE_HASH:
        raise RuntimeError("SDK discovery source/executable differs from inspected code")
    if importlib.util.find_spec("rocm_sdk_devel") is not None:
        raise RuntimeError("SDK path discovery may modify an installed developer package")
    return {"executable": str(executable), "executable_sha256": SDK_EXE_HASH,
            "source_sha256": actual, "developer_package_present": False,
            "safe_branch": "ModuleNotFoundError before expansion/link reconciliation"}


def block_external_actions(sdk_executable=None, discovery_records=None, scratch_dir=None):
    if sdk_executable and sys.platform == "win32":
        # Seed the stdlib platform cache from the Windows API. Otherwise its
        # fallback can run `cmd /c ver` before/inside native imports. This
        # preflight allows no platform-discovery subprocess besides the SDK CLI.
        version = sys.getwindowsversion()
        platform._uname_cache = platform.uname_result(
            system="Windows", node=os.environ.get("COMPUTERNAME", ""),
            release=str(version.major),
            version=f"{version.major}.{version.minor}.{version.build}",
            machine=os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"))
    else:
        # Retain the historical non-native proof's behavior.
        platform.system()
    sdk = verify_sdk_discovery(sdk_executable) if sdk_executable else None
    scratch = Path(scratch_dir).resolve() if scratch_dir else None
    if scratch:
        scratch.relative_to(ROOT)

    def scratch_paths_only(paths):
        if not scratch:
            return False
        try:
            for path in paths:
                if isinstance(path, int):
                    return False
                Path(os.fsdecode(path)).resolve().relative_to(scratch)
            return True
        except (TypeError, ValueError, OSError):
            return False
    sdk_active = False
    original_check_output = subprocess.check_output
    original_run = subprocess.run
    records = discovery_records if discovery_records is not None else []

    def checked_output(command, *positional, **kwargs):
        nonlocal sdk_active
        requests = (["rocm-sdk", "path", "--root"],
                    [sdk["executable"], "path", "--root"]) if sdk else ()
        if not isinstance(command, (list, tuple)) or list(command) not in requests:
            return original_check_output(command, *positional, **kwargs)
        if positional or kwargs.get("shell", False) or set(kwargs) - {"stderr", "timeout", "shell"}:
            raise RuntimeError("Only the verified SDK discovery arguments are allowed")
        timeout = kwargs.get("timeout")
        timeout = 5.0 if timeout is None else min(float(timeout), 5.0)
        if not 0 < timeout <= 5:
            raise RuntimeError("SDK discovery timeout must be positive and at most five seconds")
        verify_sdk_discovery(sdk["executable"])
        command = [sdk["executable"], "path", "--root"]
        child_env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            child_env.pop(key, None)
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        entry = {"command": command, "timeout_seconds": timeout}
        records.append(entry)
        start = time.monotonic()
        try:
            sdk_active = True
            result = original_run(command, executable=sdk["executable"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout, env=child_env, shell=False)
            entry.update(returncode=result.returncode,
                         stdout=result.stdout.decode(errors="replace"),
                         stderr=result.stderr.decode(errors="replace"))
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command,
                                                    result.stdout, result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired as exc:
            entry.update(error="TimeoutExpired", stdout=(exc.stdout or b"").decode(errors="replace"),
                         stderr=(exc.stderr or b"").decode(errors="replace"))
            raise
        finally:
            sdk_active = False
            entry["elapsed_seconds"] = round(time.monotonic() - start, 3)

    def audit(event, args):
        if event == "subprocess.Popen" and sdk and sdk_active:
            expected = subprocess.list2cmdline([sdk["executable"], "path", "--root"])
            if args[0] == sdk["executable"] and args[1] == expected:
                return
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn",
                     "socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Proof forbids external process/network action: " + event)
        if sdk and event == "open":
            mode, flags = args[1], args[2]
            writing = (mode and any(char in mode for char in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing and not scratch_paths_only([args[0]]):
                raise RuntimeError("Native import proof forbids file writes: " + str(args[0]))
        if sdk and event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename",
                             "os.link", "os.symlink", "os.truncate", "os.chmod"):
            paths = args[:2] if event in ("os.rename", "os.link", "os.symlink") else args[:1]
            if not scratch_paths_only(paths):
                raise RuntimeError("Native import proof forbids filesystem mutation: " + event)
    sys.addaudithook(audit)
    if sdk:
        subprocess.check_output = checked_output
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "", "FLEX_GEMM_USE_AUTOTUNE_CACHE": "0",
        "FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE": "0"})
    return sdk


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
