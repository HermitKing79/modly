"""Previously inspected SDK discovery and offline/filesystem guards, reused by preparation workers."""
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

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
        scratch.relative_to(Path(os.environ["WORKSPACE_DIR"]).resolve())

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


