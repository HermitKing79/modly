"""Seven isolated import-only workers; no adapters, models, or GPU operations.

The supervisor uses system Python. Each worker uses the explicitly selected
existing AMD venv. SDK discovery and native processes have separate timeouts.
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
import subprocess
import sys
import tempfile
import time
import traceback

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
NAMES = ("cumesh._C", "cumesh._cubvh", "cumesh._xatlas",
         "flex_gemm.kernels.cuda", "o_voxel._C", "nvdiffrast.torch", "_nvdiffrast_c")
MARKER = "PREFLIGHT_RESULT="


def load_audit():
    spec = importlib.util.spec_from_file_location("proof_audit", SCRIPT.with_name("amd_trellis2_audit.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    return audit


def worker(name, scratch_dir):
    record = {"module": name, "path": None, "status": "FAIL",
              "python": platform.python_version(), "executable": sys.executable,
              "prefix": sys.prefix, "torch": None, "hip": None, "cuda": None,
              "gpu_initialized_before_import": None, "gpu_initialized_after_import": None,
              "sdk_discovery": []}
    torch = None
    try:
        if scratch_dir:
            os.environ["TMP"] = os.environ["TEMP"] = str(scratch_dir.resolve())
            record["scratch_dir"] = str(scratch_dir.resolve())
        audit = load_audit()
        record["sdk_verification"] = audit.block_external_actions(
            sdk_executable=Path(sys.prefix) / "Scripts/rocm-sdk.exe",
            discovery_records=record["sdk_discovery"], scratch_dir=scratch_dir)
        import torch
        record.update(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
                      gpu_initialized_before_import=torch.cuda.is_initialized())
        if not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized():
            raise RuntimeError("Worker requires uninitialized ROCm torch")
        def forbid_gpu(*args, **kwargs):
            raise RuntimeError("Native preflight prohibits PyTorch GPU initialization")
        torch.cuda._lazy_init = forbid_gpu
        torch.cuda.init = forbid_gpu
        module = importlib.import_module(name)
        record["path"] = module.__file__
        Path(module.__file__).resolve().relative_to(Path(sys.prefix) / "Lib/site-packages")
        if torch.cuda.is_initialized():
            raise RuntimeError("Import initialized a GPU context")
        record["status"] = "PASS"
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if torch is not None:
            record["gpu_initialized_after_import"] = torch.cuda.is_initialized()
        print(MARKER + json.dumps(record), flush=True)
    return 0 if record["status"] == "PASS" else 1


def dependency_snapshot(args):
    audit = load_audit()
    result = audit.inventory(args)
    for label, venv in (("amd", args.amd_venv), ("modly", args.modly_venv)):
        site = venv / "Lib/site-packages"
        result[label]["all_distributions"] = sorted([
            {"name": dist.metadata["Name"], "version": dist.version,
             "metadata_sha256": hashlib.sha256((dist.read_text("METADATA") or "").encode()).hexdigest(),
             "record_sha256": hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()}
            for dist in metadata.distributions(path=[str(site)])], key=lambda item: item["name"].lower())
        for binary in result[label]["binaries"]:
            with (site / binary["path"]).open("rb") as binary_file:
                binary["sha256"] = hashlib.file_digest(binary_file, "sha256").hexdigest()
    extra = [args.amd_source / "nodes.py.before-profiler-fix",
             args.amd_venv / "Scripts/rocm-sdk.exe"]
    extra += [args.amd_venv / "Lib/site-packages" / name for name in audit.SDK_SOURCE_HASHES]
    result["preflight_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extra}
    return result


def run_child(command, timeout):
    start = time.monotonic()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        return {"returncode": result.returncode, "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"), "timed_out": False,
                "elapsed_seconds": round(time.monotonic() - start, 3)}
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for its child before raising.
        return {"returncode": None, "timed_out": True,
                "stdout": (exc.stdout or b"").decode("utf-8", errors="replace"),
                "stderr": (exc.stderr or b"").decode("utf-8", errors="replace"),
                "error": str(exc), "elapsed_seconds": round(time.monotonic() - start, 3)}


def supervise(args):
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    before = dependency_snapshot(args)
    (output / "inventory-before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    records = []
    for index, name in enumerate(NAMES, 1):
        with tempfile.TemporaryDirectory(prefix="import-scratch-", dir=output) as scratch:
            command = [str(args.amd_venv / "Scripts/python.exe"), "-B", "-I", str(SCRIPT),
                       "--worker", name, "--scratch", scratch]
            capture = run_child(command, 30)
        basename = f"{index:02d}-{name.replace('.', '_')}"
        for stream in ("stdout", "stderr"):
            (output / (basename + "." + stream + ".txt")).write_bytes(capture[stream].encode("utf-8"))
        lines = [line[len(MARKER):] for line in capture["stdout"].splitlines() if line.startswith(MARKER)]
        if lines:
            record = json.loads(lines[-1])
        else:
            record = {"module": name, "path": None, "status": "FAIL",
                      "error_type": "WorkerTimeout" if capture["timed_out"] else "WorkerExit",
                      "error": capture.get("error", "Worker exited without a diagnostic record")}
        record.update(worker_returncode=capture["returncode"], timed_out=capture["timed_out"],
                      elapsed_seconds=capture["elapsed_seconds"], worker_timeout_seconds=30,
                      stdout_file=basename + ".stdout.txt", stderr_file=basename + ".stderr.txt")
        records.append(record)
        print(name + ": " + record["status"], flush=True)
    after = dependency_snapshot(args)
    (output / "inventory-after.json").write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    unchanged = before == after
    passed = all(r["status"] == "PASS" and r["worker_returncode"] == 0 and
                 r["gpu_initialized_after_import"] is False for r in records)
    result = {"classification": "PASS" if passed and unchanged else "PARTIAL",
              "inventory_unchanged": unchanged, "imports": records,
              "scope": "Native imports only; no GPU stability, geometry or texturing validation"}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Classification: " + result["classification"] + "; inventory unchanged: " + str(unchanged))
    return 0 if passed and unchanged else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=NAMES)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--amd-venv", type=Path)
    parser.add_argument("--modly-venv", type=Path)
    parser.add_argument("--amd-source", type=Path)
    parser.add_argument("--extension-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return worker(args.worker, args.scratch) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
