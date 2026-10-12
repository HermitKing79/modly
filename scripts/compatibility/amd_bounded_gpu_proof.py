"""Single-shot AMD GPU proof. No models, downloads, builds, or runtime repairs.

Gate 1 allows metadata initialization but rejects torch tensor operations and
Triton launches. Gate 2 requires preserved Gate 1 PASS and runs one inspected
single-triangle CuMesh boundary-component fixture.
"""
import argparse
import datetime
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
import traceback

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
MARKER = "GPU_PROOF_RESULT="
IMPORTS = ("flex_gemm.kernels.cuda", "o_voxel._C")
CACHE_KEYS = ("TMP", "TEMP", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
              "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
              "TRITON_CACHE_DIR", "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR",
              "AMD_COMGR_CACHE_DIR", "HIP_CACHE_DIR", "MIOPEN_USER_DB_PATH",
              "MIOPEN_CUSTOM_CACHE_DIR", "FLEX_GEMM_AUTOTUNE_CACHE_PATH")


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT.with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_sibling("amd_native_import_preflight")


def scratch_environment(scratch):
    env = {key: str(scratch / key.lower()) for key in CACHE_KEYS}
    env.update(HIP_VISIBLE_DEVICES="0", ROCR_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0",
               FLEX_GEMM_USE_AUTOTUNE_CACHE="0", FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def require_gate_one(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    workers = result.get("workers", [])
    if (result.get("classification") != "PASS" or result.get("gate") != 1 or
            result.get("inventory_unchanged") is not True or
            result.get("observer_healthy") is not True or
            [worker.get("module") for worker in workers] != list(IMPORTS) or
            not all(worker.get("status") == "PASS" and worker.get("worker_returncode") == 0
                    and worker.get("gpu_initialized_after") is True for worker in workers)):
        raise RuntimeError("Gate 2 requires complete, preserved Gate 1 PASS evidence")
    return result


def validate_boundary_result(count, ids, edges):
    normalized = sorted(sorted(edge) for edge in edges)
    if count != 1 or ids != [0, 0, 0] or normalized != [[0, 1], [0, 2], [1, 2]]:
        raise RuntimeError("Boundary result differs from the single-triangle fixture expectation")


def emit(event, **detail):
    print("GPU_PROOF_EVENT=" + json.dumps({"event": event, **detail}), flush=True)


def boundary_fixture(torch, record):
    import cumesh
    record["fixture"] = {"vertices": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                         "faces": [[0, 1, 2]], "vertices_dtype": "float32", "faces_dtype": "int32"}
    def stage(name, function):
        record["last_stage"] = name
        emit("stage-start", stage=name)
        result = function()
        torch.cuda.synchronize(0)
        emit("stage-complete", stage=name)
        return result
    # CPU fixture values are copied to device; no random input or larger mesh.
    vertices = stage("fixture-vertices-to-device", lambda: torch.tensor(
        record["fixture"]["vertices"], dtype=torch.float32, device="cpu").to("cuda:0"))
    faces = stage("fixture-faces-to-device", lambda: torch.tensor(
        record["fixture"]["faces"], dtype=torch.int32, device="cpu").to("cuda:0"))
    mesh = stage("CuMesh-constructor", cumesh.CuMesh)
    stage("init", lambda: mesh.init(vertices, faces))
    for name in ("get_edges", "get_boundary_info", "get_vertex_edge_adjacency",
                 "get_vertex_boundary_adjacency", "get_manifold_boundary_adjacency"):
        stage(name, getattr(mesh, name))
    record["manifold_boundary_adjacency"] = stage("read_manifold_boundary_adjacency",
        lambda: mesh.read_manifold_boundary_adjacency().cpu().tolist())
    stage("get_boundary_connected_components", mesh.get_boundary_connected_components)
    count, ids = stage("read_boundary_connected_components", mesh.read_boundary_connected_components)
    record["component_count"] = count
    record["component_ids"] = stage("component-ids-to-cpu", lambda: ids.cpu().tolist())
    edges = stage("read_edges", lambda: mesh.read_edges().cpu().tolist())
    boundary_indices = stage("read_boundaries", lambda: mesh.read_boundaries().cpu().tolist())
    boundary_edges = [edges[index] for index in boundary_indices]
    record.update(edges=edges, boundary_indices=boundary_indices, boundary_edges=boundary_edges,
                  num_vertices=mesh.num_vertices, num_faces=mesh.num_faces, num_boundaries=mesh.num_boundaries,
                  gpu_peak_tensor_memory_bytes=torch.cuda.max_memory_allocated(0))
    validate_boundary_result(count, record["component_ids"], boundary_edges)
    if (mesh.num_vertices, mesh.num_faces, mesh.num_boundaries) != (3, 1, 3):
        raise RuntimeError("CuMesh topology differs from the deterministic triangle")


def worker(name, scratch):
    record = {"module": name, "status": "FAIL", "path": None,
              "python": platform.python_version(), "executable": sys.executable,
              "prefix": sys.prefix, "sdk_discovery": [], "gpu_initialized_before": None,
              "gpu_initialized_after": None}
    torch = None
    try:
        scratch = scratch.resolve()
        scratch.relative_to(ROOT)
        env = scratch_environment(scratch)
        for key in CACHE_KEYS:
            if key != "FLEX_GEMM_AUTOTUNE_CACHE_PATH":
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        record["scratch_environment"] = env
        os.environ.update(env)
        audit = preflight.load_audit()
        record["sdk_verification"] = audit.block_external_actions(
            sdk_executable=Path(sys.prefix) / "Scripts/rocm-sdk.exe",
            discovery_records=record["sdk_discovery"], scratch_dir=scratch)
        # Override only the earlier proof's deliberately hidden visibility.
        # All other process/network/write controls remain active.
        os.environ.update(env)
        import torch
        record.update(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
                      gpu_initialized_before=torch.cuda.is_initialized())
        if (platform.python_version() != "3.12.10" or torch.__version__ != "2.13.0+rocm10.0.0"
                or not torch.version.hip or torch.version.cuda is not None
                or record["gpu_initialized_before"]):
            raise RuntimeError("Unexpected Python/PyTorch/AMD runtime provenance")
        from torch.utils._python_dispatch import TorchDispatchMode

        class MetadataOnly(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise RuntimeError("Gate 1 forbids torch tensor operations: " + str(func))

        with MetadataOnly():
            import triton
            def forbid_launch(*args, **kwargs):
                raise RuntimeError("Proof forbids Triton compilation/launch/autotuning")
            triton.compile = forbid_launch
            triton.runtime.JITFunction.run = forbid_launch
            triton.runtime.Autotuner.run = forbid_launch
            record["triton"] = triton.__version__
            emit("import-start", module=name)
            module = importlib.import_module("cumesh._C" if name == "cumesh-boundary" else name)
            record["path"] = module.__file__
            Path(module.__file__).resolve().relative_to(Path(sys.prefix) / "Lib/site-packages")
            properties = torch.cuda.get_device_properties(0)
            record["gpu"] = {"name": properties.name, "device_count": torch.cuda.device_count(),
                             "architecture": getattr(properties, "gcnArchName", None),
                             "total_memory_bytes": properties.total_memory,
                             "multi_processor_count": properties.multi_processor_count,
                             "properties": str(properties)}
            if "RX 9070 XT" not in properties.name:
                raise RuntimeError("Selected device is not the intended RX 9070 XT")
            record["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(0)
            record["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(0)
            if record["gpu_memory_allocated_bytes"] or record["gpu_memory_reserved_bytes"]:
                raise RuntimeError("Gate 1 unexpectedly allocated GPU tensor memory")
            record["native_import_paths"] = {
                key: value.__file__ for key, value in sys.modules.items()
                if key.startswith(("cumesh", "flex_gemm", "o_voxel", "nvdiffrast"))
                and getattr(value, "__file__", "").endswith(".pyd")}
            for native_path in record["native_import_paths"].values():
                Path(native_path).resolve().relative_to(Path(sys.prefix) / "Lib/site-packages")
        if name == "cumesh-boundary":
            # Only this branch leaves the tensor-operation guard, after the
            # same provenance/device/import checks as the metadata workers.
            boundary_fixture(torch, record)
        record["status"] = "PASS"
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if torch is not None:
            record["gpu_initialized_after"] = torch.cuda.is_initialized()
        print(MARKER + json.dumps(record), flush=True)
    return 0 if record["status"] == "PASS" else 1


def normalize_metadata_rows(rows):
    # Windows directory allocation size can change in read-only cold/warm
    # traversal. Keep names, mode and mtime, and keep all regular-file sizes.
    return [[path, None if stat.S_ISDIR(mode) else size, mtime, mode]
            for path, size, mtime, mode in rows]


def snapshot(args, manifest_path=None):
    result = preflight.dependency_snapshot(args)
    result["venv_file_metadata"] = {}
    raw = {}
    for label, venv in (("amd", args.amd_venv), ("modly", args.modly_venv)):
        entries = []
        for directory, dirs, files in os.walk(venv, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                info = path.lstat()
                entries.append([path.relative_to(venv).as_posix(), info.st_size,
                                info.st_mtime_ns, info.st_mode])
        raw[label] = sorted(entries)
        encoded = json.dumps(normalize_metadata_rows(raw[label]), separators=(",", ":")).encode()
        result["venv_file_metadata"][label] = {
            "entries": len(entries), "tree_sha256": hashlib.sha256(encoded).hexdigest(),
            "fields": "relative path, file size (directory allocation size normalized to null), mtime_ns, mode; no followed links"}
    if manifest_path is not None:
        manifest_path.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    site = args.amd_venv / "Lib/site-packages"
    paths = [site / "torch/cuda/__init__.py"]
    for package in ("cumesh", "flex_gemm", "o_voxel", "triton"):
        paths += sorted((site / package).rglob("*.py"))
    result["inspected_package_sources"] = {
        str(path.relative_to(site)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return result


def windows_observation(since):
    # Passive WMI and System-event log queries; no GPU workload or repair.
    script = (
        "$ErrorActionPreference='Stop'; $result=@{}; "
        "$env:PSModulePath=Join-Path $PSHOME 'Modules'; "
        "try {$diagnostics=Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Diagnostics/Microsoft.PowerShell.Diagnostics.psd1'; "
        "Import-Module $diagnostics -ErrorAction Stop; "
        "$probe=Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop | "
        "Select-Object Id,RecordId,ProviderName,TimeCreated; "
        "$result.observer=@{edition=$PSVersionTable.PSEdition;version=$PSVersionTable.PSVersion.ToString();"
        "diagnostics_module=(Get-Module Microsoft.PowerShell.Diagnostics).Path;system_log_probe=$probe}} "
        "catch {$result.observer_error=$_.Exception.Message;$result.observer_error_id=$_.FullyQualifiedErrorId}; "
        "try {$result.controllers=@(Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,DriverDate,PNPDeviceID,Status)} "
        "catch {$result.controller_error=$_.Exception.Message}; "
        "try {$events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime="
        f"[datetime]'{since}';Id=@(4101,17,18,41)}} -ErrorAction Stop); "
        "$result.events=@($events | Select-Object TimeCreated,Id,ProviderName,Message)} "
        "catch {if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') "
        "{$result.events=@()} else {$result.event_error=$_.Exception.Message}}; "
        "$result | ConvertTo-Json -Depth 6 -Compress")
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    capture = preflight.run_child([str(executable), "-NoProfile", "-NonInteractive",
                                   "-Command", script], 10)
    if capture["returncode"] == 0:
        try:
            capture["observation"] = json.loads(capture["stdout"])
        except ValueError:
            pass
    return capture


def healthy_observation(capture):
    observation = capture.get("observation", {})
    return (capture.get("returncode") == 0 and capture.get("timed_out") is False and
            isinstance(observation, dict) and "event_error" not in observation and
            "controller_error" not in observation and "observer_error" not in observation and
            isinstance(observation.get("observer"), dict) and
            observation["observer"].get("system_log_probe", {}).get("RecordId", 0) > 0 and
            observation.get("events") == [] and
            isinstance(observation.get("controllers"), list) and
            any("RX 9070 XT" in gpu.get("Name", "") and gpu.get("Status") == "OK"
                for gpu in observation["controllers"]))


def supervise(args):
    gate_one = require_gate_one(args.gate_one) if args.gate == "boundary" else None
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = snapshot(args, output / "file-metadata-before.json.gz")
    save("inventory-before.json", before)
    prerequisites_unchanged = True
    if gate_one is not None:
        prerequisites_unchanged = before == json.loads(
            args.gate_one.with_name("inventory-after.json").read_text(encoding="utf-8"))
    observation_before = windows_observation(start)
    save("windows-before.json", observation_before)
    records = []
    observer_healthy = healthy_observation(observation_before)
    names = IMPORTS if args.gate == "init" else ("cumesh-boundary",)
    for index, name in enumerate(names if observer_healthy and prerequisites_unchanged else (), 1):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-scratch-", dir=output) as scratch:
            command = [str(args.amd_venv / "Scripts/python.exe"), "-B", "-I", str(SCRIPT),
                       "--worker", name, "--scratch", scratch]
            if name == "cumesh-boundary":
                command += ["--gate-one", str(args.gate_one.resolve())]
            capture = preflight.run_child(command, 30)
        basename = f"{index:02d}-{name.replace('.', '_')}"
        for stream in ("stdout", "stderr"):
            (output / (basename + "." + stream + ".txt")).write_bytes(capture[stream].encode())
        lines = [line[len(MARKER):] for line in capture["stdout"].splitlines()
                 if line.startswith(MARKER)]
        record = json.loads(lines[-1]) if lines else {
            "module": name, "status": "FAIL", "error": capture.get("error", "No worker record")}
        record.update(worker_returncode=capture["returncode"], timed_out=capture["timed_out"],
                      exit_status_hex=hex(capture["returncode"] & 0xffffffff)
                      if capture["returncode"] is not None else None,
                      elapsed_seconds=capture["elapsed_seconds"], worker_timeout_seconds=30,
                      stdout_file=basename + ".stdout.txt", stderr_file=basename + ".stderr.txt")
        records.append(record)
        print(name + ": " + record["status"], flush=True)
        if record["status"] != "PASS" or record["worker_returncode"] != 0:
            break
        observation = windows_observation(start)
        save(basename + ".windows.json", observation)
        if not healthy_observation(observation):
            observer_healthy = False
            record["observer_unhealthy"] = True
            break
    after = snapshot(args, output / "file-metadata-after.json.gz")
    save("inventory-after.json", after)
    observation_after = windows_observation(start)
    save("windows-after.json", observation_after)
    events = observation_after.get("observation", {}).get("events", [])
    passed = (len(records) == len(names) and all(
        r["status"] == "PASS" and r["worker_returncode"] == 0 and
        r["gpu_initialized_after"] is True and not r.get("observer_unhealthy") for r in records))
    observer_healthy = observer_healthy and healthy_observation(observation_after)
    unchanged = before == after
    result = {"classification": "PASS" if passed and unchanged and observer_healthy else "PARTIAL",
              "gate": 1 if args.gate == "init" else 2, "inventory_unchanged": unchanged, "workers": records,
              "observer_healthy": observer_healthy,
              "gate_one_inventory_matches": prerequisites_unchanged if gate_one else None,
              "system_events_observed": events, "started_utc": start,
              "scope": "Metadata initialization and native imports only; no torch tensor operations"
                       if args.gate == "init" else "One deterministic triangle boundary-component fixture; no hole filling or models"}
    save("results.json", result)
    print(f"Gate {result['gate']}: " + result["classification"] + "; inventories unchanged: " + str(unchanged))
    return 0 if result["classification"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=IMPORTS + ("cumesh-boundary",))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--gate", choices=("init", "boundary"), default="init")
    parser.add_argument("--gate-one", type=Path)
    for name in ("amd-venv", "modly-venv", "amd-source", "extension-source", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.worker == "cumesh-boundary":
        if args.gate_one is None:
            raise RuntimeError("Boundary worker requires Gate 1 evidence")
        require_gate_one(args.gate_one)
    return worker(args.worker, args.scratch) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
