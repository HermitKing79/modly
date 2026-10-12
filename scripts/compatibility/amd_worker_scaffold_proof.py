"""Bounded offline integration proof. No native imports, models or tensor ops.

The supervisor inventories both existing environments before/after a disposable
host running Modly's registry, ExtensionProcess and runner with the AMD adapter.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]


def run_host(command, env, timeout):
    """Keep the owned host alive until tree termination on timeout completes."""
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = {"timeout": False, "tree_cleanup_verified": True, "cleanup": None}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        result["timeout"] = True
        killer = Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"
        try:
            cleanup = subprocess.run([str(killer), "/PID", str(proc.pid), "/T", "/F"],
                                     capture_output=True, timeout=10)
            result["cleanup"] = {"exit": cleanup.returncode,
                                 "stdout": cleanup.stdout.decode(errors="replace"),
                                 "stderr": cleanup.stderr.decode(errors="replace")}
            result["tree_cleanup_verified"] = cleanup.returncode == 0
            stdout, stderr = proc.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["tree_cleanup_verified"] = False
            result["cleanup"] = {"error": repr(exc)}
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
    result.update({"exit": proc.returncode, "stdout": stdout, "stderr": stderr})
    return result


def classify_result(run, unchanged, ipc):
    complete = isinstance(ipc, dict) and \
        [row.get("node") for row in ipc.get("routes", [])] == ["generate", "refine"] and \
        ipc.get("hard_stop_restart") is True and \
        {row.get("case") for row in ipc.get("negative_tests", []) if row.get("rejected") is True} == \
        {"missing-checkpoint", "incompatible-runtime"}
    return "PASS" if run["exit"] == 0 and not run["timeout"] and \
        run["tree_cleanup_verified"] and unchanged and complete else "PARTIAL"


def ipc_proof(args):
    scratch = args.scratch.resolve()
    outputs = scratch / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    os.environ.update({"MODELS_DIR": str(scratch / "models"), "WORKSPACE_DIR": str(outputs),
                       "EXTENSIONS_DIR": str(ROOT / "experimental/extensions"),
                       "MODLY_ENABLE_EXPERIMENTAL_AMD": "1",
                       "MODLY_AMD_RUNTIME_CONFIG": str(scratch / "runtime.json")})
    prefix = scratch / "fixture-checkpoint"
    Path(str(prefix) + ".json").write_text('{"name":"scaffold_fixture_only","args":{}}')
    header = b'{"__metadata__":{"fixture_only":"true"}}'
    header += b" " * (-len(header) % 8)
    Path(str(prefix) + ".safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    config = {"python": str(args.amd_venv / "Scripts/python.exe"),
              "source_root": str(args.amd_source),
              "models": {"generate": [str(prefix)], "refine": [str(prefix)]}}
    config_path = scratch / "runtime.json"
    def save_config(value):
        config_path.write_text(json.dumps(value), encoding="utf-8")
    save_config(config)
    sys.path.insert(0, str(ROOT / "api"))
    from services.generator_registry import GeneratorRegistry
    from services.extension_process import ExtensionProcess
    from services.amd_runtime import CONTRACT
    registry = GeneratorRegistry()
    registry.initialize()
    if registry.load_errors():
        raise AssertionError(registry.load_errors())
    assert set(registry._generators) == {"trellis2-amd/generate", "trellis2-amd/refine"}
    # Tiny existing GLB container fixture, not an AI-generated geometry asset.
    glb_json = b'{"asset":{"version":"2.0"}}'
    glb_json += b" " * (-len(glb_json) % 4)
    mesh = outputs / "input-fixture.glb"
    mesh.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(glb_json)) +
                     struct.pack("<I4s", len(glb_json), b"JSON") + glb_json)
    results = {"host_python": sys.version, "host_executable": sys.executable,
               "checkpoint_fixture_only": True, "routes": [], "negative_tests": []}
    for node in ("generate", "refine"):
        worker = registry._generators["trellis2-amd/" + node]
        try:
            worker.load()
            identity = worker.runtime_identity
            assert identity["gpu_initialized"] is False and identity["weights_loaded"] is False
            pid = worker._proc.pid
            progress = []
            params = {"mock": True}
            if node == "refine":
                params["existing_mesh_path"] = str(mesh)
            output = worker.generate(b"zombie image transport fixture", params,
                                     lambda pct, label: progress.append([pct, label]))
            receipt = json.loads(output.read_text())
            assert receipt["route"] == node and receipt["asset_generated"] is False
            assert progress and output.suffix == ".json"
            try:
                worker.generate(b"fixture", {"mock": False})
            except RuntimeError as exc:
                rejected = str(exc)
                assert "mock=true" in rejected
            else:
                raise AssertionError("Inference request was accepted")
            recovered = worker.generate(b"fixture", params)
            assert worker._proc.pid == pid and recovered.is_file()
            worker.unload()
            assert not worker.is_loaded()
            worker.load()
            worker.shutdown()
            assert worker._proc is None and not worker.is_loaded()
            results["routes"].append({"node": node, "identity": identity, "receipt": receipt,
                                      "progress": progress, "rejected_inference": rejected,
                                      "same_process_recovery": True, "unload_reload": True,
                                      "cooperative_shutdown_reaped": True})
        finally:
            worker.stop()
    manifest = {"id": "trellis2-amd/generate", "node_id": "generate", "ext_id": "trellis2-amd",
                "experimental": True, "runtime_contract": CONTRACT}
    ext = ROOT / "experimental/extensions/trellis2-amd"
    cases = (
        ("missing-checkpoint", {**config, "models": {"generate": [str(prefix) + "-missing"], "refine": [str(prefix)]}}, "unavailable"),
        ("incompatible-runtime", {**config, "python": str(args.system_python)}, "AMD dependency torch:"),
    )
    for label, bad_config, expected in cases:
        save_config(bad_config)
        worker = ExtensionProcess(ext, manifest)
        worker.outputs_dir = outputs
        worker.model_dir = scratch / "models/generate"
        try:
            try:
                worker.load()
            except (RuntimeError, ValueError) as exc:
                diagnostic = str(exc)
                assert expected in diagnostic, diagnostic
                results["negative_tests"].append({"case": label, "rejected": True, "diagnostic": diagnostic})
            else:
                raise AssertionError(f"{label} was accepted")
        finally:
            worker.stop()
    save_config(config)
    worker = ExtensionProcess(ext, manifest)
    worker.outputs_dir = outputs
    worker.model_dir = scratch / "models/generate"
    try:
        worker.load()
        original_pid = worker._proc.pid
        worker.stop()
        assert worker._proc is None
        worker.load()
        assert worker._proc.pid != original_pid
        worker.generate(b"fixture", {"mock": True})
        worker.shutdown()
        results["hard_stop_restart"] = True
    finally:
        worker.stop()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("amd-venv", "modly-venv", "amd-source", "extension-source", "output", "scratch", "system-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ipc", action="store_true")
    args = parser.parse_args()
    if args.ipc:
        try:
            result = ipc_proof(args)
            (args.scratch / "ipc-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.scratch.resolve().relative_to(ROOT.parent)
    spec = importlib.util.spec_from_file_location("bounded_snapshot", ROOT / "scripts/compatibility/amd_bounded_gpu_proof.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    before = audit.snapshot(args, args.output / "file-metadata-before.json.gz")
    (args.output / "inventory-before.json").write_text(json.dumps(before, indent=2))
    command = [str(args.modly_venv / "Scripts/python.exe"), "-B", "-I", str(Path(__file__).resolve()), "--ipc"]
    for name in ("amd-venv", "modly-venv", "amd-source", "extension-source", "output", "scratch", "system-python"):
        command += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    env = os.environ.copy()
    env.update({"TMP": str(args.scratch), "TEMP": str(args.scratch), "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    # Each child worker has finite IPC deadlines. The host has an independent
    # outer deadline. Host termination is a blocker, never an automatic retry.
    run = run_host(command, env, 150)
    code = run["exit"]
    (args.output / "ipc.stdout.txt").write_bytes(run.pop("stdout"))
    (args.output / "ipc.stderr.txt").write_bytes(run.pop("stderr"))
    ipc_path = args.scratch / "ipc-results.json"
    ipc = json.loads(ipc_path.read_text()) if ipc_path.exists() else None
    after = audit.snapshot(args, args.output / "file-metadata-after.json.gz")
    (args.output / "inventory-after.json").write_text(json.dumps(after, indent=2))
    unchanged = before == after
    result = {"classification": classify_result(run, unchanged, ipc),
              "host_exit": code, "supervision": run, "inventory_unchanged": unchanged, "ipc": ipc}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "ipc"}))
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
