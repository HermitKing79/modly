"""Bounded mock IPC validation of explicit alpha-only input, never inference."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]


def child(scratch):
    sys.path.insert(0, str(ROOT / "api"))
    from PIL import Image
    from services.extension_process import ExtensionProcess
    from services.amd_runtime import CONTRACT
    scratch = scratch.resolve()
    scratch.relative_to(ROOT)
    os.environ.update(MODLY_ENABLE_EXPERIMENTAL_AMD="1", WORKSPACE_DIR=str(scratch))
    prefix = scratch / "metadata-only-fixture"
    prefix.with_suffix(".json").write_text('{"name":"fixture","args":{}}')
    header = b'{"__metadata__":{"fixture":"true"}}'
    prefix.with_suffix(".safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    config = {"python": "D:/AI/Trellis2AMD/venv/Scripts/python.exe",
              "source_root": "D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD",
              "models": {node: [str(prefix)] for node in ("generate", "refine")}}
    config_path = scratch / "runtime.json"
    config_path.write_text(json.dumps(config))
    os.environ["MODLY_AMD_RUNTIME_CONFIG"] = str(config_path)
    glb_json = b'{"asset":{"version":"2.0"}}'
    glb_json += b" " * (-len(glb_json) % 4)
    mesh = scratch / "mesh-container-fixture.glb"
    mesh.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(glb_json)) +
                     struct.pack("<I4s", len(glb_json), b"JSON") + glb_json)
    image = Image.new("RGBA", (12, 10), (40, 80, 120, 0))
    for x in range(2, 10):
        for y in range(2, 8):
            image.putpixel((x, y), (40, 80, 120, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    opaque = io.BytesIO()
    Image.new("RGBA", (12, 10), (40, 80, 120, 255)).save(opaque, format="PNG")
    rows = []
    for node in ("generate", "refine"):
        manifest = {"id": "trellis2-amd/" + node, "node_id": node, "ext_id": "trellis2-amd",
                    "runtime_contract": CONTRACT, "experimental": True}
        worker = ExtensionProcess(ROOT / "experimental/extensions/trellis2-amd", manifest)
        worker.outputs_dir = scratch
        worker.model_dir = scratch / ("mock-model-" + node)
        try:
            worker.load()
            pid = worker._proc.pid
            params = {"mock": True, "image_mode": "alpha-only"}
            if node == "refine":
                params["existing_mesh_path"] = str(mesh)
            try:
                worker.generate(opaque.getvalue(), params)
            except RuntimeError as exc:
                rejection = str(exc)
                assert "alpha-only" in rejection
            else:
                raise AssertionError("Opaque alpha-only input was accepted")
            receipt = json.loads(worker.generate(stream.getvalue(), params).read_text())
            assert receipt["image_mode"] == "alpha-only" and receipt["alpha_input"]["rembg_required"] is False
            assert receipt["weights_loaded"] is False and receipt["asset_generated"] is False
            assert worker._proc.pid == pid
            identity = worker.runtime_identity
            assert identity["gpu_initialized"] is False
            worker.shutdown()
            assert worker._proc is None
            rows.append({"route": node, "receipt": receipt, "identity": identity,
                         "opaque_rejection": rejection, "same_process_recovery": True,
                         "shutdown_reaped": True})
        finally:
            worker.stop()
    (scratch / "ipc-results.json").write_text(json.dumps({"classification": "PASS", "routes": rows}, indent=2))


def supervise(output):
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("scaffold_supervisor", Path(__file__).with_name("amd_worker_scaffold_proof.py"))
    scaffold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scaffold)
    scratch = output / "scratch"
    scratch.mkdir()
    command = ["C:/Users/wattz/Documents/Modly/extensions/trellis2/venv/Scripts/python.exe",
               "-B", "-I", str(Path(__file__).resolve()), "--child", "--output", str(scratch)]
    env = {**os.environ, "TMP": str(scratch), "TEMP": str(scratch), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = scaffold.run_host(command, env, 90)
    for name in ("stdout", "stderr"):
        (output / (name + ".txt")).write_bytes(result.pop(name))
    ipc_path = scratch / "ipc-results.json"
    ipc = json.loads(ipc_path.read_text()) if ipc_path.is_file() else None
    passed = result["exit"] == 0 and not result["timeout"] and result["tree_cleanup_verified"] and ipc is not None
    result.update(classification="PASS" if passed else "PARTIAL", command=command, timeout_seconds=90, ipc=ipc)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ("ipc", "command")}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
