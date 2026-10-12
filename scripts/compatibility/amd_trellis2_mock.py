"""Bounded IPC proof. Uses no GPU packages, real model load, or inference.

Run under the existing Modly Python 3.11 with --python312 pointing to a
standalone Python 3.12. Creates only a temporary, pip-free mock venv here.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
MOCK = '''
import json, platform, sys
from services.generators.base import BaseGenerator
class MockGenerator(BaseGenerator):
    def load(self): self._model = "mock only"
    def unload(self): self._model = None
    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        assert "torch" not in sys.modules
        progress_cb(50, "Zombie transport \\u2192 fixture")
        path = self.outputs_dir / (self.MODEL_NODE_ID + ".json")
        path.write_text(json.dumps({"python": platform.python_version(),
            "image_hex": image_bytes.hex(), "params": params,
            "node": self.MODEL_NODE_ID, "model_dir": str(self.model_dir)}),
            encoding="utf-8")
        return path
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python312", type=Path, required=True)
    parser.add_argument("--gguf-extension", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "api"))
    from services.extension_process import ExtensionProcess, _venv_python

    # Avoid inheriting live application paths or package injection.
    for key in ("PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(key, None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory(prefix="amd-ipc-proof-", dir=ROOT.parent) as tmp:
        scratch = Path(tmp)
        ext = scratch / "extension"
        ext.mkdir()
        os.environ["MODELS_DIR"] = str(scratch / "models")
        os.environ["WORKSPACE_DIR"] = str(scratch / "outputs")
        os.environ["EXTENSIONS_DIR"] = str(scratch)
        subprocess.run([str(args.python312), "-B", "-I", "-m", "venv",
                        "--without-pip", str(ext / "venv")], check=True, timeout=30)
        manifest = {"id": "amd-proof", "generator_class": "MockGenerator",
                    "nodes": [{"id": "generate"}, {"id": "refine"}]}
        (ext / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (ext / "generator.py").write_text(MOCK, encoding="utf-8")
        results = []
        for node in ("generate", "refine"):
            worker = ExtensionProcess(ext, {**manifest, "id": "amd-proof/" + node,
                                           "node_id": node})
            worker.model_dir = scratch / "models" / node
            worker.outputs_dir = scratch / "outputs"
            # A startup fault must fail this proof, never trigger package repair.
            def forbid_install(*unused):
                raise AssertionError("Automatic package repair forbidden in proof")
            worker._install_missing_package = forbid_install
            # Production startup waits indefinitely. Bound this test externally.
            timer = threading.Timer(20, worker.stop)
            timer.start()
            try:
                worker.load()  # Mock load only; no model code or GPU packages.
                progress = []
                output = worker.generate(b"zombie transport fixture",
                    {"existing_mesh_path": "mock-input.glb", "label": "zombie → GLB"},
                    lambda pct, step: progress.append([pct, step]))
                record = json.loads(output.read_text(encoding="utf-8"))
                assert record["python"].startswith("3.12.")
                assert record["image_hex"] == b"zombie transport fixture".hex()
                assert record["params"]["existing_mesh_path"] == "mock-input.glb"
                assert record["params"]["label"] == "zombie → GLB"
                assert record["node"] == node and progress[0][0] == 50
                assert "→" in progress[0][1]
                worker.unload()
                assert not worker.is_loaded()
                results.append({"node": node, "child_python": record["python"],
                                "transport": "PASS", "mock_output_suffix": output.suffix})
            finally:
                timer.cancel()
                worker.stop()

        # The real official generator's module/constructor/ready boundary only.
        # Never call its load(), generate(), or unload() methods.
        real_manifest = json.loads((args.gguf_extension / "manifest.json").read_text())
        (ext / "manifest.json").write_text(json.dumps(real_manifest), encoding="utf-8")
        shutil.copyfile(args.gguf_extension / "generator.py", ext / "generator.py")
        worker = ExtensionProcess(ext, {**real_manifest, "id": "trellis2/generate",
                                       "node_id": "generate"})
        worker.model_dir = scratch / "models" / "generate"
        worker._install_missing_package = forbid_install
        timer = threading.Timer(20, worker.stop)
        timer.start()
        try:
            worker._start()
            assert worker._proc.poll() is None
            real_ready = "PASS (ready only; no dependencies or weights loaded)"
        finally:
            timer.cancel()
            worker.stop()
        print(json.dumps({"host_python": platform.python_version(),
            "child_executable_layout": "venv/Scripts/python.exe",
            "pip_installed": (ext / "venv" / "Scripts" / "pip.exe").exists(),
            "results": results, "official_generator_ready": real_ready}, indent=2))


if __name__ == "__main__":
    main()
