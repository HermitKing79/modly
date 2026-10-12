"""Passive inventories for this machine's bounded offline model audit.

No package imports, tensor payload reads, installers or GPU queries. The
existing inventory helper hashes selected native/source files and walks metadata.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "docs/compatibility/amd-offline-model-audit-evidence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    parser.add_argument("--output", type=Path, default=output)
    options = parser.parse_args()
    phase = options.phase
    output = options.output.resolve()
    output.relative_to(root)
    spec = importlib.util.spec_from_file_location("existing_snapshot", root / "scripts/compatibility/amd_bounded_gpu_proof.py")
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    args = argparse.Namespace(amd_venv=Path("D:/AI/Trellis2AMD/venv"),
                              modly_venv=Path("C:/Users/wattz/Documents/Modly/extensions/trellis2/venv"),
                              amd_source=Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD"),
                              extension_source=root.parent / "gguf-source")
    output.mkdir(parents=True, exist_ok=True)
    result = proof.snapshot(args, output / f"venv-metadata-{phase}.json.gz")
    trees = {}
    for directory in [Path("D:/AI/Trellis2AMD/ComfyUI/models"), Path("C:/Users/wattz/Documents/Modly/models"),
                      Path("C:/Users/wattz/.cache/huggingface/hub"), Path("C:/Users/wattz/.cache/torch/hub")]:
        entries = []
        if directory.is_dir():
            for current, dirs, files in os.walk(directory, followlinks=False):
                for name in dirs + files:
                    path = Path(current) / name
                    info = path.lstat()
                    entries.append([path.relative_to(directory).as_posix(), info.st_size, info.st_mtime_ns, info.st_mode])
                    if len(entries) > 20000:
                        raise RuntimeError("Model metadata scope exceeds bound")
        trees[str(directory)] = sorted(entries)
    result["model_cache_metadata"] = {
        name: {"entries": len(rows), "normalized_sha256": hashlib.sha256(
            json.dumps(proof.normalize_metadata_rows(rows), separators=(",", ":")).encode()).hexdigest()}
        for name, rows in trees.items()}
    (output / f"model-cache-metadata-{phase}.json.gz").write_bytes(gzip.compress(json.dumps(trees).encode(), mtime=0))
    (output / f"inventory-{phase}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"phase": phase, "venv_trees": result["venv_file_metadata"],
                      "model_cache_trees": result["model_cache_metadata"]}))


if __name__ == "__main__":
    main()
