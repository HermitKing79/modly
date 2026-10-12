"""Read one registered operator schema without executing it or initializing GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compatibility"))
import amd_headless_import_proof as proof
import amd_trellis2_audit as audit

scratch = Path(sys.argv[1]).resolve()
scratch.relative_to(ROOT)
audit.block_external_actions(sdk_executable=Path(sys.prefix) / "Scripts/rocm-sdk.exe", scratch_dir=scratch)
import torch
def reject(*args, **kwargs):
    raise RuntimeError("Schema inspection forbids GPU initialization")
torch.cuda._lazy_init = reject
torch.cuda.init = reject
site = Path(sys.prefix) / "Lib/site-packages"
actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in proof.VERSION_QUERY_FILES}
if actual != proof.VERSION_QUERY_FILES or torch.__version__ != "2.13.0+rocm10.0.0" or torch.version.cuda is not None:
    raise RuntimeError("Unexpected schema-probe provenance")
torch.ops.load_library(str(site / "torchvision/_C.pyd"))
operation = torch.ops.torchvision._cuda_version.default
print(json.dumps({"operator": str(operation), "schema": str(operation._schema),
                  "gpu_initialized": torch.cuda.is_initialized(), "operator_executed": False,
                  "weights_loaded": False, "sha256": actual}), flush=True)
if torch.cuda.is_initialized():
    raise RuntimeError("Schema probe unexpectedly initialized GPU")
