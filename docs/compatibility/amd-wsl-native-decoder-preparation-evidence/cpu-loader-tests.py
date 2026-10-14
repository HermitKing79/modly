"""Capture the two approved tiny-checkpoint CPU regression suites."""
import contextlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
EV = Path(__file__).resolve().parent
pattern, label = sys.argv[1:]
expected = {"test_amd_shape_decoder_loader.py": 8, "test_amd_strict_loader.py": 5}[pattern]
sys.path[:0] = [str(ROOT / "scripts/compatibility")]
started = time.monotonic()
with (EV / (label + ".stdout.txt")).open("w", encoding="utf-8") as stdout, (EV / (label + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts/compatibility"), pattern=pattern)
        result = unittest.TextTestRunner(stream=stderr, verbosity=2).run(suite)
import torch
forbidden = [name for name in sys.modules if name.split(".")[0] in ("flex_gemm", "trellis2", "cumesh", "o_voxel", "nvdiffrast")]
record = {
    "command": [sys.executable, "-B", str(Path(__file__).resolve()), pattern, label],
    "test_pattern": pattern, "tests_run": result.testsRun, "expected_tests": expected,
    "passed": result.wasSuccessful() and result.testsRun == expected and not forbidden and not torch.cuda.is_initialized(),
    "elapsed_seconds": time.monotonic() - started,
    "runtime": {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                "hip": torch.version.hip, "cuda": torch.version.cuda, "gpu_initialized": torch.cuda.is_initialized()},
    "execution_scope": "tiny synthetic CPU/meta tensors; GPU lazy initialization denied in each test",
    "forbidden_native_or_decoder_modules": forbidden,
    "real_checkpoint_tensor_loading": False,
    "stdout": label + ".stdout.txt", "stderr": label + ".stderr.txt",
}
(EV / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record))
raise SystemExit(0 if record["passed"] else 1)
