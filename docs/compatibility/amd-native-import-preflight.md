# AMD native-import preflight

**Result: PARTIAL — 12 October 2026.** Five of seven ordinary imports passed from the intended AMD venv. The SDK subprocess guard limitation is resolved. FlexGEMM and o-voxel reach an import-time GPU device-name query, where this preflight's conservative GPU-initialization guard stops them. This is a precisely identified **test boundary**, not a demonstrated native DLL failure.

No GPU kernels, GPU tensors, TRELLIS models, installations or rebuilds were run. All seven workers recorded `gpu_initialized_before_import=false` and `gpu_initialized_after_import=false`. Native generation, texturing and the previously observed CuMesh 719 crash remain unvalidated.

## Preservation and scope

The clean local checkout started on `codex/amd-trellis2-bounded-proof` at `874a843e240d6c96b6149f7908c1734472ec0dcc`, with `origin=https://github.com/HermitKing79/modly.git`. Remote main was `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`; the remote proof branch matched the supplied HEAD. A new branch, **`codex/amd-native-import-preflight`**, was created before editing.

The AMD source checkout remained at `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5` with its existing modified `nodes.py` and untracked `nodes.py.before-profiler-fix`. Neither installed environment nor that source checkout was edited. Changes are limited to compatibility harness code, guard regression checks and evidence/documentation. The previous proof report and evidence remain historical records.

## SDK discovery safety

The inspected Triton driver calls exactly:

```python
subprocess.check_output(["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL)
```

The current PATH resolves a global Python installation's `rocm-sdk.exe`. The harness instead pins **`D:\AI\Trellis2AMD\venv\Scripts\rocm-sdk.exe`**, whose embedded launcher invokes that venv's Python and `rocm_sdk.__main__:main`. SHA-256 checks pin the executable, four SDK source files and the Triton AMD driver to the inspected versions; those hashes appear in each worker's evidence.

**The command is not universally read-only.** `_do_path()` calls `_devel.get_devel_root()`, which can expand developer files, reconcile hardlinks and rewrite package RECORDs, even after initial expansion. In this environment, `rocm_sdk_devel` is absent. The inspected function raises `ModuleNotFoundError` before those operations. The harness permits only this verified missing-package state and refuses discovery if the package becomes importable or the inspected source/executable hashes change.

The allowlisted request has a **5-second maximum timeout**, no shell, no additional arguments, no caller-provided environment/executable, and captures complete stdout/stderr. Its observed return code is **1**, with the missing `rocm_sdk_devel` diagnostic, in approximately 0.23 seconds. It does not obtain an SDK developer root or install one. Triton catches this `CalledProcessError` and continues to its existing fallback path.

Other subprocess requests, installers and network connections/DNS resolution remain blocked. Windows platform metadata is cached from `sys.getwindowsversion()` without launching `cmd /c ver`. GPU visibility is suppressed, GPU initialization is guarded, and FlexGEMM autotune cache loading/saving is disabled for the test. Temporary writes are confined to a disposable workspace scratch directory for each worker; installed directories and other paths remain write-blocked.

## Actual results

Every final worker used **Python 3.12.10**, **PyTorch 2.13.0+rocm10.0.0**, `torch.version.hip="7.15.26333"`, and `torch.version.cuda=None`. The HIP build string is recorded as reported by torch; it is distinct from the ROCm package version `10.0.0`.

Each import ran in its own `-B -I` subprocess with a **30-second timeout**. No worker timed out or crashed. Complete structured diagnostics and raw stdout/stderr are retained in [final results](amd-native-import-preflight-evidence/final/results.json).

Paths below are relative to `D:\AI\Trellis2AMD\venv\Lib\site-packages`.

| Module | Result | Path / boundary | Seconds |
|---|---|---|---|
| `cumesh._C` | PASS | `cumesh\_C.cp312-win_amd64.pyd` | 4.734 |
| `cumesh._cubvh` | PASS | `cumesh\_cubvh.cp312-win_amd64.pyd` | 6.438 |
| `cumesh._xatlas` | PASS | `cumesh\_xatlas.cp312-win_amd64.pyd` | 4.328 |
| `flex_gemm.kernels.cuda` | BLOCKED / incomplete | Parent package initialization stops before returning a module path; intended binary is `flex_gemm\kernels\cuda.cp312-win_amd64.pyd` | 4.797 |
| `o_voxel._C` | BLOCKED / incomplete | Parent package initialization stops through FlexGEMM; intended binary is `o_voxel\_C.cp312-win_amd64.pyd` | 5.984 |
| `nvdiffrast.torch` | PASS | `nvdiffrast\torch\__init__.py` | 4.328 |
| `_nvdiffrast_c` | PASS | `_nvdiffrast_c.cp312-win_amd64.pyd` | 4.312 |

The two blocked records have worker exit code 1, error `RuntimeError: Native preflight prohibits PyTorch GPU initialization`, and full tracebacks. The other five have exit code 0 and verified paths inside the intended venv.

The exact unresolved chain is:

```text
flex_gemm import
  -> kernels.triton.spconv.config.get_autotune_config()
  -> kernels.triton.utils.get_gpu_name()
  -> torch.cuda.get_device_name()
  -> get_device_properties() -> _lazy_init()
  -> preflight GPU-initialization guard
```

`o_voxel.__init__` imports `postprocess`, which imports FlexGEMM and reaches the same chain. No device name was fabricated, lazy initializer bypassed, backend code patched or native import retried with GPU initialization enabled. The trace does **not** establish that the metadata query launches a compute kernel; the harness conservatively keeps the GPU runtime uninitialized. Ordinary imports with metadata-driven initialization permitted may behave differently, but that was not tested.

## Harness verification and inventories

The discovery regression check first failed against the original guard with `Proof forbids external process/network action: subprocess.Popen`. After the narrow allowlist change, the guard checks passed. The first seven-worker run then exposed `filelock._strict._probe_link_follow_symlinks()`, which creates and removes a throwaway file/hardlink. Its trace is preserved in [initial results](amd-native-import-preflight-evidence/results.json).

After inspecting that probe, one controlled harness correction confined its temporary writes to workspace scratch directories. Five final regression tests passed: verified discovery reaches the missing-package diagnostic; other commands/shells are rejected; invalid timeouts/untrusted executables are rejected; a sleeping worker is killed within its bound; and writes outside scratch are rejected. [Guard test output](amd-native-import-preflight-evidence/guard-tests-final.stderr.txt) records exit 0. The controlled final native run returned exit 1 / PARTIAL for the initialization boundary above. There were no blind retries.

The [final before inventory](amd-native-import-preflight-evidence/final/inventory-before.json) and [final after inventory](amd-native-import-preflight-evidence/final/inventory-after.json) are identical. The initial and final inventories also match each other. Snapshots include selected wheel/PE dependencies, all installed distribution versions and METADATA/RECORD hashes in both venvs, inspected native binary hashes, SDK discovery source/executable hashes, and AMD source/backup hashes. Scratch directories were removed. This verifies the inventoried dependency/source files, not a full filesystem snapshot.

Reproduce only this bounded preflight from the proof checkout:

```powershell
& 'C:/Users/wattz/AppData/Local/Programs/Python/Python312/python.exe' -B -I scripts/compatibility/amd_native_import_preflight.py --amd-venv 'D:/AI/Trellis2AMD/venv' --modly-venv 'C:/Users/wattz/Documents/Modly/extensions/trellis2/venv' --amd-source 'D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD' --extension-source '../gguf-source' --output 'docs/compatibility/amd-native-import-preflight-evidence/reproduction'
```

**Stopped at this boundary.** No AMD adapter, worker implementation, GPU test or runtime compatibility patch is included. A native-import PASS has not been earned, and a successful import would still not validate geometry generation, texturing or GPU stability.
