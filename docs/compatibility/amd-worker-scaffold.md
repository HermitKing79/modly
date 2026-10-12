# Experimental AMD worker scaffold

**PASS for this scaffold milestone — 12 October 2026.** Modly's existing registry,
`ExtensionProcess` and `runner.py` launched the selected AMD Python 3.12 worker.
Both Generate Mesh and Texture / Refine mock routes passed IPC and lifecycle
checks. **No AI models, geometry or textures were generated.** Outputs are JSON
diagnostic receipts, not GLBs.

Work is on `codex/amd-worker-scaffold`, based on the clean
`codex/amd-bounded-gpu-proof` commit
`521d3d29e06fdde2e324e7d1bce7e03ef10dfaa8`. Main and the three prior proof
branches/reports are preserved. The existing AMD source remains at
`3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, with its modified `nodes.py` and
untracked `nodes.py.before-profiler-fix` intact.

## Integration and runtime contract

The extension lives under `experimental/extensions/trellis2-amd`, separate from
production extensions. Discovery requires `MODLY_ENABLE_EXPERIMENTAL_AMD=1`.
An absolute `MODLY_AMD_RUNTIME_CONFIG` JSON file explicitly selects the Python
executable, existing AMD source root and local safetensors checkpoint prefixes
for each node. See the [configuration and usage notes](../../experimental/extensions/trellis2-amd/README.md).
It has no installer or automatic discovery/fallback path.

The host forces this contract into subprocess mode even without an
extension-local venv. The worker uses `-B -I`; startup checks precede readiness.
The parent verifies the requested interpreter, ROCm torch metadata, format,
mock-only status and exact node identity. Missing/incompatible dependencies
produce errors without pip repair or another interpreter/device selection.
Existing production extension behavior remains outside this opt-in contract.

The [actual IPC evidence](amd-worker-scaffold-evidence/final/results.json) records:

| Boundary | Observed result |
|---|---|
| Host | Installed Modly Python 3.11.9, running this checkout's API code only |
| Worker | `D:\AI\Trellis2AMD\venv\Scripts\python.exe`, Python 3.12.10 |
| PyTorch | `2.13.0+rocm10.0.0`; HIP `7.15.26333`; CUDA `null` |
| Native provenance | AMD64 PE/cp312 binaries; HIP imports for GPU components; package paths within the selected venv |
| CPU UV utility | Existing `_xatlas` binary accepted only at its inventoried SHA-256; no CPU inference fallback |
| Source | Selected AMD checkout commit and pipeline/model source SHA-256 values recorded; no upstream imports |
| GPU/model state | Readiness measured GPU context uninitialized; weights not loaded; tensor operations and GPU initialization guarded |

CuMesh `1.0+rocm10.0`, FlexGEMM `1.0.0+rocm10.0`, nvdiffrast `0.4.0+rocm10.0`,
o-voxel `0.0.1+rocm.10.0`, ROCm `10.0.0` and triton-windows `3.8.0.post29` are
pinned. Native binaries and Python wrappers are inspected as files, not imported
in this milestone. The RX 9070 XT identity and tiny CuMesh result come from the
[earlier bounded GPU proof](amd-bounded-gpu-proof.md), not a new GPU query/run.

## Acceptance evidence

| Scaffold acceptance criterion | Result and limit |
|---|---|
| Python 3.12 worker launch and verified identity | PASS, both independent node workers |
| Generate and Texture / Refine IPC | PASS, image-byte transport, progress and JSON mock receipts; Refine validates an existing tiny GLB container fixture |
| Headless adapter import | PASS, own adapter module imports without torch, TRELLIS or ComfyUI; runtime preflight later imports torch metadata only |
| Offline paths / errors | PASS, paired local JSON/safetensors headers; GGUF, remote/UNC/device/mapped network paths, reparse points and missing files rejected |
| No CUDA/CPU fallback or installation | PASS, pinned AMD contract; an alternate Python's torch `2.12.0+rocm7.14.0` rejected before torch import |
| Termination / recovery | PASS, rejected inference request followed by successful same-process mock, unload/reload, cooperative shutdown, verified hard stop/restart |
| Bounded failures | PASS, startup/load/response waits 30/10/30 seconds, separate five-second write limit; timeout/cancellation tests and synthetic host-plus-child cleanup |

Model-path validation reads at most 1 MiB of JSON and safetensors header metadata
per checkpoint. The proof uses explicitly labeled metadata-only checkpoint
fixtures. It does **not** validate real tensor shapes, all required pipeline
weights, architecture compatibility or GLB geometry. A mock `loaded` flag means
only scaffold readiness. Manifest mesh ports reserve the eventual integration;
mock JSON receipts cannot serve as mesh workflow inputs. The graphical workflow
and GLB viewer were not exercised.

The worker prohibits network/subprocess actions and writes outside its workspace.
Offline environment variables and caches point to disposable workspace scratch.
The runtime never calls upstream `from_pretrained`, whose missing-local-file
behavior would otherwise download from Hugging Face. Safetensors and GGUF are
explicitly separate; [earlier GGUF findings](amd-trellis2-bounded-proof.md) remain
unchanged. Python audit/dispatch guards constrain this trusted scaffold; they
are not a security sandbox for arbitrary native code.

## Tests, review and preservation

The final focused **61 tests** and **2 synthetic supervisor tests** passed. The
broader API run completed 160 tests with **12 import errors and 4 skips**, due
to missing `fastapi`, `trimesh` and `starlette` in the system test interpreter.
Node's first stage passed all 47 tests; its second stage passed 4 and failed 29
because `esbuild` / `jsdom` are absent. These unrelated dependencies were not
installed or repaired. The broader application suites are not passing.
See [test results](amd-worker-scaffold-evidence/tests/) for exact commands and
complete stdout/stderr. Read-only code review identified unbounded IPC writes,
host-tree cleanup, cancellation handle loss and network-path acceptance. Those
were fixed and covered by focused regression tests. PASS classification also
requires clean supervision, complete IPC evidence and an unchanged inventory.

Two interrupted validation attempts are preserved: the first conservatively
rejected CPU `_xatlas` for lacking HIP linkage; static inventory confirmed its
exact AMD-wheel hash. The second reached both mock routes but its negative-test
expectation assumed the alternate interpreter lacked torch; its actual
incompatible version was correctly rejected. The harness expectation was
corrected. Neither attempt was an inference run, native failure or blind retry.
Both inventories remained unchanged. Windows sandbox path-resolution failures
in initial unit verification were resolved by running the same focused tests
with workspace-only scratch outside that sandbox; no runtime guards were removed.

The [final before](amd-worker-scaffold-evidence/final/inventory-before.json) and
[final after](amd-worker-scaffold-evidence/final/inventory-after.json) inventories
are structurally equal. They cover both installed venvs' distributions,
METADATA/RECORD hashes, native binary hashes, inspected package/source hashes and
full file-tree metadata. Compressed raw file manifests are retained; directory
allocation size normalization follows the previous proof. This confirms the
inventoried environment/source state, not a full content hash of every file.
No installed application, package, driver or existing source edit was modified.

## Next milestone and remaining risk

The smallest next milestone is a **bounded headless safetensors pipeline import
and complete offline model-configuration audit** inside this worker. It should
resolve the actual ComfyUI `folder_paths` / `comfy.utils.ProgressBar` import
boundary, enumerate every required local model pair and auxiliary model, and
prove no network fallback before any separately approved model-loading test.
This scaffold intentionally does not import the upstream pipeline package, which
pulls ComfyUI and native components with GPU metadata side effects.

The earlier CuMesh error 719/native crash during real geometry hole filling is
still unresolved. The successful tiny boundary fixture says nothing about that
input, full geometry, AI textures, VRAM pressure or a correctly shaped textured
GLB. The eventual zombie-image-to-textured-GLB objective remains unproved.

**Stopped after the scaffold. No automatic model loading, GPU testing or inference.**
