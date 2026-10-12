# AMD bounded GPU compatibility proof

**Result: PASS for the two approved gates — 12 October 2026.** The existing AMD runtime initialized on the RX 9070 XT, imported FlexGEMM and o-voxel, and executed one deterministic CuMesh boundary-component fixture correctly. This does **not** validate full TRELLIS generation, hole filling, texture baking, or general GPU stability. The previously observed error 719/native crash on a real mesh remains unresolved.

## Preservation and interrupted-run recovery

The proof checkout started clean at `be8a90379c08f5fd21c941bcb9092da6b7cd69fa`, with `origin=https://github.com/HermitKing79/modly.git`. Work is confined to compatibility scripts, documentation and evidence on **`codex/amd-bounded-gpu-proof`**. Main (`f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`), the first proof branch (`874a843e240d6c96b6149f7908c1734472ec0dcc`), and native-import branch (`be8a903…`) were preserved. The [bounded proof](amd-trellis2-bounded-proof.md) and [native preflight](amd-native-import-preflight.md) remain unchanged historical records.

The interrupted [Gate 1](amd-bounded-gpu-proof-evidence/gate1/results.json) remains **PARTIAL**, with zero workers, unhealthy monitoring and unequal inventory digests. Its five artifacts were preserved byte for byte. Recovery did not restart GPU testing until reconciliation, independent observer tests and review completed.

Structural inventory comparison found exactly two differences: `/venv_file_metadata/amd/tree_sha256` and `/venv_file_metadata/modly/tree_sha256`. All distribution versions, METADATA/RECORD hashes, selected native binaries, SDK files and inspected source hashes were identical. Two passive scans reproduced **4,565 AMD and 4,242 Modly directory size changes**, from zero to positive allocation sizes; names, file sizes, timestamps and modes were identical. Both warm-scan digests exactly matched the historical after digests. [Full differences and original artifact hashes](amd-bounded-gpu-proof-evidence/recovery/reconciliation.json), [reconciliation summary](amd-bounded-gpu-proof-evidence/recovery/reconciliation-summary.json), and both full metadata scans are retained.

The harness now normalizes only directory allocation size to `null`; it still compares directory names/modes/timestamps and every file's name/size/mode/timestamp. Future raw manifests are also saved as compressed JSON. Historical per-directory values were not saved and cannot be reconstructed; the passive reproduction explains the aggregate mismatch rather than recovering those missing values.

## Observer repair and controls

[Diagnosis](amd-bounded-gpu-proof-evidence/observer-original-diagnosis.json) established that Python-launched Windows PowerShell 5.1 inherited PowerShell 7 module paths and selected Core Diagnostics 7. Its `GetEvent.types.ps1xml` failed an AuthorizationManager check. A sandboxed event read separately returned `UnauthorizedAccessException`; reliable monitoring requires the approved host execution context.

The harness pins stock Windows PowerShell, confines **that child process's** module path to `$PSHOME/Modules`, and explicitly imports stock Diagnostics. No execution policy, module installation or permission setting changed. A positive read of the latest real System event must succeed before the filtered reset/WHEA/power-event query is accepted. [Two independent repaired observer checks](amd-bounded-gpu-proof-evidence/observer-fixed-check-1.json) and [second check](amd-bounded-gpu-proof-evidence/observer-fixed-check-2.json) passed. Missing observations, errors, timeouts, unhealthy controller status or matching System events stop the gate.

Each native worker has a **30-second timeout**; observer queries have 10 seconds and the verified SDK discovery command 5 seconds. Processes, network access and Python-audited writes outside workspace scratch remain blocked. Temporary/home/cache paths are redirected to disposable scratch, bytecode writes and FlexGEMM cache loading/saving are disabled. Gate 1 rejects torch tensor operations and Triton compilation/launches. These are source-reviewed harness controls, not OS-level containment of arbitrary native code. No packages, drivers or source checkouts were edited; no models, inference, weights, installers or extension builds were used.

## Exact runtime and results

Workers used `D:\AI\Trellis2AMD\venv\Scripts\python.exe`: **Python 3.12.10**, **torch 2.13.0+rocm10.0.0**, HIP **7.15.26333**, CUDA **None**, ROCm distribution **10.0.0**, Triton Windows **3.8.0.post29**. The selected device was **AMD Radeon RX 9070 XT / gfx1201**, with 17,095,983,104 reported memory bytes. Windows driver version was **32.0.31041.1004**, controller status OK.

| Approved worker | Result | Time / process result |
|---|---|---|
| `flex_gemm.kernels.cuda` | PASS; `flex_gemm\kernels\cuda.cp312-win_amd64.pyd` | 6.141 s; exit 0 |
| `o_voxel._C` | PASS; `o_voxel\_C.cp312-win_amd64.pyd` | 7.594 s; exit 0 |
| CuMesh triangle boundary fixture | PASS; `cumesh\_C.cp312-win_amd64.pyd` | 4.125 s; exit 0 |

All module paths were verified inside the AMD venv. Gate 1 contexts changed from uninitialized to initialized, with **zero PyTorch tensor bytes allocated/reserved**. Both imports ran once. [Resumed Gate 1 evidence](amd-bounded-gpu-proof-evidence/gate1-resumed/results.json) records complete metadata and SDK diagnostics; the missing-devel SDK command still exits 1 and Triton handles that existing fallback without installation.

For Gate 2, the installed Python wrapper matched the normalized Git blob of the [inspected ROCm source](https://github.com/dmonkman/CuMesh-ROCm/blob/e28fe419201ce134deeccf0947bb97b8228f2e82/cumesh/cumesh.py). The [HIP reference implementation](https://github.com/dmonkman/CuMesh-ROCm/blob/e28fe419201ce134deeccf0947bb97b8228f2e82/src/connectivity.hip#L837) derives manifold boundary adjacency, hooks/compresses connected labels, then compresses IDs. The wheel does not embed its compiled source commit; reference inspection is distinct from binary identity. Tested `_C` SHA-256: `32526ca0976822800b2320ee3cc12db5ef3fc8fb4641d721dabca8714e5b71ac`. [Inspection/provenance](amd-bounded-gpu-proof-evidence/native-source-inspection.json) records prerequisites and source references.

The single fixture had float32 vertices `[[0,0,0],[1,0,0],[0,1,0]]` and int32 face `[[0,1,2]]`, contiguous on one HIP device. Operations were `init`, `get_edges`, `get_boundary_info`, vertex-edge/boundary adjacency, manifold boundary adjacency, and **one** `get_boundary_connected_components()` call. Each stage flushed diagnostics and synchronized before advancing. No boundary-loop extraction, hole filling or remeshing ran.

Actual result: **1 component, IDs `[0,0,0]`, 3 boundary edges** (`[[1,0],[2,0],[2,1]]`), unchanged 3 vertices/1 face. [Gate 2 result](amd-bounded-gpu-proof-evidence/gate2/results.json) and [stage log](amd-bounded-gpu-proof-evidence/gate2/01-cumesh-boundary.stdout.txt) preserve the exact output. No exception, timeout, native crash, error 719 or matching System event was observed. PyTorch's tracked peak tensor memory was 2,048 bytes; native HIP allocations and total VRAM were not measured.

## Verification and next milestone

Six [supervisor regression checks passed](amd-bounded-gpu-proof-evidence/supervisor-tests-final.json), including fail-closed observation/Gate 2 checks, bounded termination, directory-size normalization that retains file-change detection, and independently expected fixture results. Broader `npm test` exited 1 on existing missing `fastapi` imports (including `test_agent_router`/`test_agent_workflow_unload`) and sandbox temporary-path permissions (including `test_download_progress_print_is_ascii_safe`); no dependency reconstruction was attempted. That run is not a GPU compatibility result.

Resumed Gate 1 and Gate 2 before/after inventories are identical, and Gate 2's baseline matches Gate 1's final inventory. They cover all distribution metadata, selected native/source content hashes and complete normalized venv file metadata; they are not a full content hash of every installed file. All scratch directories were removed. The AMD source stayed at `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, preserving modified `nodes.py` and untracked `nodes.py.before-profiler-fix` with unchanged hashes. Both existing AI environments remain unchanged within those checks.

**Next justified milestone: a separately approved offline AMD Modly worker scaffold**, with real Modly IPC under Python 3.12, explicit runtime/source provenance, offline model-path validation, actionable dependency errors and bounded process shutdown. Its acceptance should initially use mock outputs and headless imports; it must not silently install dependencies or claim generation readiness. No specific native failure was reproduced in this tiny fixture. The earlier real-mesh 719 failure still requires separate targeted validation before enabling geometry/texturing. GGUF model compatibility, full pipeline imports, 16 GB generation capacity and image-to-textured-GLB quality remain unproved.

**Stopped after the bounded proof. No adapter implementation or additional GPU run is included.**
