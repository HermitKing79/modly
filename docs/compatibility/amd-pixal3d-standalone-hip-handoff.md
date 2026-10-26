# Pixal3D model-free standalone HIP proof

**Prepared for one deliberate user-run proof after Codex closes. Actual HIP compatibility is UNTESTED. No generation is authorized.** This implements the approved model-free step from `d9191fda03aa9e79da4da2ee42db1c7ca71c9906`, retaining the failed-admission evidence at `9cbc5709753170f108d08e036750ebc99d3762a2` and the latest immutable geometry findings at `25a9a6797339b298c4113d3f6223246ff0cc4b2d`.

Launcher: `scripts/compatibility/Invoke-Pixal3DHipProof.ps1`. The original [approved contract](amd-pixal3d-native-single-image-proof-contract.json) is unchanged; the [model-free execution lock](amd-pixal3d-standalone-hip-contract.json) reuses its software, hardware and resource pins plus the already qualified runtime-wheel manifest. There was no dependency re-resolution or repeat model research.

## Before closing Codex

- Keep the checkout on the exact published handoff commit and retain the existing qualified Windows Python 3.12.10, Git and `Modly-AMD-Ubuntu-24.04` WSL distro/user `modly` with Python 3.12.3 and installed ROCm/ROCDXG runtime. No setup or WSL launch is needed now.
- The fixed isolated roots must be unused: Windows `D:\AI\Modly-Pixal3D-Proof-v1`, Linux `/home/modly/pixal3d-native-proof-v1` and its `.venv`. The Windows root was verified absent during preparation. Linux paths were deliberately not queried in this session; the live controller refuses reuse.
- HTTPS access is needed for pinned missing payloads. D: must retain the existing 50 GiB floor plus the calculated staging/install headroom; output volume and guest retain 2 GiB floors. Exact wheel expansion is checked before installation.
- Copy the final handoff command into a normal, non-administrator PowerShell window. Close Codex Desktop and ChatGPT Desktop; close RAM-heavy browsers manually if needed. The script refuses those desktop processes and never closes applications. Do not start WSL manually to prepare for this test.

The command supplied with the verified GitHub commit has this shape:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\wattz\Documents\ChatGPT\modly\proof\scripts\compatibility\Invoke-Pixal3DHipProof.ps1" -Run -ExpectedCommit "<exact-published-40-character-commit>"
```

Without `-Run`, PowerShell performs no readings, child launch, download, environment creation or GPU work. A mismatched checkout/source/interpreter stops. `ExecutionPolicy Bypass` applies only to that PowerShell process, not persistent settings.

## What the live invocation does

1. Save fresh private evidence, use the original qualified `GlobalMemoryStatusEx` reader and original three-reading implementation: **each reading >=22 GiB, four seconds apart**. A first/later failure stops immediately. Take a final 22 GiB reading before payload execution. **24 GiB stays preferred, not mandatory.** No neural-attempt marker is used.
2. In a bounded owned Windows child, reuse verified cache files or download only exact locked HTTPS wheel/source URLs. A present cache with wrong identity stops; it is not replaced. Inspect the Kitchen wheel first for a Linux x86-64 HIP ELF and gfx1201 architecture manifest. Missing/incompatible payload stops before WSL provisioning; no build fallback.
3. Start the named distro as `modly` with an owned nonce-bound Linux process group. It writes its PID/start-ticks/group receipt and waits for host GO. The Windows client must be attached to its Job first. Recheck final 22 GiB before each provisioning/operator GO. Guest root exclusivity, disk capacity, system Python and baseline runtime identity must pass.
4. Create a new independent `.venv` with system site-packages disabled. Install only locked prebuilt wheels using `--no-index --no-deps --only-binary=:all: --require-hashes --no-compile`; no resolver, APT, driver or kernel build. Stage three exact pinned ComfyUI operator source files. Flush/request eviction of only this newly created tree's file cache; there is no global cache clearing and physical RAM recovery is not guaranteed. Final 22 GiB still applies before the GPU child.
5. Run one synthetic operator worker. Require exactly one unmasked RX9070XT/gfx1201 device, the qualified Torch/HIP identities, genuine Kitchen HIP extension identity/registration and HIP selection. Disable CUDA native loading, global Triton execution and eager fallback for required HIP operations. Any failure stops the remaining operators.
6. Verify independent finite CPU references, collect memory/timing/dispatch evidence, require zero owned Torch allocated/reserved bytes before child exit, then verify natural owned-group/client cleanup. A forced or unverified cleanup cannot yield PASS. Never shut down WSL or terminate the shared distro.

No automatic retry occurs. If an admitted stage fails, keep its used roots and evidence for review; do not delete/rename them to replay the test. A late final-RAM failure stops even if provisioning already succeeded.

## Exact installation and download scope

The new environment uses Torch `2.9.1+rocm7.2.1.lw.gitff65f5bc` (runtime `2.9.1+rocm7.2.1.gitff65f5bc`), qualified runtime dependencies, NumPy 2.2.6 and Kitchen 0.2.40. Thirteen wheel archives total **2,032,899,944 bytes**. Eleven existing cached runtime wheels were hash-verified read-only during preparation. With those caches present, the missing Kitchen/NumPy downloads total **84,338,837 bytes**. Three native ComfyUI source files total 62,205 bytes and have verified local source caches. If an approved cache is absent, only that exact pinned URL may be downloaded. Any missing URL/hash mismatch stops without a new version or source build.

The pinned Triton wheel is part of the existing Torch runtime dependency closure; its import/execution is disabled in the operator worker. No full ComfyUI frontend/server, custom nodes, Manager, native extension compiler or new ROCm/Torch version is installed. The existing system ROCm libraries are read-only runtime prerequisites; existing site-packages are not shared with the new venv.

**No Pixal3D/DINO/MoGe/VAE weights, image conditioning, occupancy, sampling, neural decode, generation attempt, remeshing, UV, baking, mesh modification or texture generation.** Zombie D and the TRELLIS.2 installations are never execution inputs or write targets for this proof.

## What is actually qualified if PASS

| Check | Test and evidence |
| --- | --- |
| HIP INT8 / ConvRot | Synthetic 64x256 inputs and 64 output channels; HIP registry selection, genuine extension hash, native quantization/GEMM call counters, GPU-resident outputs and HIP events; independent Python quantization/Hadamard/integer-dot references. BF16 tolerance abs0.02 + rel0.02 is fixed before execution. |
| PyTorch attention | FP16 GPU math SDPA compared with independent Python dot/softmax/value loops, abs/rel0.002. Flash/memory-efficient attention is not qualified by this probe. |
| NAF | Exact pinned `na2d_pure`, FP16 two-head HR/LR neighborhoods and independent zero-padding/attention loops, abs/rel0.002. |
| Sparse operations | Exact pinned BF16 sparse gather/GEMM function on eight occupied voxels; independent neighborhood convolution loops, abs/rel0.02. |
| Conversion | Exact pinned dual-grid function, four cells/two known triangles; literal coordinate/index reference, exact comparison. No source mesh is used. |

Native ComfyUI definitions are extracted unmodified from hash-verified source, with a read-only free-VRAM query adapter. Model classes and the full application are not imported. This qualifies these tiny operators, not full ComfyUI integration, a 512-resolution model run, quality or 16GB model fit. A PASS permits only preparing a separately reviewed single-image generation handoff; the runner cannot continue to generation.

## Safeguards and telemetry

Retained limits: 12 GiB Windows live-host floor; D:50 GiB and artifact/guest2 GiB free; original8 GiB download/20 GiB guest-storage growth bounds; 13 GiB initial free VRAM,3 GiB live free VRAM and12 GiB Torch allocator cap. Windows client/payload children have6 GiB process commit caps; Linux process-tree RSS is independently polled against6 GiB. Payload/provisioning stages have180-second bounds, operator overall360 seconds, CPU phase120 seconds and GPU-release phase180 seconds. No bound is raised after failure.

Record Windows host availability and observer PID/working set/private commit; keep host and payload-child memory journals separate. Windows Job peaks refer to Windows clients, not Linux operators. Linux child `getrusage` records OS peak RSS, while the controller records sampled process-tree RSS sums at0.25-second intervals. GPU allocator peaks/release, free VRAM, per-operator errors/counters, GPU event intervals excluding CPU references, wall times, exit codes and nonce-bound cleanup receipts remain private. Host availability includes concurrent activity; the script does not claim complete physical WSL-memory reclamation or attribute all host changes to this proof.

## Reopen Codex and inspect

The console prints the exclusive evidence directory under local `artifacts/amd-pixal3d-standalone-<UTC>-<nonce>`. Inspect **REPORT.txt** first, then **summary.json**, `admission.jsonl`, host-memory journals, `provision/` and `operators/`. Individual operator JSON files and `operators/child-result.json` retain partial results even if a later stage fails. Interrupted or truncated result JSON must not suppress the final FAIL report.

Preparation exercised only mocks/synthetic CPU tests and no-execution dry-runs. Actual Windows admission, WSL provisioning, HIP extension import and operator execution remain untested until the deliberate user run. Independent code review and preservation results are recorded in the accompanying [verification](amd-pixal3d-standalone-hip-verification.json).
