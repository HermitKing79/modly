# Genuine turret neural output saved — PASS

Base: `487085651fd0624db7f6168991ad5258bc2d2da2`. Branch: `codex/amd-wsl-turret-artifact-writer-ready`. The reviewed execution harness and successful CPU writer/readiness evidence were published and GitHub-verified at **`e48f03a6809ac58d197c27edcd4dce77a45527d2`** before the single genuine decoder attempt on 15 October 2026 UTC.

**PASS: the full neural turret output is saved, reopened, independently verified before release, and retained unchanged after worker termination.** This is neural data at grid resolution 512, not an extracted mesh or proof of detailed shape fidelity. No extraction or texturing was attempted.

## Durable local artifacts

Directory: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-wsl-turret-writer-ready-4870856`

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `turret-neural-output.safetensors` | **96,433,760** | `dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22` |
| `provenance.json` | **22,024** | `8f428dea1c98ffabbea24a1986d388094a29bb9c9123a3a0dc48a55f9229a625` |

The ten tensors contain F32 features **`[1842968,7]`**, I32 coordinates **`[1842968,4]`**, and four pairs of BF16 subdivision logits `[parents,8]` / I32 parent coordinates `[parents,4]`. Parent counts are 3,743, 17,015, 79,169 and 378,891. Coordinates preserve original `[batch,D,H,W]` C2S ordering. Payload size is 96,412,768 bytes. Finiteness, coordinate uniqueness/bounds, mask-to-next-stage linkage, metadata, tensor equality and raw tensor-byte readback passed.

The Windows verifier independently read the complete file, bounded header, all ten payload hashes, provenance sidecar and attempt identity while the worker still held its output. The writer required that acknowledgement before returning. The same independent check passed again **after worker termination**, with identical hashes. [Post-termination integrity receipt](amd-wsl-turret-artifact-writer-ready-evidence/neural-post-termination-integrity.json) includes every tensor hash and exact Windows paths. The large output remains local outside Git; evidence and hashes are published.

## Correction qualified before GPU execution

The trusted Windows host created only the exact absent `modly\artifacts` parent. Checks verified its resolved path, current-user ownership/permissions and absence of reparse redirection. WSL confirmed the matching existing `/mnt/c/.../modly/artifacts` mapping. ImportPolicy still permits only each exact worker leaf, scratch and evidence roots; the parent was **not** whitelisted. Leaf creation is nonrecursive and exclusive.

One genuine CPU-only WSL acceptance exercised the **same shared production writer**, real unchanged ImportPolicy, Torch and safetensors. Ten clearly labelled tiny fixture tensors (344 payload bytes; 1,720-byte file) passed save/load, metadata, provenance, byte equality and independent verification before release and after exit. GPU-entry calls, initialization, native calls, model loading and policy rejections were all zero. The fixture is isolated under `artifacts\cpu-writer-acceptance-4870856` and is not genuine generated data.

Four focused host tests passed, including unchanged computation/validator identity, nonrecursive leaf creation, path-traversal rejection and complete atomic receipt publication. Independent review approved CPU acceptance and then reviewed its actual result before approving the conditional decode. The eleven existing chunk-equivalence tests were **not repeated**: `admission.py`, `chunked_final.py` and `runtime.py` are byte-identical to the proven candidate; the genuine forward and output-validator functions are unchanged. Earlier host-only PowerShell/module issues were resolved using the already installed PowerShell Core executable, without changing execution policy or environments; they preceded the single WSL acceptance.

## One genuine execution

| Measurement | Observed |
| --- | --- |
| Parent forwards / original convolutions / native global maps | **1 / 40 / 5** |
| Learned support | **3,743 → 17,015 → 79,169 → 378,891 → 1,842,968** |
| Original C2S calls / final convolution tiles | **8 / 57**, tile size 32,768; tail 7,960 |
| Complete forward, GPU synchronized | **11.389517421 s** |
| GPU stage, including output persistence and release | **19.445147131 s** |
| Recorded worker / supervisor duration | **43.224 / 60.625 s** |
| Initial device-free VRAM | **14,634,979,328 bytes** |
| Peak Torch allocated / reserved | **5,588,655,616 / 7,291,797,504 bytes** |
| After release: Torch allocated / reserved | **0 / 0 bytes** |
| Device-free VRAM after release, before process exit | **14,485,303,296 bytes** |

The genuine checkpoint hash remained `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. All 292 tensors loaded strictly; the reviewed 216 BF16 / 76 F32 runtime configuration, original full turret inputs, false TunableOp flags, learned masks, global neighbour maps and final row-tiled explicit-GEMM computation were retained. No model stage was repeated beyond the one approved decoder forward.

Ubuntu WSL2 used Python 3.12.3, Torch `2.9.1+rocm7.2.1.gitff65f5bc`, Triton `3.5.1+rocm7.2.1.gita272dfa8` and staged safetensors 0.6.2 on RX 9070 XT/gfx1201. The unchanged 13 GiB initial admission, 12 GiB allocator ceiling, operational reserves, scoped final-row exception and 60/120/240-second native/CPU/GPU deadlines remained enforced; guest/host limits remained 450/480 seconds.

[Raw supervisor result](amd-wsl-turret-artifact-writer-ready-evidence/run/results.json), [durable operation journal](amd-wsl-turret-artifact-writer-ready-evidence/run/guest/phases.jsonl) and [CPU writer result](amd-wsl-turret-artifact-writer-ready-evidence/writer-run/results.json) preserve the actual counters, timings, validation and receipts.

The [independent outcome review](amd-wsl-turret-artifact-writer-ready-evidence/independent-outcome-review.json) accepted PASS after checking the raw result, exact execution commit, unchanged code identities, persisted file hash and both independent receipts.

## Release, preservation and next boundary

All tracked model, parameter and GPU input/output references were collected; BLAS workspace clearing occurred once. Worker exit was **0**, Windows observation remained healthy, the owned Job was empty without forced cleanup, and only named Modly Ubuntu was terminated. Both distros are stopped and Docker remains default.

Before/after audits matched both Windows environments, AMD source edits/backups, native binaries, checkpoint/configuration and saved turret inputs. Linux baseline venv/packages, 922 staged payloads, 23 packaging files, 80 native source files and protected ROCm dependencies matched. Only resource availability and fresh owned scratch/output differed. Driver remained `32.0.31041.1004`. **719 historical files, five control files, two loader files, main and all 59 previous local/remote branches** remained unchanged. Existing untracked bytecode caches were preserved outside the commit. No installation, compilation, precision fallback, download or source modification in either installed application occurred.

The smallest next milestone is separately approved **native O-Voxel extraction from this saved output**, after qualifying the required native binary and bounding its resource requirements. Reuse the pinned decoder's sigmoid vertex transform, channels `3:6 > 0` intersection flags, softplus split weights, grid 512 and AABB ±0.5. Do not rerun the neural decoder. Native extraction, mesh fidelity, repair and texturing remain unproved; no mesh or GLB was produced here. Stop for approval.
