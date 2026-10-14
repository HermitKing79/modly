# Genuine WSL turret neural decode — PARTIAL

Base: `d55a7abff2ad47c5cffdfec20866a76255e2358e`. Branch: `codex/amd-wsl-turret-neural-decode-vram-ready`. Reviewed candidate published at `295b6ce86bcd577ab5a0c5f3a9e817e965119a95`; exact execution harness and successful live-readiness evidence published and GitHub-verified before the genuine attempt at **`948e98a53d0eb3846716dc94c4ef8dbb3100e899`**.

**PARTIAL:** exactly one genuine `SparseUnetVaeDecoder.forward(..., return_subs=True)` was entered on the complete saved turret. **39/40 sparse convolutions returned and three learned subdivision stages completed.** The unchanged fourth-stage C2S admission guard stopped execution before expansion. No full neural output or artifact was produced. No retry, algorithm/precision change, guard relaxation, extraction or repair followed.

## Actual progress

| Measurement | Result |
| --- | --- |
| Metadata-only live readiness free VRAM | 14,518,063,104 bytes / 13.521 GiB |
| Actual decoder admission free VRAM | 14,618,472,448 bytes / 13.6145 GiB; required 13 GiB |
| Genuine TunableOp flags / environment | All false / all three strings `"0"`, both after Torch import and before GPU placement |
| Genuine FlexGEMM qualification | PASS, original binary hash, 18 callable bindings; 0.273 s |
| Strict checkpoint loading | PASS, 292 entries; no missing/unexpected entries; strict-load/widen 10.176 s |
| Runtime precision | 216 BF16 / 76 F32, reviewed conversion after strict original F16/F32 validation |
| Full original input | F32 `[3743,32]`, I32 `[3743,4]`; unchanged hashes and row order |
| Genuine forwards / convolutions returned / native neighbour maps | 1 / **39** / **4** |
| Learned support growth | **3,743 → 17,015 → 79,169 → 378,891** |
| Completed C2S calls | Six: two original expansions per completed learned stage |
| Peak allocated / reserved Torch bytes | **5,588,655,616 / 7,291,797,504** (5.205 / 6.791 GiB) |
| Post-release allocated / reserved bytes | **0 / 0** |
| Worker / supervisor / outer duration | 31.046 / 45.578 / 46.603 s |

The last completed convolution was `blocks.3.4.conv1`, at 378,891 rows and 7.004 s since GPU-stage entry. Cleanup synchronization succeeded. The journal interval from GPU-stage entry through worker finish was 7.273 s; **no complete-forward synchronized duration exists**, because the forward did not return. Intermediate feature finiteness and final geometry quality were not established.

## Precise stop and missing observation

The original fourth-stage pre-hook reached `admission.py:190`, then `c2s_plan()` refused with:

`AdmissionStop: PARTIAL: C2S parent/child rows or channel widths outside admitted bounds`

The parent count is 378,891; source/configuration establish feature/skip/output widths 512/128/64. Those parent and channel checks satisfy the unchanged contract. The failing condition is the authentic boolean-mask child count:

`0 < children <= min(8 * 378891, 1000000)`

The count was computed before the original expansion but **not preserved before this guard raised**. It can therefore be **0 or 1,000,001–3,031,128**. The evidence does **not** distinguish empty support from excessive expansion; “exceeded one million” alone is unproven. The fourth expansion and its subsequent memory-plan calculation were never executed. This was not an observed OOM, timeout, driver reset or native crash. Peak usage below the allocator ceiling does not justify bypassing the row limit.

[Worker](amd-wsl-turret-neural-decode-vram-ready-evidence/run/guest/results.json), [durable operation journal](amd-wsl-turret-neural-decode-vram-ready-evidence/run/guest/phases.jsonl), [supervisor](amd-wsl-turret-neural-decode-vram-ready-evidence/run/results.json), and [verified summary](amd-wsl-turret-neural-decode-vram-ready-evidence/verified-result.json) preserve genuine progress and the original traceback. The loader's inherited `model_forward_calls=0` diagnostic describes its loading interface; the separate explicit parent invocation is recorded as `forward_calls=1`.

## Safety, preservation and stop

Ten Windows observations were healthy. Worker exit 1 was a controlled refusal. Job accounting was 0→0 with no forced cleanup; only named Modly Ubuntu termination succeeded. Model/component/parameter/input references were collected; BLAS workspace clearing occurred once; allocator release reached zero. Both distros are stopped and docker-desktop remains default.

Only fresh paths changed in the reviewed decoder; runtime/admission code remained byte-identical. The separate readiness adapter performed metadata queries without tensors, weights or native/model imports and exited cleanly. Two focused metadata/path tests, five existing startup checks, read-only preflight and independent source/outcome reviews passed. All original deadlines, 12 GiB allocator cap, reserves, one-million-row ceiling, original `explicit_gemm`, learned masks and containment were retained.

Before/after preservation matched Windows environments/binaries, AMD edits/backups, all four turret artifacts, decoder configuration/checkpoint identity, baseline WSL venv/packages, 922 staged payloads, 23 packaging files, 80 native source files and protected ROCm dependencies. All 523 protected historical/bytecode files and main/56 prior local and remote branches are preserved. No packages, models, drivers or production worker changed.

The smallest missing observation is the rejected fourth-stage child count. Any next decision must distinguish zero support from an over-limit count and assess the original allocation plan before considering a budget change. **Do not assume a numerical defect or raise the ceiling based on this run.** Stop here for approval; no automatic rerun, neural output recovery, geometry extraction or texturing.
