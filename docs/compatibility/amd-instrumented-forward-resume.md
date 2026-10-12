# Resumed instrumented AMD sparse-decoder forward

Date: 2026-10-12. Base: `e2842ffc0e443840066aa839cc1cfb98c16969d3`.
Branch: `codex/amd-instrumented-decoder-forward-resume`.

**PASS for one genuine sparse-decoder forward on RX 9070 XT.** Exactly one
strict checkpoint load and one complete decoder invocation produced the expected
finite float32 `[1,1,4,4,4]` output, released owned tensors and exited cleanly.
No warm-up, retry, other model, mesh extraction or asset generation occurred.

## Reviewed and published before execution

The [diagnostic correction](amd-forward-diagnostic-correction.md) and prior
[instrumented attempt](amd-instrumented-sparse-decoder-forward.md) were read first.
The candidate changed only three evidence-path references in two harness files;
all reviewed diagnostics, guards, cache procedures, ownership, observers and
45/120/180-second limits were retained. A new canonical run directory and exclusive
marker protect every historical attempt.

[52 focused CPU tests](amd-instrumented-forward-resume-evidence/cpu-candidate.json)
passed with AI imports blocked, including complete GPU identity, PyTorch's
version representation and actual-expression lifecycle replay. The
[focused read-only review](amd-instrumented-forward-resume-evidence/pre-attempt-review.md)
found no blocking issue. The exact harness was pushed and independently verified
on GitHub at [`4f51dfdced5c2da37289c45f1e0b2cc48c0503a4`](https://github.com/HermitKing79/modly/commit/4f51dfdced5c2da37289c45f1e0b2cc48c0503a4)
before launch; the supervisor checked all 19 execution dependencies against it.

[Cache preparation](amd-instrumented-forward-resume-evidence/cache-plan.json)
reverified 524 historical files, container/target/database validity and four
installed ROCm library attestations. The same nine eligible files were copied
into fresh writable cache space; 515 uncertain/temporary files were excluded.
Historical files were never runtime cache paths. The passive Windows preflight
was healthy, followed by fresh supervisor and periodic live monitor gates.

## Actual result

The [raw result](amd-instrumented-forward-resume-evidence/run/results.json),
[durable trace](amd-instrumented-forward-resume-evidence/run/trace.jsonl) and
[independent CPU analysis](amd-instrumented-forward-resume-evidence/attempt-interpretation.json)
establish:

| Boundary | Recorded result |
| --- | --- |
| Runtime | Python 3.12.10; PyTorch 2.13.0+rocm10.0.0; HIP 7.15.26333; CUDA metadata null; existing AMD venv |
| Device | AMD Radeon RX 9070 XT; gfx1201; 17,095,983,104 total bytes |
| Checkpoint | Same audited `ss_dec_conv3d_16l8_fp16`; all 74 entries / 147,585,412 payload bytes strictly matched, no missing/unexpected keys; CPU load once in 1.406s |
| CPU gate | GPU uninitialized; working set after load 644,169,728 bytes; private commit 1,086,836,736 bytes |
| Placement | 0.218s; exact model manifest verified on AMD |
| Transfers / calls | Input transfer 1; output transfer 1; root decoder and single-use permit each 1 |
| Input / output | Deterministic `[1,8,1,1,1]` float32 input; `[1,1,4,4,4]` float32 output on AMD; no gradients, inference tensor |
| Forward | 25.375s including explicit final GPU synchronization; all 64 values finite |
| Output statistics | Min -121.12086486816406; max -71.14271545410156; mean -96.97495913505554; population stddev 12.631798850835327 |
| Output SHA256 | `6a37c05abc213094b5a7e788f34539e2893c67450e1f9714ab22f12df8dd7afe` independently checked by packing the saved 64 float32 values |
| Final progress | Synchronization returned, CPU output validation passed, release verified, final worker trace `complete` (sequence 85) |
| Worker / containment | Exit 0 in 34.156s; no timeout or forced cleanup; owned Job active count 0; root/tree exit verified |
| Monitoring | Six healthy periodic Windows captures, maximum recorded gap 5.016s; pre/CPU-gate/post checks healthy, no matching driver-reset/error events; all readers/monitors/journals stopped cleanly |

All [checkpoint before/after records](amd-instrumented-forward-resume-evidence/run/checkpoint-after.json)
are identical. The independent analysis recomputes output finiteness/hash and
checks exact call/transfer counts, paired events, synchronization, resource
release, clean termination and monitor cadence from saved evidence only.

| VRAM boundary | Allocated bytes | Reserved bytes | Available bytes |
| --- | ---: | ---: | ---: |
| Before placement | 0 | 0 | 16,937,582,592 |
| Before forward | 152,836,096 | 174,063,616 | 16,756,441,088 |
| After forward | 152,836,608 | 174,063,616 | 16,655,187,968 |
| After model/tensor release | 0 | 0 | 16,822,960,128 |

Peak PyTorch allocation was **153,074,176 bytes**; peak reservation was
**174,063,616 bytes**. Model/input/output weak references were collected. Global
available memory was measured while the worker's native context still existed;
it did not exactly return to its pre-placement value. No post-exit GPU query or
second model operation was performed. Process exit is independently confirmed.

## What explains the delay

The 85 worker events include durable Conv3d module entry/return markers. The
first `module:input_layer` host call lasted **19.344s**; the first middle-block
`conv1` call lasted **5.843s**. Remaining module calls returned quickly. These
are host boundaries, not measurements of individual GPU kernel completion;
the explicit synchronization after root return confirms preceding GPU work
completed. The retained ATEN operator instrumentation emitted **zero** separate
operator markers in this run, so operator-level attribution is not established.

[Inert scratch inspection](amd-instrumented-forward-resume-evidence/scratch-side-effects.json)
records 526 files / 702,359,890 bytes, including **512 new COMGR cache entries**.
507 new entries have mtimes within the first Conv3d host call and four within
the first middle block; 511 fall within the root-forward interval. The copied
MIOpen database grew from two FP32-configured records to four records, adding
gfx1201 FP16-configured entries for `naive_conv.cpp.obj` and
`MIOpenIm3d2Col.cpp.obj`. The new naive-convolution record shares its decoded
binary hash with the original FP32-configured record, so new metadata does not
necessarily mean a distinct compiled binary. Database reads were
immutable/read-only, blobs bounded/decompressed as inert bytes, never executed.
Temporary/lock products and these new entries are preserved, not automatically
approved for future reuse.

Fourteen CPU samples observed only the launcher and Python worker. The worker's
cumulative CPU time rose from 3.125 to 22.125 seconds; **no separate compiler
child was observed**. Short-lived children between samples cannot be excluded.

First-use cache population coincided directly with the long convolution-module
calls. New COMGR products and FP16 MIOpen records strongly support compilation
contributing to the delay, including possible in-process compilation. This is
not a compiler profile or controlled cold/warm comparison: it cannot assign all
delay to compilation or prove the cause of the historical 30-second timeout.
This attempt demonstrated no native computation failure, guard error or timeout.

## Preservation and next step

[Preservation](amd-instrumented-forward-resume-evidence/preservation.json) confirms
byte-identical installed dependency inventories and compressed venv/model/cache
metadata, all 996 prior compatibility/evidence/cache files, historical local and
remote branch heads, AMD source HEAD and existing modified `nodes.py`/backup.
Main remains `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`. Runtime-created caches
are confined to this owned scratch directory and remain preserved locally.
No package, driver, installed application, production worker or existing AMD
source edit was changed. No inference pipeline, other checkpoint, CuMesh,
texture or GLB ran. [Read-only result review](amd-instrumented-forward-resume-evidence/post-attempt-review.md)
independently confirmed the bounded PASS and the telemetry/attribution limits.

The shortest next image-conditioned proof is **one offline DINOv3 conditioning
pass on a validated transparent RGBA image**, using the metadata-audited local
DINOv3 assets, established alpha-only preprocessing and strict lazy-loading
contract. Real DINO loading/inference remains unproved; the current preparation
worker keeps it disabled. With separate approval, a narrow test should verify
strict local loading, expected finite conditioning-feature shape, VRAM and clean
release under bounded monitoring; no flow/shape/texture model should load.
This advances the actual image input rather than repeating the synthetic decoder.
Full geometry generation, texturing and the earlier CuMesh 719 failure remain
unresolved. This task stops after result publication.
