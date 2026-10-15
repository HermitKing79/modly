# Genuine turret texture-SLat sampling

Base: `4f845cd306c839d0e833efe488de44b65d82584c`. Branch: `codex/amd-turret-texture-sampling`.

**PASS — one authentic image-conditioned texture latent is permanently saved.** Exact execution harness `759508067ee23ad677f8c1e55408108a52c45f08` was reviewed, pushed and GitHub-verified before launch. One sequence completed **12 steps / 12 positive forwards / zero negative forwards**, including all 30 original transformer blocks and both attention modules per forward. No retry or earlier model stage ran.

## Actual output and measurements

Windows file:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-4f845cd\latent\texture-latent.safetensors`

**1,539,912 bytes**, SHA-256 **`5506c4d0199bef64423b41b8bed560e2b7def9d29c5bb1c3072c0ea31ee1ddc4`**.

The four tensors are finite F32 `raw`, `denormalized`, `noise`, each `[3743,32]`, and original I32 `coords` `[3743,4]`. Coordinates match the saved shape sample byte-for-byte. Raw output spans −4.356615 to 5.191279. The metadata sidecar in the same directory, `texture-latent-metadata.json`, is 39,171 bytes, SHA-256 `5d1d79c35c8c25e1765cccf4edc678335ad0cde3bd51a68590f7d9412ffae1bc`. Both safetensors embedded provenance and sidecar retain source/model/config/input hashes, exact parameters, fresh-noise identity, statistics, counters, timings and memory observations.

| Measurement | Actual result |
| --- | ---: |
| Strict CPU load, including input preparation | 10.453 s |
| Sampling plus final GPU synchronization | **7.109 s** |
| Complete supervised worker | 29.078 s |
| Peak Torch GPU allocated / reserved | 2,909,045,248 / 3,080,716,288 bytes |
| Peak host working set / private commit | 5,726,339,072 / 6,861,062,144 bytes |
| Initial free VRAM | 16,897,998,848 bytes |
| Lowest step-boundary free VRAM | 13,748,142,080 bytes |
| Available host RAM admission | 19,466,649,600 bytes |
| Torch allocated / reserved after release | **0 / 0 bytes** |

Runtime was the unchanged Windows Python **3.12.10**, Torch **2.13.0+rocm10.0.0**, HIP **7.15.26333**, RX 9070 XT **gfx1201**. All three genuine TunableOp flags were false. Model and owned tensors were collected; BLAS workspace cleanup ran once. Worker exit **0**, owned Job empty with **no forced cleanup**, all monitors/journals stopped cleanly, Windows observation healthy. Initial allowance remained the full 12 GiB because measured free VRAM admitted the additional reserves.

The production reader verified persisted data while the worker remained alive and again after clean termination. Independent CPU outcome verification reopens the same actual artifact; no output was regenerated. [Result summary](amd-turret-texture-sampling-evidence/verified-result-summary.json) · [raw execution evidence](amd-turret-texture-sampling-evidence/run/results.json) · [post-exit verification](amd-turret-texture-sampling-evidence/post-exit-artifact-verification.json).

## Reviewed execution contract

One genuine 512-resolution texture-flow sample only. Reuse the original RGBA/DINO provenance, denormalized F32 `[3743,32]` shape latent and I32 `[3743,4]` coordinates. No geometry or conditioning rerun. The texture checkpoint SHA-256 is `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d`.

- Strict 640-entry BF16 checkpoint validation, source-derived **420 BF16 / 220 F32** runtime parameter schema with verified reversible widening; genuine CPU RoPE attributes, no blanket model cast.
- Normalize saved shape exactly once with shape statistics. Genuine model concatenates `[noise/state, shape_condition]` internally; the real input projection is checked for exact F32 `[3743,64]` order. Texture output uses separate texture statistics.
- Canonical 12 Euler steps, **12 positive / zero negative forwards**, sigma minimum `0.00001`, guidance `1`, rescale `0`, interval `[0.6,0.9]`, `rescale_t=3`, DINO lock `0`. Fresh CPU noise seed **20261015**. Explicit installed SDPA; compilation, unapproved native kernels/subprocesses, downloads and other checkpoints prohibited. The existing source-verified read-only SDK discovery exception is retained.
- Three TunableOp environment controls are `0` before Torch import; genuine flags must all remain false at import and immediately before forward.
- Minimum **12 GiB available host RAM / 13 GiB free GPU memory**. Torch allocator ceiling remains 12 GiB; effective allowance is the smaller of that ceiling and initial free VRAM minus **2 GiB reserve + 1 GiB opaque-workspace headroom**. Continue checking that margin at step/block boundaries. CPU/GPU/overall deadlines remain **180/240/480 seconds**, with Windows observation and owned Job containment.

## CPU acceptance and durable storage

**30 focused tests: 29 passed, one Windows symlink-privilege skip**, across five isolated CPU processes. Real installed meta architecture, sampler/controller/dispatcher lifecycle, normalization, negative guards, complete production provenance and safetensors writer/independent reader were tested. No real weights or GPU forwards were used. An initial combined runner exposed incompatible global test audit hooks; its diagnostics are preserved. Isolated tests retain every guard.

The trusted host created this exclusive directory outside all Git repositories:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-4f845cd`

The production writer and reader completed a full-shape synthetic CPU persistence probe there, including fsync, SHA-256 and byte-exact readback, followed by independent reopening after writer exit. Actual results use its separate `latent` subdirectory and contain raw/denormalized F32 texture latents, original I32 coordinates, initial noise and complete provenance. Synthetic probe data cannot substitute for real output.

[CPU qualification](amd-turret-texture-sampling-evidence/cpu-qualification.json) · [read-only model/input identity](amd-turret-texture-sampling-evidence/preflight-identity.json) · [storage probe](amd-turret-texture-sampling-evidence/persistence-preflight.json) · [preservation baseline](amd-turret-texture-sampling-evidence/baseline-preservation.json).

[Independent outcome review](amd-turret-texture-sampling-evidence/independent-outcome-review.json) confirmed the actual artifact with both the production reader and a separate public-format decoder, exact texture denormalization, all 91 module counts, memory release and natural process exit. [Post-run preservation](amd-turret-texture-sampling-evidence/preservation-after.json) matched the original mesh/neural artifacts, inputs, previews, AMD sources/backup, both Windows environments' package metadata and selected native binaries. WSL was not started. Historical Git branches remain preserved.

This PASS establishes genuine texture **latent** generation and durable persistence. It does not establish decoded material quality or a textured mesh. The next smallest milestone is separately qualifying the genuine guided texture decoder's F16/tiling contract, then decoding this saved latent with the four existing subdivision records. Do not repeat DINO, shape generation or texture sampling. No texture decoder, mesh repair, geometry change or GLB export ran here.
