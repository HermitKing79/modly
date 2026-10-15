# Genuine turret texture-SLat sampling

Base: `4f845cd306c839d0e833efe488de44b65d82584c`. Branch: `codex/amd-turret-texture-sampling`.

## Reviewed execution contract

One genuine 512-resolution texture-flow sample only. Reuse the original RGBA/DINO provenance, denormalized F32 `[3743,32]` shape latent and I32 `[3743,4]` coordinates. No geometry or conditioning rerun. The texture checkpoint SHA-256 is `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d`.

- Strict 640-entry BF16 checkpoint validation, source-derived **420 BF16 / 220 F32** runtime parameter schema with verified reversible widening; genuine CPU RoPE attributes, no blanket model cast.
- Normalize saved shape exactly once with shape statistics. Genuine model concatenates `[noise/state, shape_condition]` internally; the real input projection is checked for exact F32 `[3743,64]` order. Texture output uses separate texture statistics.
- Canonical 12 Euler steps, **12 positive / zero negative forwards**, sigma minimum `0.00001`, guidance `1`, rescale `0`, interval `[0.6,0.9]`, `rescale_t=3`, DINO lock `0`. Fresh CPU noise seed **20261015**. Explicit installed SDPA; compilation, native bindings, external processes/downloads and other checkpoints prohibited.
- Three TunableOp environment controls are `0` before Torch import; genuine flags must all remain false at import and immediately before forward.
- Minimum **12 GiB available host RAM / 13 GiB free GPU memory**. Torch allocator ceiling remains 12 GiB; effective allowance is the smaller of that ceiling and initial free VRAM minus **2 GiB reserve + 1 GiB opaque-workspace headroom**. Continue checking that margin at step/block boundaries. CPU/GPU/overall deadlines remain **180/240/480 seconds**, with Windows observation and owned Job containment.

## CPU acceptance and durable storage

**30 focused tests: 29 passed, one Windows symlink-privilege skip**, across five isolated CPU processes. Real installed meta architecture, sampler/controller/dispatcher lifecycle, normalization, negative guards, complete production provenance and safetensors writer/independent reader were tested. No real weights or GPU forwards were used. An initial combined runner exposed incompatible global test audit hooks; its diagnostics are preserved. Isolated tests retain every guard.

The trusted host created this exclusive directory outside all Git repositories:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-4f845cd`

The production writer and reader completed a full-shape synthetic CPU persistence probe there, including fsync, SHA-256 and byte-exact readback, followed by independent reopening after writer exit. Actual results will use its separate `latent` subdirectory and contain raw/denormalized F32 texture latents, original I32 coordinates, initial noise and complete provenance. Synthetic probe data cannot substitute for real output.

[CPU qualification](amd-turret-texture-sampling-evidence/cpu-qualification.json) · [read-only model/input identity](amd-turret-texture-sampling-evidence/preflight-identity.json) · [storage probe](amd-turret-texture-sampling-evidence/persistence-preflight.json) · [preservation baseline](amd-turret-texture-sampling-evidence/baseline-preservation.json).

Execution results will be added after the single approved attempt. This preparation is not a texture-generation PASS. Stop before texture decoding, mesh changes or GLB export.
