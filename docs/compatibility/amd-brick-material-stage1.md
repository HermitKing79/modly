# Genuine brick learned materials — Stage 1 preparation

Base `df416889bc65ad43f0f928c563c6d99c64e730a8`, branch `codex/amd-repeatable-generation-v1`. Geometry is accepted for experimental materials only; its known open/non-manifold topology is unchanged.

**CPU preparation PASS; the genuine sampling attempt has not yet run in this commit.** The existing fresh-image `amd_generation_texture.py` adapter and seven previously qualified texture source files are unchanged. Original brick DINO conditioning, 4,430-row shape latent/coordinates and source lineage are authenticated. The texture checkpoint's full SHA-256 is `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d`: 2,584,672,728 bytes, 640 BF16 entries, retaining the reviewed 420 BF16 / 220 F32 runtime contract.

Ten focused CPU-only adapter checks and the previously filesystem-blocked hardlink case pass, with Torch absent. The actual production writer/reader has completed a fresh 4,430-row synthetic persistence probe outside Git, with independent payload readback. Windows observation is healthy. Three RAM readings are above the unchanged 12 GiB requirement. An initial metadata-only diagnostic used the wrong distribution name `triton`; the installed package is `triton-windows 3.8.0.post29`. Correcting that query changed no runtime or execution policy and consumed no attempt marker.

Private real output is reserved at:

```text
C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-material-stage1-df41688\latent
```

One canonical 12-step / 12-positive / zero-negative sample is conditionally cleared after publication and complete committed-source verification. Shape normalization, noise-first concatenation, texture statistics, SDPA, TunableOp controls, 13 GiB initial free VRAM, 12 GiB allocator, 2 GiB reserve, 1 GiB opaque headroom, 180/240/480-second deadlines, observation and owned-process/zero-allocation release remain unchanged. No automatic retry is allowed.

[CPU identity/resources](amd-brick-material-stage1-evidence/cpu-preflight.json), [focused tests](amd-brick-material-stage1-evidence/fresh-cpu-tests.stdout.txt), [hardlink regression](amd-brick-material-stage1-evidence/hardlink-cpu-test.stdout.txt), [independent pre-execution review](amd-brick-material-stage1-evidence/independent-texture-sampling-review.txt), [preservation](amd-brick-material-stage1-evidence/preservation-before.json).

Guided material decoding is not authorised for GPU execution. Only after a genuine sample PASS may its brick-support contract be assessed on CPU. Images, model outputs and original geometry remain private and unchanged; this preparation publishes textual evidence only.
