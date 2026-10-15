# Genuine brick learned materials — Stage 1

Base `df416889bc65ad43f0f928c563c6d99c64e730a8`, branch `codex/amd-repeatable-generation-v1`. Geometry is accepted for experimental materials only; its known open/non-manifold topology is unchanged.

**Genuine texture-latent sampling PASS. Guided PBR decoder execution remains BLOCKED by its existing brick-support/memory contract.** Exactly one sampling attempt ran, using the unchanged fresh-image `amd_generation_texture.py` adapter at prepublished commit `6875a1e7b1f27e3a79d4b5ce2ddd939dccbd4a69`. All 67 committed source/input-contract identities were verified before execution. No geometry, conditioning or shape sampling was repeated.

Original brick DINO conditioning, denormalized shape latent and coordinate order were authenticated. Flow checkpoint SHA-256: `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d`, 640 original BF16 entries, retaining 420 BF16 / 220 F32 runtime parameters. Shape conditioning was normalized once; fresh noise occupied the first 32 input channels. The genuine SDPA path completed **12 steps / 12 positive / zero negative forwards**. Seed `20261015`, sigma minimum `0.00001`, guidance `1.0`, rescale `0`, interval `[0.6,0.9]`, `rescale_t=3.0` and DINO lock `0` were unchanged.

The permanently saved private output is:

```text
C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-material-stage1-df41688\latent\texture-latent.safetensors
```

It contains finite raw/denormalized F32 `[4430,32]` latents, unchanged I32 `[4430,4]` coordinates and initial noise. Size **1,814,784 bytes**; SHA-256 **`4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289`**. Production readback passed before release and after exit; a separate stdlib/NumPy reader verified all payloads, coordinate order, denormalization and provenance. Material appearance remains untested.

Synchronized sampling took **7.031 s**, CPU loading **13.844 s**, worker supervision **41.0 s** (excluding earlier preflight). Peak GPU allocated/reserved bytes: **2,942,394,880 / 3,137,339,392**; peak worker working set/private commit: **5,729,325,056 / 6,863,831,040**. Initial/minimum step-boundary free VRAM: **16,897,998,848 / 13,695,713,280 bytes**. Cleanup measured **zero allocated/reserved bytes**, collected model/tensors, exited `0` naturally and emptied the owned Job. Monitoring stayed healthy; all original limits remained unchanged.

Ten focused adapter checks, the hardlink regression and a fresh synthetic production-writer persistence probe passed before inference.

[CPU preflight](amd-brick-material-stage1-evidence/cpu-preflight.json), [focused tests](amd-brick-material-stage1-evidence/fresh-cpu-tests.stdout.txt), [hardlink regression](amd-brick-material-stage1-evidence/hardlink-cpu-test.stdout.txt), [published source closure](amd-brick-material-stage1-evidence/published-harness.json), [pre-execution review](amd-brick-material-stage1-evidence/independent-texture-sampling-review.txt), [independent actual outcome and raw-receipt hashes](amd-brick-material-stage1-evidence/independent-texture-sampling-outcome.json).

The following decoder assessment is CPU-only. No guided PBR forward or downstream processing ran. Images and model outputs remain private; only source-safe text/code is published.

The genuine guides reconstruct **4,430 → 21,391 → 97,313 → 438,684 → 1,957,891** rows. The authenticated `brick-1957891-v1` contract and ten neural tensors passed readback. Decoder config/header-only inspection confirms 284 F16 entries and **208 F16 / 76 F32** runtime parameters; no decoder weights were loaded.

**Increasing the final-row limit alone is unsafe.** Historical input/hooks/admission/output consumers still pin turret support. Untiled F16 `blocks.3.4.conv1` at 438,684 rows projects **8,511,785,652 visible / 10,639,732,065 padded bytes**. The unchanged cap/headroom leaves **1,171,427,999 retained bytes**; model plus two source-live F16 feature tables already require **1,173,173,656**, before coordinates/guides/caches. This proves refusal by the conservative admission formula, **not measured GPU OOM** or AMD numerical failure.

The CPU preparation script reuses original formulas for 40 convolution/8 C2S plans: five expected native maps, **60 final tiles** (last **24,579 rows**) and **125,305,024 output-payload bytes**, excluding headers/working copies. These are projections. Six focused regressions pass, including genuine guide linkage and historical writer/reader rejection of unsupported rows. Historical defaults remain unchanged.

Next: a separately approved **CPU-only F16 guided-decoder integration**, reusing the proven global-neighbour row-tiling approach at the six large-support sites and adding a single authenticated brick profile through input/guide, admission, output and independent-reader paths. It must retain historical defaults, precision and all memory/deadline limits. Only after that qualification should one genuine guided PBR forward be proposed. The BF16 geometry runner cannot be substituted for the F16 material decoder.

[CPU decoder inputs/plans and blockers](amd-brick-material-stage1-evidence/decoder-cpu-readiness.json), [final focused verification](amd-brick-material-stage1-evidence/decoder-cpu-final-verification.stdout.txt), [independent decoder review](amd-brick-material-stage1-evidence/decoder-cpu-independent-review.txt), [CPU test execution notes](amd-brick-material-stage1-evidence/decoder-cpu-test-execution-notes.json), [CPU preparation source](amd-brick-material-stage1-evidence/decoder_cpu.py), [preservation before](amd-brick-material-stage1-evidence/preservation-before.json), [preservation after](amd-brick-material-stage1-evidence/preservation-after.json). The 100 protected files, source edit/backups, prior artifacts and selected environment inventories remained unchanged; the preservation receipt states its bounded inventory scope.
