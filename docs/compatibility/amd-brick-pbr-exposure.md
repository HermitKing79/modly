# Brick learned-PBR exposure comparison — PASS

Base: `b0b7be18c492dfbd13aa34a8dd0276027fdcc84a`.
Branch: `codex/amd-brick-pbr-exposure-comparison`.

**Exposure is a substantial cause of the original preview's dark presentation.**
At +2 EV, the same learned colours, brick variation and pale surface mottling are
much easier to see. The image retains colour variation rather than washing out to
white. No material values, lights or geometry changed. Recommend **+2 EV for
visual review under this particular studio rig**, without changing an application
default or regenerating materials.

This does not uniquely distinguish insufficient illumination from darker albedo:
exposure scales both. It demonstrates that the existing learned RGB already
contains useful colour/detail obscured by the baseline presentation, not that
missing colours were recovered. Photo/model colour differences remain uncalibrated.

## Controlled comparison

One Blender 5.2.1 session imported the original mesh once and rendered exactly
three +X frames: **0, +1, +2 EV**. All are 1024×1024, Cycles CPU, 32 samples/four
threads with CPU OpenImageDenoise. Camera, lights, geometry, six learned material
channels, alpha, native `clamp(RGB,0,1)**2.2` conversion and AgX settings remained
identical. Actual settings were saved before/after each frame and rejected any
change other than exposure. EV0 was restored for final validation.

The new EV0 image's **pixels are identical** to the previous accepted baseline;
PNG container hashes differ. The material and complete vertex/face payload hashes
are unchanged. No separate colour-management variant was added, keeping this a
single-variable experiment.

| Fixed-region display statistic | EV0 | +1 EV | +2 EV |
| --- | ---: | ---: | ---: |
| Mean encoded luma, 0–1 | 0.14069 | 0.21790 | 0.32038 |
| Dark pixels, all channels ≤16/255 | 8.03% | 3.08% | 1.35% |
| Any channel ≥250/255 | 0 | 0 | 0 |
| Mean display HSV saturation | 0.757 | 0.709 | 0.646 |

The fixed 148,521-pixel region comes from contrast against the background in the
authenticated prior clay image. It is a labelled proxy, not a complete geometric
silhouette. Luma uses Rec.709 weights on encoded PNG code values; these are display
statistics, not linear radiance or reflectance. Saturation decreases as exposure
and tone mapping brighten the surface, but no near-white clipping was observed.
The photograph's foreground mean luma is 0.30835 for context only: different
projection, lighting and visible surfaces prevent a colour-fidelity score.

## Private deliverables

Directory: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-pbr-exposure-b0b7be1`

- `exposure-comparison.png`: reference and three variants side by side,
  2000×638; SHA-256 `35bebaeaae7281755453351e5c7875a9bd61a50f8a00a1c0f37c319235d499b1`.
- `contact-sheet.png`: mobile-friendly 2×2 layout, 1000×1178; SHA-256
  `87be0536a1481116c4dac6224fc5daa29951d0226eb58b5219f06d737b2608df`.
- `detail-comparison.png`: identical enlarged image crop at all three exposures;
  no extra render or fabricated surface detail.
- `renders\view-pos-x.png`, `view-pos-x-ev1.png`, `view-pos-x-ev2.png`:
  original full-resolution PNGs.

All six PNGs independently reopened. [Paths, sizes, hashes and statistics](amd-brick-pbr-exposure-evidence/image-verification.json).
New comparison images remain private and outside Git; no source photographs or
model binaries are added to this publication.

## Execution and preservation

Renderer **54.886 s**; supervisor **57.484 s**; full invocation **58.203 s**.
Peak committed memory **2,979,557,376 bytes**, working set **2,779,996,160 bytes**.
Minimum sampled Windows available RAM **18,194,538,496 bytes**; 57 live resource
checks passed. The original **6 GiB / 180 s**, **12 GiB RAM / 2 GiB disk** limits
were retained. Composition took **1.391 s**, with a 650,489,856-byte Job-recorded
peak per-process commitment within its separate **1 GiB / 60 s** bound. Both Jobs
emptied naturally with exit 0 and no forced cleanup. Blender's optional CUDA-loader
warning did not change the verified CPU render/denoising settings.

Nine focused CPU tests passed. [Preflight](amd-brick-pbr-exposure-evidence/preflight-final.json),
[pre-execution review](amd-brick-pbr-exposure-evidence/preexecution-review.json),
[actual renderer metadata](amd-brick-pbr-exposure-evidence/render-metadata.json),
[host result](amd-brick-pbr-exposure-evidence/host-result.json),
[independent outcome review](amd-brick-pbr-exposure-evidence/independent-outcome-review.json).

Before/after audits preserve all 100 protected files, original mesh/material,
historical evidence, original caches and selected environment/source identities.
The 86 pre-existing remote heads are checked at publication; the final publication
receipt remains outside Git. No Torch, WSL, model loading, GPU inference, geometry
change, bake, export or frontend work occurred. **Stop for visual review.**
