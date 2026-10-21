# Portable vertex-colour foliage GLB — PASS

Base `cfe852b476725afe9deecb5d9ad44892794ce2e1`; isolated branch
`codex/amd-foliage-colour-glb`. One GLB export and one actual Blender import.
Private geometry/images remain outside Git. No UVs, bake, neural work, WSL,
geometry modification or full-model export.

## Verified portable file

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-foliage-colour-glb-cfe852b\export\foliage-learned-colour.glb`

**165,848 bytes**; SHA-256:
`c24055f7ef2fbe144c1a73ba46242cbeb6b2a2c27397440be13a3d56b8f6fe73`.

Fresh independent parsing verified **3,177 vertices / 4,531 original triangles**,
exact original F32 positions after the proper glTF Y-up rotation, index order,
winding, multiplicity and original source IDs. Saved F32 linear `COLOR_0` bytes
are exact; no new colour fit, gamma conversion or export quantisation.

One embedded glTF 2.0 triangle primitive, unit base factor, metallic
`0.00011082693761409814`, roughness `0.999995986777544`, OPAQUE and double-sided.
No NORMAL, UV, images, textures, external resources or compression extensions.
OPAQUE approximates this measured nearly opaque patch; arbitrary cutouts or
translucency are not represented. Original spatial PBR data remain unchanged.

## Actual Blender and numerical evidence

Blender **5.2.1 LTS** imported the file once into a fresh scene. Actual geometry,
flat shading, shader links, constant factors, unit alpha and double-sided material
passed readback. The independently loaded reference uses original geometry and
saved F32 colours, rather than data reconstructed from the imported object.

The installed importer stores colours as **BYTE_COLOR / CORNER**. It was left
untouched. Its encoded corner-colour residual versus F32 reference was RMSE
**0.00113588**, p95 **0.00186279**, maximum **0.00201713**. This viewer quantisation
is separate from the byte-exact F32 GLB transport.

All **58,951 existing** source comparisons were replayed without new spatial
queries. Exported F32 predictions are unchanged. Actual imported colours also
pass every original cohort gate: worst RGB RMSE **0.00688415**, p95 **0.01505871**,
maximum **0.06669773**, against **0.02 / 0.05 / 0.15**. Each fixed metallic,
roughness and actual OPAQUE-alpha error remains below the **0.02** scalar limit.

Four matched 640² Cycles CPU renders used the existing studio, +2 EV exposure,
32 samples, fixed seed, flat geometry and CPU denoising. Far orthographic scale
is exactly four times near. Transparent film measures silhouettes without
altering OPAQUE material semantics. Near/far silhouette IoU is **1.0**; visible
RGB RMSE is **0.00130049 / 0.00087789**, maximum **1/255**. Whole-image maximum
is **2/255**, including antialiased fringe pixels.

Visual inspection shows corresponding surfaces, gaps and colours in both
versions, with no new missing foliage or back-face loss apparent. Existing holes,
thin fragments and faceted geometry remain. Framing follows the prior 3D-bounds
camera: only 6,747 near / 439 far pixels have alpha >=128, so these images do not
establish fine-detail, continuous spatial-field or all-viewer equivalence.

## Private comparison and resources

Private root:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-foliage-colour-glb-cfe852b`.

- `inspect\comparison.png`: labelled reference/import near/far contact sheet.
  SHA-256 `52c043dd6076d451477e3ddc449e376defa5d736b71b5e945d1359df7b93c8e2`.
- `render-qualified\reference-near.png`, `imported-near.png`,
  `reference-far.png`, `imported-far.png`: actual individual renders.
- `artifact-manifest.json`: complete output paths, sizes and SHA-256 values.

| Successful stage | Worker seconds | Owned-process seconds | Peak process commit |
|---|---:|---:|---:|
| Export |1.844|3.000|130,691,072 B|
| Independent GLB/query verification |1.625|2.609|137,945,088 B|
| Import and four renders |18.884|21.563|894,861,312 B|
| Independent PNG/imported-query readback |1.953|3.110|175,968,256 B|

Maximum process/job commitment **853.41 MiB**, below 6 GiB. Lowest sampled
available RAM **17,901,940,736 B**, above 12 GiB; destination free space remained
above 113.56 GB, against 2 GiB. Every successful worker exited naturally inside
180 seconds; owned Jobs emptied without forced cleanup and watched sources stayed
unchanged. These are measured CPU resources, not GPU or full-model projections.

## Preserved launch failure, tests and review

Initial Blender startup failed at argparse because the command omitted `--`.
It reached **zero imports and zero renders**, exited naturally, and remains
BLOCKED in its original `render/` leaf and marker. Source commit `75498ad` retains
that complete candidate. A focused command-boundary correction, three CPU
regressions and independent review preceded the one intended actual import in
fresh `render-qualified/`. No GLB export, genuine import or render was repeated.

[27 focused checks](amd-foliage-colour-glb-evidence/focused-tests.txt),
[three launcher regressions](amd-foliage-colour-glb-evidence/launch-correction-tests.txt),
[independent reader review](amd-foliage-colour-glb-evidence/independent-reader-review.json),
[corrected candidate review](amd-foliage-colour-glb-evidence/independent-launch-correction-review.json),
[measured results and file hashes](amd-foliage-colour-glb-evidence/execution-results.json)
and [independent outcome](amd-foliage-colour-glb-evidence/independent-outcome.json)
with [preservation AFTER](amd-foliage-colour-glb-evidence/preservation-after.json)
are source-safe textual evidence.

All 100 protected files, 38 frozen trees / 995 files / 17 reports, selected
environments, AMD edits/backups, old caches and quarantine match BEFORE. Checkpoint
checks were stat/header-only, with zero weight payload read. Stop for visual
review of this portable patch; no full-model export or retopology is authorised.
