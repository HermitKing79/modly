# Foliage export decision — vertex-colour candidate PASS

Base `7f1b1675913dcdd91175caa8169383236f8ed7eb`; isolated branch
`codex/amd-foliage-export-decision`. One bounded CPU numerical comparison.
No selection, UV solve, packing, bake, export, renderer, model or GPU execution.

**Decision:** separately prove a portable vertex-colour version of this exact
4,531-triangle foliage patch, using genuine learned RGB and constant material
factors. Its measured errors meet every predeclared gate. No target-mesh change
is needed to justify that next proof. This is not a completed portable asset,
visual equivalence proof or general qualification of vertex colours.

## Actual no-UV material comparison

Authenticated all 3,177 original vertex rows, original triangles/winding/source
IDs, the complete geometry/material coordinate correspondence, saved learned
field and 815-chart capacity evidence. The original CPU eight-neighbour sampler
uses raw F32 encoded RGB/scalars before any clamp; native support ordering and
partial-support renormalisation remain unchanged. Original data are untouched.

The derived glTF candidate clamps only its display attributes, applies exact IEC
sRGB decode, stores F32 linear COLOR_0, interpolates in linear light and encodes
again for RGB error measurement. This follows the [glTF colour contract](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc).
It does not substitute encoded-RGB interpolation or native-renderer gamma 2.2.
Reference errors use the spatial sample clipped afterward to [0,1]. No byte
quantisation, exposure change or per-vertex colour fitting was introduced.

PCG64 seed 20261021 generated 10,000 area-weighted points in addition to every
face centroid. All 2,221 shared-edge face pairs cut by the saved forest were
tested at five edge and five inward positions on both faces, retaining
non-manifold multiplicity and cycle seams. **58,951 total queries**, zero
unsupported samples; minimum positive support 1, raw denominator 0.22857945.

| Cohort | Points | Encoded RGB RMSE | p95 absolute | Maximum absolute |
|---|---:|---:|---:|---:|
| Every centroid |4,531|0.00410611|0.00892909|0.03414900|
| Area-weighted surface |10,000|0.00518673|0.01136657|0.04874728|
| Candidate seam edges |22,210|0.00682222|0.01491802|0.06687320|
| Near those seams |22,210|0.00675605|0.01479700|0.06592792|

All cohorts, including the combined 14,531 required points, meet the unchanged
**RMSE 0.02 / p95 0.05 / maximum 0.15** gates. All 11,105 paired exact-edge locations
have zero world-position, spatial-reference and vertex-RGB gap. This establishes
sampled correspondence, not continuous-field equality or shaded-viewer parity.

## Scalars, ranges and alpha

One set of constants was fitted from mean(clipped 10,000 weighted targets), then
tested on every cohort. Each channel is gated separately at RMSE 0.02; no pooled
scalar metric hides a failure. Per-cohort best-constant floors are also retained.

| Channel | Fitted constant | Sampled raw range | Worst cohort RMSE |
|---|---:|---|---:|
| Metallic |0.00011082694|−0.000941591 to 0.000986980|0.0001331791|
| Roughness |0.99999598678|0.999484003 to 1.001425147|0.0000331863|
| Alpha |0.99998668395|0.999374151 to 1.000743508|0.0000473689|

All pass. Derived sample clipping counts are 20,001 negative metallic, 56,266
roughness above 1 and 49,704 alpha above 1; no RGB clipping. These counts include
the declared repeated seam population. Raw values remain preserved. Full-field
and original-vertex ranges/counts are separately published in numeric evidence.

OPAQUE ignores stored alpha, but its actual opacity 1 has worst sampled alpha
RMSE 0.0000493658 here, also below 0.02. This measured patch is nearly
opaque, nearly fully rough and nearly nonmetallic. That does not qualify cutout/translucency
behaviour elsewhere. Standard COLOR_0 cannot generally encode spatial metallic
or roughness; constant factors work for this measured patch only. No emission,
AO, normal or displacement detail is invented.

## Multi-atlas cost and recommendation

The 815 saved disk charts require at least **four groups** under 256 charts per
atlas and the existing 64² owner policy. Actual conformal UVs, packing, distortion,
mip-safe occupancy and padding waste remain unproved; four is not a fit promise.
For four 1024² atlases, base RGBA8 + MR RGB8 costs **28 MiB base / 37.3333 MiB full mip
chains**; if both GPU textures occupy RGBA8, **32 / 42.6667 MiB**. These uncompressed
arithmetic estimates cover LOD0–10 storage, not actual GPU allocation or a new
LOD0–10 isolation proof. Mesh/staging/driver overhead and compression are excluded.
The conventional representation needs at least four material groups/eight maps;
actual draw calls are engine-dependent and unmeasured.

For comparison, this selection's raw F32 RGB attribute is 38,124B (37.2KiB) before
exporter vertex duplication. No final GLB size is predicted. Multi-atlas PBR is
an optional high-fidelity route pending separate feasibility, not a game-ready
default for this small patch or the original roughly 4.17M-triangle model.

**Single next action:** one separately approved portable vertex-colour foliage
proof using this exact selection and verified material factors, with fresh import
and viewer validation. Do not start multi-atlas baking or geometry changes now.

## Execution, review and preservation

Worker 1.875s; supervisor 3.265s; peak process commit **193,458,176B (184.50MiB)**
under 6 GiB / 180s. Three live samples: minimum available RAM **19,233,742,848B**
(12 GiB floor), destination free **104,666,472,448B** (2 GiB floor). Exit 0,
naturally empty owned Job, no forced cleanup; all 77 watched identities unchanged.

[18 focused tests](amd-foliage-export-decision-evidence/focused-tests.txt),
[source contract](amd-foliage-export-decision-evidence/independent-source-contract.json),
[exact candidate review](amd-foliage-export-decision-evidence/independent-candidate-review.json),
[actual measurements](amd-foliage-export-decision-evidence/execution-results.json)
and [atlas cost model](amd-foliage-export-decision-evidence/cost.json) are source-safe.
[Independent outcome verification](amd-foliage-export-decision-evidence/independent-outcome.json)
reopened all arrays and reproduced the measurements. A separate coordinate-tuple
oracle on 64 existing saved queries agreed within 2.44e-7 (limit 2e-6).
Private fixed queries/predictions and raw receipts:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-foliage-export-7f1b167\run`.
No images or model arrays enter public Git.

[Preservation AFTER](amd-foliage-export-decision-evidence/preservation-after.json)
matches 100 protected files, 36 frozen trees / 954 files / 16 reports, old caches and
quarantine, selected environments and AMD edits/backups. 1,440 fresh full hashes;
checkpoint checks are stat/header-only, zero weight-payload reads. Three old
disposable fixture copies retain labelled stat-only checks. Historical evidence
and all original generated assets remain intact. Stop after this decision.
