# Bounded zombie retopology feasibility

> Use superpowers:executing-plans and one independent final review. User specification authorizes the bounded experiment and method selection; do not request another permission round.

**Goal:** Determine whether the immutable orphan baseline can yield a usable lower-resolution posterior torso patch and chest-wound patch, rather than optimize defect counters alone.

**Architecture:** Keep the path-bound checkout on a new isolated branch and exclusive private output root. Reuse unchanged native CPU owner (6 GiB commit, 180 seconds/stage, 12 GiB live RAM, 2 GiB disk, GO/EOF, owned Job cleanup). Source is the orphan-cleaned 377a15dc... PLY, never the winding experiment. New arrays/meshes only.

## Fixed experiment design

- Inspect installed Blender build options and operator/modifier RNA without asset imports. Consider voxel, QuadriFlow, decimation and a small view-supported chart reconstruction.
- Test once per crop: ordinary Blender voxel remesh with voxel length twice the crop's median source edge length, no smoothing/adaptivity/component deletion. This is a control, not a presumption that closed output is good.
- Test once per crop: exterior first-hit surface sampling independent of source winding, regular grid spacing twice median source edge length, dense in-cell hit/depth witnesses. Omit unsupported/discontinuous cells rather than bridge openings. Orient faces toward the known outside viewing half-space. This single-chart approximation explicitly cannot represent hidden/overlapping sheets or arbitrary undercuts; oblique views must test that limitation.
- Posterior box x[-.045,.045], y[.02,.12], z[.17,.23]. Chest-wound box x[-.005,.115], y[-.12,.055], z[.125,.255]. Complete source triangles only; report IDs and artificial crop boundaries using full-source edge incidence. Do not close crops before remeshing.
- Record all rejected chart cells, candidate component counts, crop/genuine/unsupported boundaries, topology, orientation, surface-distance and matched front/oblique depth/coverage differences. Retain every candidate component and original reference payload.
- Render original/voxel/chart under matched neutral light and backface diagnostics from straight and two oblique cameras for each crop. Retain cavity, rim and ragged-detail differences; no texture or geometry-normal smoothing.

## Tasks

- [x] Verify starting commit and preserve refs, protected identities and historical assets.
- [x] RED/GREEN fixtures for crop-boundary attribution, outside-facing chart construction, missing/depth-jump rejection and immutable source data.
- [x] Probe installed CPU capabilities; authenticate and save two reference crops with source mappings.
- [x] Execute fixed voxel and surface-chart experiments, once per region; measure all candidate outcomes honestly, including failure/empty output.
- [x] Matched-view CPU render and independent metric/readback verification. Identify corrected versus omitted/concealed defects; decide GO/NO-GO using visible cavity/detail preservation.
- [ ] Independent final review, preservation checks and safe-code/text publication. Assets/screenshots remain private.

Interfaces: reference crops carry full-source vertex/face IDs and global boundary classification. Candidate evaluation compares the same test domains, reports unsupported regions and all components, and must not treat artificial crop boundaries as source damage. Processing and rendering are separate owned stages. No full-character rebuild, neural/WSL, UV, baking, texturing or production integration.

Review focus: no winding-variant input, no silent hole fill/fragment removal, chart witness limitations, crop boundaries, visibility versus true exterior classification, reduction versus feature loss, exact source preservation, native resource bounds and evidence-supported decision. No iterative parameter search if both methods fail.

Findings: native voxel control loses most open crop surfaces; the single first-hit chart improves orientation and counters but visibly steps/enlarges wound openings and omits cavity walls. Only 554/988 frontal deep-wound witness pixels retain deep hits, with 416 lost and 18 shallowed. It also interpolates across three sampled void pixels inside the wound ROI despite nine supporting samples per cell. No continuous opening-preservation claim is valid. NO-GO for extending either tested method to a larger character region.

Verification correction: initial compose stopped because its quantile recomputation retained F32 interpolation whereas the recorder explicitly uses F64. Saved arrays reproduce all recorded values exactly in F64. Added a RED/GREEN regression and matched the dtype rather than loosening tolerance. Initial failed compose receipt and two partial sheets remain unchanged; only verification/presentation reran under a new owned output path. No mesh processing or rendering was repeated.
