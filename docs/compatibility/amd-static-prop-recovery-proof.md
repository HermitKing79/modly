# Phase 20B — controlled static-prop UV/export recovery

Parent: `aec918d34198adc5d2be9e12d0675bf49fef44a4`. One fresh CPU attempt, 31 October 2026. Private evidence: `artifacts/amd-static-prop-recovery-20261031-v1/`.

**UV: NO-GO. GLB export/readback: NO-GO / NOT EXECUTED. Separate static-prop export proof: NO-GO.** The exact UV failure is now measured and preserved. All eight appearance combinations completed, producing 24 individual images and a contact sheet; **appearance acceptability remains INCONCLUSIVE and USER_APPEARANCE_APPROVAL_PENDING** because the tight views provide limited feature information.

The original Phase 11 high-fidelity benchmark remains permanently NO-GO. The separately authorised LOWER-DETAIL STATIC PROP concession remains specific to this saved candidate's 524 microfacet faces / 0.0024493% source area. No general noise-removal rule, asset promotion or game-ready claim follows.

## What changed

The historical runner and its `LIVE_EXECUTION_HOLD=True` were left byte-for-byte unchanged. A new first-party controller and small Blender adapter use a fresh exclusive directory. They reuse the reviewed Blender 5.2.1 implementation, including its prepared `shade_flat()` presentation initialisation. The original Blender entry consumes the sole GO/EOF handshake after owned Job attachment. No old main/launcher is resumed.

The stale failure handling was replaced with current structured evidence: the exact UV diagnostic classification, failed conditions, measured ratio and preserved `uv.npy` SHA-256 must match the settled exception. Resource, ownership, unrelated script failures, missing/mismatched layout data and any unexpected GLB/readback state are refused. Appearance-only admission required a separate independent review receipt pinning the actual failure and audit artifacts.

UV coordinates and complete diagnostics are saved **before** enforcing the unchanged gates. The new review page reports the retained numerical ratio and actual outcome; it never claims those values were lost. The generated manifest and contact sheet were preserved unchanged, with a separately reviewed `final-manifest.json` / `final-appearance-assessment.json` supplement recording visual limitations.

## Measured UV result

The single Smart UV Project invocation retained angle **66 degrees**, island margin **0.01**, area weight **0**, aspect correction **true**, scale-to-bounds **false**. No alternative settings or retry occurred.

| Measurement | Actual result |
|---|---:|
| Candidate | 502 unique geometric vertices, 1,000 triangles |
| Maximum/minimum UV-area density ratio | **2.133809957682501 — FAIL against 1.1** |
| Independent Float64-cross density ratio | 2.1338098418980778 |
| Maximum relative difference of per-face density calculations | 1.493766842067483e-6 |
| Minimum density; candidate face ID | 0.0913838386513771; **531** |
| Maximum density; candidate face ID | 0.19499574488555949; **289** |
| Charts / seam edges | 10 / 149 |
| UV boundary segments | 298 |
| Boundary vertices with degree other than two | **3 — retained diagnostic requiring review** |
| Finite UVs, bounds, zero-area UV triangles | finite; all within [0,1]; zero degenerate triangles |
| Complete continuous positive-area overlaps | **0**, exact SAT over 3,005 AABB candidate pairs |
| Nonoverlapping in-bounds atlas area | 0.5976134403802308, approximately **59.7613%** |

Density units are UV-unit² per original asset-unit²; its ratio is dimensionless. IDs are zero-based rows in the authenticated 1,000-triangle candidate, not rows in the two-million-triangle neural reference. Every triangle's signed/absolute UV area, original 3D area, density, validity, chart and both density extremes are retained. Seams and boundary segments include original vertex indices and UV endpoints. Coverage is a Float64 area sum after complete nonoverlap, not a texel/mip-padding certificate. The three boundary-degree anomalies are not silently classified PASS or repaired.

Acceptance retains Phase20 arithmetic: Float32 cross product followed by Float64 norm. The independent CPU path converts positions to Float64 before crossing. Both exceed 1.1 by a large margin, so ordinary arithmetic variation does not explain the failure. The worst chart alone has density ratio **2.1338016**; uniform whole-island rescaling cannot remove that within-chart variation. This does not authorise another unwrap experiment or establish which alternative method would pass.

The prepared geometry hash is `befc39ccf34044c2d26a9dd3b67ebee0da3463904e71181160e66efa11876c1c`. Saved UV SHA-256 is `cb61017ac6aa7c43d3d4efa6c7c932b325f0bcfecc80ceaa09273cf046e7e7db`. The independent CPU audit reproduced the saved metrics and original correspondence before appearance-only admission.

## Appearance evidence and limits

Matte roughness **0.8** and glossy roughness **0.12**, metallic zero, were compared at whole-object azimuths 45/225 degrees and source-frozen detail regions 37/8. Every combination contains source / raw wrap / final candidate. All use CPU Cycles, four threads, 16 samples, no denoising, flat face shading and identical light/camera/scale within the comparison. All six prepared input-array hashes and 24 image identities/camera records were verified after execution. No source geometry was regenerated, repaired, remeshed, simplified or smoothed.

Whole-object silhouettes and the major bevel bands remain recognisable and visually close. The detail37 panels are nearly white with little useful microfacet/highlight structure; detail8 is nearly uniform dark gray and low contrast. Very bright tops and 16-sample gloss noise further limit fine comparisons. **The images do not establish microfacet or highlight acceptability.** The precise cause of the low information was not measured, so no claim is made that exposure, clipping or occlusion alone caused it. No rerender or lighting retuning was attempted. User appearance approval remains pending.

Private paths, relative to the evidence root:

- **Contact sheet:** `review/contact-sheet.png` — matte left, glossy right; each half has source/raw/final columns and whole45/whole225/detail37/detail8 rows.
- **Review page:** `review/index.html`; separate `review/matte-sheet.png` and `review/glossy-sheet.png`; all original 512-square PNGs remain in their named material/view directories.
- **UV coordinates and full diagnostics:** `asset/uv.npy`, `asset/uv-diagnostics.json`.
- **Independent audit:** `uv-audit.json`.
- **Final qualification manifest:** `final-manifest.json`; `final-appearance-assessment.json` records the independent visual limitation and binds the original generated manifest/sheet hashes.
- **GLB:** none. The failed UV guard prevented export, neutral test texture generation and readback. Conditional exporter/parser code was not live-qualified.

Normal baking remained SKIPPED: signed source normals and ray correspondence are unqualified. S&box is NOT TESTED. No finished AI texture, normal map or recovered silhouette detail is claimed.

## Resources, preservation and decision

Fourteen separately bounded stages took **196.638 seconds total**, excluding development/review. The UV/export stage stopped naturally after **9.625 seconds**, using **287,617,024 bytes** peak owned Job commit. The independent UV audit took **3.015 seconds**. All eight appearance stages passed in **11.843–16.250 seconds** each; the sheet took **9.235 seconds**. Largest owned Job commit was **892,510,208 bytes**. Minimum observed Windows available RAM was **16,689,319,936 bytes**.

All stages retained **12 GiB host floor, 6 GiB process/Job, four CPU cores, 180-second per-stage deadline and 2 GiB disk floor**. Every worker/tree exited naturally with an empty Job; no forced cleanup, retries or raised limits occurred. No WSL, ROCm, GPU, neural, CGAL binary, installation, download or external third-party add-on/script was invoked.

Authentication and final preservation checked 166 Phase19 files, 314 Phase18 files, 552 Phase14 files, 47 original source bindings and 188 protected identities, plus **all 44 historical Phase20 evidence files**. Candidate OFF SHA-256 remains `2e3522850b76d300fcf2c92dac9ccc32272ed1ed8b24ae0839a2e5cc8aa04b0b`. Original source meshes/tensors, environments, backup, criteria and historical holds are unchanged. **Seven new focused offline tests passed before execution and again afterwards**, covering retained diagnostics, invalid UVs, exact failure/ratio matching, coverage limits, original settings/hold and sole GO ownership. Reused code remains at its reviewed parent identities.

Independent reviews approved pre-execution code, the settled UV failure, appearance-only continuation and final outcome. The useful recovery is now a reproducible **layout-quality failure**, with actual diagnostics and preserved material views. It is not an export success. The shortest justified next technical decision is a separately authorised UV-only approach addressing within-chart distortion, followed by export only after qualification; simple island rescaling is insufficient. Better close-up evidence and explicit user approval also remain necessary. No follow-up is initiated.

Remaining readiness includes a qualified UV layout/boundary assessment, actual neutral-PBR GLB export and independent readback, informative material/distance views, production textures and padding/mips, S&box units/import/collision/LOD checks, and a separate licence decision before any CGAL binary product integration. This private proof grants no distribution rights or organic-asset qualification.
