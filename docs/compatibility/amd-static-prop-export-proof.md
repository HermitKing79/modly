# Phase 20 — optional static-prop UV/export proof

Parent: `2eb2f40b7976bdc5029dd99c9d407770ac3840f9`. CPU-only execution, 31 October 2026.

**Result: NO-GO for this export attempt. New appearance evidence: INCONCLUSIVE. No GLB or new comparison images were produced. No automatic retry was made.**

The separate **LOWER-DETAIL STATIC PROP** profile is defined in `amd-static-prop-profile.json`. It accepts the documented 524 coherent microfacet faces / 0.0024493% source-area trade-off only for this saved non-deforming candidate, conditional on user inspection at the intended viewing distance/material. All original topology, intersection, distance, full-surface support, 1,000-triangle and eight-view gates are retained. **The original Phase 11 high-fidelity result remains permanently NO-GO.** No prior result or asset was promoted.

## Actual outcomes

| Requirement | Outcome |
|---|---|
| Reauthenticate original AI, saved wraps and criteria | PASS |
| Create unchanged rendering/export copies | PASS, within declared Float32 serialization tolerance |
| CPU UV unwrap | Operation completed; **layout NOT QUALIFIED** |
| UV scale consistency | **FAIL validation**, numerical ratio unrecorded |
| UV coverage, chart boundaries and complete layout receipt | NOT COMPLETED |
| Source-normal detail bake | SKIPPED — signed normals/ray ownership unqualified |
| Experimental GLB and independent readback | NOT EXECUTED; no GLB exists |
| New matte/glossy whole-object and detail views | INCONCLUSIVE — first render guard stopped before an image |
| S&box import | NOT TESTED — no existing qualified import-validation route found |
| Original-source preservation | PASS |

## UV failure

One headless Blender 5.2.1 CPU invocation used Smart UV Project with angle 66 degrees, island margin 0.01, area weight zero, aspect correction enabled and scale-to-bounds disabled. The declared scale check required a finite maximum/minimum **UV-area / 3D-triangle-area** density ratio of at most 1.1. This was a conservative engineering choice for the new proof, not an inherited Phase 11 geometry threshold.

Blender stopped at `ValueError: Unequal UV area density`. By code order, finite/in-bounds/nonzero UV and continuous overlap checks had returned before that failure. The overlap predicates use exact dyadic-integer separating-axis tests over every AABB-overlapping triangle pair, permitting shared boundaries. This is stronger than texel-centre sampling, but **there is no completed UV qualification receipt**.

The error was raised for either a nonfinite ratio or one above 1.1; the original message did not retain its value. UV coordinates were scheduled to be saved after validation, so they were lost when the process exited. We cannot report the actual ratio, measured coverage, island count, a diagnosed cause of distortion or a successful checker. The result does not establish that Blender UV tools are generally unsuitable. Neither the threshold nor unwrap settings were changed and retried.

## Appearance failure and offline correction

A separately reviewed continuation allowed only the eight already-declared source/raw/final material comparisons. It bound the exact failed export logs, host receipt, source snapshot and prepared copies; it could not authorise another unwrap/export or fabricate a verification result.

The first `matte-whole45` invocation stopped at the combined `CPU flat unmodified render required` guard. That guard covered CPU device, modifiers and smooth-face flags but did not log the individual condition. No image was written. **This is a harness refusal, not evidence that the material appearance is unacceptable.** The other seven comparisons and contact sheet were not executed.

Independent inspection of installed Blender `_bpy_types.py` demonstrates a missing presentation initialisation: `Mesh.from_pydata(..., shade_flat=True)` calls `Mesh.shade_flat()`, which sets the face-domain `sharp_face` attribute. Our bulk constructor omitted that call. Installed source SHA-256: `2cf02a4ba02f01f7ccac59b99b7d8404b6868cae25bee91abba1c1a17aef8bb5`. This establishes the omission, but not which combined guard condition failed in the saved process.

The published code contains **offline-only, unexecuted corrections**: explicit `data.shade_flat()`, separate device/modifier/smooth-count diagnostics, retained UV coordinates before validation, and a numerical ratio in future failure messages. Flat shading changes presentation attributes; the existing coordinate/index identity check follows it. An explicit **LIVE_EXECUTION_HOLD** rejects further preparation/export/render stages. No correction is claimed live-verified.

## Geometry, materials and bake boundaries

The immutable candidate remains 502 vertices / 1,000 triangles, with OFF SHA-256 `2e3522850b76d300fcf2c92dac9ccc32272ed1ed8b24ae0839a2e5cc8aa04b0b`. Source/raw/final copies were schema-checked, input-hash-bound and converted to Blender/glTF Float32. Maximum Euclidean vertex displacement is zero for the original AI Float32 data and **2.2316831e-8 D** for each saved wrap, below the predeclared 1e-7D serialization bound. Original files are unchanged. This attempt does not provide a new exported-surface intersection or GLB fidelity result.

The prepared neutral test material/export verifier is **unqualified live code**, not finished AI texture output. Its offline checks cover oriented triangle/corner UV correspondence, continuous overlap on both working and serialized UVs, seam-duplicate accounting, flat normals, scene transforms, bounded accessors, neutral PBR factors and packed texture pixels. No actual GLB reached those checks.

Normal baking was deliberately skipped: unsigned feature comparisons do not establish signed exterior normals, and the source has inconsistent winding, non-manifold edges and inconclusive contact diagnostics. No qualified ray ownership/cage correspondence exists. No normal map was produced, and no silhouette restoration is claimed.

## Resources, preservation and review

| Stage | Seconds | Peak owned Job commit |
|---|---:|---:|
| Authenticate | 22.687 | 18,051,072 bytes |
| Prepare copies | 3.016 | 152,834,048 bytes |
| UV/export invocation, failed UV gate | 10.500 | 290,275,328 bytes |
| First appearance invocation, failed guard | 8.016 | 501,399,552 bytes |
| Preservation | 23.875 | 17,616,896 bytes |

Minimum observed available Windows RAM was **17,108,664,320 bytes**. Every stage retained the 12 GiB host floor, 6 GiB process/Job cap, four-core affinity, 180-second deadline and 2 GiB disk floor. Children exited naturally with empty owned Jobs; no forced cleanup, unrelated process termination, GPU/WSL operation or resource escalation occurred.

Authentication and final preservation checked 166 Phase 19 evidence files, 314 Phase 18 files, 552 Phase 14 files, 47 original source bindings and 188 protected identities. Blender executable SHA-256 remained `284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb`. Existing environments, checkpoints, historical evidence and backup were not modified. **61 targeted offline tests passed**, including eight new SAT/correspondence/material/bounds/scratch/failure-routing/hold tests. These are not substitutes for missing live export results. A direct hardlink fixture was unavailable under Windows sandbox permissions; no physical-link test success is claimed.

Private evidence is `artifacts/amd-static-prop-export-20261031-v1/`, including `asset-manifest.json`, original failed `export-run` / `matte-whole45-run` logs, immutable review receipts, prepared copies and preservation receipts. Earlier Phase 19 neutral comparisons remain in `artifacts/amd-alpha-features-20261031-v1/detail-review/index.html`; they are **earlier evidence**, not new Phase 20 matte/specular validation.

Independent review inspected the source, exact failures, installed Blender implementation, tests, provenance and preservation. Publication contains source/tests/text only. CGAL binary product integration and proprietary distribution remain unapproved; this task did not invoke or rebuild CGAL.

**Next technical action:** a separately authorised fresh CPU UV/export attempt can use the corrected flat-shading initialisation and persistent UV diagnostics. Its UV ratio/layout must be qualified before any GLB is accepted; the current evidence does not justify a threshold relaxation or a specific alternative unwrap setting. Remaining readiness work includes successful export/readback, material/distance appearance approval, production texture/UV-padding checks, S&box units/import/collision/LOD validation and finished PBR materials. No reconstruction or broader integration is proposed here.
