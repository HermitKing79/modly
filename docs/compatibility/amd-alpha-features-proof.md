# Phase 19 — Alpha Wrapping feature preservation

Parent: `0d80e8adadd96fe72d827e2924c1bbe30820f2ba`. Read-only CPU investigation, 31 October 2026. **Frozen research result: NO-GO. No new reconstruction or simplification was executed.** The saved 1,000-triangle result remains an **AI-DERIVED RECONSTRUCTION**, not a raw neural PASS or a game-ready asset.

## Decision

The principal remaining failures are **real, coherent, extremely narrow source-normal strips**, not demonstrated random noise or an edge-counting error. Magnified views show a narrow source bevel strip replaced by a sharper crease in both saved wraps. The major planes, main bevel bands and full-object silhouette remain close. Whether these microfacets matter for an intended material and viewing distance is a separate asset decision; their small area does not waive the frozen criterion.

**STOP before Stage B.** Almost every failure predates simplification. The evidence does not separate alpha tessellation effects from offset/concavity effects sufficiently to choose one justified smaller alpha while keeping offset and the final 1,000-triangle budget. A finer wrap might help, but that is currently an untested hypothesis. Narrowness below the offset is not a mathematical impossibility proof: some convex planar facets can survive offsets. We have not proved that 1,000 triangles and 10 degrees cannot coexist; neither saved output demonstrates that they can. Another parameter run would be speculative.

## Authentication and scope

Reauthenticated 314 Phase 18 evidence files, 552 Phase 14 files and 47 original source bindings; retained 188 protected file identities. Frozen criteria SHA-256 remains `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0`.

| Immutable input | SHA-256 |
|---|---|
| Original raw AI PLY | `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3` |
| Original vertices NPY | `fa52e8076fe99cc251db578911f1e0b6eb6224a5a1118aeb18e6a28ab9926e60` |
| Original faces NPY | `3b530ebc0a00e73a2a8192ad855f136734e2170cb5357d28e669df3064a400fc` |
| Saved raw wrap, 2,808 triangles | `3da7a965d5769399de477c1b035c42ed6a0c60526681101a0e100a3f6c30c9ad` |
| Saved simplified candidate, 1,000 triangles | `2e3522850b76d300fcf2c92dac9ccc32272ed1ed8b24ae0839a2e5cc8aa04b0b` |

The frozen normal test tracks 1,925,348 of 2,047,210 original faces: 5,776,044 vertex incidences in 51 patches. The remaining 121,862 faces are not silently called feature-PASS; all original triangles participate in source connectivity and geometric context. Both saved outputs were compared against every tracked incidence using the original unsigned 10-degree and 0.002D guards. D is the original AI diagonal, not a procedural reference dimension.

## Complete comparison

| Incidence result | Count |
|---|---:|
| Pass both outputs | 5,774,476 |
| Fail raw, pass simplified | 0 |
| Pass raw, fail simplified | 5 |
| Fail both | 1,563 |

Thus 99.681% of the final failed incidences already fail in the raw wrap. Independent NumPy point-to-triangle and angular calculations reproduced all **3,136 output/incidence checks** for the failure union. Binary masks, row identities, regions, corner membership and aggregate transitions were independently reconciled.

The union is **524 distinct source faces**, comprising **523 failing in both outputs and one failing only after simplification**. Some additional failed incidences occur on faces already failing elsewhere. Deduplicated affected area is **0.0000435956718127 D²**, or **0.0024493137%** of the original surface area. These are face areas, not repeated per-corner areas.

The five new failures are original face/corner pairs **500016/0, 500017/0, 500017/2** in region 34 and **1553457/0, 1553457/2** in region 8. Their nearest locally available normal angles change respectively from approximately 9.883→10.573, 9.316→10.010, 9.316→10.010, 9.953→10.425 and 9.953→10.425 degrees. These are modest threshold crossings, not the explanation for the other 1,563 failures.

The final failed-incidence nearest-local-angle range is **10.00997–19.81844 degrees**, median **17.43880 degrees**. The raw range is **10.00967–19.82246**, median **17.44054**. Original face 506773 remains an independently checked witness: unrestricted distance is small, but its locally available normals differ by about 18.9 degrees.

## What the local evidence establishes

- **520 affected faces** lie in residual regions 34–50. These are long, thin source structures, predominantly connected chains: area-weighted normal p95 is only **1.83–4.84 degrees** from each region's dominant unsigned normal. Their affected lengths along Y are approximately **0.01775–0.03125D**; X extents are **0.000119–0.000168D**, with still smaller Z extents. They are not isolated arbitrary normals. The fixed wrapping offset was **0.0005D**. Those dimensions explain why a positional error can be tiny while an angular feature test fails; they do not by themselves identify which alpha/offset mechanism removed the facets.
- The other four affected faces are one in region 8, one in region 18 and two in region 24. Regions 18/24 already fail raw; region 8 is simplification-only. All 20 affected regions have individual measurements and matched views. The failed-face graph has **22 components**, using only genuine two-incidence source edges; component face lists, areas and bounds are retained.
- Complete original index incidence again reproduces **1,421 boundary edges and 519 greater-than-two-incidence edges**. **None of the 524 failed faces directly touches either kind of edge.** This rules out direct edge-junction association for this failure set, not all possible source defects nearby.
- **Source-contact classification is INCONCLUSIVE.** CGAL EPIK constructed triangle intersections missed known shared-index contact in 5,641 directed affected-face/neighbor records (not necessarily distinct physical contacts). Four affected faces carried computed beyond-shared indicators and 464 carried near-shared indicators; these cannot be promoted to proven defects. Likewise, absence of reported non-index contact does not prove absence of overlapping sheets. The independent normal/distance and index-incidence results do not depend on that unreliable contact construction. We did not rerun the census or hide this limitation.

Each affected-face record includes original indices, exact native positions (D=100 normalization with preserved original coordinate transform), source area/normal, patch, edge fans and 22,003 probe-selected neighboring triangle records. Every failed corner includes unrestricted nearest triangle, globally nearest normal-compatible triangle/distance, and minimum angle among triangles actually within 0.002D. Local context is explicitly selected by vertex/centroid probes, not claimed to be a complete triangle-pair intersection survey.

## Visual evidence and optional asset profile

Private evidence: `artifacts/amd-alpha-features-20261031-v1/`.

- `review/index.html`: 20 matched full-region triptychs, original / raw / simplified; neutral flat shading and wireframe rows.
- `detail-review/index.html`: five additional tighter views of regions 8, 18, 24, 34 and 37, including source witness 506773 and a simplification-added incidence.
- `census/affected-faces.jsonl`, `tracked-face-outcomes.bin`, `analysis.json`, `face-classification.json`, `visual-completion.json`: complete measurements and explicit reliability limitations.

Cameras, scale and lighting match within every triptych; camera derivation uses source geometry only. The full original mesh participates in depth testing; these are not cutaways, repaired geometry or smoothed normals. The close-ups reveal genuine microfacet loss. At the eight frozen full-object 1024-square views, the previous verified silhouette IoU remains 0.995111–0.999895 and boundary discrepancy at most one pixel. Those views do not establish close-range or specular-material equivalence.

A separately proposed **static hard-surface geometry profile** can retain the existing 1,000-triangle, closed oriented manifold, one-component, complete self-intersection, readback, bidirectional distance, full-surface 0.002D support and eight-view silhouette requirements. The saved candidate already satisfies those technical gates. For this profile, accepting the explicitly documented microfacet losses would require visual sign-off at the intended viewing scale/material rather than silently exempting small faces or lowering the frozen normal threshold. This is a **post-hoc profile proposal**, not an independent benchmark PASS or a reusable automatic noise rule.

**Assessment: promising lower-detail static-prop geometry; optional-profile qualification remains conditional on that explicit appearance decision.** UV layout, PBR materials, normal/detail baking, engine import, collisions and other asset classes remain unqualified. No claim is made for organic bodies, wounds or fingers. The shortest credible next action is that profile/appearance decision using these existing meshes and views, followed only if accepted by separate asset-pipeline qualification—not another speculative wrap run.

## Verification, resources and limitations

53 targeted offline tests passed, including nine new guard, correspondence, distance, occlusion, PNG and serialization tests; native contact/normal fixtures also passed. An initial broad filename discovery included an unrelated image-preparation test requiring unavailable Pillow; it was not treated as a pass or installed. The explicitly scoped reconstruction/ownership test suite passed.

The first rendering stage completed all 20 images and camera metadata, then **FAILED** while serializing a NumPy integer in its final receipt. That failed receipt and logs remain unchanged. The defect was reproduced offline and corrected with a native integer counter. A separately reviewed read-only finalisation validated all original PNG CRCs, decoded pixel lengths and region/camera coverage, and generated the five distinct tighter views. No original image was overwritten or rerendered. `visual-completion.json` is the new completion receipt; it does not rewrite the failed host result.

| Stage | Seconds | Peak owned Job commit |
|---|---:|---:|
| Complete native census | 40.969 | 492,101,632 bytes |
| Independent numerical assessment | 5.031 | 218,386,432 bytes |
| Initial 20-view rendering, receipt failure | 46.922 | 167,997,440 bytes |
| Existing-image validation and five detail views | 8.234 | 160,608,256 bytes |

Maximum build Job commit was 906,772,480 bytes. Minimum observed available host RAM was 16,241,451,008 bytes. All stages retained 6 GiB process/Job, four-core affinity, 180 seconds, 12 GiB host floor and 2 GiB disk safeguards. Runtime children exited naturally with empty owned Jobs; two build/configuration stages used the existing exact-identity compiler-helper cleanup. No resource cap was raised, no reconstruction retry occurred, and no WSL, GPU, neural operation, installation or environment change was performed.

Independent review approved bounded Stage A execution, checked the actual local images and source-contact limitation, and supports stopping before Stage B. Final preservation reauthenticated the original inputs and historical evidence; all 28,795 authenticated dependency-file identities and 204 historical Git refs remained unchanged. CGAL remains a privately used GPL/commercial dependency under the previously reviewed terms; this result grants no proprietary binary-distribution permission. Only new source, tests and text are published; all meshes and images remain private.
