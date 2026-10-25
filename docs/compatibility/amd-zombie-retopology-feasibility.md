# Zombie two-region retopology feasibility

Base commit: `983f5115bd1529571bc062b14974f1e3f3ccb985`. Branch: `codex/amd-zombie-retopology-feasibility`.

## Decision

**NO-GO for extending either tested method to a larger character region.** The surface-chart method demonstrates actual orientation improvement and useful coarse exterior sampling on ordinary skin. It does not preserve the chest wound well enough: openings become stepped and enlarged, cavity walls disappear, and separated surface pieces remain. The voxel control produces closed fragments after losing most of the reference surface. Zero non-manifold edges is therefore not sufficient evidence of usable retopology.

Stop this experiment without a parameter sweep or another sequence of small repairs. The blocker is explicit reconstruction of separate exterior skin, wound-rim and cavity-wall sheets with feature constraints and joins. A single projected surface cannot represent all overlapping or hidden cavity sheets. Volume remeshing an open, inconsistently oriented crop does not supply that missing interpretation. These results do not establish that full-body voxel remeshing or all sheet-aware methods must fail; neither is qualified by this bounded test.

The original high-detail zombie remains the immutable detail reference. No candidate replaces it. No full-character rebuild, neural/WSL execution, UVs, detail baking, texture work or production integration occurred.

## Reference and method selection

Input is the **approved orphan-cleaned** PLY, SHA-256 `377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5`, with 325,807 vertices and 673,604 triangles. The winding-flipped experimental variant is not used. Original neural output SHA-256 `e589fec9f5ea58ec7bdb07c00c3c40911b6191d08055611fac17ddee65aa7918` and original run contract `13a140490d933d0c9f21af701af36caf54384b0fe0e5dc9c4714d0328dea6343` are authenticated using the existing CPU-only provenance checks; no model weights are loaded.

Installed Blender is **5.2.1 LTS**, build `9e2066aef7ef`, with OpenVDB enabled. RNA inspection exposes Voxel, Blocks, Smooth and Sharp Remesh modes, QuadriFlow boundary/sharp options, and BVH two-sided ray queries. QuadriFlow's operator is exposed but its execution is not qualified by this test. No dependency was installed.

| Method | Disposition | Reason |
|---|---|---|
| Native voxel remesh | Tested once on each crop | Cheapest built-in volume control; no smoothing, adaptivity or component filtering. No artificial caps added before processing. |
| First-hit surface chart with discontinuity seams | Tested once on each crop | Cheap test of visibility-selected exterior geometry and explicit unsupported seams, independent of source winding. Uses existing Blender BVH and small NumPy helpers. |
| QuadriFlow | Considered; not executed | Does not itself clean intersecting source sheets; boundary/sharp options cannot identify the intended wound topology. |
| Decimation alone | Considered; not executed | Reduction alone leaves the source's ambiguous sheets/incidences unresolved. |
| Feature-led multi-sheet retopology | Considered; not built | Requires explicit wound rim/cavity interpretation and deformation flow; beyond this bounded automatic proof. |

Blender documents the volume nature of voxel remeshing and that automatic remesh output is generally insufficient for final deformation topology. See the [official Blender retopology manual](https://docs.staging.blender.org/manual/en/latest/modeling/meshes/retopology.html) and [Remesh Modifier documentation](https://docs.blender.org/manual/en/4.3/modeling/modifiers/generate/remesh.html). Installed RNA and measured results, rather than the documentation version, determine the actual settings below.

## Reproducible scope and settings

Complete triangles only; local crops preserve the reference triangle coordinates, corner order and source face/vertex ID mappings exactly.

| Region | World-space crop box | Reference V / F | Measured median edge | Voxel size / requested chart step |
|---|---|---:|---:|---:|
| Ordinary posterior torso | x[−.045,.045], y[.02,.12], z[.17,.23] | 2,266 / 4,330 | .0018623676224381 | .0037247352448761 |
| Chest wound | x[−.005,.115], y[−.12,.055], z[.125,.255] | 9,743 / 19,157 | .0018305829011647 | .0036611658023295 |

One configuration per method/crop; twice measured median edge sets a local sampling scale, not a global polygon target. Voxel mode uses adaptivity 0, smooth shading false and disconnected filtering false. The posterior chart uses 25×17 cells, actual x/z steps .0036/.003529411765; the wound chart uses 33×36, steps .003636363636/.003611111111.

The chart casts two-sided first-hit rays from +Y for the posterior and −Y for the wound, ignoring original winding. Its exterior direction is inferred from the known outside viewing half-space and visible first hits; it is not a global inside/outside certificate. Each cell requires all nine dense samples and rejects adjacent sample distances above twice its largest grid step. Accepted cells become consistently oriented triangle pairs. Disconnected vertex fans are separated only on the new chart, preserving those candidate triangle coordinates (two duplicated chart vertices in the wound). No source vertex, triangle or component is changed.

These nine witnesses are **not a continuous guarantee against bridging small holes**. Frontal evaluation detected four newly covered source-void pixels in the wound crop, three inside the wound ROI. They are reported rather than interpreted as successful hole repair. A single chart does not recover hidden sheets or overhangs. All candidate components are retained and counted; there is no largest-component filter.

## Topology and triangle reduction

Counts describe each crop/candidate, not changes to the full source character. Boundary counts include crop cuts and must not all be interpreted as defects.

| Region / method | V | Triangles | Reduction | Components | Boundary edges | >2-face edges | Winding conflicts | Invalid vertex links |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Posterior reference | 2,266 | 4,330 | — | 1 | 240 | 34 | 1,429 | 37 |
| Posterior voxel | 8 | 12 | 99.72% | 1 | 0 | 0 | 0 | 0 |
| Posterior chart | 378 | 650 | 84.99% | 1 | 112 | 0 | 0 | 0 |
| Wound reference | 9,743 | 19,157 | — | 3 | 1,362 | 746 | 6,571 | 841 |
| Wound voxel | 430 | 796 | 95.84% | 16 | 0 | 0 | 0 | 0 |
| Wound chart | 823 | 1,280 | 93.32% | 4 | 374 | 0 | 0 | 0 |

No zero-area triangles were measured in any output. Fewer conflicts in a replacement candidate do not mean the corresponding source sheets were repaired in place. The chart avoids ambiguous layers and replaces sampled visible surfaces; the voxel output loses almost everything. Both change membership, shape and boundaries of their separate candidates.

### Artificial cuts versus original defects

Global edge incidence gives exact attribution for reference-crop boundary edges:

- Posterior: **192 artificial cuts and 48 genuine unpaired source edges**.
- Wound: **420 artificial cuts and 942 genuine unpaired source edges**. One artificial cut intersects an originally non-manifold edge.

"Genuine" means already unpaired in the original source, not necessarily intentional wound anatomy. For chart output, geometry near the outer crop rectangle is labelled crop margin; all other boundary edges remain unclassified support/discontinuity seams. Posterior: 70 crop-margin / 42 internal seam edges. Wound: 87 / 287. These labels are not one-to-one source-defect correspondences. Voxel outputs have no boundary edges because the surviving pieces are closed; this conceals loss instead of preserving the open surface.

## Matched visual and surface measurements

Private comparison sheets show reference / voxel / chart from −35°, 0° and +35°, with matched cameras, flat neutral lighting, front/back colors and backface-transparent tests. Fifty-four 512×512 Cycles CPU frames use eight samples/four threads; no normal smoothing or render-time modifier. All imported mesh payloads match saved arrays before and after rendering. Separate 128×128 two-sided ray arrays measure coverage and depth; these are sampled tests, not continuous silhouette or Hausdorff proofs. This task cannot establish full-character silhouette preservation.

At the straight view:

| Measure | Posterior voxel | Posterior chart | Wound voxel | Wound chart |
|---|---:|---:|---:|---:|
| Reference hit pixels retained | 0/5,670 | 4,579/5,670 | 401/6,341 | 4,461/6,341 |
| New hits where reference is void | 0 | 0 | 8 | 4 |
| Reference/candidate hit-mask IoU | 0 | .8076 | .0632 | .7031 |
| First visible candidate hits facing camera | No hits | 4,579/4,579 | 409/409 | 4,465/4,465 |
| Shared-hit absolute depth error, p95 | No overlap | .000291908 | .004537463 | .001074672 |

The posterior reference has 1,329 camera-facing and 4,341 backfacing first hits. The chart genuinely improves the visible side assignment: all 4,579 observed candidate hits face the camera, including oblique tests. Its ordinary skin is much more usable under culling, with close depth agreement where present. Nevertheless, border recession and enlarged internal gaps remain. Excluding two grid steps at the artificial crop margin still loses **164/3,755** reference-hit pixels. Lower counters do not eliminate that surface loss.

The wound chart has **416 missing deep-hit pixels and 18 shallowed deep hits** out of **988** reference deep-hit witnesses; only **554** retain a deep hit. The voxel control retains **3**, loses **961** and shallows **24**. These witnesses use x[.03,.10], z[.15,.23], with depth relative to surrounding reference skin plus two sampling steps. The reference ROI has 626 void pixels; chart covers three and voxel six. This geometric classifier is not a semantic wound mask or hidden-cavity proof.

The matched images show the chart replacing ragged wound edges with large stair steps, widening some openings and dropping interior walls/slivers. The right-oblique view exposes missing layered surfaces particularly clearly. Wound-core coverage still loses **1,269/5,061** reference-hit pixels after excluding crop margins. Thus this failure is not merely an artificial crop-edge effect. The voxel control leaves small closed remnants rather than a faithful torso/wound surface; its clean topology counters are misleading.

The wound chart also exposes **103 backfacing first hits** in the −35° view (2,473 front-facing hits). A +Y/−Y chart-normal rule is therefore not a complete three-dimensional exterior classification. The straight-view orientation result must not be generalized to every view or cavity surface.

Sampled candidate-to-reference p95 distances can look small even for a bad result: the posterior voxel remnant lies near the source (.000004074), but visible-reference-to-candidate p95 is .07204 because almost all source coverage is missing. The chart's corresponding bidirectional p95 values are .000236/.001812 posterior and .000572/.005804 wound. These tests explicitly expose omission rather than relying on a one-way proximity score.

## Suitability and next decision

The chart demonstrates cleaner local orientation, low triangle count and a manifold open patch; that is a useful limited exterior-sampling primitive. It is **not ready for character UV layout, detail baking or deformation-oriented retopology**: artificial seams, missing cavity sheets and absent anatomical edge flow must be resolved first. No UVs or baking were attempted. Quad topology alone would not address those missing surfaces.

Neither tested output meets the appearance/feature-preservation acceptance, so **do not extend these settings to a larger character region**. The next substantive approach would require explicit feature-constrained exterior/rim/cavity sheets and controlled joins, potentially artist-guided retopology. That is a different design decision, not authorization to begin another repair series here. Preserve the original high-detail reference for later projection/baking when a suitable retopology exists.

## Execution and preservation

Existing native limits remain 6 GiB process commit, 180 seconds/stage, 12 GiB live host RAM and 2 GiB output disk, with GO/EOF, exact owned Job attachment and natural process cleanup. All established neural gates/resource safeguards remain unchanged. CPU settings and measured phase resources are in the companion verification receipt; no GPU-intensive work occurred.

Measured peak process commit was **470,925,312 bytes**; minimum sampled available host RAM was **20,283,047,936 bytes**. Capability inspection took 4.109 seconds, reference preparation 4.109, both methods on both regions plus sampled-distance collection 4.500, independent mesh readback 1.844 and all 54 matched CPU renders/depth measurements 16.547. Each stage stayed within its original limits and settled naturally. Individual modifier/chart construction times are recorded as part of the processing receipt, separately from supervision and sampling overhead.

An initial image-summary verification stopped on an F32/F64 percentile interpolation mismatch. Recorded raw arrays reproduced the original metric exactly with its explicit F64 interpolation; a regression test now enforces matching dtype. The check tolerance was not relaxed. Failed receipt and partial sheets remain intact; only the summary/verification stage reran in a fresh output path. Neither mesh processing nor rendering was repeated.

Private root: `artifacts/amd-zombie-retopology-feasibility-20261025-v1` outside the published repository. It contains full source mappings, reference crops, four candidates, settings, topology/readback, ray/depth witnesses, renders and verified comparison sheets. All original assets and historical evidence are retained. Only safe source/tests/textual results are published; no character geometry, image or private process details enter Git. Independent review and preservation/publication results are recorded in the companion receipt when complete.
