# Phase 14 — Mesh-aware, end-to-end CGAL reconstruction

30 October 2026. Parent `6b8e1cbacededbd8072fb768a452b2a526663360`.

**A genuine closed 96-triangle AI-derived candidate was produced. Asset promotion: NO-GO.** The single SCIP solve reached verified optimal status. Topology, complete self-intersections, sampled fidelity, all eight silhouettes and source-feature retention passed. One selected face group demonstrably extends outside its compatible finite source patch; another remains uncertified. The candidate is retained as failed evidence, not delivered as a qualified game asset. Raw TRELLIS.2 geometry remains NO-GO.

## Actual method and execution

Reused the authenticated CGAL 6.2.1 / SCIP 10.1.0 / SoPlex 8.1.0 stack and existing CPU NumPy/compiler. No downloads, installations, dependency rebuilds, GPU, neural or WSL execution occurred. Only new private adapters were compiled. The [official generic, mesh and point-set region-growing implementations](https://doc.cgal.org/6.2.1/Shape_detection/index.html) were inspected. A generic face-index adapter avoids repairing the defective triangle soup into a `Surface_mesh`.

Original index-edge adjacency reproduced 1,421 boundary and 519 greater-than-two-incidence edges; both block traversal. Fitting uses real triangle-area integration (`Dimension_tag<2>`), unsigned normals within 10 degrees and maximum triangle-vertex distance .002D. Final region members are rechecked after refitting. No box dimensions, symmetry, orthogonality, fixed face count, ground truth, hull substitution, source edits or RANSAC sweep were used.

The native pass recovered 51 finite source patches, labelling 1,925,348 of 1,926,786 eligible triangles in 2.843 seconds. The extra 120,424 locked triangles were excluded from fitting and propagation. Genuine small corner patches were recovered; their sparse original sampled support initially failed. The directive explicitly allowed additional real samples, so one predeclared area-stratified barycentric pass added 12,032 genuine points, using the same rule for every patch. Historical heldout spatial cells and locked triangles were rejected. Original-only support was an overly restrictive interpretation, corrected without lowering any threshold or fitting heldout coordinates.

This qualified 27 planes, with 261–8,966 strictly assigned training points each. The historical holdout was unchanged at **13,962/13,962 coverage**; additional locked validation reached **65,534/65,536**, both above 99.5%. Source-wide nonlocked geometry informs fitting beneath some historical points, so that set is not blind spatial generalisation. Locked triangles were absent from training/propagation; nearby surfaces remain correlated.

CGAL's actual mandatory refit/merge output was observed before optimisation. Every final training member passed; all 51 source patches retained 100% strict triangle/area support against surviving planes. Finite domains remained original triangle-index sets, with disconnected regions, holes and junctions preserved. No convex hull or bounding box replaced their support.

| Single global reconstruction | Measured result |
| --- | ---: |
| Candidate arrangement faces | 1,105; cap 20,000 |
| Solver | SCIP 10.1.0 / SoPlex 8.1.0 |
| Actual status | Optimal, zero gap, one node, 649 LP iterations |
| Variables / constraints | 2,975 / 6,282 |
| Output | 50 vertices, 27 polygons; 96 triangles after copy-only triangulation |
| Owned stage including live source screening | 29.313 s; peak Job commit 224,722,944 bytes |

The exact in-memory CGAL object was screened and then solved after GO+EOF under one owned Job. An exclusive marker enforces one actual MIP. A preceding harness invocation refused an OFF byte mismatch before any solve; its cause is not asserted as harmless ordering. The same-object protocol removes the need to assume serialization equivalence. All refused evidence remains intact.

## Independent candidate acceptance

| Frozen requirement | Result |
| --- | --- |
| At most 1,000 triangles | PASS: 96, from 2,047,210 source triangles |
| Boundary / non-manifold / winding-conflict edges | PASS: 0 / 0 / 0 |
| Vertex links / face-bearing components | PASS: every link one cycle; one component |
| Unused / duplicate / zero-area / nonfinite geometry | PASS: zero; no duplicate coordinate vertices |
| Complete self-intersections | PASS: zero, whole-candidate CGAL exact-predicate test |
| Source-to-candidate RMS / p99 / maximum, divided by D | PASS: .000014051 / .000072927 / .000255941 |
| Candidate-to-source RMS / p99 / maximum, divided by D | PASS: .000018249 / .000074592 / .001051205 |
| Eight 1024-square silhouettes | PASS: minimum IoU .999809468; maximum boundary distance 1 pixel |
| Source bevel/corner/residual retention | PASS: all 51 patches retain compatible output faces; every patch vertex within .002D |
| Original-unit PLY readback | PASS: exact vertices and triangles |
| Complete compatible finite output footprints | **NOT PASS: one demonstrated extension; one unresolved group** |

Each direction used 65,536 actual triangle-surface samples. Independent NumPy brute-force distances for 256 dispersed source samples agreed with native AABB distances within 1.40e-16D. Maximum distance of every original source vertex to the candidate is .000318910D. The entire output surface was additionally bounded within .002D of some real source geometry using adaptive 1-Lipschitz triangle-cell certification, not just point sampling.

Eight CPU neutral-lit, two-sided flat-shaded views share source-frozen cameras, scale and lighting. Elevation is 20 degrees, azimuths 0 through 315 in 45-degree steps, with no candidate fitting. Inspection shows preserved overall shape, bevel widths and corners, with the raw mesh's small dark seam artefacts largely absent. This visual improvement does not override the failed support gate.

## Exact remaining failure

Proximity to any source surface is insufficient for assigned-patch support. Every output triangle was matched to its actual refined plane; separate complete certificates used only genuine source triangles whose vertices and unsigned normals meet that plane's unchanged .002D / 10-degree criteria. **25 of 27 groups certified; 90 of 96 triangles certified before refusal in the other groups.** The remaining six are not all claimed to be proven defective.

Group 6 has only 907 compatible source triangles but contributes four output triangles. An independently evaluated output witness is **.006533590D from its compatible source domain**, exceeding .002D, while only **.000156269D from the entire source**. Thus the global arrangement extends a plane beyond its observed compatible patch while remaining visually close to another source surface. This is a concrete finite-support failure, not missing source geometry, a failed solver, a topology defect or a sampling-only objection.

Group 13 also lacks a complete certificate. Limited independent probes did not establish a violation there; it remains unresolved. The confirmed group-6 witness already prevents promotion, so no further geometry experiment, threshold change, optimisation or repair was attempted.

**Interpretation:** mesh-aware segmentation plus global selection demonstrably produces very light, closed, oriented geometry with excellent measured shape preservation. Upstream's soft point-coverage objective does not impose a hard finite-patch footprint constraint. Correct plane recovery and optimal solver exit therefore cannot establish asset quality alone.

## Evidence, preservation and next decision

Private evidence is under `artifacts/amd-mesh-aware-reconstruction-20261030-v1/`, including raw candidate OFF, triangulated/original-unit copies, provenance, solver/runtime logs, domain witnesses and eight-view previews. `validation/review.html` and `validation/matched-eight-view-contact.png` show the matched comparisons. Candidate assets are failed experimental evidence, not a promoted deliverable.

Original decoded archive SHA-256 remains `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975`; PLY remains `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3`; frozen criteria remain `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0`. Reauthentication covered 47 original source/provenance files and 35,021 selected dependency payload files (770,322,338 bytes), plus pinned archives. Original assets, 188 protected records and existing Git refs were checked unchanged; no VHDX/backup access occurred.

All stages retained 6 GiB process-and-Job commit, four-core affinity, 180-second deadlines, 12 GiB live host and 2 GiB output disk floors; SCIP retained 120 seconds, 4,096 MiB and one thread. Recorded minimum host availability was 16,636,485,632 bytes. Peak operational Job commit was 543,330,304 bytes; maximum including compilation was 972,570,624 bytes. Analysis and solver processes exited naturally with empty Jobs. Qualified compiler-helper cleanup remained limited to its pinned owned handle. An inherited binary-size check failure and a later Python routing failure are preserved; neither was a resource breach or extra MIP. No safety gate was relaxed.

Independent review covers method, actual geometry, preservation, limitations and the final NO-GO. Thirty-seven offline tests cover routing, source isolation, edge incidence, unsigned normals, numerical distances, frozen gates and retained-handle cleanup. GPL/private-use and linked-dependency limits from the [Phase 13 licence review](amd-cgal-global-reconstruction-proof.md#safety-licensing-and-decision) remain applicable. No dependency headers, binaries or private meshes/images are published; proprietary integration and binary distribution are not authorised.

**One next technical action, separately authorised:** add a hard candidate-face eligibility veto from the existing compatible finite-domain certificate before SCIP selection, then permit one bounded solve or report infeasibility. Reuse this segmentation/backend; no fitter sweep, new engine, mesh repair or forced closure. The engine merits that narrowly scoped continuation for piecewise-planar props, buildings and machinery. Broader performance is untested, and no claim is made for zombies or organic surfaces. Stop here; no follow-up is executed.
