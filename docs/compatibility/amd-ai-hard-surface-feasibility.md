# Phase 11 — AI-derived hard-surface feasibility

30 October 2026. Base `35810dfc7bec02fb3d63fbec570909d1711b99e8`.

**NO-GO for producing a clean-mesh candidate with this ad hoc reconstruction route. No candidate was constructed.** This is not proof that the AI geometry is unreconstructable. The measurements show strong planar information, but the fixed plane model fails its declared validation screen, and finite face boundaries/junctions remain unverified. Under the project-lead engineering-priority addendum, a box template would be a poor use of further effort.

## Authenticity and separation from ground truth

Fresh CPU authentication passed the original ten tensors, hashes/linkages, 3,888-row provenance, original raw NPY/PLY identity and saved image/topology comparison files. Neural SHA remains `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975`; PLY SHA remains `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3`.

Authentication and surface analysis ran in **separate CPU children**. The assessment accepts only the AI mesh arrays, a filtered authentication receipt and frozen criteria. Its audit policy rejects the procedural benchmark module/reference contract and other artifact folders, including the original ground-truth mesh and dimension-bearing comparison metadata. The old image comparison is authenticated, not used for fitting. All measured bounds, scale, planes and residuals derive from AI positions/triangles. No original dimensions, box generator, imposed symmetry/orthogonality, fixed face count, halfspace clipping or convex hull entered the estimator.

## Criteria declared before measurement

[Frozen criteria](amd-ai-hard-surface-criteria.json), SHA-256 `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0`, were copied before source assessment and checked at stage boundaries/end.

- At most **one candidate and 1,000 triangles**, labelled **AI-DERIVED RECONSTRUCTION**. This is a simple-prop budget, not an organic/game-wide polygon target.
- Source-only scale `D`: diagonal of measured AI vertex bounds. Plane distance ≤0.002D and unsigned normal difference ≤10°; spatial holdout coverage ≥99.5%.
- Candidate/source bidirectional triangle-surface distance: 65,536 samples each way; RMS ≤0.0015D, p95 ≤0.003D, p99 ≤0.005D, sampled maximum ≤0.01D. Point-to-infinite-plane distance cannot substitute for this.
- Eight frozen source-derived 1024² views: silhouette IoU ≥0.995 each, silhouette-boundary distance ≤2 pixels; significant bevels/corners must remain visibly and geometrically supported.
- One closed face component; zero boundaries, non-manifold edges, winding conflicts, unused vertices, duplicate/zero-area faces or nonfinite geometry; one cyclic link at every vertex; complete candidate triangle-intersection verification. Unassessed is not PASS.
- Finite face footprints and junctions must be supported. A method must handle supported nonconvex arrangements or refuse them; a good box-shaped average fit never authorizes convexification.

## AI-only measurements

One deterministic pass sampled **65,536 points by original triangle area**, using geometric normals modulo sign. No winding correction occurred. A spatial-block holdout reserved 13,962 points; the other 51,574 fitted unsigned-normal/offset plane hypotheses. Maximum 48 planes/96 hypotheses and all distance/angle limits stayed fixed. Residual samples were retained.

| Measurement | Result |
| --- | ---: |
| AI vertices / triangles | 1,023,995 / 2,047,210 |
| Fitted plane hypotheses | 21 |
| Large surface-like fits | 6, each 5,495–9,267 training samples |
| Bevel-like support bands | 12, each 425–834 training samples |
| Small corner-like fits | 3, each 30–34 training samples |
| Training coverage | 51,519/51,574 = **99.8934%** |
| Spatial holdout coverage | 13,871/13,962 = **99.3482% — below 99.5%** |
| Held-out misses | 28 without compatible normal; 63 beyond plane-distance limit |

The six large fits have training p95 residuals approximately 4.2e-6–1.2e-5 source units; bevel-like fits approximately 2.3e-5–1.5e-4. These describe observed plane support, not verified polygon boundaries or outward sheets. Area weights include overlapping source faces and are not independent exterior-area evidence.

Spatial blocks can withhold an entire tiny corner patch. Therefore the failed holdout screen does **not** establish missing source geometry or the failure of every full-data reconstruction. Nor does the residual tail, which can match an unmodeled corner to a distant angularly compatible parallel plane, measure Hausdorff/surface error. There was no threshold tuning, full-data refit, inferred corner cap or fallback.

## Engineering choice: optimise overall progress

| Approach | Implementation effort / elapsed time | Resources | Preservation and expected useful scope |
| --- | --- | --- | --- |
| Bespoke box/plane assembly | Low for a convincing one-off; substantial additional work for general finite faces. Fit assessment took 1.594 s; construction untested. | Measured 108 MiB peak commit; CPU. | Risks replacing unknown corners/concavities with assumptions. Low value beyond the shapes its template encodes. **Do not pursue.** |
| Global plane-arrangement face selection, PolyFit/CGAL family | More initial binary/solver/data qualification; data-dependent optimisation time, unmeasured on Modly. Prefer this investment over repeated local edits. | CPU plus MIP solver; plane/candidate growth needs explicit caps. | Jointly evaluates data support and topology; a stronger evidence-backed direction for **closed piecewise-planar props**, not organic meshes. Modly success remains unproven. |
| Upstream narrow-band remeshing/projection | Existing algorithm, but native backend qualification is unresolved; no qualified CPU equivalent here. AMD time/memory unmeasured. | Native GPU dependencies; no execution admitted. | Broader surface class, but wound/thin-feature/opening preservation remains unproven. It is not an available drop-in fix. |

CGAL's [polygonal reconstruction method](https://doc.cgal.org/latest/Polygonal_surface_reconstruction/index.html) constructs face hypotheses from detected planes, then optimizes their selection under topology constraints. It requires all necessary major planes and a MIP solver. This supplies the global finite-face reasoning missing from the greedy estimator; it does not make insufficient input or intentional openings disappear safely.

The [PolyFit project](https://github.com/LiangliangNan/PolyFit#obtaining-polyfit) lists prebuilt Windows executables/Python bindings and an open-source SCIP option. Those are **availability claims, not local qualification**. Current metadata/spec checks find no PolyFit/CGAL bindings or external SCIP executable. SciPy includes a HiGHS MILP interface, but that alone supplies neither plane arrangements nor a verified reconstruction implementation. No solver was executed. PolyFit is GPL-3.0 software; distribution suitability and exact binary/dependency provenance must be reviewed before integration. No commercial-solver fallback or charges are authorized.

**Recommended next engineering step, separately approved:** qualify one existing global plane-reconstruction implementation in isolation, preferably an authenticated CPU executable with the open-source SCIP path, against finite-patch/concave-feature fixtures and this immutable AI source under the frozen geometry gates. Do not write a new box fitter or a broad solver framework. This is a more substantial, potentially more computationally expensive route, but it addresses a useful asset class beyond one box. It still does not solve characters, wounds or arbitrary smooth assets; those require a separate preservation-aware route. TRELLIS.2 remains unchanged.

## Result and preservation

**Reconstruction: NOT ADMITTED. Candidate topology, vertex-manifoldness, intersections, bidirectional fidelity and candidate/source matched-view tests: NOT RUN, never PASS.** No low-poly mesh, GLB, texture or game-asset candidate is delivered. The original raw neural geometry retains its NO-GO classification.

Source authentication took 3.313 s/100,851,712 peak committed bytes; assessment took 1.594 s/113,225,728 bytes. Both owned Jobs exited naturally and empty under the unchanged 6 GiB process/180 s stage/12 GiB live-host/2 GiB free-output limits. Six focused tests cover arbitrary tilted/parallel planes, retained curved residuals, input immutability, GT-read rejection, owned GO and frozen-criteria tampering. Independent review checked method, receipts and decision. Source hashes, protected identities and previous Git refs remain preserved; no whole-environment byte audit is implied.

Private inspection samples/settings/results remain in workspace `artifacts/amd-ai-hard-surface-assessment-20261030-v1/`. No source geometry was edited, no repair variant was exported, and no WSL, GPU, neural inference, model load, installation, smoothing/remeshing, financial charge or purchase occurred. Stop after this reviewed feasibility result.
