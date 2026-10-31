# Phase 18 — native CGAL Alpha Wrapping

**Final asset: NO-GO under the unchanged feature-preservation gate. Native CPU reconstruction: demonstrated.** The one AI wrap and one predeclared simplification produced a closed 502-vertex, 1,000-triangle model. Topology, intersections, bidirectional distances, complete surface support, eight silhouettes and readback passed. Local normal-compatible feature retention failed; the candidate is preserved as **AI-DERIVED RECONSTRUCTION — UNQUALIFIED**, never raw neural geometry PASS.

Parent: `46990a85f2a71aa24a76177aa552cc810e9b5f7a`. Private evidence: `artifacts/amd-alpha-wrap-20261031-v1`. Protocol was frozen and independently reviewed before execution in [the plan](amd-alpha-wrap-plan.md). No parameter sweep, reconstruction retry or alternative engine followed.

## Implementation and provenance

We reused CGAL 6.2.1, the existing isolated Boost 1.87 headers, CMake 3.31.10, MSVC 19.44 and CPU NumPy 2.3.5. No installation or download was needed. SCIP, SoPlex, Eigen, GPU and WSL were not used. The complete pinned archive hashes and every selected installed payload were verified before compilation: 28,799 files across five dependency stages. These separately bounded stages avoided an unbounded aggregate verification pass.

The [official triangle-soup example](https://github.com/CGAL/cgal/blob/v6.2.1/Alpha_wrap_3/examples/Alpha_wrap_3/triangle_soup_wrap.cpp) was compiled unchanged and executed once on its authenticated `blobby-shuffled.off` sample. Its actual output had 760 vertices and 1,516 triangles; independent checks confirmed a valid, closed, outward-oriented single component with no non-manifold vertices, duplicate/degenerate geometry or self-intersections.

The AI adapter calls the same triangle-soup overload on **all 1,023,995 original vertices and 2,047,210 triangles**. It does not construct a manifold mesh from the input or repair its indexing/winding. The pre-existing authenticated source-only normalization centres the AI bounds and scales their diagonal to 100. Original units are restored on export. No procedural ground truth, prescribed box dimensions or 27-plane arrangement enters reconstruction.

The published `amd_alpha_wrap.CMakeLists.txt` is the executed private build project with the source-directory prefix replaced by `${MODLY_PROOF_SCRIPTS}`; expanding that variable reproduces the executed file byte-for-byte after newline normalization. The controller records the exact Visual Studio x64/v143 configuration, disables GMP and package-registry writes, builds one target at a time, and pins each resulting executable. The existing evidence paths are consumed one-shot paths, not a new execution handoff.

Original AI identities remain:

- PLY: `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3`.
- Vertex NPY: `fa52e8076fe99cc251db578911f1e0b6eb6224a5a1118aeb18e6a28ab9926e60`.
- Face NPY: `3b530ebc0a00e73a2a8192ad855f136734e2170cb5357d28e669df3064a400fc`.
- Original ten-tensor archive: `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975`.
- Frozen criteria: `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0`.

## One configuration and one conditional simplification

The AI run used alpha **0.05D**, offset **0.0005D**, seed **20261030**. The official CLI uses denominators, hence 20/2000 rather than literal fractional arguments. The offset leaves margin below the 0.0015D RMS limit. Alpha remains a feature-access/complexity choice, not a guarantee that details survive. The [upstream manual](https://doc.cgal.org/6.2/Alpha_wrap_3/index.html) explains that wrapping approximates an offset surface and can suppress narrow cavities or details; those effects must be measured.

The raw wrap had **1,406 vertices / 2,808 triangles** and passed independent topology/intersection checks. It exceeded the frozen budget, so exactly one predeclared CGAL Lindstrom–Turk pass ran with the official polyhedral-envelope and bounded-normal-change filters. Its envelope was **0.0005D of the original AI source**, measured against the initial wrap. It stopped at **1,000 actual triangles**, with 502 vertices. Neither envelope nor target was changed after seeing results. No other smoothing, filling, welding or cleanup was added.

## Measured final acceptance

| Gate | Result |
|---|---|
| At most 1,000 triangles | PASS — exactly 1,000 |
| Closed, outward, consistently oriented; one component | PASS |
| Boundary / non-manifold edges / winding conflicts | 0 / 0 / 0 |
| Invalid vertex links / duplicates / unused vertices / zero-area triangles | All zero |
| Complete candidate self-intersections | Zero |
| Source → candidate RMS / p95 / p99 / max | 0.000487394 / 0.000508502 / 0.000534531 / 0.000690973 D |
| Candidate → source RMS / p95 / p99 / max | 0.000487348 / 0.000508354 / 0.000534046 / 0.000983355 D |
| Eight silhouette IoUs | 0.995111–0.999895; all ≥0.995 |
| Eight silhouette boundary distances | 1 pixel each; limit 2 |
| Whole candidate surface within 0.002D of original triangle union | CERTIFIED — all 1,000 triangles; 3,059,704 queries |
| Whole original surface within 0.002D of candidate | CERTIFIED — all 2,047,210 triangles; 2,047,210 queries |
| Original-unit PLY readback | Bit-exact exported coordinates and indices |
| Independent 256-point brute-force distance cross-check | Maximum disagreement 1.125e-16 D |
| Normal-compatible source features | **FAIL — 20 of 51 patches** |

Each sampled direction contains 65,536 points. Frozen limits remain RMS 0.0015D, p95 0.003D, p99 0.005D and max 0.01D. All eight views use the original fixed cameras, 20-degree elevation, 1024-square resolution, shared scale/light and neutral two-sided flat shading. No smoothing or texture masks the geometry.

Plane-fitting counts, assigned plane IDs and plane-specific candidate domains are **NOT APPLICABLE**, not PASS. Their pre-reviewed replacement certifies complete triangle footprints against the actual finite source-triangle union in both directions, at unchanged 0.002D. Longest-edge subdivision uses a conservative 1-Lipschitz bound, 1e-8 native guard, depth 20 and a 32-million-query ceiling per direction. These are guarded numerical certificates, not symbolic exact geometry. Neither direction exhausted its bound. Unsupported bridges beyond that distance would fail; features smaller than the declared tolerance are not thereby proven preserved.

## Exact remaining blocker

The original 51 tracked source patches comprise **5,776,044 vertex incidences**. Each incidence must be within 0.002D of a final triangle whose unsigned normal is within 10 degrees of that **actual original source face**, retaining the prior conservative angle/distance guards. This is not comparison against a fitted plane normal.

**1,568 incidences fail (0.02715%)**, in regions **8, 18, 24 and 34–50**. The latter are small residual patches. Regions 8, 18 and 24 have respectively 2, 3 and 6 failures. The remainder occur in regions 34–50. No failure was waived because its area or visual effect appears small. An intermediate progress message overstated the affected region count as 21; the verified count is 20.

Independent review directly sought all final triangles for original source face **506773**, region **37**, using a separate small standard-library brute-force calculation. Its three vertices have unrestricted distances of approximately 0.000422D, 0.000459D and 0.000418D to the final mesh, yet every nearby triangle normal differs by at least **18.9038 degrees**. The nearest 10-degree-compatible triangles are about 0.485D away. The raw 2,808-triangle wrap already fails this same normal witness, so this particular failure is **not solely caused by simplification**. A passing face in region 34 was independently reproduced as a mapping/control check.

The contact sheet retains the major box silhouette and bevel bands, with cleaner-looking surfaces and small edge/shading changes. It does not establish conspicuous loss of the major shape. Conversely, visual similarity cannot override the failed local orientation/detail criterion. The evidence supports a localized feature-representation failure, not wholesale shape destruction. It does not yet attribute every failed incidence to wrapping versus simplification.

## Resources and preservation

| Stage | Wall time | Peak owned Job commit |
|---|---:|---:|
| AI wrap, including admission/worker overhead | 8.281 s | 825,008,128 bytes |
| One simplification | 6.156 s | 28,647,424 bytes |
| Native geometry/distance/feature/view validation | 29.063 s | 543,723,520 bytes |
| Independent Python assessment/export | 3.203 s | 115,568,640 bytes |

The wrap algorithm itself reported 5.877 s. Across all 22 authentication/build/execution/verification stages, measured elapsed time summed to 282.889 s; this excludes development/review time. The largest build Job used 1,031,929,856 bytes. Minimum observed Windows available RAM was **15,965,425,664 bytes**, above the unchanged 12 GiB floor. Every stage stayed within 6 GiB process/Job, four-core affinity, 180 seconds and 2 GiB free disk.

Runtime processes exited naturally with empty owned Jobs. Six configuration/build stages used the previously qualified, exact-identity compiler-helper cleanup; this is recorded separately from natural runtime exit. No forced Job cleanup, retry, raised limit or unrelated process termination occurred.

Final preservation rehashed the 47 original source bindings, 273 Phase 16 and 552 Phase 14 evidence files; 188 protected identities and the Phase 17 evidence manifest also matched. All 28,799 selected installed dependency file identities/lengths/timestamps remained unchanged after execution (complete payload hashes were checked before execution). All 203 recorded historical local refs remained unchanged before publication. No working AI environment, original asset, checkpoint, backup or VHDX was written.

**77 offline regression tests passed**, with zero failures/errors/skips. Independent pre-execution review approved the parameter/feature protocol and exact stage/file bindings; final review checked actual receipts, output identities, source preservation, images and the independently reproduced feature witness.

## Private outputs and decision

Under `artifacts/amd-alpha-wrap-20261031-v1/`:

- `raw-wrap.off`: `3da7a965d5769399de477c1b035c42ed6a0c60526681101a0e100a3f6c30c9ad`.
- `simplified.off`: `2e3522850b76d300fcf2c92dac9ccc32272ed1ed8b24ae0839a2e5cc8aa04b0b`.
- `validation/AI-DERIVED-UNQUALIFIED.ply`: `573b2ed867758595db91ade052270ea264e4d88ff583bef8e6e737f055476884`.
- `validation/matched-eight-view-contact.png`, `validation/review.html`, `result.json` and complete stage receipts.

The engine demonstrated a practical, inexpensive native CPU route from a defective two-million-triangle soup to a very close, closed 1,000-triangle model. **It merits consideration as an optional finishing component, but this candidate does not pass the frozen acceptance contract and is not approved for integration.** The algorithm is not a box-template fitter; this proof nevertheless qualifies only this benchmark. It establishes nothing about preserving zombie wounds, fingers or other organic cavities.

The single next decision is whether to authorise targeted work on the demonstrated local feature-orientation loss, with the acceptance criteria explicitly retained or separately reviewed. No further parameter attempt, engine, repair campaign or criterion change is initiated here.

Alpha Wrapping and the used high-level CGAL algorithms are GPL-3.0-or-later/commercial components; support headers also carry LGPL terms and Boost carries its own license. The [CGAL licensing policy](https://www.cgal.org/license.html) distinguishes private use from distribution obligations. This experiment does not authorise proprietary Modly binary distribution. Only our source/tests/text are published; CGAL/solver binaries, meshes, images and private source evidence remain local. A separate licensing decision is required before product integration.
