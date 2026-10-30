# Phase 13 — Modern CGAL reconstruction proof

30 October 2026. Base `eeecc0d45ae37b00407f27d8f18ba95ef82e5040`.

**Native backend: PASS. AI-derived box reconstruction: NO-GO at plane support. Modly integration: NO-GO pending a qualified AI result.** One real official-sample optimisation succeeded. The single declared AI plane extraction failed the unchanged coverage gate, so no AI optimisation or candidate was produced. This is a segmentation failure, not evidence that the original AI source lacks the missing geometry.

## Qualified implementation and actual execution

Used the official [CGAL 6.2.1 release](https://github.com/CGAL/cgal/releases/tag/v6.2.1), [SCIP Optimization Suite 10.1.0 Windows SDK](https://github.com/scipopt/scip/releases/tag/v10.1.0), existing MSVC 19.44.35228, portable CMake 3.31.10, Boost 1.87 headers and Eigen 3.4.0. All new runtime/build files are private under workspace `artifacts/amd-cgal-global-proof-20261030-v1/`; working Python, TRELLIS.2, ROCm and WSL installations were not modified. Published release digests were verified; exact source/build/binary identities and commands remain in local receipts. No executable or dependency archive is published with Modly.

The official `polyfit_example_user_provided_planes.cpp` and its original 31,374-point `ball.ply` were unchanged. **Two private audited header overlays** add only resource/candidate caps, runtime-module logging and acceptance of `SCIP_STATUS_OPTIMAL` rather than any incumbent. Objectives and reconstruction algorithms remain upstream. The SDK's exported CMake target contains publisher-build `C:/deps` paths; a standard private SHARED IMPORTED target instead binds its actual DLL, import library and headers with no system fallback. This changes installation metadata, not solver code.

| Official sample measurement | Result |
| --- | ---: |
| Candidate faces / MIP variables / constraints | 793 / 2,097 / 4,478 |
| Actual runtime | SCIP 10.1.0, SoPlex 8.1.0 |
| Optimisation | Optimal; 111 LP iterations; zero gap |
| Export/readback | 36 vertices, 20 polygons, 68 fan-equivalent triangles |
| Boundary / non-manifold edges / winding conflicts | 0 / 0 / 0 |
| Face-bearing vertex components / bad vertex links | 1 / 0 |
| Wall time / peak total Job private commit | 2.625 s / 63,127,552 bytes |
| Exit and cleanup | Natural; owned Job empty |

Eleven observed SDK DLLs were loaded from the private SDK and independently matched to authenticated archive payloads. Windows platform modules account for the remainder. This snapshot was captured at adapter entry, not as a trace of every possible later delay-load. Independent OFF parsing and topology checks passed. Complete sample self-intersections and sample surface fidelity were not measured: this is solver/export/basic-topology qualification, not an AI asset-quality PASS.

## One AI-only plane extraction

Reauthenticated the ten decoded tensors, 3,888-row input provenance, raw PLY/NPY geometry and original comparison receipts. Original decoded SHA remains `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975`; PLY SHA remains `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3`. Frozen Phase 11 criteria SHA remains `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0`.

The original 65,536-point area-stratified sampling was replayed from authenticated AI triangles: original triangle IDs and spatial split match; F64 positions/normals agree within 2e-15. Only the 51,574 training points entered CGAL Efficient RANSAC. The original 13,962 held-out points were excluded. No procedural dimensions, ground-truth mesh, box template, symmetry or convex-hull substitution was used.

The method and settings were frozen before native AI input: seed 20261030, probability .001, minimum support 24, distance .002D, unsigned normal angle 10°, cluster distance .02D and at most 48 output planes. Reversible source-derived computational units use `q=100*(p−AI_bounds_center)/D`; inverse coefficients and original-unit checks preserve physical shape. The old greedy estimator's 96-hypothesis knob is not a CGAL RANSAC API parameter. Native sample IDs survive reordering; complete coordinate/normal membership was verified. Actual detected coefficients were used without recentering. CGAL's potentially 3×epsilon final membership was independently screened at the strict frozen distance/angle limits.

| AI screen | Result |
| --- | ---: |
| Detected planes | 17 |
| Strict training support per plane | 217–9,267; all ≥24 |
| Training coverage | 49,569/51,574 = 96.1124% |
| Original spatial holdout coverage | **13,343/13,962 = 95.5665%**, required ≥99.5% |
| Uncovered held-out points | 619; all lack a compatible plane normal |
| Native extraction wall time / peak Job commit | 2.828 s / 45,764,608 bytes |

No thresholds or parameters were changed after observing these results. All 2,005 unassigned training points and their identities remain in evidence. Infinite-plane coverage cannot establish finite face support or surface fidelity. Necessary feature coverage is unverified, so forcing global closure would be unjustified. This result does not prove those surfaces absent from the original neural mesh.

**AI MIP, output mesh, triangle budget, candidate topology/vertex links/intersections, bidirectional fidelity, bevel/corner preservation and eight matched views: NOT RUN / NOT PASS.** The raw 2,047,210-triangle neural mesh retains its existing NO-GO classification. No generated mesh was repaired or substituted.

## Safety, licensing and decision

Predeclared bounds: 6 GiB process **and Job** commit, four logical CPUs, 180 s per stage, 120 s/4,096 MiB/one thread per solver, 20,000 candidate faces, 12 GiB live host floor and 2 GiB free output disk. Neural 22 GiB admission, GPU/disk and backup protections remain unchanged. Setup encountered preserved staging/configuration failures; these are not included in the sample timing. An owned compiler `vctip.exe` remained after otherwise successful builds. Independently reviewed cleanup uses its retained handle, complete Job enumeration, stage/output success, creation identity, canonical installed path and pinned SHA; it is explicitly labelled nonnatural compiler-helper cleanup. It never applies to sample/solver runs. No global telemetry or installation settings were changed.

Licensing is mixed: CGAL reconstruction/shape detection GPLv3+, solver interface LGPLv3+; modern SCIP/SoPlex Apache-2.0; Eigen MPL-2.0-only build; Boost BSL; GMP LGPLv3+/GPLv2+ alternatives; MPFR LGPLv3+; ZIMPL LGPL; AMPL/MP HPND and bundled notices; Ipopt EPL; MUMPS CeCILL-C/BSD exceptions; Intel compiler runtimes under their internal-development EULA. NumPy/OpenBLAS/LAPACK/GCC-runtime notices and exception terms are retained. Actual SDK Boost is 1.92.0, separate from application headers 1.87.0. [SCIP's licensing inventory](https://www.scipopt.org/index.php#license), [Intel's grant](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html) and [MUMPS terms](https://mumps-solver.org/index.php?page=dwnld) supplement actual package notices. Private execution is qualified; proprietary integration, binary redistribution and a complete redistribution compliance package are **not authorised or qualified**. No paid solver, cloud service or purchase was used.

Independent review and 21 offline tests cover numerical transforms, strict support, topology discrimination, Windows exact spawn events and positive/negative retained-handle ownership checks. Original source hashes, protected file records and previous Git refs were rechecked. Private build recipes, overlays, binaries, meshes, point clouds, logs and resource receipts remain local. No neural/GPU/WSL execution occurred.

**Next technical direction, separately authorised:** use mesh-adjacency/normal plane segmentation with explicit small-feature and residual coverage, rather than another RANSAC tuning loop. Preserve the frozen assessment and fidelity gates; do not quietly replace the examined holdout or infer absent planes. The now-working global backend is potentially useful for adequately segmented, piecewise-planar buildings, machinery and props, as [upstream describes](https://doc.cgal.org/6.2.1/Polygonal_surface_reconstruction/index.html). No AI asset has passed, broader asset-class success is unproven, and this is not a method for zombies or organic detail. Stop here.
