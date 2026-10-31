# Phase 17 — finite-patch subdivision feasibility

**NO-GO: subdivision of the existing supporting planes cannot meet the frozen fidelity gate.** No production solve, reconstructed mesh, or geometry modification was performed. This is a representation limitation, not proof that the original AI surface lacks adequate geometry.

Parent: `b5343362982e540bf05140ae13b70c693cda3e79`. Private evidence: `artifacts/amd-patch-subdivision-20261031-v1`, outside the published repository. Classification: **NO_GO_FIXED_PLANE_SUBDIVISION_FIDELITY**. The raw neural benchmark remains NO-GO.

## Why splitting supported portions cannot close this arrangement

The architecture gate in Stage A takes precedence over constructing an unusable subdivision engine. We verified the premise rather than assuming it: all 1,105 recorded candidates are full, connected convex cells of the complete arrangement of the 27 refined source planes, clipped by CGAL's six source-derived proxy planes. There are no missing interior plane cuts.

For any finite, conforming subdivision of one such cell, a new internal edge has only coplanar subfaces on its two sides. No different supporting plane passes through the cell's interior. A closed, embedded surface therefore requires both subfaces to be selected or neither: `x_left = x_right`. These equalities propagate through the full cell. Unsupported portions omitted from the subdivision are zero-selected complement regions; their boundary still forces adjoining selections to zero. Disconnected domains, holes and concave patch outlines do not escape this condition.

Consequently, a closed supported selection on these unchanged planes must select whole original cells. It inherits the original coarse incidence constraints and unsupported-cell exclusions. Splitting shared edges conformingly cannot introduce a new transverse surface to close a finite patch boundary. Duplicated coincident sheets or T-junctions would violate the frozen embedded-manifold/intersection requirements.

This corrects the earlier suggestion that finite-patch subdivision alone might recover a valid closed selection: that suggestion had not established whether the arrangement already contained every plane intersection. Here it does. The conclusion does **not** apply to arrangements with missing plane cuts, newly supported transition surfaces, changed plane hypotheses, or unrestricted nonplanar reconstruction.

## Authenticated geometry and independent measurements

The new read-only audit authenticated 273 Phase 16 evidence files, 552 Phase 14 files, all 47 original source snapshot entries, the immutable criteria, both previous adapter binaries, and 188 protected identities. The final preservation pass matched them again. Original AI tensors, source arrays, failed candidates and historical evidence remain unchanged.

CGAL's original source was independently checked against the complete [official 6.2.1 library archive](https://github.com/CGAL/cgal/releases/download/v6.2.1/CGAL-6.2.1-library.zip), SHA-256 `db49ffa9d2c5a3386e0725983bf3478e3fd2ade43553f8fc3900e18554decd49`. Both inspected installed headers match their authenticated archive payloads:

- `Polygonal_surface_reconstruction/internal/hypothesis.h`: `eb1c781eba23c7098f7c78d5474cbde301606fee462d3ee584e81dc552d2ed94`.
- `Polygonal_surface_reconstruction.h`: `6e7ac0100275203d19f2aefbdf250f18421a18cef3b11784ca59c9733a680aa9`.

The official implementation constructs source-derived bounding proxies, cuts the plane arrangement pairwise, and identifies adjacency through cached triple-plane intersections. Our audit separately reconstructed exact rational intersections from the recorded F64 plane coefficients, with all rounding differences bounded. These analytical vertices were never exported or substituted for AI geometry.

| Check | Result |
|---|---:|
| Candidate polygons / refined source planes | 1,105 / 27 |
| Vertex occurrences / unique triple-plane identities | 4,412 / 650 |
| Missing interior cuts / ambiguous vertices / duplicate cells | 0 / 0 / 0 |
| Reconstructed geometric edge fans | 1,607; exact recorded face-set multiset match |
| Fan incidence | 935 four-face fans; 672 proxy-boundary singleton fans |
| Source-plane proxy domains | All 27 have one complete boundary cycle |
| Source points used to reproduce CGAL proxy bounds | 60,527 |
| Maximum proxy-bound discrepancy | 1.066e-14 native units |
| Maximum analytical vertex displacement | 3.079e-14 native units |
| Unsupported witnesses transferred to exact cells | All 968 |
| Possible original cells after necessary closure propagation | 104 |

Native coordinates use the existing source-diagonal normalization `D = 100`; this is not a prescribed box size. Proxy bounds were recomputed from authenticated AI training points using the official formula. No procedural reference or ground-truth dimensions were consumed.

The unsupported-witness transfer independently establishes exact distance at most `1e-10` from each witness to its recorded polygon, and at most `1e-10` from that polygon to its analytical cell. Subtracting their sum from the inherited guarded source-distance witness still exceeds the unchanged `0.2 + 1e-8` native threshold for every excluded face. Source-triangle distances remain inherited guarded CGAL measurements; they are **not** described as exact rational certificates. The exact arithmetic covers cell identity, halfspace relations, incidence and the decisive enclosing-box distance.

## Complete frozen-source feasibility screen

Every admissible subdivision selection is contained in the same 104 surviving cells. Their outward-expanded enclosing boxes supply conservative necessary distance bounds; these boxes never construct replacement geometry.

| Measurement | Result |
|---|---:|
| Frozen samples examined | 65,536 / 65,536 |
| Samples beyond `0.01D + 1e-10D` necessary distance | 65,265 (99.5865%) |
| Necessary RMS distance lower bound | 0.2064947714D |
| Necessary 95th / 99th percentile bounds | 0.3209113958D / 0.3595782100D |
| Exact decisive witness | Source row 59,133 |
| Decisive distance lower bound | 0.401424758954D |
| Frozen maximum distance | 0.01D |

Population summaries use F64 arithmetic with outward-rounded bounds. The decisive witness uses an exact rational squared-distance comparison to all surviving analytical boxes, expanded by `1e-8` native units. Its failure is about 40 times the maximum allowable distance; the decision does not depend on population rounding or borderline samples. Small percentile differences from Phase 16 reflect the explicitly used nearest-rank summary convention.

No new representation on these fixed planes can remove this obstruction while remaining closed and supported. Candidate complexity, SCIP optimisation and the final 1,000-triangle budget were therefore not reached. There is no new candidate for eight-view validation, readback or asset promotion. The original historical 96-triangle candidate remains failed finite-support evidence.

## Adjacency, objective and validation scope

We verified the geometry-to-original-face-variable mapping and all geometric fans without reusing CGAL's pointer identity as the answer. We did not insert arbitrary vertices into `extract_adjacency()` or invoke an unchanged MIP on subdivided geometry.

An actual subdivision adapter would also need explicit coplanar two-face equality constraints: upstream treats non-four-face fans as zero-selected. Its complexity coefficient depends on `adjacency.size()`, and support-point ownership and per-face coverage are not automatically additive after splitting. Blindly rebuilding fans and retaining those coefficients would change the optimisation meaning. None of these objective changes can overcome the feasible-set obstruction proved here, so no speculative coefficient or source-certificate machinery was implemented.

Fifteen new synthetic tests cover complete cells, missing cuts (explicit refusal), duplicate/degenerate/coincident configurations, exact distances, a narrow oblique patch, conforming shared edges, disconnected/concave/holed domains, unsupported complements, invalid indices and complete population screening. Exhaustive binary assignments independently check two-sided internal-edge closure on connected subdivisions. These are applicability/closure tests, **not** a claim to have built or qualified a polygon-clipping engine. Combined with existing incidence, integrity, resource, ownership and native-routing safety regressions: **81 tests passed; zero failures, errors or skips**.

Source domains and arrangement decisions already consumed source-wide information, including historical held-out areas. This is not blind validation. The frozen 65,536 samples were not used to fit new parameters; no new parameters were fitted. The independent reviewer reproduced the source-level exact-cell reasoning and synthetic tests, then reviewed the measured outputs and preservation receipts.

## Resources, incomplete runtime check and preservation

The source-only cell audit completed in **9.250 s**, peak committed process/Job memory **28,581,888 bytes**, with natural exit and an empty owned Job. Admission available RAM was 17,548,214,272 bytes. Original limits remained 6 GiB process/Job, four-core affinity, 180 s per stage, 12 GiB live host floor and 2 GiB free output disk. No neural thresholds changed.

Separately, complete native-runtime reauthentication hit its unchanged **180-second deadline**; peak Job commit was 93,511,680 bytes. Owned forced cleanup passed. The timeout is preserved, was not retried, and does not qualify current native execution. Its performance cause was not established. The subsequent independently authenticated source-only audit required no CGAL/SCIP execution or installed third-party package import. This separates a runtime admission failure from the demonstrated geometric obstruction.

No solver, compiler, installation, download, GPU, neural, WSL, backup or environment mutation occurred. Existing source files and all 202 recorded historical local refs were preserved; only new source/tests/text are published. Private geometry and binary evidence stay local. Existing CGAL/solver licensing restrictions remain; this work grants no proprietary distribution permission.

## Decision and next boundary

**Stop this fixed-plane subdivision route.** Its blocker is closed-surface representability and resulting fidelity, not a failed optimiser or an established lack of source geometry. The prior subdivision recommendation is withdrawn for this arrangement.

Any future continuation would need genuinely source-supported transition geometry or revised plane hypotheses that permit different closed surfaces; adding samples, clipping harder, or running the same solver cannot do that. Whether the AI source supports those changes remains unproven. No alternative strategy or further execution is authorised or begun by this result.
