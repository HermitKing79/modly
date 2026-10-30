# Phase 16 — complete certification and closure feasibility

**Result: NO-GO — candidate geometry/fidelity failure for the current arrangement.**
Parent: `967ec6baac435a8cef4189d58430770383f47542`.
Private evidence: `artifacts/amd-closure-feasibility-20261030-v1`.

Certification is complete. No new mesh was exported. One optimistic diagnostic
SCIP solve ran; **zero production solves** ran. This is neither an empty optimum
nor insufficient certification, and it does not establish inadequate original AI
geometry. Small nonempty incidence-feasible selections exist, but every permitted
closed selection fails the frozen whole-asset fidelity requirement.

## Measurements and decision

| Stage | Measured result |
|---|---|
| Historical states | 79 eligible, 719 unsupported, 307 unresolved |
| Replayed witnesses | All 719 confirmed against complete compatible source-triangle domains and actual polygon footprints |
| Optimistic edge screen | 386 potentially allowed faces; 134 survive necessary conditions |
| One diagnostic MIP | SCIP optimal, status 1, gap 0; 32 selected faces; no export |
| Complete depth-20 pass | 136 eligible, 967 unsupported, 2 depth-unresolved; 2,153,223 queries; zero resource-exhausted faces |
| Targeted depth-28 continuation | Group 10 certified at depth 21; group 13 has a real unsupported witness at distance 0.200088506074568 native units |
| Final certification | **137 eligible, 968 unsupported, 0 unresolved**, all 1,105 faces examined |
| Certified edge screen | 33 eligible faces necessarily excluded; **104** remain potentially usable |
| Whole-asset obstruction | Exact rational lower-bound witness **40.1424758954 native units**, versus frozen maximum **1.0** |

The diagnostic used 2,975 binary variables and 7,002 constraints: the original
6,282 constraints, 719 zero-veto constraints and one diagnostic-only nonempty
constraint. Original objective, manifold constraints and solver implementation
were unchanged. At diagnostic execution, its selection contained 15 historically
eligible and 17 unresolved faces. Under the subsequent complete depth-20
classification, those 32 faces comprised 30 eligible and two depth-unresolved
faces. It was never an asset-quality result.

The 32-million-query allowance changed an operational work limit, not geometry
acceptance. The complete pass used only 2.153 million queries. One targeted
continuation of the two unresolved identities used 184,525 queries; all other
certificates were retained unchanged. The unchanged support requirement is
0.002D: native diagonal 100, distance 0.2, numerical margin 1e-8 and planarity
guard 1e-10. The depth limit is not a frozen Phase 11 geometry criterion.
Finite-domain certificates use guarded floating-point CGAL calculations; exact
rational arithmetic applies to the final bounding-box obstruction.

## Why no production solve was justified

CGAL's actual intersection fans were captured directly, using its internal
triple-plane intersection identities. No coordinate-based edge welding was used.
All polygon-edge occurrences are accounted for: 935 four-face fans and 672
one-face fans. CGAL requires four-face incidence sums to be zero or two; other
fans have sum zero. Iterative necessary-condition elimination, starting with the
968 unsupported faces, leaves 104 possible faces. A replayable elimination
certificate is saved. This screen alone is not a closure proof.

Every selected polygon, and any triangulation of it, lies inside its vertex
bounding box. We expanded every one of the 104 boxes outward by 1e-8 native units,
then measured distance to their **entire union**, which is more permissive than
any possible selected mesh. Frozen source sample 59,133 is still at least
40.1424758954 native units away. An independent exact-Fraction calculation from
the recorded binary64 coordinates checked all 104 boxes and proved its squared
lower bound exceeds even `(1.00000001)^2`. The frozen maximum is 1.0 native units,
or 0.01D; the proven lower bound is approximately **0.401425D**.

Therefore **every** permitted closed selection fails at least one mandatory
source-to-candidate distance sample. No additional optimisation can change that
conclusion while keeping this arrangement and its eligibility constraints.
Floating-point summaries find 65,265 of 65,536 source samples beyond the maximum;
the decisive independently exact check is the one witness above. Bounding boxes
are used only to prove failure, never to certify surface fidelity or topology.

The obstruction reflects whole arrangement cells crossing genuine finite source
patch boundaries. Vetoing a whole unsupported cell also removes its supported
portion; closure constraints then exclude additional cells. These counts are
**not counts of missing neural surfaces**. Group 13's previously uncertified
extension is now demonstrably unsupported. Group 10's previous refusal was only
insufficient subdivision depth.

## Identity, execution and preservation

Fresh CGAL construction reproduced the exact refined plane coefficients and
memberships. The identity-forensics run matched 401 historical polygon bit-level
keys; the subsequent diagnostic matched 1,038. All 1,105 polygons had unique
same-plane cyclic correspondence, with maximum measured vertex displacements
1.4648213755e-14 and 1.2306961193e-14 native units respectively. Historical fan
identities were never recorded and are not claimed identical. Old eligible
certificates were not transferred: all current polygons were reclassified.
Unsupported witnesses were independently rechecked on actual live footprints.
The certification and final obstruction use the same frozen diagnostic polygons
and fans, pinned through the earlier reviewed input hashes.

The initial exact-key capture refusal is preserved. An earlier CMake configuration
completed, but its cleanup receipt remains FAIL because the new directory lacked
the compiler-helper identity pin. Its owned Job removed the leftover helper.
The verified pin was staged; configuration was not repeated. Subsequent builds
used the qualified retained-handle compiler-helper cleanup. All geometry-analysis
workers terminated naturally with empty owned Jobs.

Unchanged limits: 6 GiB per process/Job, four-core affinity, 180-second stages,
12 GiB live host floor and 2 GiB output-disk floor; SCIP 120 seconds, 4,096 MiB,
one thread. Maximum Job commit including compilation: **1,254,912,000 bytes**.
The separate 22 GiB neural admission gate and all original neural safeguards
remain unchanged and were not invoked.
Minimum observed available host RAM: **16,885,100,544 bytes**. Diagnostic stage:
3.812 seconds / 322,621,440 bytes peak Job commit. Complete certification:
6.234 seconds / 313,413,632 bytes. Targeted continuation: 2.015 seconds /
313,634,816 bytes. Final fidelity screen: 1.219 seconds / 101,744,640 bytes.
These are supervised stage times, not isolated solver timings.

Reauthentication/preservation passed for 552 Phase 14 files, 156 Phase 15 files,
47 original source hashes and 188 protected identity records. Seven approved
runtime archives and their 35,021 selected installed files were reauthenticated.
All 201 historical local refs remain unchanged. Original neural tensors, raw
geometry and the earlier 96-triangle failed candidate remain unchanged. No WSL,
GPU, neural work, downloads, installation, source repair or backup operation ran.
Both historical and locked source geometry inform finite-domain constraints;
neither validation set is described as blind to those constraints.

**66 offline tests pass.** Independent review reproduced input correspondence,
witness checks, constraint feasibility, certification counts, graph elimination
and the exact rational obstruction. No candidate exists, so post-candidate
topology, vertex links, intersections, original-units readback and eight-view
validation are **NOT RUN / NOT PASS**. No solver-optimal diagnostic is promoted.

## Single recommended next action

For separate authorisation, evaluate **finite-domain-aware candidate subdivision
with conforming shared edges before face selection**, preserving supported parts
of large cells rather than vetoing them wholesale. Keep this engine, source
geometry and frozen criteria. This is an architectural feasibility direction,
not a promise that subdivision can close the source or repair organic assets.
No such work began here. CGAL/solver licensing remains the previously qualified
private-experiment scope; no dependency headers, executables or assets are
published, and proprietary integration still requires a separate decision.
