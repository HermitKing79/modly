# Phase 15 — Finite-patch-constrained global reconstruction

30 October 2026. Parent `0b9dd14fc33fa63f9624b5f930d8dcc7847e631f`.

**Asset result: NO-GO — no new mesh. Intrinsic source/closure feasibility: INCONCLUSIVE.** One constrained SCIP solve reached verified optimal status with every face variable zero. The veto mechanism worked, but the bounded classifier exhausted its internal query budget before examining all necessary surface groups. This is not evidence that the source lacks geometry or that a nonempty closed reconstruction is mathematically infeasible.

## What was implemented and actually tested

Reused the authenticated AI input, unchanged 27 refined planes, training samples, finite triangle domains and qualified CGAL 6.2.1 / SCIP 10.1.0 / SoPlex 8.1.0 runtime. No segmentation, sampling, neural generation, WSL/GPU execution, downloads or installations were repeated. Only the new private adapter was built; dependencies and all Phase 14 files stayed intact.

Four minimal private hooks classify the actual untouched `target_mesh` after `f:index` assignment, impose vetoes after binary-variable creation, gate the solve, and verify selection before CGAL deletes unselected hypotheses. Removing these hooks reproduces the original header text exactly. The objective, existing manifold constraints and solver implementation remain unchanged. The published patch generator reproduces the compiled private header without publishing dependency headers.

Each candidate record binds its live face descriptor, first-block MIP index, actual vertex coordinates, exact hexadecimal geometry key and live `f:supp_plane` pointer to the observed refined group. Neither copied-OFF row ordering nor dangling `f:supp_segment` pointers are used. Index uniqueness, geometry and plane assignment are checked before constraints and after solution recovery.

The classifier verifies simple convex polygon footprints with CGAL predicates and covers them with complete triangle fans, including boundary subsets. Adaptive longest-edge subdivision uses the 1-Lipschitz distance bound to genuine compatible source triangles. The support limit remains **.002D** (.2 in authenticated native D=100 units), with a 1e-8 native numerical margin and planarity rejection above 1e-10. The margin reserves the bounded deviation between alternate triangulations of nearly planar cells. These are guarded floating-point certificates, not exact-arithmetic distance proofs.

Every face receives `CERTIFIED_ELIGIBLE`, `CONFIRMED_UNSUPPORTED` or `UNRESOLVED`. An unsupported result requires a point witness outside the guarded distance limit. Query/depth exhaustion remains unresolved. Every noneligible face receives an explicit equality **`x_i = 0`**; binary upper bounds alone would not work with the qualified adapter. The complete returned solution and all vetoes are checked before any output conversion.

### Native synthetic fixtures

| Actual CGAL/SCIP fixture | Result |
| --- | --- |
| Supported closed tetrahedron | Optimal; four faces selected; closed output |
| Same plane arrangement with one genuinely smaller source patch | Extension vetoed; optimal empty selection; excluded variables zero |
| Same unsupported case with a **fixture-only** positive-selection constraint | Verified SCIP infeasible, status code 2 |
| Exhausted query budget / excessive planarity residual | Unresolved and refused |
| Duplicate face index / spoofed excluded selection | Rejected |

The production formulation keeps CGAL's original allowance for all-zero selection. The positive-selection constraint was confined to the infeasibility fixture. Three synthetic solves do not count as AI reconstruction attempts. Fifty-three offline tests passed, including Windows audit routing, separator/collision handling, exact patch roundtrip, numerical checks and owned-process safeguards.

## The single AI solve

| Measurement | Result |
| --- | ---: |
| Actual candidate faces / complete eligibility records | 1,105 / 1,105 |
| Certified eligible | 79 |
| Confirmed unsupported extent | 719 |
| Unresolved | 307 |
| Explicit zero-veto constraints | 1,026 |
| MIP variables / total constraints | 2,975 / 7,308 |
| Original constraints / added vetoes | 6,282 / 1,026 |
| Actual SCIP status | Optimal, code 1, zero gap, primal=dual=0 |
| Selected faces | **0** |
| Native stage / peak total Job commit | 6.438 s / 304,508,928 bytes |
| Exit | Natural; owned Job empty |

All 2,975 solution values were zero. No excluded face was selected. One exclusive AI-MIP marker exists; there was no second solve, unconstrained fallback, geometry modification or substitute output.

### Why this does not settle geometric feasibility

The global classifier budget of **2,000,000 reference-distance queries** was an implementation bound inherited from the earlier output-footprint checker. It was not a frozen geometry requirement or demonstrated hardware limit. It was undersized for complete classification of this 1,105-cell arrangement.

- **305** unresolved cells were refused specifically for `QUERY_LIMIT`.
- **Two** cells, in groups 10 and 13, reached depth/numerical uncertainty. They are not proven unsupported.
- The budget expired during group 15, leaving 24 cells there unresolved.
- All **281** cells in groups 16–26 were unqueried. These groups correspond to three bevel patches (source regions 14, 6 and 8) and all eight corner patches (regions 18–25). Their source geometry is preserved; it was not established as missing or unsupported.

The 719 unsupported cells are arrangement hypotheses with demonstrated unsupported extent, not 719 defects in the source mesh. Group 6 retained two eligible cells and rejected 42 unsupported ones. Those exclusions correctly prevent the earlier unsupported extension from being selected, but an incomplete global mask cannot establish a useful complete-asset result.

The executed parent admitted a solve after only the minimum four-certified-face check. That was insufficient evidence of complete-surface readiness. **The internal work budget and admission were underqualified for this experiment.** RAM and the 180-second stage deadline were not exhausted. The empty optimum is a valid result for the consumed mask; it does not prove nonempty infeasibility. Presolve eliminating all variables is also not such a proof, because optimization reductions can eliminate dominated choices.

## Offline correction and acceptance scope

After preserving the actual result, a small **prospective, offline-only** guard was added: a future GO refuses any `UNRESOLVED / QUERY_LIMIT` record after checking complete and unambiguous face records. Local depth uncertainty retains its declared veto policy. Exactly reaching the query count is allowed if every record is nevertheless conclusive. Regression tests show the saved case would now be refused before GO. No native process, classification or MIP was rerun under this guard. The executed entry source, approved receipts and original outcome remain unchanged in evidence.

A Windows separator lookup also caused a preserved pre-native provenance refusal before the successful invocation; its normalization and collision checks were tested and independently reviewed. It consumed zero native classifications or solves. Two private adapter builds capture the initial implementation and reviewed planarity/depth fixes; no dependency was rebuilt.

No candidate exists. Therefore **topology, intersections, bidirectional distances, source-feature retention, compatible output footprints and eight-view output comparison: NOT RUN / NOT PASS for Phase 15**. Phase 14's 96-triangle candidate remains historical failed evidence and was not relabelled or substituted.

The declared prospective feature test retains the old preferred-group check as a diagnostic and evaluates actual source-vertex proximity to the union of finite output triangles compatible with each source triangle's unsigned 10-degree normal. This removes an unfrozen hypothesis-ID requirement; it does not certify triangle interiors. It was frozen before the AI solve, independently reviewed and not exercised because no output exists.

Full source domains include geometry from historical heldout areas and locked triangles. Solver eligibility consumed that geometry. **Neither validation set is blind to these constraints**, even though their sample coordinates were not fitted and their identities remain unchanged.

## Preservation, resources and decision

Private evidence: local `artifacts/amd-finite-patch-reconstruction-20261030-v1/`, including the plan, hashes, exact private header patch, build/runtime logs, native fixtures, every eligibility record, veto/selection vectors, actual empty result and postrun correction receipt. No new model or visual comparison is available.

Rehashing verified all **552 Phase 14 files** and **47 original source/provenance files** unchanged, including the old 96-triangle candidate and original neural/PLY/NPY data. All **188 protected records** and historical Git refs were checked. The reused dependency archives and 35,021 selected installed payloads were authenticated; 11 actually loaded SDK modules independently matched approved archive bytes. No VHDX/backup access or working-environment modification occurred.

The 6 GiB process-and-Job limit, four-core affinity, 180-second CPU stage deadline, 12 GiB live Windows floor and 2 GiB output disk floor remained enforced. SCIP retained 120 seconds, 4,096 MiB and one thread. Minimum observed host availability was 17,181,614,080 bytes. Peak Job commit including compilation was 1,251,762,176 bytes. All operational processes exited naturally; compiler-helper cleanup retained its exact owned-handle checks. Neural RAM/VRAM/storage gates were untouched.

CGAL/private-use and linked-dependency restrictions from the [qualified licence review](amd-cgal-global-reconstruction-proof.md#safety-licensing-and-decision) remain applicable. Only owned source/tests and this report are published, without dependency headers, binaries or private assets. Proprietary integration or executable distribution is not authorised.

**Decision:** veto implementation and safety tests PASS; clean-asset delivery NO-GO; feasibility with a complete eligibility mask INCONCLUSIVE. Before considering any future solve, complete the necessary certification work within the existing physical CPU/RAM limits and enforce the new work-completion admission. This requires a separate decision. No retry, larger budget, alternative strategy or execution handoff is initiated here.
