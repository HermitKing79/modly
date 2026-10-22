# Census verification PASS; selected patch BLOCKED

Base `523ac20600ea78c5c90beeb1312ce42f72a04661`; isolated branch
`codex/amd-brick-census-recovery-patch`. One corrected readback and one frozen
patch qualification ran. **No census rerun, simplification or alternative
selection occurred.** Original geometry and historical failed evidence remain
unchanged. Private arrays were not published.

## Corrected readback — PASS

The preserved verifier used the face-mask name `certified_edge_touch_faces`
as a summary key. The producer correctly calls that summary count
`certified_edge_face_population`. A new verifier copy uses an explicit,
one-to-one mapping for all 15 masks and validates the exact 34-field positive
summary. Missing/unknown fields, duplicate JSON keys/mappings, incorrect
mapping targets, invalid integer types and count/work inconsistencies reject.
Sealed producer bytes reject any alteration to any of its 34 stored values.
The original zero-prelink producer branch has a different schema; this
experiment does not invent defaults for it.

The exact old production loop reproduced the KeyError in a focused RED
probe. Fourteen readback checks passed before execution. The independent
[source review](amd-brick-census-recovery-evidence/independent-verify-review.json)
bound six sources, the plan and log. The genuine corrected worker then passed:

* All 29 saved artifact size/hash/schema checks and the original source closure.
* All 12,520,752 occurrence/permutation/key/incidence/winding checks.
* Mask, protected-face/vertex and complete one-ring guard relations and counts.
* Endpoint/protection conditions for all 758,314 saved eligible edges.
* All 32 frozen independent full-global-star link witnesses and link arithmetic.
* All original-component and octant distribution recounts with separate
  face/vertex/edge denominators.

The global fan graph was not rebuilt. Its classification and the recorded
maximum-free-degree/fan-component statistics remain authenticated,
source-reviewed producer outputs; recorded padded-work arithmetic is checked.
This is the unchanged original readback scope, not a new all-edge link solve.
The [independent readback outcome](amd-brick-census-recovery-evidence/independent-readback-outcome.json)
accepts the explicit PASS receipt. The old verifier's BLOCKED receipt remains
SHA-256 `1e278c38d9fc9b4642f2eae7323ee1860991e1772410f0fb9f48fc2b2f770dd7`.

## Single original patch — BLOCKED by crop guards

The [policy](amd-brick-census-recovery-evidence/recovery-plan.md) was frozen
before selection. Use the largest certified-edge-connected face component,
ties lowest original face ID, then its lower-median original edge key. Include
complete original endpoint stars and exactly three synchronous all-incidence
edge-face rings. Never truncate, change seed/rings or exceed 6,000 faces.

The selected component has **24,396 faces / 35,127 certified edges**, among
9,431 eligible-face components. Its minimum original face is 289,566. Median
rank 17,563 selected original edge-cache row **1,889,720**, endpoints
**612,841–612,852**. Its original two faces are 1,270,262 and 1,270,266.
These are source identifiers, not collapse ancestry or altered geometry.

Complete seed stars contain 10 faces. Rings add 8, 10 and 16, producing
**44 original triangles / 34 vertices**, one original vertex component
(label 1), one raw edge-connected component and one oriented sheet. Positions,
oriented faces, multiplicity, source IDs and accepted native RGBA rows retain
exact original correspondence. No colours were fitted or field values queried.

| Eligibility stage | Original edges |
|---|---:|
| Globally certified edges touching the crop | 77 |
| Both original incident faces present | 55 |
| Complete original endpoint stars represented | 23 |
| Survive actual crop and inherited guards | **0** |
| Strict eligible / disjoint initial opportunities | **0 / 0** |

There are **no original boundary or non-manifold edges in this patch**, no
winding conflict, aliases or multiple local fans. All inherited global
protected-face/vertex/lock masks are zero. The crop creates 22 artificial
boundary edges. The third ring's 16 faces become immutable, involving 29
protected vertices; the required one-ring guard locks **all 34 vertices**.
The genuine pinned local classifier and separately merged/global-star checks
agree. Equality between the selected unique incident faces and the verified
global face-degree counts certifies complete candidate endpoint stars; no
cropped-only link was represented as a full-global link.

The 16 protected faces leave 28 faces in a **loose face-count-only ceiling**.
That 28/44 (63.64%) bookkeeping bound is not admissible reduction: with every
vertex locked, the allowed initial reduction and fixed-lock collapse capacity
are **zero (0% of the patch)**. No collapse operation occurred. No private
diagram was produced because the frozen diagram gate requires a qualified
patch. This single failed crop does **not** establish that other regions, or
a context-complete crop, are unreducible.

Before the real patch, 24 focused tests passed, including the complete
qualification handoff with the genuine pinned local classifier on a synthetic
grid and rejection under inherited global locks. The independent
[patch source review](amd-brick-census-recovery-evidence/independent-patch-review.json)
bound all ten sources, plan, log and qualified readback receipts. The
[independent patch outcome](amd-brick-census-recovery-evidence/independent-patch-outcome.json)
records the actual negative result and scope. There was no second choice.

## Measured execution and preservation

| Owned CPU phase | Worker / host seconds | Peak process / Job committed bytes | Minimum observed Windows available RAM |
|---|---:|---:|---:|
| Corrected readback | 9.406 / 12.391 | 201,289,728 / 202,129,408 | 18,509,889,536 |
| Patch qualification | 3.250 / 5.984 | 309,858,304 / 310,693,888 | 19,359,543,296 |

Both preserved the **6 GiB process AND Job caps, 180-second deadline, 12 GiB
live RAM floor and 2 GiB destination disk floor**. All observations admitted;
minimum destination free space across both phases was 110,649,819,136 bytes.
Both exited naturally with empty owned Jobs, unchanged watched sources and
no forced cleanup. The patch host correctly reports BLOCKED because its
worker's admission verdict is BLOCKED; this was not a timeout or containment
failure. Projection figures in the private receipt are explicitly projections;
the table gives actual committed-memory peaks.

Private root:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-census-recovery-523ac20`.

* `verify/result.json` SHA-256
  `cd11a416d1d3a54187530e9990b838373af1d73bae2cc5ee2d375c7e6e7ba221`
* `patch/result.json` SHA-256
  `f1a3dbe764f73757153edacbe2e782c99426870798d2b9f79c2985b81a8a7865`
* `patch/selection.json` and private NPYs retain all source IDs/context and
  exact geometry/colour readback; no mesh or texture asset was exported.

[BEFORE](amd-brick-census-recovery-evidence/preservation-before.json) and
[AFTER](amd-brick-census-recovery-evidence/preservation-after.json) retain all
100 protected files and the frozen 61 historical trees / 1,665 files / 28
reports, including all 65 previous census files. Original/rejected models,
GLBs, colours, scene, selected environments/source backups, caches and
quarantine match. The AFTER audit performed 2,165 fresh full hashes over
3,647,963,836 bytes, with zero issues. Three checkpoint headers/stats were
checked, with zero checkpoint payload reads. All 108 prior remote heads matched
before publication; only the new experimental branch is to be added.

**Next step:** separately approve one guard-aware context qualification around
this same frozen seed, within 6,000 faces and all unchanged protection rules.
Determine enough original context to leave an interior before attempting any
reduction. Do not simplify, search alternate seeds, rerun the census or expand
this crop automatically.
