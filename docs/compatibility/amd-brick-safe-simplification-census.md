# Original full-mesh safe-simplification census — BLOCKED

Base `e262ad7c37f108e1cea456775b9b8c4b11cc9ed8`; isolated branch
`codex/amd-brick-safe-simplification-census`. One original-only CPU census
completed. The separate independent readback stopped on a verifier metadata
key mismatch, so **no patch is qualified**. Neither worker was retried. There
was no simplification, geometry change, renderer, WSL, GPU or model execution.

## Reported global measurements

These are the completed census's measurements, **not a completed independent
qualification of its global link population or distributions**.

| Stage / population | Count |
|---|---:|
| Original vertices / triangles | 1,957,891 / 4,173,584 |
| Referenced / unused original vertices | 1,952,142 / 5,749 |
| Original unique edges | 6,167,053 |
| Non-manifold / boundary edges | 387,068 / 267,652 |
| Vertices incident to non-manifold edges | 398,221 |
| Triangles touching those vertices | 1,775,350 |
| Original boundary vertices | 328,000 |
| Triangles touching boundary vertices / actual boundary edges | 1,308,882 / 250,841 |
| Oppositely wound / conflicting incidence-two edges | 4,789,797 / 722,536 |
| Original face-bearing vertex-connected components | 207 |
| Oriented edge-connected sheets | 278,013 |
| Vertices with multiple local fans / multiple sheets | 1,002,093 / 947,994 |
| Exact referenced position aliases | 0 |
| Immutable protected faces / their vertices | 2,979,405 / 1,493,828 |
| Remaining pre-guard interior triangles | 1,194,179 |
| One-ring locked vertices | 1,663,142 |
| Guard-affected / fully unlocked interior triangles | 721,547 / 472,632 |
| Free-endpoint, unprotected-face edges entering full link tests | 758,405 |
| Rejected common-neighbour links | 91 |
| Rejected shared link edges (tetrahedron case) | 0 |
| Census-reported full-link-certified initial edges | 758,314 |
| Triangles touched by those initial edges | 571,541 |

Junction, boundary, winding, local-fan, sheet and exact-position interfaces
freeze their incident faces. Their complete vertex neighbourhood is guarded.
A face with a guarded corner may still have an eligible opposite edge; the
721,547 guard-affected faces are **not all immutable**. Face, vertex and edge
counts have different denominators. The source-safe
[reported component/octant distributions](amd-brick-safe-census-evidence/reported-distributions.json)
retain all 207 original components and eight fixed AABB octants, with those
denominators labelled. Their independent recount was not reached.

Full link tests require exactly the two distinct opposite common neighbours
and no shared link edge. They do not merely test incidence two. Maximum free
degree was 10; 75,840,500 padded comparisons were performed in batches of at
most 8,192 edges. The initial edges overlap and cannot be added up as
simultaneous collapses or a guaranteed reduction. Geometric collapse error,
appearance and a usable patch remain unqualified. No self-intersection search
or claim of watertightness is made.

## Qualification failure and stopping point

Seventeen focused CPU checks passed, including equality with the pinned prior
local classifier, winding/fan/unused-row cases, tetrahedron rejection, cache
corruption, cancellation and isolated Python helper resolution. An independent
[GO source review](amd-brick-safe-census-evidence/independent-candidate-review.json)
bound the exact seven sources, plan and test log before the census. A premature
read-only preflight initially found that the review receipt was not yet present;
it reserved no marker or worker. The completed preflight passed before execution.

The independent readback then raised `KeyError: 'certified_edge_touch_faces'`
at `verify_worker.py:164`. That is a mask name; the producer's summary count
is named `certified_edge_face_population`. This is our verification-schema
defect, not a demonstrated mesh discrepancy. The failed source and receipt are
preserved unchanged. The focused fixtures did not cover this complete summary
handoff.

Before that failure, control flow passed all saved-file hashes/schemas, all
12,520,752 occurrence/permutation/endpoint-key/incidence/winding checks, degree
identity, protection/one-ring/stage boolean replay, the 758,405 pre-link
endpoint count and membership constraints on all 758,314 reported edges.
Vertex-summary comparisons, broader count arithmetic, sheet-population count,
independent full-link witnesses and component/octant recount were not reached.
The [independent outcome](amd-brick-safe-census-evidence/independent-outcome.json)
records this scope explicitly. A host execution PASS means containment and
natural settlement, **not** a successful verifier or mission PASS.

No selection, patch geometry or diagnostic image was generated. Patch tests
would not replace the incomplete global readback. There is no evidence here
that the whole original model is unreducible: the observed positive population
is a useful screening result, pending completion of qualification.

## Measured resources and preservation

| Owned CPU phase | Worker / host seconds | Peak process / Job committed bytes | Minimum observed Windows available RAM |
|---|---:|---:|---:|
| Census | 9.015 / 11.844 | 725,606,400 / 726,450,176 | 19,446,022,144 |
| Independent readback, incomplete | 7.500 / 10.140 | 201,101,312 / 201,945,088 | 19,540,250,624 |

Both retained the 6 GiB process **and** Job caps, 180-second limit, 12 GiB live
RAM and 2 GiB disk floors. All observed resource samples passed. Minimum
destination free space across both phases was 110,674,776,064 bytes. Both
workers exited naturally with empty owned Jobs, no forced cleanup and unchanged
watched sources. The 5 GiB preflight admission was a conservative projection;
the table records actual committed-memory peaks. No old topology cache was
regenerated or overwritten; the new private occurrence/fan indices were derived
only because the historical cache lacks face IDs and directions.

Private evidence root:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-safe-census-e262ad7`.

* `diagnose/result.json` SHA-256
  `69c2133a0ae6c04bd5aabc8c924a5587d13980cb2e5e0c81abd59eb946b64c4b`
* `verify/result.json` SHA-256
  `1e278c38d9fc9b4642f2eae7323ee1860991e1772410f0fb9f48fc2b2f770dd7`

The original mesh remains SHA-256
`ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd`.
[BEFORE](amd-brick-safe-census-evidence/preservation-before.json) and
[AFTER](amd-brick-safe-census-evidence/preservation-after.json) preserve the
100 protected files, original/rejected models, colours, saved scene, selected
environment/source-backup identities, caches and quarantine. The fixed scope
contains 59 historical trees, 1,600 files and 27 reports; all match. The AFTER
audit performed 2,099 fresh full hashes over 3,484,143,985 bytes with zero issues.
Three checkpoint headers/stats were checked; no checkpoint payload was read.
All 107 previous remote heads matched before publication. Only this new
experimental branch is to be published; models and images remain private.

**Next step:** separately approve the narrow summary-key correction and one
fresh CPU readback of the already-saved census. Only after it passes should
the single frozen topology-selected patch be admitted. Do not rerun the global
census, simplify the mesh or switch reconstruction methods on this evidence.
