# Final MeshLib rejection attribution

**Decision: CAUSE UNRESOLVED; park this MeshLib route unless new implementation or rejection-state evidence appears.** The saved constructor's lossless-transport **NO-GO** remains unchanged. This investigation passed independent saved-data verification; it did not qualify another native experiment or a repair.

Base: `b0d6fcf9adf92152c4c3784fa9e9abbbd782817f`. Branch: `codex/amd-meshlib-rejection-attribution`.

The original 457-vertex/781-face brick patch, source mappings, all 13 partial readbacks, overwritten skipped counter of four, failed regions and prior rejection-membership witness were authenticated. Installed MeshLib 3.1.4.297's 78-file pinned closure was unchanged. Exactly one read-only NumPy analysis and one separately authored verifier ran. No MeshLib import, constructor, simplifier, remesher, repair, renderer, model or GPU operation ran.

## Four original rejected faces

W1–W4 denote ascending original local FaceId. Exact local/global triangle and vertex IDs, oriented corners, positions, normals, every incident edge/vertex star, original/prefix/retained link graphs and native left/right/halfedge memberships are private in `artifacts/amd-meshlib-rejection-attribution-b0d6fcf/analysis/witnesses.json` (134,904 bytes; SHA-256 `d9945d0779cab5cb73d180f4a8c805780c1ec066fba17b4b5eb24ac33e14bf28`). No raw geometry or actual source IDs are published.

All four have three distinct vertices, finite positions and positive area. There is no demonstrated invalid-triangle explanation.

| Witness | Original boundary / incidence-2 / incidence-3 edges | Vertices with branching original links | Earlier same-direction edge competitors | Observed effect |
|---|---|---:|---:|---|
| W1 | 0 / 2 / 1 | 2 | 0 | One original three-face edge loses this face |
| W2 | 1 / 2 / 0 | 1 | 0 | No non-manifold edge on the face; removal still changes its vertex context |
| W3 | 0 / 2 / 1 | 2 | 0 | Its removal leaves the S2 same-direction pair described below |
| W4 | 2 / 0 / 1 | 2 | 1 | Its private third vertex belonged only to this face and becomes unreferenced |

The three incidence-3 witnesses each touch a different original non-manifold edge. Each such edge necessarily contains repeated directed use among its three faces. That association does not identify which rejection branch ran.

**W2 is not an isolated manifold counterexample.** Its full vertex stars contain 5, 3 and 5 faces. The first vertex touches a different incidence-3 edge outside W2; its original link is connected but branching, with maximum degree three. The other two links have maximum degree two. W2 itself has no co-directed shared-edge conflict. Its strict source-row prefix contains four vertex-incident faces but no edge-incident face. These facts defeat a blanket explanation based solely on a rejected face's own non-manifold edge or an earlier same-direction edge. They do not establish the native insertion state or a unique vertex rejection condition.

Original link graphs at all witness vertices are connected; none is being presented as a recovered semantic sheet partition. Source-row prefixes include earlier rejected faces and are explicitly **not** execution traces.

## The two split endpoint pairs

| Pair | Original / retained face incidences | Actual native groups | Demonstrated interpretation |
|---|---:|---|---|
| S1 | 3 / 3 | One opposite-directed two-face pair plus one singleton | All three source faces survive through two native edge IDs; incidence greater than two is not universally rejected |
| S2 | 3 / 2 | Two separate singleton edge IDs | W3 is omitted; the two retained uses have the same direction, so they are not the two opposite sides of one paired native edge |

The halfedge/left/right records establish those actual pairings and boundaries. They do not authenticate which original sheets should have been paired. The earlier independently verified global result remains: 777 retained faces, 456 valid referenced vertices, 457 unchanged allocated positions, two native face-adjacency components, and 151 native boundary edges. Retained source endpoint adjacency alone remains connected; it is not interchangeable with native edge pairing.

## What remains unknown

The [installed-declaration audit](amd-meshlib-rejection-attribution-evidence/source-constraints.json) documents possible state-dependent face admission. `VertInfo.duplicationNeeded` explicitly describes a single chain/ring, no triangles **or two open chains** as acceptable; its internal triangle-pair classification is not our descriptive link multigraph. `allowNonManifoldEdge=True` does not document unconditional preservation of arbitrary vertex-fan/orientation configurations.

Focused installed-package and workspace searches located no matching C++ constructor implementation or saved rejection trace. Exact internal rejection causality, insertion scheduling and intended sheet continuation therefore remain unproved. No minimal, non-destructive correction is supported. Random reorder, four-face omission, automatic fan duplication, welding or relaxed transport criteria would not resolve that proof gap.

**Recommendation:** park this constructor route. Reopen it only if version-matched implementation/rejection-state evidence identifies one specific falsifiable correction that preserves every original oriented triangle and interface. No further MeshLib execution is authorised by this report.

## Verification and resources

[Nineteen focused synthetic tests](amd-meshlib-rejection-attribution-evidence/focused-tests.txt) passed. They include complete producer/independent-checker handoffs, source/record corruption rejection, directed edges, disconnected/branching links, split endpoints, lost references, native reciprocity and exact typed Float64 comparisons. The independent worker verified all four witnesses, both split pairs, the lost vertex and all 28 context faces directly against the authenticated arrays; it imported no analysis functions.

| Owned CPU phase | Worker / host elapsed | Peak process / Job committed bytes | Minimum available Windows RAM |
|---|---|---|---:|
| Analysis | 1.203 / 5.234 s | 125,198,336 / 126,038,016 | 17,822,011,392 |
| Independent verification | 1.172 / 3.453 s | 126,578,688 / 127,418,368 | 18,544,951,296 |

Both retained the separate 6 GiB process and Job ceilings, 180-second hard/150-second cooperative deadline, 12 GiB available-RAM and 2 GiB destination-free floors, binary GO, monitoring and exclusive markers. Destination free space stayed at least 106,596,384,768 bytes. Both exited naturally with empty Jobs, no forced cleanup, healthy monitoring and unchanged watched sources. These are CPU process/Job committed-memory measurements, not GPU measurements.

[Execution summary](amd-meshlib-rejection-attribution-evidence/execution-summary.json), [analysis](amd-meshlib-rejection-attribution-evidence/analysis-result.json), [independent verification](amd-meshlib-rejection-attribution-evidence/verification-result.json), [analysis source review](amd-meshlib-rejection-attribution-evidence/independent-analysis-review.json) and [verifier peer review](amd-meshlib-rejection-attribution-evidence/independent-verify-review.json) bind exact sources, inputs and private receipts.

The [preservation AFTER](amd-meshlib-rejection-attribution-evidence/preservation-after.json) passed with no issues: protected100, original400, SDK78, partial13, original source lineage, environments, backups, caches/quarantine, 71 historical trees/2,037 files/33 reports and all 113 pre-existing remote heads unchanged. Checkpoint checks read bounded headers only; no weight loading occurred. Private evidence remains local. Publication adds only this report and source-safe evidence on the isolated branch; final Git publication is verified separately.
