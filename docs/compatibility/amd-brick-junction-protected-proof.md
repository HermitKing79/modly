# Junction-protected original patch: BLOCKED before simplification

Base `4a033425d55947e080eee27df2757139ba7bec31`; isolated branch
`codex/amd-brick-junction-protected-proof`.

**The frozen patch contains244 original triangles and115 vertices. Every face
requires protection; zero eligible collapse edges remain. No simplification
was called, and no candidate was produced.** This is an admission BLOCKED,
not a failed simplifier or a NO-GO verdict for all parts of the model.

## Selection and measured protection

Only the accepted original mesh/colour rows were loaded as patch data. The
rejected50% candidate was authenticated for preservation, never used as input.
The complete original endpoint stars of edge1456609–1456613 contain30 faces.
Exactly three original edge-adjacent rings add48,65,101 whole faces, totaling244;
the last ring is the frozen crop buffer. Original IDs, F32 positions, oriented
corners, multiplicity and native colour association are retained exactly.
The four original edge faces are3033048,3033067,3033078,3033080.

Independent readback confirms **all244 faces already touch at least one of
the109 globally non-manifold vertices**, before any boundary, sheet/fan,
outer-ring or guard expansion. Complete junction-star preservation alone
therefore protects the entire patch; the guard ring is not the cause of the
zero eligible count. Endpoint incidence degrees15 and19, with four common
faces, independently certify the complete30-face seed union. Cached global
incidence also certifies completeness of all three saved expansion rings.

| Measured original-patch property | Count |
|---|---:|
| Original triangles / vertices | 244 /115 |
| Vertices incident to global non-manifold edges | 109 |
| Patch edges globally incidence>2 | 138 |
| Genuine global boundary edges in patch | 5 |
| Edges missing some global incidents because of the crop | 134 |
| Complete opposite-wound incidence-two interior edges | 128 |
| Inconsistent complete incidence-two edges | 1 |
| Consistently oriented edge-connected diagnostic sheet pieces | 116 |
| Vertices with multiple local face fans | 74 |
| Equal-position vertex aliases | 0 |
| Protected faces / vertices | **244 /115** |
| Guard-locked vertices / free vertices | **115 /0** |
| Strict eligible edges / mutable eligible faces | **0 /0** |

Crop incidences are not physical holes in the master. Sheet pieces and local
fans describe the cut incidence graph, not116 semantic object parts; local
fan fragmentation can also result from the crop. All ambiguous interfaces
are conservatively protected. Global cached incidence prevents crop-hidden
junctions from being misclassified as ordinary manifold interiors.

## Enforceable API, insufficient mutable geometry

The already-installed meshoptimizer0.22 experimental `simplifyWithAttributes`
has a real Uint8Array vertex-lock argument. Its installed source fixture also
shows why locks alone are insufficient: an unlocked neighbour can collapse
into a locked vertex and change incident connectivity. Thus the proposed
route protects every required face's vertices, their complete one-ring
neighbours and native equal-position classes. No experimental API or WASM
runtime was invoked. Stable `LockBorder` was not mistaken for arbitrary locks.

The census protects complete local junction stars, original boundaries,
crop/interface triangles, orientation conflicts, multiple sheets/local fans,
position interfaces and the outer ring. Eligibility additionally requires
two opposite-wound global/local faces, a common identified sheet, free endpoints
and the full simplicial link condition. Independent source review caught and
resolved both the tetrahedron shared-link-edge case and local fans that reconnect
elsewhere in one global sheet. Eight focused synthetic tests passed.

Here the protected-face floor already equals the entire244-face input. The
requested20% eligible-interior reduction has an empty denominator; it cannot
be counted as a vacuous PASS. No target/error search, alternate patch, Blender
fallback, source-face removal, repair, colour fitting or protective-limit
relaxation followed. Geometry/material/silhouette tests for a candidate were
not reached and are not claimed to pass.

## Execution and preservation

The owned read-only census took **3.015s**; supervisor and post-checks
**5.297s**. Measured peak committed process/Job memory was
**166,780,928 /167,620,608bytes** (about159.1/159.9MiB). Minimum observed
available Windows RAM21,224,292,352bytes; destination free111,629,754,368bytes.
Unchanged6GiB process AND Job,180s,12GiB RAM and2GiB disk gates passed.
The Job emptied naturally; no forced cleanup, GPU/model execution, neural
work, simplification, render, bake, repair, package change or export occurred.

Private exact selection, source IDs, original colours and admission masks:
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-junction-protected-4a03342/diagnose/`.
These are original-patch diagnostic arrays, not a reduced asset.

Scoped audits preserve protected100,400 original identities,57 historical
trees/1,554files/26reports, original and rejected assets, selected environments,
source backups and caches/quarantine. No checkpoint weight payload was read.
All106 prior remote heads remain unchanged; final publication verifies the
new diagnostic branch separately. Models/images remain private.

[Frozen feasibility plan](amd-brick-junction-protected-evidence/diagnostic-plan.md) ·
[API/source review](amd-brick-junction-protected-evidence/api-review.json) ·
[Counts and private artifact hashes](amd-brick-junction-protected-evidence/outcome.json) ·
[Independent outcome](amd-brick-junction-protected-evidence/independent-outcome.json) ·
[Preservation](amd-brick-junction-protected-evidence/preservation-after.json).

**Next:** park collapse trials at this fully protected junction. Before another
reduction, separately qualify a genuinely manifold part with nonempty free
interior. This result does not establish feasibility or impossibility for the
whole model. No further patch search or full-model reduction is started.
