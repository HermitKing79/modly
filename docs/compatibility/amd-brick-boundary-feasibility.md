# Local brick boundary feasibility

**BLOCKED — no suitably isolated opening has been established at A1 or secondary A3. No repair is authorised.** Base `677d741eba18baa667f726aa38ce2c4f52720a8d`; branch `codex/amd-brick-boundary-feasibility`. The original mesh, faces, neural geometry, PBR field, camera and previous diagnostic were authenticated. Original data remain unchanged. [Preflight](amd-brick-boundary-feasibility-evidence/preflight.json), [measured source IDs, memberships, normals and depth lists](amd-brick-boundary-feasibility-evidence/result.json).

## A1: an open chain meeting junctions

The exact center ray through PNG pixel **(534,372)** first hits original face **2369396**, vertices **[1143327,1143329,1154379]**. Its opposite-edge incidences are **[2,3,1]**. The incidence-1 edge belongs to complete boundary component **1014637**:

`1143327 — 1143329 — 1143348`

These are **two edges and three vertices**, with degrees **1,2,1**. The chain is genuinely open; it was traced globally rather than cropped into an apparent loop. Both endpoints also meet incidence-3 edges:

- **[1143327,1154379]**, witnessed by faces **2369391,2369396,2369398**.
- **[1143348,1154395]**, witnessed in the primary face-star data by face **2369402**; complete incidence is three.

The complete vertex star of the hit triangle has twelve original faces, retaining their original winding and normals. This is a real topological discontinuity beside non-manifold junctions. Closing the two-edge chain would require inventing a new rim segment and deciding how junction sheets connect; that is not an isolated hole fill.

Chain length is **0.00450053 world units**. Its XYZ extent is **0.00047511 × 0.00197017 × 0.00198746**, projecting to PNG bounds **[533.718,371.538]–[535.059,372.891]**. These are canonical model units, not physical metres. **Opening area is undefined** because there is no closed rim; no closing edge or patch was manufactured to assign one.

## Surrounding boundaries and secondary A3

The fixed A1 window **[522,360,546,384]** intersects 1,274 original boundary edges across **261 complete graph components**. Exact global screening finds **233 open/branching components**, one closed component touching a non-manifold junction, and **27 closed/no-junction cycles**, all fully traced. None of those 27 cycles' projected bounding boxes contains the A1 pixel center; the nearest box is **0.76955 px** away. This excludes them as complete projected rims surrounding that exact center, not as possible defects elsewhere in the crop.

A1 did not qualify, so A3 **(538,499)** was inspected second. Face **3356903**, vertices **[1603076,1603057,1603077]**, has incidences **[1,2,2]**. Its boundary component **1408151** is a **17-edge / 18-vertex open chain**, endpoints **1603033** and **1603136**, touching two more incidence-3 edges: **[1418566,1603033]** and **[1603136,1603337]**. Chain length is **0.02757496 world units**. Full ordered membership is recorded in `result.json`.

A3's window intersects 1,020 boundary edges and 198 components: **173 open/branching**, one closed/junction-touching and **24 closed/no-junction cycles**. All 24 were traced; none of their projected boxes contains the center (nearest **1.00005 px**). No eligible cycle was discarded by the trace budget. [Exact junction incidences](amd-brick-boundary-feasibility-evidence/nm-incidences.json).

## Depth, appearance and unresolved safety

All **114 deterministic rays** preserve every hit, sorted by depth and original face ID, without truncation or recorded tie/clip/projected-degeneracy ambiguity. There are six original face hits on each center ray. At A1, the nearest hit is at **X=0.090744**, with signed normal-to-camera **−0.82377** and learned encoded RGB **(.504,.275,.149)**. Three pixels left, the nearest hit is at **X=0.241231**. The original five-ray neighborhood spans **0.150486 world units** in depth. Thus actual sheets/depth transitions and brown material are present; a black learned texel alone does not explain the observation.

This does **not** distinguish an intentionally open brick joint, a missing nearer sheet, folded rear-facing geometry or shadowing conclusively. No area-light shadow reconstruction was performed. Non-manifold incidence is not itself proof of a triangle self-intersection.

The 51 traced no-junction cycles have measured best-fit-plane metrics, with **zero detected non-adjacent segment crossings in those projections**. All are nonplanar under the numerical planar-certificate tolerance. **Nonplanarity alone does not make a 3D loop irreparable.** Full 3D triangle collisions, continuous clearance, vertex-link manifoldness and brick-versus-foliage intent were not certified. A hypothetical patch could bridge sheets, change a joint or affect thin foliage; those risks remain unproven. Original vertex/material correspondence survives this investigation, but no new-face interpolation has been qualified.

## Private close-up and execution

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-boundary-feasibility-677d741\a1-boundary-closeup.png`

SHA-256 **`fad7a930407e4ec771000e59144f898375ba9810e3b6ba0a03dcbae427c4bb5b`**, **64,025 bytes**, **1160×820**. The left panel is an untouched 12× nearest-neighbour crop, with A1 at its center. The right panel enlarges the actual projected source edges and IDs; it is not a surface or lighting render. [Image metadata/readback](amd-brick-boundary-feasibility-evidence/image-verification.json).

The first CPU launch stopped at the 256-component trace limit, exited naturally, and is preserved with its exact source. A source-equivalent graph screen then screened out only components failing the necessary closed/no-junction graph tests before detailed tracing. The cap remained **256**, with every direct-hit component and every potentially admissible cycle retained. Sixteen focused CPU tests passed. [Recovery preflight](amd-brick-boundary-feasibility-evidence/recovery-preflight.json), [independent preparation review](amd-brick-boundary-feasibility-evidence/independent-pre-review-recovery.json).

The completed measurement took **6.125 s**, with **546,172,928 bytes** peak per-process commitment; minimum sampled Windows RAM was **20,404,535,296 bytes**. The original **6 GiB / 180 s**, **12 GiB RAM / 2 GiB disk** envelope remained enforced. Private image composition took **0.703 s / 32,296,960 bytes**, inside **1 GiB / 60 s**. Both Jobs exited zero and emptied naturally without forced termination. Process execution PASS is distinct from **boundary qualification BLOCKED**. [Execution](amd-brick-boundary-feasibility-evidence/host-result.json), [outcome review](amd-brick-boundary-feasibility-evidence/independent-outcome-review.json).

[Before](amd-brick-boundary-feasibility-evidence/preservation-before.json) / [after](amd-brick-boundary-feasibility-evidence/preservation-after.json) preserve the original assets, all 100 protected files, 400 scoped identities, recorded environment/source edits/backups, original caches and 20 historical evidence trees. All 88 pre-existing remote heads, including main, are verified at publication. Images and model assets remain private.

**No disposable-patch experiment is recommended for A1 or A3 on this evidence. Stop for visual review; do not modify geometry or materials.**
