# Zombie chest-wound source sufficiency audit

Base: `9d76ef59cadc10160e0a468d930a81286ce73e4e`. Branch: `codex/amd-zombie-wound-source-audit`. This is a bounded CPU-only, read-only investigation. No replacement geometry, reconstruction, repair, remesh, winding edit, neural/WSL work, UVs, baking or textures were produced.

## Decision

**NO-GO for reliable automated multi-sheet reconstruction of the complete chest wound from the current source/evidence.** The mesh contains exterior torso triangles, ragged opening edges and real visible internal strips/flaps. It does not establish a reliably separated set of exterior, rim and cavity-wall surfaces. Open gaps often reveal more distant torso geometry; those hits must not be treated as a local cavity floor or used to invent missing walls.

Recommend improving upstream wound geometric coherence and explicit cavity-wall/rim definition before another automated reconstruction attempt. This is a recommendation, not authorization to generate again. It is not a proof that an artist could not interpret or rebuild the wound, nor a mathematical impossibility claim about every separation method. No reconstruction or upstream generation follow-up starts here.

## Important correction to the inherited depth evidence

The prior experiment's **988 “deep-wound” samples are depth witnesses inside a rectangular ROI, not a semantic cavity mask**. On a matched frontal reference, a visually traced wound outline contains only **51** of those sample centers; **937 lie outside**, mostly on curved ordinary torso beside/below the wound. The reference's constant skin-Y threshold mistakes this curvature for cavity depth.

The original prior report already described these as geometric witnesses, but its wound terminology and 554/988 retention number must not be used as measured cavity-wall coverage. Its visual NO-GO for stepped/missing wound detail remains supported by the saved views. This investigation corrects the interpretation without rewriting historical reports or evidence.

A preliminary depth-only analysis here initially suggested a 107-face skin-to-deep path. Visual validation found **zero path face centroids inside the visible wound outline**. It crosses ordinary torso below the wound and is rejected. Its raw receipts, images and unauthorised preliminary proposal remain unchanged for traceability. Their semantic/GO interpretation is explicitly superseded by `semantic-validation.json`; they are not execution instructions. Published code labels such intermediate results preliminary and requires visual validation.

## Source authentication and immutable data

Input is the approved orphan-cleaned high-detail PLY, **325,807 vertices / 673,604 triangles**, SHA-256 `377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5`. The experimental winding variant is never used.

Original high-detail PLY remains `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c`. Neural tensor output remains `e589fec9f5ea58ec7bdb07c00c3c40911b6191d08055611fac17ddee65aa7918`, with the existing ten-tensor provenance validation and original contract `13a140490d933d0c9f21af701af36caf54384b0fe0e5dc9c4714d0328dea6343`. No checkpoint weights are opened.

The 189-entry seal from the prior retopology investigation authenticates original crop arrays, mappings, views and measurement receipts. Its inventory hash is `396c604b378389ff11a3aabb00d475af252cf6dbabab2be1853265067a216768`. All baseline PLY/NPY bytes and every vertex/corner payload remain unchanged, including through the six new full-source display renders. Only materials are assigned on an in-memory display copy. No mesh is exported.

## Layered, winding-independent tracing

Reuse the original −35°, 0° and +35° cameras and 128×128 ray grids. Full-source two-sided BVH tracing covers **49,152 rays**, at most 16 events per ray, with continuation epsilon **9.5367431640625e−7** model units. No ray reaches the cap with unrecorded further hits. Maximum observed event count is 11. Source normal sign is never used to choose the exterior or order layers.

BVH continuation cannot resolve coincident/sub-epsilon layers. To audit that limit, all 988 legacy deep rays are independently intersected in F64 against every full-source triangle whose projected XZ box could meet the ray. Shared-depth hits are grouped; one shared-depth tie event is retained as ambiguity rather than counted as another sheet.

| Legacy depth witness check | Result |
|---|---:|
| Saved source triangle intersection recovered | 988/988 |
| Full-source first-hit depth equivalent | 988/988 |
| Nearer full-source occluder | 0 |
| Rays with multiple separated depth groups | 979/988 |
| BVH/F64 grouped-depth disagreements | 0 |

Grouped depth counts are 9 rays with one group, 916 with two, 20 with three, 42 with four and one with seven. In 99 rays, more than one event lies within the old front-crop Y limit; many other second events are much farther back. **Ray events are not automatically physical or semantic sheets.** These checks authenticate the measured source geometry; because the legacy points originate from that source, triangle recovery alone is not independent proof of a reconstructible cavity.

## Visual constraint and what geometry is present

The final inspection uses a manually traced outer wound outline on the authenticated 512×512 full-source frontal view. Its coordinates, clear exterior seed rectangles and probable internal-surface polygons are stored exactly with the reference-image hash. The trace follows the visible wound, not the inherited rectangular depth ROI. It is an inspection aid, not a precise anatomical segmentation or a closed 3D rim.

Clear exterior seeds must be seen from at least two matched views and must avoid faces incident to >2-face edges. Probable rim candidates require visible genuine source opening incidence near the traced outline. Probable internal candidates require frontal and another-view support inside the visually identified wound strips; they may still be folded rim/skin flaps rather than distinct cavity walls. Branch-ambiguous, unseen and unsupported faces stay uncertain. No label propagates across the source to force a complete partition.

The color-coded inspection marks blue exterior candidates, orange probable rim, green probable internal/cavity-surface candidates and purple uncertainty; gray is outside the audited source crop. The final sampled candidate masks contain:

| Source-front-crop role | Triangles |
|---|---:|
| Supported exterior seeds | 329 |
| Probable rim candidates | 36 |
| Probable internal/cavity-surface candidates | 173 |
| Uncertain / unsupported ownership | 18,619 |

The large uncertain count reflects conservative, sparse visual support and unresolved ownership; it is **not a count of missing triangles**. There is no complete reliable rim or cavity mask. The central strip and lower lip/flap are visibly represented by triangles, but calling them anatomical cavity walls rather than folded exterior material is unsupported. Ragged openings coexist with many incidental small holes/non-manifold structures. A wound-looking silhouette alone cannot separate them.

Within the visually traced wound, the frontal grid has **1,884 samples**:

| First full-source support | Samples |
|---|---:|
| Source within the old front-crop Y bound | 977 |
| Farther source beyond that bound | 896 |
| No full-source hit | 11 |

The Y bound .055 is the inherited crop limit, not an anatomical cavity classifier. Farther hits do not prove a local wound wall/floor. No-hit samples do not prove an accidental defect: they may be intended through-openings or source incompleteness. Their intent remains uncertain; no caps or floor are invented.

Of the legacy ROI's 626 crop-void samples, 554 hit full-source geometry beyond that front crop and 72 still have no full-source hit. This explains why a cropped black area is not necessarily an absent whole-character surface. It also explains why simply collecting all deeper hits could incorrectly add a distant torso sheet to the wound reconstruction.

## Connectivity and separability

The prior front crop has 19,157 triangles in three crop components. Its boundary attribution remains **420 artificial crop cuts / 942 genuine source-unpaired edges**; “genuine” does not mean intentionally part of the wound. The crop includes a previously identified detached component; observed candidate skin/rim/internal labels in the preliminary source connectivity data belong to the main global component, not proof of an anatomically separate cavity.

For the legacy depth witnesses, 981/988 have a local two-face path to an exterior depth-proxy seed; 958 retain a path when all faces incident to non-manifold edges are excluded. Those are real connectivity facts, but the rejected below-wound path shows that they do not prove skin-to-wound-wall ownership. Label-subset disconnected groups must not be confused with actual global detached fragments. Connectivity, depth order and visible anatomy must agree before they can define a source sheet.

The investigation therefore finds **partial represented features, but inadequate evidence of reliable complete wound separation**. It does not conclude that every visible feature is absent or that the entire zombie is unusable. It specifically withholds a multi-sheet reconstruction proposal that would require guessing wall ownership, interpreting flaps as cavity surfaces or capping open regions. Improving upstream geometric coherence and explicit wound-wall definition is the supported next direction, outside this task's authorization.

## Evidence, safeguards and verification

Private output root: `artifacts/amd-zombie-wound-source-audit-20261025-v1`, outside the published repository. Authoritative final files are `semantic-validation.json` and `semantic-inspection/validated-source-role-inspection.png`, `validated-wound-support.png`, and `visual-landmark-constraints.png`. Layer arrays, F64 per-ray hits, raw preliminary data and rejected interpretations are retained. The initial `inspection-sheets/` and depth-only proposal are preliminary and superseded, not final cavity evidence.

The existing native CPU owner retains 6 GiB process commit, 180 seconds/stage, 12 GiB live host RAM, 2 GiB output disk, GO/EOF, exact Job attachment and natural owned cleanup. Existing 22 GiB neural gates, preferred 24 GiB headroom and all other protections are untouched. Every stage is CPU/read-only; no voxel/chart/winding experiment is rerun. New tests cover two-sided layers, depth ties, uncertain roles, the visual-ROI counterexample, no geometry-write/repair calls and resource/import gating. Focused tests and independent review/preservation/publication receipts accompany this report.
