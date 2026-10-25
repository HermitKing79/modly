# Zombie geometry diagnosis: unchanged topology and normals comparisons

Starting revision: `86d4cf52670c4c3962c82874f6c32805baaf4617`. The original PLY is unchanged at SHA-256 `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c`, 326,232 vertices and 673,604 triangles. Original PLY/NPY payloads, all ten neural tensors, run contract, input lineage and source image were authenticated before analysis. No extraction, repair, neural execution, WSL command, UV work or texture generation occurred.

**Decision: GO for one future copy-only orphan-vertex cleanup experiment.** Remove only the 425 exact unreferenced vertex IDs, remapping retained indices without changing any triangle coordinates. This is a tightly controlled first cleanup, with zero surface changes expected. The experiment has not been performed. Automatic hole filling, blanket fragment deletion, crown trimming, global manifold conversion and production smooth shading are **NO-GO** on current evidence.

## Independent topology measurements

An independent structured-pair sort counted all triangle edges. Counts agree with the retained extraction analysis:

| Incident faces | Unique edges |
| --- | ---: |
| 1: boundary | 19,512 |
| 2 | 950,250 |
| 3 | 24,996 |
| 4 | 6,453 |
| More than 2, total | 31,449 |

There are 4,437 connected boundary-edge networks; 883 are simple connected degree-two cycles. Other networks include chains and branches, so treating the entire boundary set as fillable hole loops would be incorrect. Non-manifold edges form 3,710 networks. **147,484 two-face edges traverse in the same direction in both incident faces**, indicating inconsistent winding under the original fixed index order. This winding count excludes edges with more than two faces. No self-intersection or vertex-link manifold classification was performed.

All 50,961 defect edges are located in the local `defect-edges.csv`: original endpoint IDs, incidence, surface component, edge-network ID, midpoint and length. `topology.npz` retains the exact edge arrays, component labels, incident non-manifold face IDs and all 425 unreferenced IDs. `edge-networks.json` gives connected-network bounds, centroids, lengths and degrees. Coordinates/lengths use the original normalized D/H/W space; no real-world body height or millimetre scale is assumed.

The following mutually exclusive geometric viewing bins locate the distribution. They are documented axis-aligned bins, **not semantic segmentation**:

| Viewing region | Boundary edges | Non-manifold edges |
| --- | ---: | ---: |
| Crown/scalp, z ≥ .43 | 2,257 | 5,145 |
| Head/neck, remaining z ≥ .30 | 3,062 | 2,613 |
| Hands, remaining abs(x) ≥ .40 | 925 | 761 |
| Arms, remaining abs(x) ≥ .145 | 1,664 | 1,581 |
| Chest/shoulders, remaining z ≥ .16 | 2,322 | 2,012 |
| Abdomen, remaining z ≥ −.035 | 2,839 | 6,857 |
| Shorts/hips, remaining z ≥ −.19 | 3,389 | 9,673 |
| Legs, remaining z ≥ −.45 | 1,770 | 1,245 |
| Feet, remaining edge midpoints below −.45 | 1,284 | 1,562 |

## Wounds, clothing and uncertain intent

The source image and matched render details support preserving the chest and abdominal gashes, torn thigh/knee areas, jagged shorts hem, open mouth and hair silhouette as distinctive character features. Their appearance is likely intentional. This does **not** establish that every local boundary, disconnected sheet or excessive-incidence junction is intended. A wound can have a manifold cavity lining; an opening label alone cannot justify leaving or closing a particular edge.

Four candidate inspection boxes are explicitly marked `?` in the local front overlay. Their bounds and exact counts remain in the analysis JSON. These boxes may overlap; their counts must not be summed as a partition:

| Candidate region | Boundary | Non-manifold | Interpretation |
| --- | ---: | ---: | --- |
| Front chest wound | 790 | 645 | Preserve visible wound shape; individual edge intent uncertain |
| Front abdominal wound | 1,021 | 1,156 | Preserve cavity and ragged lip; disconnected internal patches uncertain |
| Clothing hem | 1,897 | 3,205 | Preserve ragged outline; dense internal junctions require sheet inspection |
| Crown projection/scalp | 966 | 3,273 | Connected protrusion plus nearby topology defects; no automatic trimming |

Defects are widespread across the head, arms, hands, feet and ordinary body surfaces, not confined to the obvious wound openings. Edges incident to three/four faces are structural non-manifold junctions regardless of their anatomical location. Filling boundary loops would not fix these junctions.

## Crown, fragments, hands and hair

The visible crown stub persists in the improved views and belongs to **C0, the main connected surface**. None of C1–C9 reaches z=.43. It is not one of the detached fragments or a vertex-only orphan. The source has irregular hair, but does not establish that this particular stub is intentional; classify its artistic intent as **uncertain** and protect it pending a separate silhouette decision.

The nine disconnected surface fragments contain 266 triangles in total. All hands, limbs and the crown remain in C0. IDs are stable within this diagnostic, sorted by triangle count with a deterministic tie break:

| ID | Triangles | Centroid x,y,z | Minimum vertex-to-main-vertex gap | Location interpretation |
| --- | ---: | --- | ---: | --- |
| C1 | 212 | .0410, −.0192, .2441 | .007155 | Upper chest/base-of-neck region; possible internal tissue, uncertain |
| C2 | 12 | −.0200, −.0506, .0763 | .003058 | Front abdominal wound region, uncertain |
| C3 | 12 | .0478, .0471, .4114 | .002934 | Posterior head/hair region, uncertain |
| C4 | 10 | −.0132, −.0505, .0772 | .001856 | Front abdominal wound region, uncertain |
| C5 | 8 | .0402, .0469, .4063 | .001946 | Posterior head/hair region, uncertain |
| C6 | 6 | −.0421, .0447, .3821 | .001758 | Posterior head/hair region, uncertain |
| C7 | 2 | −.0492, −.0472, .0783 | .005473 | Front abdominal wound region, uncertain |
| C8 | 2 | −.0460, .0393, .4027 | .003405 | Posterior head/hair region, uncertain |
| C9 | 2 | .0408, .0479, .4125 | .002201 | Posterior head/hair region, uncertain |

These are vertex-to-vertex distances, **not nearest-surface distances or proofs of non-intersection**. A topologically disconnected fragment can overlap or lie within the main surface. Largest C1 has 40 boundary and 25 non-manifold edges; C3 has one non-manifold edge. Blanket deletion of the small components is not justified by size alone.

Hand closeups show retained finger/thumbnail-like channels and a thumb, but the fingers remain joined/simplified and their tips coarse. Smooth normals do not recover missing finger separation. Hair tips, wound lips and clothing contours stay ragged in both presentations. These are geometric limitations rather than flat-shading artifacts. Some fine black speckle is low-sample noise or shading, so individual dark pixels are not classified as holes without a matching measured cyan edge.

## Flat versus smooth presentation

Sixteen matched full-body renders use the original eight camera positions/scales: flat versus smooth vertex-normal shading, identical neutral material and lights. AgX with exposure −.75 and soft 180/140/160 W area lights reduces the previous clipped highlights. Cycles CPU, four threads, no denoising or mesh modifiers were used. Original coordinate/index payloads were verified bit-exact before/after each rendering batch; the PLY hash, size and timestamp stayed unchanged.

Smooth shading softens some planar transitions but introduces strong banding/striping across the torso, scalp, hair and limbs. This is consistent with ambiguous normal averaging on the measured winding-conflicted/non-manifold surface; the relative contribution of each cause was not experimentally isolated. It is not a repaired or production-ready normals solution. Wound openings, crown stub, coarse hand silhouettes, rough hair/clothing edges and all topology counts remain unchanged at the same eight angles.

Colour overlays use **cyan for boundary edges** and **magenta for more-than-two-face edges**, as separate CPU-rendered curve objects. Their display radius is .00035 normalized units; they are annotations, not added asset geometry. Components have distinct diagnostic colours. Isolated-fragment views hide the main component for visibility using a separate subset display object containing only existing source triangles. The original source object remains intact and payload-verified. X-ray leader-line markers and blue orphan-vertex projections are explicitly labelled unoccluded annotations; they do not imply that every marked fragment is visible from the front.

Local sheets include matched eight-angle comparisons, eight topology views, crown/hair details, wound/hand details, clothing, fragment locations, unreferenced vertex locations and uncertain-region boxes. Images and diagnostic geometry arrays stay local.

## Converter attribution

A separate dictionary-based enumeration read the saved learned edge flags directly, without calling the converter or importing its Torch/GPU module. It recovered exactly **336,802 valid source quads**, in the same order as the saved triangle pairs, with **22 missing-corner flags**. The pre-triangulation quad network independently reproduces the exact boundary-edge ID set and the exact 31,449 non-manifold edge ID set. Every one of the **336,802 inserted diagonals** has incidence two.

Therefore diagonal selection, CPU key sorting and the choice of split are not the origin of these excess-incidence edges in this saved mesh. The observed over-connected network is already specified by the learned flags plus the qualified global quad-assembly policy. This does not establish that the neural model alone caused it. The converter faithfully preserves that network and supplies no manifold-sheet separation or globally consistent winding repair; its policy consequently carries the defects into the mesh.

The installed reference was read only, never imported, and matched its qualified SHA-256 `2538315098531d5306e2cb5bf84942335e114c23fd1192dbd1920800c1c9b7a2`. Its offsets and inference split-weight rule agree with the unchanged CPU translation. No converter implementation, thresholds, flags or decoder parameters changed.

## Smallest next experiments and later asset strategy

1. **GO: one copy-only orphan cleanup control.** Select only the 425 saved unreferenced IDs. Expect 325,807 retained vertices and 673,604 triangles. Require bit-exact per-triangle coordinates/order, unchanged bounds, wounds and all eight silhouettes, unchanged ten surface components, and unchanged 19,512/31,449 edge counts. Keep the original and the variant separately with source hash and an index-remapping receipt. This cleans the container; it does not solve the main surface topology.
2. **NO-GO: automatic hole filling, global welding or keep-largest-component deletion.** Wounds and ragged clothing contain uncertain intentional openings; fragments can be internal character detail. The crown is attached and must not be treated as a detached island.
3. For the first structural surface experiment, manually identify a small ordinary-surface patch away from wounds, face, fingers and silhouette. Investigate incident face fans and consistent winding there on a copy. A source-quad-informed sheet split can duplicate indices at identical coordinates instead of moving/deleting triangles, but which faces belong together requires local evidence. Do not propagate a guessed fan pairing across all 31,449 junctions. Reassess normals after sheet/winding decisions, without using global smooth shading to conceal unresolved structure.
4. Later retopology should retain this genuine mesh as the immutable detail reference, with protected masks around wounds, mouth/eyes, fingers, ears, hair tips, hem and silhouette. Prioritise coherent sheets and useful edge flow through deformation regions; preserve cavity lips and decide their intended lining explicitly. Work in local reviewed patches rather than replacing the full character speculatively.
5. Derive LOD budgets from measured target-camera distances, silhouette/feature screen error, shading stability and runtime cost. Preserve distinctive wound openings and the T-pose silhouette at each level. No arbitrary polygon target is set. UV layout, baking and textures remain separately authorised later work.

## Bounds and preservation

Every stage retained the original CPU precedent: 6 GiB process commitment cap, 180-second stage deadline, 12 GiB live available host RAM and 2 GiB output-volume disk floor, owned GO/EOF gating and bounded cleanup. The neural 22 GiB admission/final guard and preferred 24 GiB headroom were untouched; no neural operation was admitted.

| Stage | Seconds | Peak committed bytes |
| --- | ---: | ---: |
| Independent topology/location analysis | 6.110 | 179,687,424 |
| Direct learned-quad attribution | 3.438 | 197,873,664 |
| Matched flat/smooth renders | 19.891 | 743,030,784 |
| Topology/components renders | 15.563 | 1,283,543,040 |
| Feature closeups | 36.000 | 1,358,467,072 |
| Labelled sheets | 1.860 | 42,336,256 |

All six workers exited 0; Jobs were empty with verified natural settlement and no cleanup errors. The lowest sampled available RAM was 21,680,242,688 bytes, above the retained CPU floor. Nine focused tests cover independent incidence, source quads, component labels, spatial bins, low-resource rejection and the renderer pre-import gate. Original numerical/converter/owner sources and original assets were preserved; the private asset inventory covers 89 files in addition to the inherited 100 protected identities/78 bounded hashes and 88 retained records.

Independent review found no Critical or Important issues. It independently checked all 46 image hashes, original PLY hash, matched cameras, six owned-stage receipts, preservation inventories and the proposed orphan cleanup scope. Its single wording correction (bins classify edge midpoints) is incorporated. Final verification also checked all 50,961 CSV rows, ten inspection sheets, 256 original source hashes and 156 prior local refs. See [source-safe verification](amd-zombie-topology-diagnosis-verification.json).

Private evidence root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-topology-diagnosis-20261025-v1`. All original meshes, tensors, images and environments remain in place. Only safe code and textual reports are published.
