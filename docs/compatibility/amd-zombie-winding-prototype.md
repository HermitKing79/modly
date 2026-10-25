# Zombie two-face winding prototype and regional feasibility

Base: `6dd749040f1c290633546af285884cc08e176b56`. CPU only; no neural, WSL, UV, baking, textures, smoothing, hole filling, welding, fragment deletion or pipeline integration.

## Decision

**The authorized two-face experiment passes its exact structural acceptance. NO-GO for a larger winding-only repair under the tested fixed-exterior conditions.** Coherence improves, but the two visible faces become backfacing. A regional orientation objective can improve counts while worsening the current rear-facing presentation. Preserve the high-detail original as the immutable detail reference and retain this variant as an experiment, not a replacement production mesh.

The next meaningful direction is sheet-aware topology/retopology planning, with explicit outward-side and feature preservation, rather than a sequence of small winding flips. Do not automatically split sheets, propagate flips, fill openings, delete fragments or rebuild the mesh. This task performs none of those operations. Winding correction can remain a useful subordinate tool once surface sheets and their intended sides are known; these measurements do not establish a globally outward orientation.

## Exact prototype acceptance

The immutable orphan baseline PLY SHA-256 is `377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5`. Only zero-based face rows **334008 and 334009** use corner permutation **[0,2,1]** in a new private variant.

Variant PLY SHA-256: `e3e954f9cac60595759ab06333cf033f8149a86215e5d1318dc3080d66d4d489` (12,666,764 bytes). Independent PLY readback agrees with saved NPY arrays.

| Measure | Before | After |
|---|---:|---:|
| Vertices | 325,807 | 325,807 |
| Triangles | 673,604 | 673,604 |
| Two-face winding conflicts | 147,484 | 147,480 |
| Boundary edges | 19,512 | 19,512 |
| Edges incident to more than two faces | 31,449 | 31,449 |
| Surface connected components | 10 | 10 |
| New winding conflicts | — | 0 |
| Changed RGB pixels across eight matched silhouette pairs | — | 0 |

Every vertex payload, including signed zeros, is bit-identical. Every face outside the two declared rows is bit-identical. Per-face triangle coordinate multisets, face membership/order, bounds, complete component labels and every edge identity/incidence are unchanged. Only the four expected perimeter-edge signed sums change. Removed conflict edge IDs: 495632, 495633, 495636, 495664, with vertex pairs (160747,160748), (160747,160756), (160748,160757), (160756,160757). The internal diagonal remains consistent. Canonical per-face coordinate-stream SHA-256 is `7c9ef99f97cdd28c0c008aa3ea4bfcd626ea2a62c690a3980b687a2866a23a5d` for both meshes. This canonical stream sorts each face's vertex IDs; it permits the two declared winding permutations while the stronger exact-row test rejects all others.

## Shading and backface comparison

Matched CPU Cycles renders use flat normals, eight samples, four threads and identical cameras/materials. Imported vertex/loop payloads match their source arrays before and after rendering; no modifier, normal smoothing or geometry movement occurs.

Both selected geometric normals invert exactly (dot product −1). Their normal-dot-direction-to-camera values change from approximately +0.79267 and +0.78249 to −0.79267 and −0.78249. Ray casts hit the same two face IDs first in both variants. They therefore change from camera-facing cyan to backfacing red in the diagnostic material; the backface-transparent comparison hides them. This is a local visual witness, not a proof of a globally outward side on an open, non-manifold mesh.

Neutral two-sided shading is pixel-identical (zero changed RGB pixels). The front/back display changes 905 pixels; the backface-transparent display changes 528. Regional context shading also has zero changed pixels. Large-scale holes, incidence defects and silhouette remain. Local coherent winding is not sufficient evidence of an improved game-facing surface.

## Regional orientation constraints — analysis only

Use the previously accepted ordinary posterior central upper-trunk box: x=[−0.045,0.045], y=[0.02,0.12], z=[0.17,0.23]. Complete triangles only: 4,330 initially. Exclude any triangle whose corners, three edge midpoints or centroid do not project inside non-contour silhouette pixels at every original camera. This leaves **1,563 triangles / 899 vertices**. Positional exclusion and a local context render keep this study away from named front wounds, clothing, scalp, face, hands and feet. The silhouette check is a sampled raster exclusion, not semantic segmentation, continuous silhouette distance or global visibility proof.

For each edge with exactly two incident faces, define flip bits x and require `x[a] XOR x[b] = 1` for an existing same-direction conflict, otherwise 0. Include **2,212 internal constraints** and **223 crossing constraints** to fixed exterior faces. There are **501 internal conflicts and 44 crossing conflicts**, 545 affected conflicts total. Other global edges remain outside the regional optimization.

The graph has six disconnected adjacency components containing 1, 1,541, 2, 6, 6 and 7 faces. This disconnection is specific to the selected two-face adjacency graph; it does not create additional global surface components. All six are internally orientable: **zero contradictory internal fundamental cycles**. The implementation emits an explicit odd-XOR cycle witness when a contradiction occurs in fixtures. Three components have incompatible fixed-exterior orientation requirements. Their exterior witness pairs are saved privately.

The largest component has 496 internal conflicts and 40 crossing conflicts. A coherent orientation can clear all internal conflicts, but its two whole-component gauges require 1,138 or 403 flips and leave 160 or 51 crossing conflicts. The better gauge introduces 28 previously absent crossing conflicts while removing 17 old crossing conflicts. Neither provides a completely consistent fixed-boundary solution. Only three small components, 14 faces total, permit complete fixed-exterior consistency; they offer four additional conflict removals.

### Exact partial-repair bound

To avoid conflating whole-component consistency with all possible partial improvements, a separate symbolic min-cut calculation minimizes remaining conflicts while preserving **every currently consistent internal/crossing edge** and fixing all exterior faces. Internal orientability supplies a parity-label change of variables; then every internal constraint becomes an equality. Existing bad constraints have unit capacity, good constraints capacity 546, which exceeds the 545-conflict feasible baseline. That combinatorial bound enforces the no-new-conflict objective; it is not a replacement memory threshold.

The measured maximum flow and cut capacity both equal **325**. The optimum therefore removes **220 of 545 affected conflicts**, introducing **zero**. A symbolic full-mesh edge check independently confirms predicted global conflicts **147,260**. No regional face array or repaired regional mesh is constructed. The saved solution uses 130 symbolic flips: 128 posterior-camera-facing faces would become backfacing, while two would turn toward the camera. Camera-facing faces would decline from 415 to 289. This is one optimum; no claim is made that all equally optimal solutions have identical visual effects.

A second, explicitly presentation-constrained bound fixes all 415 currently posterior-camera-facing faces. Its maximum flow and cut capacity both equal **541**: at most **four** conflicts can be removed without new conflicts or reversing those presentation witnesses. This is evidence for stopping the tested winding-only route; it is not a newly imposed host safety gate or a globally outward-normal classifier.

Orientation has meaningful *numerical* potential (220 removals), but the tested constraints do not demonstrate meaningful *surface-quality* improvement. Keeping the current rear-facing presentation restricts the benefit to another tiny edit. A larger winding-only prototype is therefore **NO-GO**, and no additional mesh edit has been performed.

The selected region touches **20 boundary edges and 8 three-face edges**. Pairwise orientation constraints cannot resolve those incidence defects. Global **19,512 boundary and 31,449 non-manifold edges remain unresolved**. Sheet ownership, vertex fans and intended openings require separate structural reasoning before a reliable outward orientation or riggable retopology can be established.

## Verification, preservation and local evidence

Fourteen focused new tests cover exact authorized edits, immutable baselines, signed-zero changes, fixed exterior versus internal cycle contradictions, disconnected graphs, min-cut certificates versus exhaustive small graphs, low-resource refusal and pre-import render gating. Nineteen retained topology/orphan tests also pass. The unchanged native CPU owner enforces 6 GiB process commit, 180 seconds/stage, 12 GiB live host availability, 2 GiB output disk, GO/EOF and owned-process cleanup. Existing 22 GiB neural gates, preferred 24 GiB headroom and all other neural/resource safeguards are untouched.

Private output root: `artifacts/amd-zombie-winding-prototype-20261025-v1` in the local workspace, outside the published repository. Receipts: source authentication, variant manifest/acceptance, independent verification, eight-view silhouette check, regional constraints/face IDs, exact cut certificates, presentation comparison and owned-stage results. Original neural tensors, original PLY/NPY, source image and orphan baseline remain unchanged. Before/after inventory checks cover 257 prior asset files; the original 100 protected identities, 78 bounded hashes and 88 retained historical records match their established snapshot.

Only new safe code, tests and this textual evidence are published on an isolated branch. Generated meshes, NPY mappings, screenshots, process telemetry and full local constraint witnesses stay private. Independent review and final publication verification are recorded in the companion verification receipt when complete.
