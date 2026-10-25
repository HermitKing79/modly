# A-body / D-hands: bounded CPU feasibility

**NO-GO for a clean copy-only wrist replacement on these meshes. No hybrid was created.** D's visibly better-separated fingers are useful, but the inspected sources do not provide a demonstrated, unambiguous transition into A's wrists. Joining them would require additional local fitting and seam reconstruction. This result does not prove that all future body-part replacement is impossible; it rejects promoting a simple cut-and-swap into a Modly feature on this evidence.

Base: `bc7160c7439a5d0ea9653e166620b6d535c498ce`. New branch: `codex/amd-zombie-ad-wrist-feasibility`. This was a one-case read-only spike, not production integration. No neural inference, WSL, model loading, dependency installation, repair, remeshing, smoothing, retopology, texturing or mesh export occurred.

## Authenticated sources

| Source | Original PLY SHA-256 | Vertices | Triangles |
| --- | --- | ---: | ---: |
| A | `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c` | 326,232 | 673,604 |
| D | `2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205` | 356,876 | 777,400 |

Both successful decode contracts, input lineage, ten-tensor payloads and original mesh manifests were reauthenticated using the existing CPU readers. Independent PLY/NPY coordinate/index agreement passed. A's original mesh was used, not its winding experiment. Its 425 unused vertices remain untouched.

## Wrist alignment and connectivity

The measurements use the original normalized coordinates; no physical body size is assumed. For each signed X side, eight fixed stations cover abs(X)=.34013 through .41013 at .01 intervals. These are viewing/inspection regions, not anatomical segmentation. Intersections are keyed by original crossed edges and do not depend on winding. Diagnostic segment calculations neither clip nor replace triangles.

At abs(X)=.38013:

| Measurement | −X side | +X side |
| --- | ---: | ---: |
| A section depth Y | .045772 | .045746 |
| D section depth Y | .026781 | .026867 |
| D / A depth | .5851 | .5873 |
| D / A section height Z | 1.0140 | 1.0180 |
| D minus A section-node mean Y | −.009926 | −.009014 |
| D minus A section-node mean Z | −.002739 | −.003648 |
| Estimated outward forearm-axis difference | 7.46° | 7.31° |

The axis estimate joins two narrow forearm-slab means; it is not a complete anatomical wrist frame. Section-node means are not area centroids. Translation alone leaves A-to-D section-point median distances of .00486/.00547 and maxima of .01276/.01204 at this station. These are point-to-point contour measurements, not nearest-surface distances or a tested fit. No transform was applied.

Only **one of 16 paired stations** has a single degree-two intersection loop in both models: −X at .37013. Even there, A/D have 172/135 section nodes and different dimensions. No +X paired station is a clean single loop in both sources. A single clean plane is not proof of a clean surrounding sheet or correspondence between meshes. Both distal viewing ROIs remain connected to their respective arms; there is no separately detachable hand component.

An independent dictionary-based edge-incidence witness examined a hypothetical whole-triangle partition at .38013. It did not create a cut mesh:

| Potential seam | New boundary edges | Degree-one ends | Branch vertices |
| --- | ---: | ---: | ---: |
| A retained body, −X | 83 | 2 | 0 |
| D donor hand, −X | 63 | 1 | 1 |
| A retained body, +X | 81 | 5 | 3 |
| D donor hand, +X | 57 | 8 | 0 |

These are **artificial boundaries that a cut would introduce**, separate from the donor hands' existing 418/418 one-face edges. The inspected contours are not matching clean seam loops. Retaining A's non-hand triangles unchanged is straightforward as data selection, but does not make an unjoined collection a usable hybrid. We did not force a weld, choose unsupported sheet pairings, fill gaps, invent a bridge or export that collection. **Actual new seam edges: zero.**

## Visual evidence and dark artefacts

Sixteen new views compare the unchanged A/D meshes under identical cameras, scale and neutral lights: front/back bodies, front/back of each hand, and front topology overlays for each hand. All source vertex positions, indices, triangle order, object transforms and counts were payload-checked before and after rendering. Cycles CPU, four threads, 16 samples, AgX exposure −.75; flat shading, no denoising or modifiers. Diagnostic curves are separate display annotations and are not asset geometry.

The close-ups show a substantially more readable spread of fingers in D; A's hand is much more slab-like in these matched views. D's shape/pose and wrist depth differ as well as its finger gaps, so it is not a geometrically interchangeable hand.

Cyan marks measured one-face edges; magenta marks edges incident to more than two faces; amber marks the diagnostic wrist section. Several black pits coincide with cyan boundary outlines, supporting actual open-edge defects in those locations. Other dark marks coincide with excessive incidence or lack a clear overlay correspondence. They remain unclassified: darkness alone does not establish an opening, and this test does not isolate shading, winding or occlusion contributions for every speckle. Intentional wounds were not relabelled or altered.

Global counts remain A: 19,512 boundary / 31,449 non-manifold / 147,484 two-face winding conflicts; D: 29,697 / 65,918 / 157,374. Better finger separation does not imply cleaner topology or game readiness.

## Bounds, evidence and stopping point

The unchanged CPU owner enforced 6 GiB process commit, 180 seconds per stage, 12 GiB live available host RAM, 2 GiB output-disk floor, four render threads and owned GO/EOF/Job cleanup. Analysis, rendering and assembly all passed with natural exits and no cleanup errors. Stage timings were 7.796 / 25.219 / 1.453 seconds; peak process commit 228,155,392 / 840,998,912 / 65,732,608 bytes. Neural 22 GiB gates and preferred 24 GiB headroom were untouched and were not used to admit any neural operation.

Five focused geometry witnesses and seven unchanged CPU-policy tests passed. Preservation verified **3,154 existing private files, 100 protected identities, 88 retained records and 168 prior local refs** unchanged. No historical tracked file was modified. Images and diagnostic arrays stay local; only new source/tests/text are published.

Independent read-only review approved the five new source/test/text files with no Critical, Important or Minor findings requiring changes. It checked the section/cut algorithms, source manifests, 16 render hashes, three sheet hashes, preservation metadata, prior refs and owned-stage receipts. It did not rerun workers or rehash the entire preservation closure; the primary preservation check did rehash the bounded historical assets and protected hashes.

Private root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-ad-wrist-feasibility-20261025-v1`.

- `review/body-front-back.png`: matched neutral body comparison.
- `review/hand-minus.png`, `review/hand-plus.png`: neutral front/back and topology close-ups.
- `analysis.json`: all 32 source sections, dimensions and winding-independent graph measurements.
- `seam-witness.json`: independent cut-incidence counts and final NO-GO.
- `authentication.json`, `renders/render-metadata.json`, stage `host-result.json` receipts and preservation manifests retain the verification chain.

Stop here. The current sources would require a separately scoped local transition reconstruction before a useful hybrid could be claimed. No further fitting or repair experiment was started.
