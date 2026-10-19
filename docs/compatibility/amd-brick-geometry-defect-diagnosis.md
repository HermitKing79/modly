# Brick geometry defect diagnosis

**PASS for measured diagnostics; individual hole/shadow causes and repair safety remain partly unproven.** Base `1d1c1a6b54050200fca870b49c1b26bb51338c6a`; isolated branch `codex/amd-brick-geometry-defect-diagnosis`. No geometry or material changed, and no new render or model execution occurred.

The original mesh, NPY arrays, neural output, material field, mapping receipt and accepted **+2 EV** image were authenticated by full SHA-256 before and after analysis. The previously verified original-row material mapping and original +X orthographic camera were retained. [Input preflight](amd-brick-defect-diagnosis-evidence/preflight.json), [actual measurements and cases](amd-brick-defect-diagnosis-evidence/diagnostic-summary.json).

## Confirmed topology

The unchanged **1,957,891 vertices / 4,173,584 triangles** contain **267,652 boundary edges** and **387,068 non-manifold edges** (maximum incidence four). No repeated-index, exact float64 zero-area or nonfinite faces were found. Duplicate triangles were not re-sorted in this bounded audit; the earlier full extraction verification remains separate evidence.

There are **207 face-bearing components**, using connectivity through shared original vertex indices. The largest contains **4,118,568 triangles** (98.68%). Another 75 components contain at most ten triangles each, totalling 292 triangles. **5,749 unused vertices** are separate isolated indices, not visible floating surfaces. Point-touching faces coalesce under this definition; edge-connected face-component counts were not measured. Detached components are real connectivity facts, but are not automatically unwanted fragments: leaves and vines may be intentionally separate.

## What the dark marks establish

Two fixed crops nominate six small dark image blobs and two broad-dark controls. All **40** center/three-pixel-neighbour rays hit original triangles, without clipping, projected-degeneracy or depth-tie ambiguity. All eight center hits belong to the largest component. The ray test uses original winding and an opaque pixel-center approximation; it does not reproduce antialiasing, near-opaque alpha, denoising, area-light shadows or AgX.

| Case / PNG pixel | Encoded learned RGB at hit | Five-ray depth span | Evidence and limit |
|---|---|---:|---|
| A1 (534,372) | .504, .275, .149 | .15049 | Rendered black, but hit material is brown. Hit face touches incidence-1 and incidence-3 edges; geometric layering/orientation contributes a plausible explanation. Exact shadow/opening cause unproven. |
| A2 (626,464) | .426, .228, .136 | .23231 | Large local depth change; center face has three incidence-2 edges. Not justified to label this pixel a simple hole or a black learned texel. |
| A3 (538,499) | .446, .244, .153 | .05466 | Hit face touches a boundary edge; nearest edge has incidence two. Boundary proximity is correlation, not causal proof. |
| A4 (607,520) | .458, .313, .130 | .00651 | Broad dark control has a hit with incidence-2 edges; illumination/orientation remains unresolved. |
| B1 (444,340) | .476, .262, .147 | .01361 | Hit face touches two incidence-3 edges. Genuine junction topology; its contribution to darkness is unproven. |
| B2 (426,346) | .424, .242, .164 | .01277 | All three face edges have incidence two. A dark pixel alone is not a topology diagnosis. |
| B3 (464,463) | .536, .324, .196 | .20188 | Center is .031 px from an incidence-1 edge; strong depth discontinuity is genuine, closed-hole status unproven. |
| B4 (453,495) | .529, .290, .081 | .31764 | Broad dark control crosses markedly different depth layers; neither a closed hole nor pure colour cause is established. |

Depth spans use original world units, with neighbouring rays only three pixels away. Six of eight center faces have signed normals facing away from +X. These are measured orientation/layering facts, **not proof of incorrect winding**. The saved shader conversion is reproduced as float32 `clip(RGB,0,1)**float32(2.2)` per vertex before interpolation. Encoded RGB is not directly comparable with shaded/exposed PNG values; no lighting inversion is claimed.

Of **47** sampled boundary/non-manifold edge midpoints, **13** match the nearest opaque surface depth within `1e-7` world units. The sparse overlay displays **ten**, separated from other markers and case centers. This confirms visible topological boundaries/junctions in the crops. It does not prove whole-edge visibility, a closed repairable hole or that a nearby dark mark was caused by that edge. A ray hit may also be a rear surface visible through an opening in a nearer sheet.

## Resources, verification and next step

The combined CPU analysis took **5.344 s** under the unchanged **6 GiB / 180 s** envelope. Job-recorded peak per-process commitment was **593,035,264 bytes**; minimum sampled available Windows RAM was **20,416,552,960 bytes**. All six live RAM/disk observations passed. The Job emptied naturally, exit zero, without forced cleanup. The small venv launcher's working set is not reported as the analysis worker's peak RAM. [Host result](amd-brick-defect-diagnosis-evidence/host-result.json).

The initial diagnostic launch stopped during SciPy import because a local hostname query was mistaken for network activity. It read no mesh and exited naturally. That evidence is retained. Only `socket.gethostname` was admitted; connection, DNS, process, GPU-import and foreign-write negatives remain rejected. Eighteen numerical fixtures and a real guarded-import closure check passed. [Initial failure](amd-brick-defect-diagnosis-evidence/initial-launch.stderr.txt), [focused checks](amd-brick-defect-diagnosis-evidence/focused-tests.txt), [independent preparation review](amd-brick-defect-diagnosis-evidence/independent-pre-review.json), [recovery review](amd-brick-defect-diagnosis-evidence/independent-pre-review-recovery.json).

**No repair experiment is yet justified.** Global hole filling, welding and fragment deletion risk closing intentional brick gaps or destroying thin foliage. After visual review, the cheapest further qualification is to inspect one selected brick-panel boundary for a genuinely closed, locally manifold loop and determine whether a nearer sheet is missing. Only that evidence could support a separately approved disposable-patch repair test. Do not infer repair eligibility from these counts or dark pixels.

## Private visual evidence

Directory: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-defect-diagnosis-1d1c1a6`.

- `diagnosis-overlay.png`: 1,024,939 bytes; SHA-256 `23eba0a1afd867ffb5a07d4501fec879c6c16c80ee31db2fcb476cdc8557e2a2`.
- `diagnosis-region-a.png`: 161,843 bytes; SHA-256 `6208a31405493b14512e2ea1c2a32e772a6b0cf32afad16d1e27ea64709c3cd1`.
- `diagnosis-region-b.png`: 145,317 bytes; SHA-256 `8c13b25661b8ab7bcdfeed0c50eeb782f2f91f77aef6bd4cab80d77bd1e6df5a`.

All three PNGs reopened independently; the two unannotated 3× crop panels are pixel-exact nearest-neighbour enlargements of the accepted original. In the overview, the B2 label overlaps B1's center; use enlarged region B and its untouched left panel for that comparison. Composition reused existing pixels and took **0.891 s**, **45,019,136 bytes** peak per-process commitment, within **1 GiB / 60 s**. Its Job also emptied naturally without forced cleanup. [Image verification](amd-brick-defect-diagnosis-evidence/image-verification.json), [composition result](amd-brick-defect-diagnosis-evidence/composition-host.json), [independent outcome review](amd-brick-defect-diagnosis-evidence/independent-outcome-review.json).

The [before](amd-brick-defect-diagnosis-evidence/preservation-before.json) / [after](amd-brick-defect-diagnosis-evidence/preservation-after.json) audit preserves all 100 protected files, all 400 scoped file identities, original assets, four original cache directories, selected installed-environment/source-edit/backup identities, and prior evidence. Checkpoints were checked by bounded header/stat/configuration identity; no model-weight loading occurred. All 87 pre-existing remote branch heads, including main, are checked at publication; the final commit/remote receipt stays private outside Git. No model arrays, source photographs or diagnostic images are included in this publication. Stop for visual review.
