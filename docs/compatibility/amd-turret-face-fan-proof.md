# Turret face-fan separation: NO-GO for full-mesh scaling

From `ad7a4363d33643baf1e791987ff0393f9cb33abc`, branch `codex/amd-turret-face-fan-proof`. CPU-only Blender 5.2.1 LTS (`9e2066aef7ef`) and NumPy. **Separation unlocked Collapse, but the first real patch fragmented and lost surface detail. Stop before full-mesh preparation or decimation.**

## Method and exact preservation

The bounded method joins face corners only across original edges with exactly two incident faces, then duplicates vertices for the resulting corner fans. It does not invent pairings at four-face junctions. Cropped edges retain their global incidence classification. All original ordered triangles, multiplicity, winding, corner positions and linear learned RGBA are preserved byte-exact before decimation; bounds are unchanged. Output edge incidence is at most two. This conservative method does not prove every possible fan-pairing strategy unsuitable.

## Results

| Measurement | Synthetic junction grid | Real forward patch |
|---|---:|---:|
| Original triangles | 1,024 | 272 |
| Vertices before → separated | 561 → 612 | 131 → 317 |
| Edges with >2 faces | 16 → 0 | 65 → 0 |
| Edge-connected components | 1 → 4 intended sheets | 1 → 34 |
| Boundary edges | 128 → 192 | 74 → 294 |
| Unchanged Collapse result | 64 triangles | 202 triangles |
| Separated Collapse result | 51 triangles | 68 triangles |

The connected synthetic mesh contains two perpendicular subdivided planes sharing 16 four-face junctions. At ratio 0.05, sampled geometric deviation was below `3.85e-8` times the original diagonal; both informative silhouettes had IoU 1.0. The edge-on view was uninformative. This gate passed.

The real patch used a fixed three-ring neighborhood around global edge `(1214655,1214693)`, near `(0.1015,-0.3169,0.1192)`. Its four incident faces and both endpoint stars were complete. All 74 original patch boundaries were artificial extraction cuts; ten global non-manifold edges were partly cropped and correctly excluded from joining. No patch was trimmed or retried.

Separation created **220 net boundary edges**, increased boundary length 3.93× and left only 38.69% of area in the four largest components. At ratio 0.25, reduction reached 75% versus 25.74% for the control, but six original sheet labels disappeared, representing 2.94% of source area. The result contained 29 edge-connected components. Three local axis silhouette IoUs were **0.7891, 0.9677, 0.9666**, below the predeclared 0.98 floor.

Geometry comparisons were restricted to corresponding sheet labels, avoiding false matches to coincident sheets. Complete source-to-result statistics are unavailable after sheet loss. Even the matched subset had normalized RMS/P95/max deviations **0.0264/0.0557/0.1308**, exceeding 0.005/0.01/0.03 limits; reverse comparison also failed. These are bounded samples, not Hausdorff distances or whole-turret silhouette measurements. Retained-surface RGB RMS/P95 **0.0081/0.0172** passed; unavailable complete-source colour statistics must not be interpreted as measured colour-error failures. Both outputs had finite data, valid indices and zero zero-area triangles.

## Verification, resources and preservation

Three focused core regressions cover closed fans, duplicate/reversed triangles and cropped global junctions. [Independent readback](amd-turret-face-fan-evidence/verification.json) verified eight saved mesh archives, hashes and exact pre-decimation corner correspondence. [Review](amd-turret-face-fan-evidence/independent-review.json) agrees with NO-GO. The synthetic executed source is preserved separately; before the real phase only stricter actual-Blender input assertions were added, without rerunning or changing the synthetic experiment.

Synthetic/real host runs took **3.42/4.27 seconds**, peaking at **289,189,888/692,035,584 bytes**, under 2 GiB caps and 60/120-second deadlines. Both exited naturally with empty owned Jobs. Eleven source artifacts, including original meshes, scenes, neural/PBR fields and both GLBs, retained their SHA-256 identities. No installed environment was changed. Full-mesh edge incidence was inspected read-only; no full-mesh preparation or decimation occurred. No GPU, WSL or GLB export occurred. The second real patch was skipped after failure.

Detailed [synthetic](amd-turret-face-fan-evidence/synthetic.json), [real patch](amd-turret-face-fan-evidence/forward-patch.json) and [host](amd-turret-face-fan-evidence/real-host.json) evidence is preserved. Arrays remain outside Git at `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-face-fans-ad7a436`.

**Recommendation:** Do not scale this separation-plus-Collapse method. A separately approved, bounded voxel-remeshing or alternative surface-reconstruction test on a derived representative patch is the next alternative; it would change the surface and needs explicit geometry/colour fidelity acceptance. None was implemented or run here.
