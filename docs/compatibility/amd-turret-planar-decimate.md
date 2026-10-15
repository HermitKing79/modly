# Planar Decimate: modest local benefit, NO-GO for escalation

Base `e32915d24cdbbaf740f4f5d33dffbf99fbae8a2c`; branch `codex/amd-turret-planar-patches`. One disposable Blender **5.2.1 LTS / 9e2066aef7ef** CPU worker tested the original saved real patch, without face-fan separation. **Both settings preserved measured quality, but neither met the predeclared 20% worthwhile triangle-reduction criterion. Stop; do not scale to the master.**

## Implementation and scope

The installed executable matched SHA-256 `284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb`. Its C++ source is not bundled. Official version-tagged [Planar implementation](https://github.com/blender/blender/blob/v5.2.1/source/blender/bmesh/tools/bmesh_decimate_dissolve.cc#L88-L155) rejects non-manifold edge-join candidates, while [later degree-two vertex cleanup](https://github.com/blender/blender/blob/v5.2.1/source/blender/bmesh/tools/bmesh_decimate_dissolve.cc#L319-L450) can still change nearby topology. This differs from Collapse's previously measured restrictions; it does not guarantee frozen junctions.

Fixed settings were **1° and 5°**, each on a fresh unchanged copy, `DISSOLVE`, boundary dissolution disabled, and `NORMAL` delimiter. Disposable patches contain XYZ, indices and genuine linear learned RGBA; they have no original UV/seam/sharp/material-boundary information. Blender supplies no colour delimiter. Counts below come from actual `calc_loop_triangles()` tessellation, not polygon counts.

## Actual measurements

| | Original | 1° | 5° |
|---|---:|---:|---:|
| Vertices | 131 | 126 | 103 |
| Effective triangles | 272 | 262 | 223 |
| Reduction | — | 3.68% | 18.01% |
| Blender polygons | 272 | 220 | 134 |
| Edges with >2 faces | 65 | 65 | 61 |
| Boundary edges | 74 | 74 | 62 |
| Edge-connected components | 1 | 1 | 1 |
| Minimum three-axis silhouette IoU | — | 0.999925 | 0.996611 |
| Worst-direction sampled geometry RMS / P95 / max, divided by original diagonal | — | 0.000076 / 0.000216 / 0.000423 | 0.001187 / 0.003015 / 0.004356 |
| Worst-direction learned RGB RMS / P95 | — | 0.000824 / 0.001521 | 0.002481 / 0.005705 |

All positions, colours and indices were valid, with no zero-area triangles, missing component labels or fragmentation. Surviving vertex positions and RGBA matched original rows byte-exact. Retriangulation nevertheless changes interpolation: the table measures the colour field rather than claiming unchanged corner topology. Boundary length changed from `0.1420236674` to `0.1420040154` at 5°. The patch's 74 original boundaries were artificial extraction cuts; these measurements describe local topology, not every global junction. Inherited extraction metadata mentioning a join policy belongs to the previous experiment; no such preparation ran here.

Quality gates were the prior geometry RMS/P95/max limits `0.005/0.01/0.03`, informative silhouette IoU ≥`0.98`, RGB RMS/P95 ≤`0.02/0.05`, no new components/non-manifold edges, and boundary-length growth ≤5%. Both trials passed. At most 512 vertex occurrences and 512 area-stratified centroids per direction were sampled. Three-axis silhouettes use 256² pixel centres; they are not full-model or Hausdorff guarantees, and coincident surfaces within one component remain ambiguous.

One independently fixed housing target `(0.24,0.15,0.10)` was inspected read-only. Its three-ring neighborhood had only **80 triangles** and **1.97% area within 10° of the selected panel axis**, failing the predeclared ≥100-triangle/60%-flat eligibility check. It was skipped without modification or reselection. **Predominantly flat armour panels therefore remain unqualified.**

## Execution, verification and decision

Two modifier calls took **0.000855/0.000954 seconds**. Worker computation took **1.234 seconds**; host supervision **4.656 seconds**, with **301,928,448 bytes peak committed memory**, under the existing 2 GiB/120-second limits. Exit was natural and the owned Job empty; no forced cleanup occurred.

[Independent archive readback](amd-turret-planar-evidence/verification.json) verified all three saved meshes, hashes, counts and surviving original values. Thirteen preserved files, including original PLYs, Blender scenes, both GLBs, neural/PBR data and prior patch evidence, retained their identities. [Independent review](amd-turret-planar-evidence/independent-review.json) found no blocking issues. No GPU, WSL, inference, installation, full-mesh modification, Collapse, separation, remeshing or GLB export occurred.

Artifacts: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-planar-e32915d`. Exact hashes and measurements: [patch results](amd-turret-planar-evidence/saved-forward.json), [host receipt](amd-turret-planar-evidence/host.json).

**Recommendation:** Park automated LOD optimisation until a separately approved surface-reconstruction or part-aware retopology milestone. This probe establishes a small quality-preserving local benefit, not impossibility for all Planar settings or flat panels, nor feasibility of 100K/30K assets. No voxel remeshing was attempted; these open cropped patches are unsuitable as enclosed-volume tests.
