# First genuine turret mesh — exported, topology stop (PARTIAL)

Base: `ad07f96d85b3845837f294e1e61446fcf6ec4db1`. Branch: `codex/amd-turret-cpu-mesh-extraction`. Reviewed CPU converter candidate: `1054cafb60d25d965a3f3952b7e28518ecdffc52`.

**The full-resolution untextured mesh exists and can be imported into Blender. Conversion and independent readback passed. Overall PARTIAL:** the authentic output contains non-manifold and boundary edges. The requested topology stop was applied; no repair, rendering, further extraction or decoder retry followed.

## Open the actual mesh

Directory: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-cpu-mesh-ad07f96`

In Blender use **File → Import → Stanford PLY**, select **`turret-full.ply`**, keep scale **1.0**, and leave vertex merging/repair disabled. Select the imported object and use **Numpad .** to frame it, then orbit normally. Canonical coordinates are unchanged; the pinned TRELLIS renderer uses +Z up. No model has been normalized, reoriented, decimated or smoothed.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `turret-full.ply` | **74,276,526** | `89549a36c6acc6f884174a2097ad7e346980d565c999117ea29f89cda9baad1a` |
| `vertices.npy` | **22,115,744** | `2c4ea4aa3ac6dc870e5305c5448a3a3e5e75f09c6ecfc41c3e93aab57819f140` |
| `faces.npy` | **48,148,448** | `453acbcd87c8190c1600b27ec3cab0052db75ad65dcb01ec9734cf8863ab5b81` |

`provenance.json` is alongside these files. Full geometry remains local outside Git; source, tests, measurements, hashes and independent evidence are published. The original neural safetensors remains unchanged at SHA-256 `dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22`.

## What was proved

The complete original ten-tensor artifact passed header/schema/provenance and payload SHA checks, finite-value checks, coordinate bounds/uniqueness and learned-stage linkage. It was opened read-only. No checkpoint or model was loaded.

The CPU converter follows the [Microsoft O-Voxel inference function, pinned at de38fdd](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/o-voxel/o_voxel/convert/flexible_dual_grid.py). It preserves the original dual-vertex sigmoid, strict positive edge flags, softplus weights, three offset groups, strict F32 weight-product comparison (ties choose split 2), winding, grid 512 and AABB ±0.5. Sorted integer keys replace the native hashmap. Each bounded batch queries the complete global coordinate table and retains every original vertex/index. Missing-corner quads are rejected exactly as required. No alternate surface algorithm or geometry repair was used.

Static inspection found an existing AMD Windows O-Voxel binary, but no qualified WSL O-Voxel installation/extractor for the pinned Torch/ROCm runtime. Ordinary package import also has unrelated GPU-probing dependencies. The CPU route avoided new native qualification, installation or builds. No native extractor was executed; no CPU/native numerical-equivalence claim is made. Float32 CPU transcendentals need not be bitwise identical to GPU implementations.

| Measurement | Result |
| --- | --- |
| Vertices / triangles | **1,842,968 / 4,012,360** |
| Accepted original quads | **2,006,180** |
| Split 1 / split 2 quads | **1,005,364 / 1,000,816** |
| Omitted flags with missing neighbours / outside-grid queries | **1,616 / 0** |
| Conversion / complete worker duration | **1.204 / 2.938 seconds** |
| Peak committed process memory | **261,132,288 bytes** |
| Allocation/time safeguards | **2 GiB hard process cap; 180 seconds; 6 GiB available RAM and 2 GiB free disk admission** |
| Invalid indices / nonfinite vertices | **0 / 0** |
| Repeated-index / zero-area / duplicate triangles | **0 / 0 / 0** |

All 1,842,968 vertices are retained, including unused ones, as in the reference. Bounds are approximately X `[-0.340970,0.341598]`, Y `[-0.500183,0.499592]`, Z `[-0.283557,0.294619]`. The slight extension below −0.5 is permitted by the original half-voxel dual-vertex margin; nothing was clipped.

Six focused tests cover known connectivity, all axes/winding, both diagonals/ties, missing and boundary neighbours, shuffled global indices across processing batches, invalid inputs, transforms/scale and PLY readback. The final suite passed; an earlier host temporary-directory permission failure was corrected by preserving its tiny fixture inside the workspace.

The [independent CPU verification](amd-turret-cpu-mesh-extraction-evidence/independent-mesh-analysis.json) checked the actual full PLY/arrays, every original vertex, **every one of the 2,006,180 quad witnesses**, flags, offsets, diagonal decisions, original ordering and completeness. It did not rerun extraction. Its process peaked at 314,077,184 committed bytes and exited naturally. Both owned Jobs were empty with no forced cleanup.

## Actual topology and mandatory stop

Independent edge-incidence analysis found:

- **164,154 boundary edges**.
- **338,449 non-manifold edges** with more than two incident faces; maximum incidence **4**.
- **5,374,787 edges** with two incident faces.

This confirms that the preserved source-faithful mesh is **not closed or edge-manifold**. No converter mismatch was demonstrated. These counts do not establish self-intersection or orientation correctness, which were not tested. They also do not identify an AMD defect or prove a cause in the neural model.

The worker's raw `PASS` means conversion/readback passed; it does not supersede this **PARTIAL** final disposition. The prepared Blender preview scripts were **never executed**: the topology stop took precedence. No PNG or GLB was generated, and no problematic faces were quietly removed. The complete unmodified PLY and arrays remain available for visual inspection.

Both installed Windows environments, native binaries, AMD source edits/backups, original turret inputs and decoder metadata/configuration match before/after audits. WSL and its packages were not started or modified. Main, previous branches, historical evidence and original neural data are preserved.

**Next action:** inspect this actual PLY in Blender. Any further topology diagnosis, grey preview or repair needs a separately scoped instruction after this stop; it can use these saved vertices/faces and neural flags without rerunning the decoder. No speculative correction is proposed here.
