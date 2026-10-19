# Brick seam-aware learned-PBR bake — PASS

Base: `87f375585ad7ae5538856f9451c3233746e8fc77`. Branch:
`codex/amd-brick-seam-uv-bake`. One genuine CPU spatial bake completed on the
same deterministically selected **781-triangle / 457-vertex** brick patch.
The portable textured patch was exported, independently reopened in a fresh
Blender process, and compared against source-sampled F32 material. No master
geometry/material changes, neural execution, WSL, GPU inference or full-model
baking occurred. Generated models, maps and pictures remain private outside Git.

## UV qualification

The previous planar collision is between original faces **3,809,836 and
3,809,838**, local faces 469/471, intersection area
**0.00012292821691067823 normalized UV²**. This establishes projected overlap,
not a physical self-intersection. The previous failed run saved no subset-ID
list; this task recovered the exact selection by replaying its pinned original
selector, then preserved all source IDs and the overlap witness privately.

Source-edge-connected islands are split deterministically at projected
collisions and packed into the original **1024×1024** budget. The result has
**four islands**, 2,343 UV corner vertices, zero positive-area interior overlap
above the declared 1e-12 UV² tolerance, and **586,848 covered texels**.
Every original triangle has at least **74 covered texels**. Four-pixel reserved
gutters give eight pixels between island content bounds. Actual F32 UVs retain
at least 19,333 texels/world-unit tangent density; geometry positions, oriented
triangle multiplicity, winding and original material-row identities remain exact.

[Independent UV qualification](amd-brick-seam-uv-bake-evidence/independent-uv-review.json)
rechecked all 2,716 analytic candidate pairs, full raster coverage, original
triangle correspondence, gutters and orientation before the bake was admitted.

## Genuine transfer and measured quality

The [previous source audit](amd-brick-uv-bake-evidence/source-semantics.json)
remains pinned. Barycentric world positions are sampled using the genuine
six-channel sparse eight-neighbour algorithm, including missing-neighbour
renormalization. Original clipped byte truncation is retained. Quantize-then-Telea
padding is applied independently inside each disjoint island rectangle; this
is an atlas-packaging extension, and covered sample bits/bytes remain unchanged.
There is no vertex-colour interpolation or nearest-neighbour substitute.

All original accuracy gates passed without relaxation:

| Measurement | Actual | Existing limit |
|---|---:|---:|
| Surface encoded-RGB RMSE | 0.00217091 | 0.02 |
| Surface RGB absolute p95 / maximum | 0.00344331 / 0.00572169 | 0.05 / 0.15 |
| Cross-seam RGB RMSE / maximum | 0.00310638 / 0.00875374 | 0.02 / 0.15 |
| Surface scalar RMSE | 0.00127177 | 0.02 |
| Byte-quantization maximum | 0.003921556 | <1/255 + 1e-7 |
| Independent F64 eight-neighbour oracle maximum | 1.56611e-7 | 2e-6 |
| Unsupported source queries | 0 | 0 |

Checks include every face centroid plus 10,000 area-weighted samples, 90 seam-edge
and 90 near-seam queries, 45 paired seam comparisons, and ownership of every
bilinear footprint. Independent readback reproduced the metrics from the saved
PNGs; see [numeric review](amd-brick-seam-uv-bake-evidence/independent-bake-outcome-review.json).

RGB remains in native encoded PNG space, with glTF sRGB decoding for rendering;
roughness uses G and metallic B in the linear MR map. Raw F32 values remain
unchanged. Only derived maps/shaders clip to [0,1]: sampled metallic is about
-0.00116–0.00080, roughness 0.99949–1.00105, and alpha 0.99946–1.00076.
Alpha is retained in F32 and RGBA, including the embedded base map; the test
material intentionally remains **OPAQUE**.

## Portable asset, comparison and resources

Blender 5.2.1 exported a **493,124-byte core glTF GLB**, with 781 triangles and
2,343 corner vertices. Fresh import verified exact oriented triangle positions,
maximum UV round-trip error 2.98023e-8, embedded map pixels, shader routing and
unchanged source inputs. Both panels use identical CPU Cycles / 32-sample /
640² / +2 EV presentation. Their RGB8 full-frame MAE is **0.1780/255** and
maximum difference **2/255**. Visual inspection found no obvious added seams
or missing detail; existing openings and patch boundaries remain visible.

The reference panel uses unquantized F32 spatial samples at the **same atlas
resolution**, with derived padding held fixed. It isolates quantization and GLB
transport; it is not a continuous-field shader. Separate continuous-surface
holdouts above measure atlas approximation error.

The first export launch stopped before GO/scene/export at an empty-scratch
assertion. Blender creates a temporary session directory at startup and removes
it on exit ([Blender source](https://github.com/blender/blender/blob/v5.2.1/source/blender/blenkernel/intern/appdir.cc#L1128-L1214)).
The adapter now admits only the exact live, empty `bpy.app.tempdir` immediate
child after the unchanged GO gate, rejecting foreign or redirected content.
The stopped attempt and original adapter are preserved; fresh export/compare
destinations were used. **The material bake was not repeated.**
[Focused independent startup review](amd-brick-seam-uv-bake-evidence/independent-render-startup-review.json)
approved this correction.

| Successful owned stage | Host elapsed seconds | Peak process commit bytes |
|---|---:|---:|
| Recover patch | 4.563 | 414,846,976 |
| UV qualification | 3.172 | 197,955,584 |
| Genuine bake | 4.781 | 495,276,032 |
| Export | 15.640 | 441,303,040 |
| Fresh import and both renders | 38.922 | 978,108,416 |
| Independent readback/contact sheet | 1.282 | 205,856,768 |

Each process retained the **6 GiB / 180-second** limits and owned Windows Job.
Minimum sampled available Windows RAM was **19,506,774,016 bytes**, above the
12 GiB floor; destination free space remained above 2 GiB. Every process,
including the stopped startup launch, exited naturally with an empty Job and
no forced cleanup. These are measured process-commit and available-host figures,
not GPU-memory measurements. [32 focused CPU tests passed](amd-brick-seam-uv-bake-evidence/focused-tests-before-export.txt).

## Private deliverables and preservation

Root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-seam-uv-bake-87f3755`

- `finish\comparison.png` — SHA256 `e56f74980b3ced18ff64204cba8d290c484d4d5136c59aeffc4dfa349415a7dd`
- `render-qualified\export\patch.glb` — SHA256 `e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21`
- `bake\base-color.png` — SHA256 `f294c9006b286747fb76ddb2f637e02ecfb0710410acfc2db68937a741d8891d`
- `bake\MR.png` — SHA256 `2900c8e6124ffbb992eb863e3a2a10bd337b4851b509645f9a6a266dfc8fea1e`
- `render-qualified\compare\direct.png` and `baked.png` — individual reference/baked renders.
- `bake\sampled-field.npy`, `reference-field.npy`, source-ID arrays and phase receipts retain F32 samples and full correspondence.

[Execution evidence](amd-brick-seam-uv-bake-evidence/execution-results.json)
contains all output hashes, settings and measured gates.
[Final independent review](amd-brick-seam-uv-bake-evidence/independent-outcome-review.json)
rechecks the portable mesh, embedded maps, actual PNGs and stated limitations.
[Preservation after](amd-brick-seam-uv-bake-evidence/preservation-after.json)
confirms zero mismatches across 100 protected files, selected environment/native
identities, AMD edits/backups, caches, source closures and 24 historical evidence
trees. Original mesh SHA remains `ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd`;
material SHA remains `8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
Three historical disposable copies retain explicitly labelled stat-only checks;
this is not a whole-installation byte audit.

This is a small-patch feasibility PASS. Arbitrary mip-level bleeding, full-model
UV packing, complex foliage, other viewers and GPU bitwise sampler parity remain
unqualified. The single next action is **visual review of the private comparison
and portable patch**, before separately approving any larger bake.
