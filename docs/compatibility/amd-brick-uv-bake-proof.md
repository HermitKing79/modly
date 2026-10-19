# Brick learned-PBR UV feasibility — BLOCKED

Base: `18c0c92fe348e8a7fc04e9b5c9443423484ee5b6`. Isolated branch:
`codex/amd-brick-uv-bake-proof`. CPU-only; no original asset changes.

The bounded patch preparation stopped at **analytic positive-area UV overlap**.
No actual texture sampling, texture images, portable patch, Blender process or
comparison render was produced. Texture detail retention is therefore unproved;
this is a UV-parameterization blockage, not a demonstrated material-quality or
AMD failure. No repair of A1/A3 or geometry changes were attempted.

## Established semantics and inputs

The original mesh SHA-256 remains
`ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd`;
the genuine six-channel material SHA-256 remains
`8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
Preparation reauthenticated the existing full-row vertex/material correspondence
receipt and compared the complete material/geometry coordinate arrays in order.

[Source audit](amd-brick-uv-bake-evidence/source-semantics.json) pins 12 installed
sources. Native baking samples the sparse six-channel field at barycentric world
positions, using eight neighbours and renormalization over found in-bounds
support. It then truncates clipped values to bytes and inpaints UV padding.
The existing direct-row preview instead interpolates vertex material values;
its RGB gamma conversion also precedes interpolation. They are not equivalent.

A bounded NumPy sampler implements the native spatial algorithm. On an unchanged
source triangle the native closest-point projection is mathematically identity.
The stock exporter itself is unsuitable here: it invokes GPU/native processing
and geometry changes. No GPU bitwise-parity or complete exporter-equivalence
claim is made. [29 focused CPU checks passed](amd-brick-uv-bake-evidence/focused-tests-final.txt),
covering an independent sparse-sampling oracle, missing support, byte packing,
UV overlap including sub-texel intersections, PNG filtering and synthetic GLB
geometry/map validation. These do not establish real baked-asset success.

## Actual bounded preparation

An initial normal-sign filter excluded the visible surface and stopped before
UV creation. One read-only nine-ray inspection established negative-X winding
on the visible brick sheet. A reviewed correction selected original face
3843320 at pixel `(580,554)`, normal cosine `-0.960421`, within the same brick
ROI, away from A1/A3. The absolute inclination criterion stayed `0.9`; only UV U
was mirrored. Original face winding and positions were preserved.

The corrected connected patch passed source-hit, size, colour-variation and
consistent-winding checks. Its planar UV chart failed the independent analytic
overlap gate at `bake_patch.py:108` / `patch_math.py:83`, before rasterization
or sampler construction. The exact overlap pair/area and patch face count were
not journalled before the exception; they are not claimed. No further attempt
or approximation followed.

| Owned CPU operation | Exit | Seconds | Peak process commitment |
|---|---:|---:|---:|
| Initial nomination | 1 | 2.282 | 343,474,176 B |
| Read-only nine-ray check | 0 | 2.282 | 280,768,512 B |
| Corrected UV preparation | 1 | 3.390 | 413,908,992 B (394.7 MiB) |

All three Jobs emptied naturally with no forced cleanup. The 6 GiB/180-second
limits and 12 GiB RAM/2 GiB disk floors remained unchanged. Corrected-run
minimum observed Windows availability was 20,479,639,552 B; destination disk
availability was 111,505,448,960 B. Commitment is a Windows Job process metric,
not whole-host RAM, working-set or GPU memory.

[Exact errors, measurements and receipt hashes](amd-brick-uv-bake-evidence/execution-results.json)
and the independent reviews accompany the code. Private failure evidence stays
under `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-uv-bake-18c0c92`.
The initial source snapshots match their pre-execution hashes; fresh corrected
paths are under its `qualified` child. No private images or model binaries are
included in this publication.

## Preservation and next step

[AFTER verification](amd-brick-uv-bake-evidence/preservation-after.json) found zero
mismatches: all 100 protected files, 42 selected environment checks, 13 source
edit/backup checks, 400 authenticated identities, prior source closures, four
cache directories/20 files, 22 historical trees/526 files and nine reports.
523 historical files received fresh hashes; three old disposable fixture copies
retain explicitly labelled stat-only checks. Original mesh, neural and material
artifacts received fresh full hashes; checkpoint weight payload bytes read: zero.
Remote-head preservation is independently checked during publication.

**One recommended next experiment:** seam-aware UV unwrapping of this same small
disposable surface, preserving every original oriented triangle, followed by the
same spatial-sampling and image-comparison gates. This addresses the demonstrated
non-injective planar projection without repairing geometry. It requires separate
approval. Do not proceed to full-model baking or infer texture quality from this
blocked preparation.
