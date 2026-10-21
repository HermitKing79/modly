# One bounded foliage selection and UV qualification

Approved base64d05aa0d40073778f316e788d5ebe595b44495f. Branch
codex/amd-bounded-foliage-uv. No bake, GLB, Blender, Torch, WSL or GPU work.

Before loading/counting actual mesh faces, visually nominate exactly one original
+X image ROI [346,393,393,436], strictly within previous [332,384,400,448].
It includes the orange leaf surface and neighbouring curved vine. Image windows
leaf[350,399,378,425] and vine[380,394,392,434] are visual coverage proxies,
not semantic segmentations or claims of complete botanical parts.
Freeze nomination.json before selection; no alternate ROI/count search.

Reuse byte-pinned historical visible_faces with only module ROI changed in memory.
Complete original nearest-visible triangles and one original-edge neighbour ring
inside that same ROI; no clipping/truncation/deletion. Max6000. On oversize stop
with actual count, no UV work. On success save all original source IDs/corner bits,
connectivity, original/global edge incidences and artificial crop boundaries.
Inspect the private source/selected-surface image before classifying representation.

Reuse unchanged dominant-normal foliage_layout.build_layout. Fixed projected
density4096*sqrt(3)*(1+1e-6), tangent floor4096; one deterministic shelf packing,
32-aligned owner rectangles and exact continuous positive box/trilinear support
LOD0..5 (including half-levels). Max256 islands,250000 clip tests,64M raster
bbox work;1024² atlas. Require both F64 and portableF32 positive area, no interior
overlaps and every face covered. Save actual arrays only if layout exists;
BLOCKED is this deterministic candidate, not a theorem about all foliage layouts.

Reuse original owned CPU supervisor:6GiB process commit,180s worker,12GiB Windows
available RAM,2GiB destination free, binaryGO after attachment, natural emptyJob.
Source/material/geometry identities authenticated before and after; actual material
coordinates must match geometry coordinates byte-exactly. Private geometry/images
outsideGit; public source/text metrics only. All Python -I -B, bytecode disabled
before imports. Retain historical caches and quarantine unchanged.

Independent pre-execution review binds all candidate sources. Independent readback
checks original face/vertex correspondence, saved arrays/images and actual layout
measurements without reselecting another ROI. Stop after one qualification.
