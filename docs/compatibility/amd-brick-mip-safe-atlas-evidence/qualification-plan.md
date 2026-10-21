# One mip-safe patch revision: declared experiment

Base e8244cb0f5e044c6749c278968da720c9f029942. Preserve all accepted files.
Phase1 authenticates original GLB/maps, all UV/bake receipts and entire previous
mip result; independently recomputes the LOD5 face3809864 roughness witness.
Before UV mutation/bake/render, prove full positive box/bilinear support for every
point of every original island through LOD5 including fractional blends.

One proposed layout only, base pixels, half-open ownership rectangles:
0:[0,0,768,1024], 1:[800,32,928,160], 2:[800,192,928,320],
3:[800,352,928,480]. UV translations in pixels:0:[0,0],1:[118,60],
2:[77,220],3:[28,380]. Integer translation retains existing density and raster
phase. Outer atlas CLAMP_TO_EDGE is part of the proof; do not require nonexistent
external image texels. Interior rectangle boundaries align to32-pixel mip blocks.
Reject if any positive contributing texel has foreign/unowned support, or any
triangle/coverage/density/provenance preservation check fails. No gutter search.

Only after independent feasibility GO, create this one disposable UV revision,
repeat genuine sparse eight-neighbour CPU spatial sampling with the original
sampler, original encoded RGB/scalar byte quantization and isolated Telea padding.
No point/nearest or vertex-interpolation material substitute. Export one standard
embedded-texture GLB using the already-qualified Blender/export path if safe.

Reuse every previous fixed query and source face/barycentric location, island-edge
cohort, seam pair and projected heights512/128/32. Exact original box/trilinear
filter in linear light, all eleven LODs0,.5,...,5; original RGB RMSE/p95/max
.02/.05/.15 and scalar RMSE.02; foreign/unowned tolerance1e-7 unchanged.
Reference remains genuine source-covered F32 samples normalized only after the
same complete footprint. Preserve LOD0 continuous-point and seam checks.

Any capacity/admission/implementation uncertainty is BLOCKED before rebake;
completed revision with failed gates is NO-GO. No retry/alternate padding/size.
One owned worker per phase,6GiB/180s,12GiB live available host RAM/2GiB destination
disk floors, binary GO, natural empty Job release. No WSL/Torch/GPU/model work.
Private payloads/images outside Git, only code and textual evidence published.
