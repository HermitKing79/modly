# Bounded portable foliage vertex-colour proof

Base cfe852b476725afe9deecb5d9ad44892794ce2e1; new isolated branch
codex/amd-foliage-colour-glb. User authorises one genuine portable proof, with
no UVs, bake, model, GPU, geometry change or full-asset export.

Authenticate exact saved 4,531 oriented triangles/3,177 vertices/source IDs and
all original geometry/material lineage through the unchanged input adapter.
Reuse the actual saved F32 linear COLOR_0 array, without fitting or re-encoding.

1. One CPU writer: proper rotation original(x,y,z) -> glTF(x,z,-y), shared original
   indices and F32 position/colour bytes; source IDs in mesh extras. One standard
   triangle primitive. No NORMAL/UV/texture/extensions/external resources.
   Unit base factor; metallic 0.00011082693761409814; roughness
   0.999995986777544; OPAQUE approximation; doubleSided true.
2. Fresh independent Python worker parses actual GLB chunks/accessors/material
   and authenticates original sources. Recheck all saved 58,951 locations using
   exported F32 values and original source-spatial samples; no new source query.
3. One Blender 5.2.1 factory-startup CPU worker imports once, keeps actual imported
   byte-colour storage unchanged, reads geometry/colour/shader/factors, and builds
   an independently authenticated in-memory FLOAT_COLOR reference. Same flat
   shading, original positions, camera, lighting and seed. Near/far (4x ortho)
   comparison: four 640-square Cycles CPU 32-sample renders, CPU denoising only.
   Film transparency is a render-output diagnostic for silhouette, not a change
   to OPAQUE material behaviour.
4. Fresh Python readback verifies every PNG and actual imported corner colour,
   repeats original source-spatial gates for all 58,951 positions and makes a
   private labelled contact sheet. Inspect actual images independently.

Unchanged RGB error gates: RMSE <=0.02, p95 absolute <=0.05, maximum <=0.15
for each centroid/weighted/combined/seam-edge/seam-near cohort. Each scalar RMSE
<=0.02; OPAQUE compared as actual opacity 1. Export colours must be byte-exact
saved F32, while Blender's known BYTE_COLOR quantisation is separately measured.
Matched rendered PNG RGB errors (whole image and visible union) use the same
0.02/0.05/0.15 limits; alpha>=128 silhouette IoU >=0.999. Geometry and winding
must remain exact after the documented proper coordinate rotation.

Every actual stage runs sequentially under the existing owned Windows Job and
resource monitor: process/job 6 GiB, 180 seconds hard deadline, 12 GiB available
RAM and 2 GiB destination-free floors. Exact binary GO follows attachment. Fresh
exclusive per-stage marker/output, no retry. Prior-stage PASS required. Natural
empty Job and unchanged watched inputs required; forced cleanup is not PASS.

Independent source/outcome review and before/after preservation. Only source,
tests, textual metrics and hashes enter public Git; GLB/arrays/images remain
under the new private artifact directory. Stop for visual review after proof;
no general asset-quality, client-universal or full-model claim.
