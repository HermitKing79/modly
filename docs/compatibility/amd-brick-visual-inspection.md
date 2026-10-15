# Genuine brick multi-angle inspection — PASS

One Blender session imported the unchanged full-resolution mesh once and produced all six axis views, two opposite isometric views and a clean silhouette. All images reopened and passed hash verification. **This is an inspection PASS; the mesh is not qualified as watertight or game-ready.** No images, source photographs or mesh binaries are included in this publication.

Base: `b536fb6d140c297ec845cb5741b19370884105a5`, branch `codex/amd-repeatable-generation-v1`. The source remains SHA-256 `ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd`, with **1,957,891 vertices and 4,173,584 triangles**. Actual Blender position and triangle-index bytes matched both original NPY arrays before rendering, after the first view, and after all views. Source files, winding, world transform and scale remain unchanged.

## Private visual evidence

Windows directory, outside the proof checkout:

```text
C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-visual-inspection-b536fb6
```

- `contact-sheet.png`: labelled 1440×1612 sheet containing all nine views; 1,036,709 bytes. SHA-256 `f87fde9ae5fb045fe290a5146e917ac73898e2cb03142eb611fe87ead2ab3417`.
- `source-comparison.png`: authenticated original RGBA beside +X clay view; 1600×890, 960,460 bytes. SHA-256 `36c7c43e87da162be12d24735930c05654317d0f6c8cb2d95650f1e5a27a266c`.
- `surface-detail.png`: labelled four-times nearest-neighbour image crop of representative +X brick gaps/surface marks; 1300×1000, 301,811 bytes. This is not an indexed-edge diagnostic or image enhancement.
- `renders\view-pos-x.png`, `view-neg-x.png`, `view-pos-y.png`, `view-neg-y.png`, `view-pos-z.png`, `view-neg-z.png`, `view-iso-a.png`, `view-iso-b.png`, and `silhouette-pos-z.png`: individual **1024×1024** PNGs.

[All twelve image paths, exact sizes and SHA-256 hashes](amd-brick-visual-inspection-evidence/image-verification.json). The sheet is provided in the private Codex conversation for viewing; no public media upload was made.

Signs label **camera position**, looking toward the mesh centre. Isometric A is `(+,−,+)`, B `(−,+,−)`. +Z is screen-up except the ±Z cameras, which use +Y. Every view uses the same orthographic scale, stored by Blender as `1.5038471221923828` (the exact F32 rounding of the computed scale). These labels do not establish the photograph's front.

The exact original 1018×1018 RGBA source and staged run copy both remain SHA-256 `a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293`, confirmed against selection, input-contract and DINO provenance. +X gives a useful source-facing comparison. Camera angle, projection and presentation scale are not photograph-calibrated; no silhouette-overlap score is claimed.

## Visual assessment

The object is clearly recognisable as staggered brickwork wrapped in vines, leaves and curled tendrils. Multiple views show a volumetric masonry core and vegetation around several sides. The upper vine arch and lower wrapped cluster broadly correspond to the reference. Fine curls survive, while some foliage is thinner, less dense or ragged compared with the source.

Dark gaps between brick courses, pinprick surface marks, perforation-like leaf marks and a few isolated-looking flecks are visible. Less vegetated −X/+Y views expose irregular brick courses and deep gaps. The single source photograph cannot validate the invented hidden sides. Disconnected components were not counted, so isolated-looking pixels are not proof of disconnected geometry.

All clay views are flat-shaded with constant neutral material, no bump/displacement/normal textures and no geometry smoothing. Contour irregularities and open-looking gaps persist across views; flat normals also contribute faceting, and shadows/occlusion can resemble holes. CPU denoising can soften fine detail. The unlit silhouette removes lighting effects but shows only the +Z outline. No spatial edge-incidence overlay was computed: **co-location of visible defects with the known 267,652 boundary and 387,068 non-manifold edges remains unproved**. No repair was attempted.

## Execution and preservation

Installed **Blender 5.2.1 LTS**, Cycles CPU, four threads, 32 samples, CPU OpenImageDenoise for the eight clay views. The silhouette uses opaque black emission against white, one sample and no denoising. Neutral camera-relative soft lights, fixed exposure and shared framing were not retuned after inspection.

The first +X render took **12.494 seconds**. The host independently reopened its PNG, checked dimensions/hash/source/counts/CPU policy and only then sent continuation at **27.015 seconds**, without reimporting the mesh. Renderer duration was **107.164 seconds**, supervised duration **114.047 seconds**. Peak working set was **2,628,403,200 bytes**; peak committed memory **2,904,969,216 bytes**, within the **6 GiB / 180-second** envelope. All **113** live admissions passed; minimum available RAM was **17,995,100,160 bytes**, minimum output-volume free space **108,242,665,472 bytes**, above the 12 GiB / 2 GiB floors. Worker exit was **0** and the owned Job was already empty, with no forced cleanup.

Cycles logged an optional CUDA-loader initialization failure; the verified rendering and denoising devices remained CPU. There was no Torch, ROCm, WSL, model execution, package change or neural work. Contact-sheet composition used a separate 1 GiB / 60-second CPU bound: **1.750 seconds**, peak working set **64,282,624 bytes**, natural exit.

Four camera and four first-frame host tests passed. Independent readback verified all nine PNGs, derived images, actual camera axes, shared F32 framing and original ordered mesh payloads. [Pre-execution review](amd-brick-visual-inspection-evidence/preexecution-review.json), [render result](amd-brick-visual-inspection-evidence/host-result.json), [readback](amd-brick-visual-inspection-evidence/independent-readback.json), [independent outcome review](amd-brick-visual-inspection-evidence/independent-outcome-review.json).

[Before](amd-brick-visual-inspection-evidence/preservation-before.json)/[after](amd-brick-visual-inspection-evidence/preservation-after.json) inventories confirm all 100 protected files, saved neural/mesh/photo inputs, source edits/backups, prior evidence and selected environment/native identities are unchanged. Publication advances only the development branch; all 76 other remote branch heads, including main, are preserved. The new public files are source-safe code, tests and textual evidence only.

**Next:** user visual acceptance of this unchanged geometry before separately authorising a material stage. Stop before learned texturing, geometry repair, GLB export or frontend integration.
