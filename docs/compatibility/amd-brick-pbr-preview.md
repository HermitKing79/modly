# First genuine brick learned-PBR preview — PASS

Base: `9755adc6c57624cb81afe2dac2d5fa404b619023`.
Branch: `codex/amd-brick-pbr-visual-preview`.

One contained Blender CPU session imported the original mesh once and produced
three **1024×1024** learned-material views. All six resulting private images,
including comparison sheets and an image-detail crop, reopened successfully.
No neural execution, WSL, model loading, geometry edits, baking or model export
occurred. No images, photographs or model binaries are added to GitHub.

## Correspondence and material semantics

The complete material coordinate table matches the original geometry **byte for
byte and in row order**, SHA-256
`690670fe58a7021c51fef9fad6457ad8e66715d13a88239adabdffa8a05d9976`.
Independent CPU comparison verified every one of the **1,957,891 vertex positions**
against the original dual-grid extraction formula and both PLY/NPY representations.
All **4,173,584 triangle indices**, multiplicities and winding match the saved
faces. Correspondence is established for every row, rather than inferred from
equal counts. [Full mapping proof](amd-brick-pbr-preview-evidence/mapping-verification.json).

The original material remains SHA-256
`8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
Its raw and transformed F32 values remain unchanged. Six unclamped transformed
channels are retained as FLOAT/POINT attributes in the disposable Blender scene.
Actual attribute values and shader links were read back before rendering, after
the first view and after all views; hashes match the independently inspected field.

The derived shader RGB follows the pinned TRELLIS.2 PBR renderer's
`clamp(RGB,0,1)**2.2` convention, with no byte quantization. This is explicitly the
native gamma approximation, not a claim of exact IEC sRGB conversion. Metallic,
roughness and alpha are connected to Principled BSDF through display-only [0,1]
clamps. No manual colour, emission, normal or displacement effects were added.
This is a direct-row, triangle-interpolated material preview, **not a spatial
trilinear PBR bake or portable textured asset**.

| Transformed channel | Actual range | Values requiring shader clamp |
| --- | --- | --- |
| RGB | R 0.21757–0.88401; G 0.06548–0.72302; B 0.01687–0.61833 | None |
| Metallic | −0.004184–0.006669 | 810,749 below zero |
| Roughness | 0.978309–1.002575 | 1,721,533 above one |
| Alpha | 0.998056–1.004114 | 1,433,895 above one; none below 0.99 |

The learned material is essentially matte, nonmetallic and opaque. Extra scalar
diagnostic renders were unnecessary for these nearly constant channels; complete
unclamped statistics and counts are preserved in the mapping evidence.

## Private images and visual assessment

Directory outside Git:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-pbr-preview-9755adc`

- `contact-sheet.png`: 1500×1178, 1,021,170 bytes; SHA-256
  `e805ba8d15a845e96146b2522d9988b9b83728e50093b35e0c34aaf3a5aadf0f`.
- `comparison-sheet.png`: original photograph, prior +X clay and learned-PBR +X;
  1500×638, SHA-256 `be60ce233c8990c1c8e8f4509fc7de85a881d5e39c92ce926ca803645d6df5ea`.
- `renders\view-pos-x.png`, `renders\view-iso-a.png`, `renders\view-iso-b.png`:
  full individual PNGs.
- `surface-detail.png`: a labelled 4× nearest-neighbour crop of +X; an image
  enlargement, not a new render, surface reconstruction or added detail.

[Every private path, size and image hash](amd-brick-pbr-preview-evidence/image-verification.json).
The source photograph remains authenticated at SHA-256
`a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293`.

The brickwork has visible red/brown variation and pale mottling, while vines and
leaves carry ochre/brown colours consistent with the reference palette. Under the
unchanged neutral lighting, the generated result appears darker than the source
photograph. Leaf/brick gaps, pinholes and ragged thin surfaces remain visible;
materials have not repaired or hidden the known topology. Hidden-side fidelity
cannot be established from one photograph. Neither projection nor lighting is
photo-calibrated, so no pixel-match or material-quality score is claimed.

## Bounded execution and preservation

Blender **5.2.1 LTS**, Cycles CPU, four threads, 32 samples, CPU OpenImageDenoise.
Exact prior +X and opposite isometric cameras, orthographic scale, flat shading,
neutral lights, AgX/exposure and original geometry were retained. The first +X
image passed independent host verification before continuation at 29.297 seconds.

Renderer: **48.782 seconds**; supervisor: **55.469 seconds**; full invocation:
**56.109 seconds**. Peak process working set: **2,875,576,320 bytes**; peak committed:
**3,074,342,912 bytes** (2.86 GiB). All 55 live resource samples passed; minimum
available Windows RAM was **18,595,221,504 bytes**, minimum output-volume free space
**112,039,383,040 bytes**. The original **6 GiB / 180-second** cap and **12 GiB RAM /
2 GiB disk** floors were unchanged. Exit was 0; the Job emptied naturally without
forced cleanup. Image composition used a separate **1 GiB / 60-second** bound:
0.953 seconds, 57,192,448-byte peak per-process committed memory recorded by the Job,
also natural exit. Blender emitted an optional CUDA-loader warning; actual render
and denoising devices remained CPU.

Blender's embedded Python created one new 30,954-byte camera-helper bytecode
cache. Its exact bytes were preserved in private preview scratch, and only its
newly created empty directory was removed. The 20 pre-existing cache files and
all original source bytes remain unchanged. [Receipt](amd-brick-pbr-preview-evidence/incidental-bytecode.json).

**15 focused CPU tests passed.** Independent pre-execution and outcome reviews,
full attribute/geometry readback and image reopening support this result.
[Preflight](amd-brick-pbr-preview-evidence/preflight.json),
[review](amd-brick-pbr-preview-evidence/preexecution-review.json),
[render result](amd-brick-pbr-preview-evidence/host-result.json),
[actual Blender metadata](amd-brick-pbr-preview-evidence/render-metadata.json),
[outcome review](amd-brick-pbr-preview-evidence/independent-outcome-review.json).

[Before](amd-brick-pbr-preview-evidence/preservation-before.json) and
[after](amd-brick-pbr-preview-evidence/preservation-after.json) audits record the
100 protected files, original mesh/geometry/material, historical evidence/markers,
selected environment identities and AMD edits/backups unchanged. All 85 existing
remote heads, including main, are checked at publication; the final receipt remains
private outside Git to avoid a self-referential commit.

**Stop for visual review.** No UV bake, retopology, export or frontend integration
is authorized by this result. The mesh remains open/non-manifold and experimental.
