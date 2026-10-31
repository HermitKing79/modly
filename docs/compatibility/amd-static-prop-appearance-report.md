# Phase 20D — practical static-prop appearance review

Parent: `9fa586d222f8fba81e6782dff38a458d07aaed80`.

This task reviews the existing Phase 20C experimental GLB. It performs no
geometry reconstruction, UV edit, normal edit, baking or re-export. The
original Phase 11 high-fidelity benchmark remains **PERMANENT NO-GO**.
The candidate is an **AI-DERIVED RECONSTRUCTION**, with user appearance
approval pending under the separate LOWER-DETAIL STATIC PROP profile.

## Authenticated inputs

- Existing GLB: 94,652 bytes; SHA-256
  `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201`.
- Original AI surface: 1,023,995 vertices / 2,047,210 triangles.
- Saved raw Alpha Wrap: 1,406 vertices / 2,808 triangles.
- Saved final candidate: 502 geometric vertices / 1,000 triangles. The GLB's
  2,695 attribute vertices retain its qualified UV/flat-normal discontinuities.
- Historical authentication includes 166 Phase 19 files, 314 Phase 18 files,
  552 Phase 14 files, 47 original source bindings, 188 protected identities,
  44 Phase 20 files, 128 Phase 20B files and 90 Phase 20C files.

Private review/evidence root: `artifacts/amd-static-prop-appearance-20261031-v1`.
Private images and models are not published.

## Why the previous close-ups were uninformative

All 24 historical images were decoded and measured; both 12-panel material
sheets were visually inspected. Informative whole-object views are reused.
Region 37 source matte/glossy panels contain 90.9985% / 90.5476% pure-white
pixels. Region 8 is not clipped, but source luminance standard deviation is
only 0.8192 / 0.5002 on a 0–255 scale. The former loses highlight information;
the latter offers little useful local contrast.

The previous Blender bridge retained view direction, target and span but
discarded the source camera's right/up roll basis. Its Standard display
transform and fixed world-space lights were unsuitable for these very tight
surface views. Two-sided centre rays hit the intended original source faces
506773 and 1553457, with zero centroid occlusion gap. This establishes
visibility only at those tested points, not over the complete region. The
retained PNGs cannot isolate every lighting contribution or recover clipped
radiance.

## Frozen comparison plan

One matte region-37 pilot must establish usefulness before the other three
region/material combinations. Maximum additional rendering: twelve images
(source/raw/final × two regions × two materials), including the pilot.

The new comparisons retain the full historical source camera basis and
direction, widen the context span to 0.006 source diagonal, and use 768-pixel
CPU Cycles images with 64 samples. AgX / Medium High Contrast / exposure 0,
camera-relative key/fill, shared neutral gray, roughness 0.8 or 0.12 and
two-sided shading are identical within each triplet. Installed OCIO config
and LUT bytes are pinned. No setting sweep or retry is permitted.

Source and raw-wrap render copies initialize the same geometric flat-face
presentation as their historical images. The final close-ups import the
actual authenticated GLB, preserving its corner normals and UVs. Before/after
hashes bind coordinates, oriented indices, corner normals, UV layers and
transforms. Review materials exist only in the disposable Blender process.

The inherited supervisor retains a 6 GiB process/Job commit cap, four cores,
180-second stage deadline, 12 GiB available Windows host floor, 2 GiB disk
floor, exclusive evidence paths, GO after Job attachment, monitored natural
exit and owned-process cleanup. No WSL, GPU, neural, CGAL or installation
route is present.

## Outcome

**PASS — informative matched visual evidence. INCONCLUSIVE — intended-use
appearance. USER_APPEARANCE_APPROVAL_PENDING.** This is not an asset promotion.

The pilot was independently judged informative before continuation. Exactly
twelve new images were rendered: source/raw/final for two regions and two
materials. Twelve useful historical whole-object images were reused; no
whole-object rerender or parameter retry occurred. All 24 historical images
were rehashed again during final image verification.

- Region 37: the original has a narrow intermediate-gray microfacet strip;
  raw and final show a largely single, shifted transition. The loss is plainly
  interpretable at this magnification.
- Region 8: the source has shallow triangular faceting and an irregular lower
  transition; raw and final are straighter/flatter. Raw versus final is not
  decisively distinguishable in the saved views. Two of the five
  simplification-added failed incidences occur here; three occur in region 34.
  These images do not isolate those individual numerical failures.
- Not every nearby visible facet or image difference is established to belong
  to the 524-face failure set. The complete earlier census remains the basis
  for that count and area measurement.
- Glossy variants remain useful for local shape, but this lighting produces
  mostly diffuse response without a strong localized glint. They do not
  establish general moving-highlight acceptability.
- Whole-object views look closely similar at their saved screen framing, but
  their bright tops are clipped. New close-ups span 0.006 source diagonal,
  versus 1.15 for the old whole views: about **192 times spatial magnification
  at equal display width**. No physical intended viewing distance or finished
  production material has been qualified.

The HTML page includes source-location markers, clearly labelled triplets,
material/region navigation and click-to-zoom/pan. Markers project source target
points and explicitly do not certify far-side visibility; no PNG was altered.
Historical rows correctly say *saved final mesh*, whereas new final close-ups
use the actual imported GLB. The saved GLB retains its original neutral test
PBR material (roughness 0.35); comparison overrides exist only in RAM.

## Resources and preservation

| CPU comparison | Elapsed seconds | Peak Job/process commit, bytes |
|---|---:|---:|
| Matte region 37 pilot | 80.672 | 989,745,152 |
| Glossy region 37 | 79.625 | 988,360,704 |
| Matte region 8 | 79.829 | 999,727,104 |
| Glossy region 8 | 81.469 | 995,676,160 |

Rendering total: 321.595 seconds. Peak commit: about 0.931 GiB. Minimum sampled
available host RAM during rendering: 13,934,804,992 bytes (about 12.978 GiB),
above the unchanged 12 GiB CPU live floor. Host availability includes concurrent
activity; the Job peak measures owned-process commit, not all host changes.
Every stage exited naturally with an empty owned Job and no forced cleanup.

Independent checks reproduced all twelve image hashes, frozen cameras and
actual CPU/material/display settings. All source/raw/final coordinates,
oriented indices, UV layers, corner normals and transforms were identical
before and after every three-image comparison stage. Historical preservation reauthentication passed,
including the 90-file Phase 20C tree and unchanged full GLB checksum. No original
mesh, image, output, environment or backup was written. Frozen neural gates
and the historical execution hold remain unchanged.

Offline verification: **29 tests PASS**, covering inherited safety/UV checks
and new two-sided visibility, camera basis, image statistics, continuation
hold, absent export/edit routes and UV/normal/camera-tampering rejection.
Independent review approved the pilot, inspected all twelve full-resolution
new images, checked every settled resource/preservation receipt and reviewed
the final assessment/report path. Only source code, tests and this textual
report are published; local manifests bind the private images and assets.

## Open the local result

Review page: `artifacts/amd-static-prop-appearance-20261031-v1/review/index.html`.

Existing GLB:
`artifacts/amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb`.

In an empty Blender scene, choose **File → Import → glTF 2.0**, select that GLB,
select the imported object and use **Frame Selected** (Numpad `.`). This imports
the existing experimental neutral-test asset; it performs no re-export. Do not
overwrite original assets. The review page also supplies the full local path.

## Interpretation limits

The documented 524 source microfacet faces cover 0.0024493% of original
source area and lack a qualifying normal-compatible counterpart. Most form
coherent thin strips; they have not been demonstrated to be noise. A positional
surface-distance pass does not mean their orientation detail was preserved.
Most failed incidences already occur in the raw wrap; simplification adds
five. This concession is specific to the saved candidate, not a general
small-feature removal rule.

Normal baking remains SKIPPED. S&box remains NOT TESTED. Intended appearance,
finished PBR, production UV padding/mips, engine import/units/collision/LOD,
asset-class generalisation and distribution licensing remain separate gates.
No fully game-ready or organic-asset result is claimed.
