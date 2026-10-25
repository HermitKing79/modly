# Zombie D reference preparation

**GO for one subsequent guarded generation using D**, subject to a separately verified handoff and fresh admission. One CPU composite was created. No inference, image-generation model, WSL, model loading or geometry operation ran. Hi3D remains NO-GO; no tooling was installed or integrated.

D uses authenticated Zombie A as its immutable base and the existing C palms/fingers as raster donors. A was not redrawn. Only two source rectangles were edited: screen-left `[0,260,162,420)` and screen-right `[1054,260,1216,420)` in the unchanged 1216×1294 canvas. The new image has genuine transparency, including 1,251,719 alpha-zero pixels.

| Qualification | Result |
| --- | --- |
| Source preservation | 19,631 changed RGBA pixels, all inside the two hand/wrist rectangles; zero changes elsewhere. |
| Actual model input | Original pinned alpha-only preprocessing and Lanczos resize; 980×980 crop to 512×512. A's known conditioning-raster hash reproduced exactly. |
| Non-hand conditioning | Zero changed pixels outside bounded hand/wrist neighbourhoods. Face, torso/wounds, clothing, legs and feet are independently pixel-identical to A. |
| Finger readability | Both hands visibly show five distinguishable tips with transparent gaps. Selected left columns show 4/6 and 7/7-pixel gaps; selected right columns show 7/7 and 4/6-pixel gaps at alpha >128. Each gap includes fully transparent pixels. These are raster probes, not a 3D digit classifier. |
| Wrists | Visually continuous, without visible detached/doubled joins. Every sampled wrist column has opaque foreground; adjacent silhouette steps are at most two source pixels. Each hand/wrist crop has one opaque foreground component. |
| Edge artifacts | No saturated red edge pixels in the effective 512 input. Inspection of transparent-checker and black-composited views found no new visible edge contamination. |

**Explicit changes beyond fingers:** C's palms face the viewer rather than retaining A's edge-on hand pose. Donor vertical scales are 0.809524 left and 0.857143 right to match A's wrist thickness. Donor-only RGB offsets are `[-3,+2,-1]` and `[0,+6,+5]`. The two 28-pixel-wide wrist/forearm transition strips are x=134–162 and x=1054–1082; their shading and local silhouette blend into A. No other forearm, body, face, wound, clothing or foot region changed. This is a controlled hand-design comparison, not finger-spacing-only causality.

The nominal floating-point crop moved half a resized-source pixel in x. Pillow rounds both boxes to the same effective integer box `[-12,10,968,990]`; actual 512 readback verifies identical non-hand pixels and framing. No body rescaling or wound-detail loss occurred. A's existing toe appearance is retained.

Local evidence, relative to the workspace:

- `artifacts/amd-zombie-d-reference-20261025-v1/zombie-d-reference.png` — the only candidate and proposed future source.
- `.../A-versus-D-512.png` — side-by-side effective inputs at native 512 pixels.
- `.../A-versus-D-details.png` — matched hands/wrists and wounds enlarged with nearest pixels.
- `.../D-checker.png`, `D-source-changed-pixels.png`, `preparation.json`, `verification.json`, `visual-review.json` — transparency, edit extent, reproducible transforms and acceptance evidence.

The original CPU owner retained its 6 GiB process cap, 12 GiB host-RAM floor, 2 GiB disk floor, 180-second deadline, GO/EOF and owned-tree cleanup. Preparation completed naturally in about 1.45 seconds with 260,096,000 bytes peak process commit; separate read-only verification also passed. Existing generation code and safeguards were not changed.

**Next comparison:** retain A's successful mesh as baseline. A future D handoff must authenticate this new source and recompute D conditioning, occupancy, shape and decode using the unchanged qualified 512 settings. Preserve three fresh RAM readings four seconds apart at ≥22 GiB, the final ≥22 GiB pre-neural reservation guard, preferred-only 24 GiB, 12 GiB live floor, existing VRAM/disk/deadline/attempt limits and owned cleanup. Compare matched eight-angle geometry for hands and retained wounds. No attempt was reserved and no generation was launched here. Clearer input is a reason to test, not evidence that the future mesh will improve.

All images and private assets stay local. The companion authority records immutable image and raster hashes; only safe text is published.
