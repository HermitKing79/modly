# Brick portable PBR mip/seam qualification — NO-GO

Base: `0f2fd9250e8a2825df240a1f948afed0b07d31fa`. Branch:
`codex/amd-brick-pbr-mipmap-qualification`. One bounded CPU filtering experiment
completed naturally. **The unchanged four-pixel-gutter atlas fails the declared
minification criteria.** LOD0–1 pass; contamination starts at LOD1.5, and numerical
accuracy gates fail from LOD2.5. This preserves the earlier accepted LOD0 result.
No UV rebuilding, new spatial-field sampling/texture bake, inference, geometry
change or retry occurred.

## Authenticated asset and filter contract

The original 493,124-byte GLB remains SHA256
`e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21`.
Both embedded PNGs match the accepted bake byte-for-byte. The genuine source
material remains SHA256 `8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
All 781 triangles, original positions/winding and four-island correspondence
were reauthenticated; actual GLB UV round-trip error remains 2.98023e-8.

Both texture samplers specify **9729 LINEAR magnification, 9987
LINEAR_MIPMAP_LINEAR minification, 33071 CLAMP_TO_EDGE**. The GLB embeds only
1024² PNGs; mip generation happens in the viewer. Khronos defines bilinear
filtering at two levels followed by interpolation between levels, but does not
prescribe a unique downsampling kernel. RGB is decoded from sRGB before filtering;
metallic/roughness are linear B/G channels. [Khronos filtering](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_filtering),
[material semantics](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material).

The [predeclared plan](amd-brick-pbr-mip-evidence/qualification-plan.md) uses an
explicit **F64 linear-light 2×2 box chain**, levels 0–5 (1024² through 32²),
including half-integer LODs. It models the sampler with no mip-storage rounding.
This is one defined CPU implementation, not a Blender/game-engine or universal
viewer qualification. Clamping protects the atlas exterior, not internal islands.

## Reference and unchanged quality gates

All 10,961 saved continuous queries plus 2,343 saved covered texel centres near
triangle edges were retained. Each fixed query was tested at all eleven LODs.
Additional projected-size views bring the total to **304,904 evaluations**.

- **A:** full accepted PNG atlas, filtered normally.
- **B:** same PNG values/kernel, restricted to the query's own padding rectangle.
- **C:** genuine saved F32 spatial samples at covered texels of that island,
  averaged over the same footprint. B/C normalize only after the full filter.

A−B isolates foreign/unowned influence; B−C distinguishes padding/quantization
error; A−C measures total footprint-matched error. Mips are not scored against
point samples. Original point and raw-seam checks are retained at LOD0; mip seam
residuals subtract the corresponding C pair gap so normal phase-dependent blur
is not mistaken for a seam. RGB RMSE/p95/max remain **0.02/0.05/0.15**;
scalar RMSE remains **0.02**, including separate metallic/roughness diagnostics.
Any foreign/unowned weight above **1e-7** fails the declared no-contamination gate.

[Independent candidate review](amd-brick-pbr-mip-evidence/independent-candidate-review.json)
and scalar-only support qualification preceded the marker. All source footprints
were positive. A direct original-base block oracle agreed within **2.95e-15**;
an independent separable-kernel readback oracle agreed within **3.82e-14**.

## Actual result

Maximum weights below are separate worst queries, not weights to add together.

| LOD | Mip side | Foreign maximum | Unowned maximum | Worst-island roughness RMSE | Decision |
|---|---:|---:|---:|---:|---|
| 0 | 1024 | 0% | 0% | 0.00283 | PASS |
| 1 | 512 | 0% | 0% | 0.00278 | PASS |
| 1.5 | 512/256 blend | 1.387% | 0.787% | 0.00275 | NO-GO: contamination |
| 2 | 256 | 2.774% | 1.573% | 0.00495 | NO-GO: contamination |
| 2.5 | 256/128 blend | 9.375% | 10.959% | 0.02701 | NO-GO: accuracy also fails |
| 3 | 128 | 18.750% | 21.918% | 0.05048 | NO-GO |
| 4 | 64 | 32.813% | 36.846% | 0.21532 | NO-GO |
| 5 | 32 | 47.156% | 55.210% | 0.26354 | NO-GO |

LOD0.5 also passes. At LOD1.5 only four fixed queries have foreign influence and
five have unowned influence; broad surface averages still pass, so they must not
hide edge failures. At LOD5, 174/13,304 fixed queries have foreign contribution
and 319 have unowned contribution. Worst-island RGB RMSE reaches **0.07153**.

One independently checked witness at LOD5 is original face **3,809,864**, saved
query 10,830, island 2: roughness is **0.447138** instead of source-footprint
**0.999989**, with 55.21% unowned and 9.55% foreign influence. The failure therefore
affects a real PBR channel, not merely ignored alpha or a dark learned RGB value.

The cause is the atlas support relative to the filter: three island rectangles
are only **41×44, 49×43 and 23×35** base pixels. Four reserved pixels become
half a texel at LOD3 and one eighth at LOD5; coarse taps mix neighbouring island
data and black unowned space. Ordinary blur remains in C and is measured
separately: interior source red variation falls only about 4% between LOD0 and5.

## Private visual evidence and resources

Root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-pbr-mip-0f2fd92\run`

- `comparison.png` — 1024×1738, SHA256 `789d5de0efc3a16311b5e133dcdafe2f2719429765762a717028c1e5f938b1c1`.
- `near-source.png` / `near-accepted.png`, corresponding `medium-*` and `far-*`.
- `near-contamination.png`, `medium-contamination.png`, `far-contamination.png`:
  red = foreign weight, green = unowned weight, black = neither on the surface.
- Saved A/B/C arrays, source support, ownership weights and query UV/LOD arrays
  allow independent numerical readback; all file hashes are in the textual evidence.

Views use the original geometry projected at **512 / 128 / 32 pixels high**,
with actual UV-derivative LODs **0.98867 / 2.98867 / 4.98867**. Enlarged panels
use nearest display enlargement after filtering. They show **unlit learned
base-colour**, not a lit PBR render; metallic/roughness are checked numerically.
Near appearance remains close. Medium/far show ordinary blur plus extra edge
darkening; similar brown island colours can conceal substantial source mixing.
Pixelated openings at far size are projection effects, not geometry edits.

The worker took **4.437 seconds**, supervised wall time **5.656 seconds**, peak
process commit **666,103,808 bytes (635.25 MiB)**. Minimum sampled host available
RAM was **18,070,278,144 bytes**; destination free space **106,515,460,096 bytes**.
Unchanged 6 GiB/180-second and 12 GiB RAM/2 GiB disk limits passed. Exit0, natural
empty Job, no forced cleanup, all watched sources unchanged. Process success is
distinct from the **NO-GO quality verdict**. [24 focused CPU tests passed](amd-brick-pbr-mip-evidence/focused-tests.txt).

[Independent outcome review](amd-brick-pbr-mip-evidence/independent-outcome-review.json)
verified every saved output hash, reopened ten PNGs and reproduced all metrics.
[Execution evidence](amd-brick-pbr-mip-evidence/execution-results.json) records
complete cohorts, witnesses, channel errors, settings and artifact hashes.
[Preservation after](amd-brick-pbr-mip-evidence/preservation-after.json) reports
zero mismatches: 100 protected files, selected environment/edit/backup/cache
identities and 26 frozen trees/694 files/11 reports. Three old disposable copies
retain labelled stat-only checks; this is not a whole-installation byte audit.

**Next action:** separately approve one disposable mip-aware atlas/padding
revision for this same patch, using the preserved genuine samples and geometry,
then apply these identical tests. No revised UVs, padding, GLB or full-model bake
was produced here. Stop for review; the original accepted GLB remains intact.
