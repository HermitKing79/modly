# Brick mip-safe atlas — PASS under the declared CPU filter

Base `e8244cb0f5e044c6749c278968da720c9f029942`; isolated branch
`codex/amd-brick-mip-safe-atlas`. One deterministic disposable atlas revision,
one genuine CPU spatial rebake, one portable GLB repack and one complete mip
regression. No alternate layout, retry, geometry change or neural/GPU work.

## Feasibility proved before revision

The original accepted GLB remains 493,124 bytes, SHA256
`e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21`.
The original material remains SHA256
`8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
All previous UV/bake producer receipts and all19 saved mip artifacts were
authenticated. The face **3,809,864**, LOD5 roughness failure independently
reproduced: **0.4471383744** versus source-footprint **0.9999886065**.

For mip level L, B=2^L and q=clamp(p/B−0.5,0,1024/B−1). Every positive
bilinear tap over a continuous axis interval [a,b] inherits base blocks from
floor(q(a)) through ceil(q(b)); exact zero-weight next taps are excluded.
Half/fractional LOD support is the union of adjacent levels. The analytical
proof bounds every point of every triangle, not just sampled queries.

The chosen ownership rectangles and integer UV translations, in base pixels:

| Island | Faces | Half-open owned rectangle | Translation | Required LOD5 support, half-open |
|---|---:|---|---|---|
|0|775|[0,0,768,1024]|[0,0]|[0,0,736,1024]|
|1|2|[800,32,928,160]|[118,60]|[800,32,896,128]|
|2|3|[800,192,928,320]|[77,220]|[800,192,896,288]|
|3|1|[800,352,928,480]|[28,380]|[800,352,864,448]|

Owned area **835,584/1,048,576 pixels (79.6875%)**, leaving212,992 unused.
All interior boundaries align to32-pixel mip blocks. The large island uses
legitimate atlas-exterior CLAMP_TO_EDGE; no external foreign pixels are assumed.
Small islands receive full isolated support despite their small surface area.
Increasing a generic gutter without this alignment/support proof would not
establish safety. No density reduction or island rescaling was needed.

[Candidate review](amd-brick-mip-safe-atlas-evidence/independent-feasibility-review.json)
preceded the bounded phase1 worker; [outcome review](amd-brick-mip-safe-atlas-evidence/independent-feasibility-outcome.json)
preceded all actual UV revision, sampling and GLB output.
Exact phase1 executed source bytes are archived privately in
`amd-brick-mip-safe-atlas-e8244cb\phase1-exact-executed-source`, matching the
original four review/attempt hashes; the published adapter subsequently gained
the separately reviewed phase2 dispatch and source pins.

## One source-faithful revision

Exact integer translations were evaluated in F64 to preserve original raster
phase/barycentric bits. All781 oriented triangles, multiplicities, original
face/vertex/material correspondence and all586,848 covered texels remain;
minimum coverage remains **74 texels per triangle**. UV area/density is retained.
The original sparse eight-neighbour spatial sampler recomputed every covered
texel from the genuine six-channel field. Recomputed covered F32 values match
the old genuine samples **byte-for-byte** after translation.

Original byte quantization, RGB encoding, MR G=roughness/B=metallic, retained
alpha and per-island byte-first Telea padding were reused. Alpha remains OPAQUE.
No neutral/invented colours, per-vertex/nearest sampling or new material model.

The existing Blender-exported GLB was **repacked in CPU code**, replacing only
UV0 and the two embedded PNG payloads. No Blender export/import was run anew.
POSITION, NORMAL and index buffers are preserved; material/sampler settings
remain unchanged: LINEAR / LINEAR_MIPMAP_LINEAR / CLAMP_TO_EDGE. No external
textures or required extensions. Portable F32 UV conversion and actual GLB
reload/support checks retain the original2e-7 tolerance; measured maximum UV
round-trip error is **2.98023224e-8**.
Independent old/new portable UV Jacobians are identical for every triangle;
minimum tangent density remains **19,333.0758 texels/world-unit**.

## Exact unchanged quality regression

Same declared **F64 linear-light 2×2 box pyramid and bilinear/trilinear** model,
LODs0,.5,...,5. Same13,304 fixed queries, all island-edge/interior/seam cohorts,
45 seam pairs, source-footprint reference and projected heights512/128/32px:
**304,904 evaluations**. Each mip reference averages genuine source samples
over the matched footprint; ordinary blur is not scored against a point sample.

RGB RMSE/p95/max gates stay **.02/.05/.15**; scalar RMSE stays **.02**;
foreign/unowned tolerance stays **1e-7**. All eleven LODs pass every gate.
**Maximum foreign and unowned weights are exactly0** throughout tested queries.
Actual exported triangle UVs also pass the analytical continuous-support proof.

| LOD | Worst-island RGB RMSE | Worst-island roughness RMSE | Decision |
|---|---:|---:|---|
|0|0.002757|0.002828|PASS|
|1|0.002057|0.002777|PASS|
|2|0.001980|0.002681|PASS|
|3|0.001929|0.002483|PASS|
|4|0.002257|0.002093|PASS|
|5|0.006196|0.001457|PASS|

All half-integer levels also pass. Direct original-base block oracle maximum
error **2.9421e-15**. Full cohorts, channels, seams, support and hashes are in
[execution results](amd-brick-mip-safe-atlas-evidence/execution-results.json).
At the original failing face/query, revised LOD5 roughness is **0.997904** versus
matched source **0.999989**, with100% own-island influence and no foreign/unowned
weight (previous accepted atlas:0.447138).
[Ten focused tests](amd-brick-mip-safe-atlas-evidence/focused-tests.txt) passed,
including real-filter impulse ancestry and strict GLB positive/negative cases.
[Independent outcome verification](amd-brick-mip-safe-atlas-evidence/independent-outcome-review.json)
reopened the actual GLB/maps/images and reproduced every reported metric with
zero discrepancy. Across all tested cohorts/islands/seams/LODs, maximum RGB
RMSE/p95/absolute error are **.006933/.014615/.035647**; scalar RMSE **.001633**.

## Private deliverables and measured resources

Root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-mip-safe-atlas-e8244cb\revision\run`

- `candidate\patch.glb`: **638,700 bytes**, SHA256
  `dc772cd55fc523c5427ab4771da27e62cd6905d5109f00b2fca47403704da857`.
- `candidate\bake\base-color.png`:540,248 bytes, SHA256
  `0048a6342df959735cb4687dc22c9a1c4f013459bb5ffb93d94d4ed0080652ad`.
- `candidate\bake\MR.png`:17,162 bytes, SHA256
  `307f5a0c17c974c4fadeac0a2ae16c00bcd0368a666c92313a943282cf05314f`.
- `comparison.png`:127,647 bytes,1024×1738, SHA256
  `8361e8810ec675c7c9c5500da6f333863d65fe65698eb4783f19f00af146b54f`.
- `near-accepted.png`, `medium-accepted.png`, `far-accepted.png`: revised views;
  corresponding `*-source.png` matched references and `*-contamination.png` maps.
- Complete UV/raster/source F32 data, A/B/C arrays and private receipts retained.

Comparison shows original accepted GLB left versus revised right, same unchanged
geometry and actual projected footprints. It is **unlit learned-RGB software
evidence**, not a lit Blender/PBR/game-engine screenshot. Metallic/roughness are
verified numerically. Near views remain close; medium/far avoid the prior extra
atlas darkening while retaining ordinary source blur. Small brown islands can
hide contamination visually, hence the numerical ownership and PBR checks.

Phase1:1.188s worker/1.843s host,185,479,168B peak process commit.
Actual revision: **7.422s worker/8.219s supervised**, **672,264,192B (641.12MiB)**
peak process commit; minimum sampled available RAM **17,976,848,384B**, destination
free **103,653,687,296B**. Unchanged6GiB/180s process,12GiB live RAM and2GiB disk
floors passed. Both workers exit0, naturally empty owned Job, no forced cleanup;
all watched sources unchanged.

The accepted GLB, textures, original master/field, protected source files,
selected installed environment/edit/backup identities and historical artifacts
remain unchanged. [After-preservation](amd-brick-mip-safe-atlas-evidence/preservation-after.json)
records zero mismatches across100 protected files and28 frozen trees/745files/
12reports; selected environments, edits, backups and caches retain their recorded
identities. Three historical disposable copies retain explicitly labelled
stat-only checks; this is not a whole-installation byte audit.
No WSL, Torch, ROCm, GPU, neural model, install, geometry repair
or full-model bake. This qualifies **only this patch/filter through LOD5**:
LOD>5, anisotropy, alternative mip-generation/storage kernels and universal
game-engine compatibility remain untested. Stop for review of this disposable
asset; do not automatically scale baking to the full model.
