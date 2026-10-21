# First full-resolution source-sampled GLB — PASS

Approved base `3db265ca500cfe46909d12f52c151924763cd2f6`; isolated branch
`codex/amd-brick-full-sampled-glb`. One genuine full CPU resampling/export and
one fresh independent verification completed naturally. No Blender/rendering,
GPU, model execution, geometry change, UV or texture bake was performed.

## Private portable output

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-full-sampled-glb-3db265c\export\brick-source-sampled-colour.glb`

- **97,073,940 bytes**; SHA-256
  `1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae`.
- All **1,957,891 original vertices / 4,173,584 original oriented triangles**.
- Standard self-contained glTF 2.0: F32 POSITION/COLOR_0, uint32 indices, one
  primitive, unit base factor, metallic `0.00011082693761409814`, roughness
  `0.999995986777544`, OPAQUE and double-sided. No normals, UVs, textures,
  compression extensions, external dependencies or invented material details.
- Source-to-glTF rotation `(x,z,-y)`, determinant +1; inverse `(gx,-gz,gy)`.
  Source coordinates, triangle order, winding and multiplicity remain exact.

Complete F32 linear vertex colours in the same directory:
`linear-COLOR0.npy`, **23,494,820 bytes**, SHA-256
`1240160cefcf961d463d9c06416e0907469c8d80a590fb44f483bb1fc220ea44`.
Original vertex index is the colour row; no duplication, fitting or reordering.
Original six-channel material data and source mesh remain unchanged.

## Actual sampling and independent readback

The authenticated original stored world positions passed full bit-exact
dual-grid reconstruction verification. One qualified sparse eight-neighbour
index sampled all vertices in **60 contiguous batches**, maximum 32,768 rows,
last **24,579**. Original F32 arithmetic/order and partial-support normalisation
were retained. Every six-channel result was finite; **zero unsupported rows**.
Minimum positive neighbour count was **1**, minimum raw weight sum
**0.140675247**. **1,371,046** rows lacked at least one neighbour and used the
original renormalisation policy; no row was replaced, repaired or discarded.

Derived encoded RGB was clipped only for the original IEC conversion to F32
linear COLOR_0. Exclusively created RGB/support/weight-sum NPYs were flushed,
reopened and checked. All **25,967** previously qualified vertex-colour rows
matched bit-exactly before the single GLB was written. Original PBR values were
preserved; this is vertex RGB plus fixed factors, not a spatial PBR texture bake.

The separately authored strict reader ran in a fresh owned process after export
exit. It read the complete file with bounded mappings and compared all vertex
and colour bits in 60 batches and every triangle index in 128 batches. Header,
chunk lengths, accessors, alignment, bounds, material, source hashes and absence
of unwanted dependencies passed. Full F32 colour transport and original
geometry/index/winding/multiplicity comparisons were exact.

Replaying all **56,910 original queries** and **2,048 frozen interior holdouts**
against the exported colours passed all **112** region/subcohort gates.
Predictions differed from the previous accepted ones by **0.0**. RGB limits
remain RMSE 0.02 / p95 0.05 / maximum 0.15; each scalar RMSE remains 0.02.
The holdout is on new positions within previously frozen faces, not unseen faces.
Numeric export integrity does not establish complete visual fidelity.

## Measured resources and bounded preparation

| Stage | Worker elapsed | Owned Job elapsed | Peak process | Peak Job |
|---|---:|---:|---:|---:|
| Full sampling/export |4.907 s|6.312 s|182,935,552 B|183,771,136 B (175.26 MiB)|
| Fresh independent verification |2.766 s|4.125 s|143,351,808 B|144,183,296 B (137.50 MiB)|

Measured sampler-index construction **0.109 s**, sampling/staging **1.547 s**,
GLB export **0.610 s**, independent full geometry/colour readback **0.157 s**.
Lowest observed available Windows RAM **19,190,124,544 B**; lowest destination
free space **112,782,860,288 B**. Original 6 GiB process AND Job caps, 180-second
per-worker deadline, 12 GiB RAM and 2 GiB free-disk floors all passed. Both Jobs
settled empty naturally, exit 0, without forced cleanup; watched sources match.

The BIN is exactly **97,072,392 B**, JSON **1,520 B**, with **28 B** framing.
No full buffer copies or giant JSON source-ID lists were needed. Actual complete
NPYs and GLB fit within the admitted 256 MiB staging allowance, preserving the
2 GiB disk floor. The 1 GiB working projection was not treated as a measurement.

Tiny preparation tests exposed a test-module import collision and a real Windows
metadata distinction: in these fixtures, path `ctime` matched birth time while
descriptor `ctime` matched the later metadata-change time. The test now imports
its exact adapter file. The reader compares
shared device/inode/size/mtime/birthtime across APIs, retaining full timestamps
before/after within each API and all regular-file/link/reparse/content checks.
The mismatch and prior failed focused logs remain private. Final **21/21 tests
passed with zero skips**, including actual hardlink rejection; no full attempt
was consumed by those preparation fixes. No historical reader was modified.

## Evidence, preservation and stop

[Qualification and resource plan](amd-brick-full-sampled-glb-evidence/qualification-plan.md),
[focused tests](amd-brick-full-sampled-glb-evidence/focused-tests.txt),
[independent candidate review](amd-brick-full-sampled-glb-evidence/independent-candidate-review.json),
[full measured results and all material cohorts](amd-brick-full-sampled-glb-evidence/execution-results.json),
[independent outcome review](amd-brick-full-sampled-glb-evidence/independent-outcome.json)
and [preservation AFTER](amd-brick-full-sampled-glb-evidence/preservation-after.json)
contain source-safe evidence. Complete model/colour/support outputs, journals,
raw material ranges and exact input identities remain private under the output
root above and its sibling `verify` directory.

All 100 protected files, original geometry/material/neural artifacts, previous
accepted GLBs, 44 frozen trees / 1,187 files / 20 reports, selected environments,
AMD edits/backups, original cache trees and quarantine remain unchanged.
Checkpoint preservation read bounded headers only, zero weight payload.
Only new source code and textual evidence are published. Main and prior remote
heads are preserved; publication is checked separately after pushing this branch.

Known original boundary/non-manifold geometry remains intact; this asset is not
claimed repaired, watertight, simplified or game-ready. Shading without exported
normals and complete appearance have not been evaluated in a viewer.
**Next action:** separately qualify a bounded Blender import and visual
inspection. Stop here; do not launch Blender automatically.
