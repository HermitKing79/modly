# Source-sampled vertex colours — bounded feasibility PASS

Approved base `ad619e54b80aef70d832b488adab41ddf69779d8`; isolated branch
`codex/amd-brick-resampled-colour`. One actual CPU probe completed naturally.
All original and position-holdout material gates passed. No full-model colour
array, GLB, Blender render, UV/bake, geometry change, model or GPU work followed.

## Confirmed cause

The original source uses its stable F32 sigmoid, then reconstructs each vertex as
`dual=F32(2)*sigmoid(geometry[:3])-F32(.5)` and
`v=(F32(coordsXYZ)+dual)*F32(1/512)-F32(.5)`.
The sparse material row belongs to the cell centre `coordsXYZ+.5`; the genuine
spatial sampler instead evaluates `q=(v+.5)*512`, with `floor(q-.5)`, eight
neighbours and renormalisation over found support. Matching coordinate-row
identity therefore preserves lineage without making a row colour the colour at
the displaced vertex. The pinned CPU sampler follows the original native source
semantics; this experiment makes no new native GPU bitwise-parity claim.

All **1,957,891** saved vertex positions were reconstructed bit-exactly in
32,768-row verification batches. Every sampled cell-centre control reproduced
its direct material row exactly. All **25,967** queried unique vertices differed
from their centres; maximum absolute axis displacement was **0.965012 grid units**.
Direct-row versus actual-vertex sampled RGB had RMSE **0.0113578**, p95 **0.0232001**,
maximum **0.1564743**. No saved positions were replaced by recomputed positions.

The original failure at face **2,045,193**, vertex **985,998**, saved query
**29,335** reproduced its **0.156474287** maximum exactly. Its displacement is
**1.181387 grid units** in length (**0.002307397 world units**). Sampling the
field at the actual stored vertex reproduces its old raw corner reference;
its corrected encoded RGB error is **4.26281e-9**.
All old corner raw targets agree exactly; IEC conversion/F32 rounding leaves
encoded corner errors below **1.91e-8**. That expected corner agreement proves
cause and mapping, not independent surface fidelity.

## Fixed-query and holdout evidence

Retained every original **56,910** query and source-spatial reference, including
16 regions and separate corner/centroid/interior cohorts. Sampled all six raw
channels once at each sorted unique original vertex, then clipped derived RGB,
applied original IEC sRGB decode, rounded linear COLOR_0 to F32, interpolated
linearly and IEC-encoded predictions. Explicit original-ID corner maps prevent
compact-table reordering or unsampled placeholders. No colours or factors were fitted.

Before any new source query or corrected prediction, froze **2,048** additional
interior positions: 64 area-weighted faces per original frozen region, two points
per face, seed **20261023**, independent PCG64 selection/interior streams.
Final F32 points had zero old/new or new/new collisions, with no redraw. This is
a **position holdout on previously selected faces**, not face-disjoint validation.
Every holdout region and each of its two interior streams is gated separately.

| Population / required metric rows | Worst RMSE | Worst p95 absolute | Maximum |
|---|---:|---:|---:|
| Original, 64 region/subcohort rows |0.00599343|0.01342099|0.04752671|
| Holdout, 48 region/subcohort rows |0.00543670|0.01192699|0.02546207|
| Unchanged limits |0.02|0.05|0.15|

All entries below PASS; complete all/corners/centroids/interiors and holdout
all/interior0/interior1 metrics are in the linked machine-readable evidence.

| Region | Original all RMSE | Original centroid RMSE | Original interior RMSE | Original max | Holdout all RMSE | Holdout max |
|---|---:|---:|---:|---:|---:|---:|
|global|0.001370|0.002077|0.001864|0.020829|0.001798|0.012001|
|brick|0.002072|0.003132|0.002825|0.018758|0.002785|0.013127|
|foliage|0.001502|0.002307|0.002027|0.020588|0.002495|0.019461|
|rear_x|0.001420|0.002227|0.001889|0.016947|0.002431|0.015199|
|backfacing_x|0.001662|0.002475|0.002285|0.028501|0.002877|0.024670|
|boundary|0.001387|0.002155|0.001856|0.022762|0.001898|0.010252|
|nonmanifold|0.001239|0.002009|0.001607|0.023872|0.003184|0.025462|
|high_variation|0.003873|0.005993|0.005201|0.047527|0.004823|0.021566|
|octant_0|0.001237|0.001906|0.001666|0.018964|0.001822|0.009254|
|octant_1|0.001299|0.001978|0.001762|0.017502|0.001970|0.019808|
|octant_2|0.001592|0.002376|0.002186|0.027399|0.001886|0.008490|
|octant_3|0.001545|0.002318|0.002115|0.033510|0.001662|0.012465|
|octant_4|0.001277|0.001885|0.001765|0.017888|0.001930|0.014316|
|octant_5|0.001321|0.002217|0.001667|0.034280|0.001509|0.006929|
|octant_6|0.001282|0.002049|0.001682|0.014149|0.002040|0.014227|
|octant_7|0.001616|0.002415|0.002218|0.024819|0.002176|0.013555|

Original fixed metallic **0.00011082693761409814**, roughness
**0.999995986777544** and OPAQUE alpha **1** passed in every cohort. Worst scalar
RMSEs across both populations were **0.000173724 / 0.000147159 / 0.000099976**,
below the unchanged **0.02** limit. No zero-support samples were discarded.
OPAQUE remains an approximation of this nearly opaque field. The geometric
rear/octant labels do not prove visibility or complete semantic segmentation.

## Measured resources and projected full-model cost

One sampler index, three calls: 25,967 actual vertices, 25,967 centre controls,
2,048 holdout references (**53,982 total evaluations**). Worker **2.203 s**,
owned Job **3.485 s**; peak process **193,679,360 B**, peak Job **194,519,040 B**
(**185.51 MiB**). Minimum sampled Windows available RAM **17,561,677,824 B**,
destination free **113,444,143,104 B**. Both unchanged live floors (12 GiB/2 GiB)
passed. Exit 0, naturally empty Job, no forced cleanup, watched sources unchanged.
Process and Job caps remain **6 GiB**, owned deadline **180 seconds**.

Full resampling is a projection, not another measurement. With one persistent
index and 32,768-row batches, original mapped input tables total approximately
206.7 MB, index/order tables 31.3 MB and a complete F32 RGB table 23,494,692 B.
Allow **1 GiB working memory** for sorting, batches, conversion, I/O and interpreter.
Measured index construction took 0.110 s; 25,967 vertex samples took 0.015 s.
Linear extrapolation is approximately 1.13 s of sampling, but the 15 ms timing
is near clock resolution and ignores full-data cache/I/O effects. A conservative
**30-second proposed resampling/conversion/readback budget** fits the existing
180-second cap; it is unmeasured and must retain real admission and cancellation.

POSITION + F32 COLOR_0 + uint32 triangle buffers require **97,072,392 B**:
23,494,692 +23,494,692 +50,083,008. A shared-vertex standard GLB should be about
**97.1 MB**, assuming no NORMAL/UV/texture buffers or vertex duplication. Full
writer admission, geometry readback and Blender import/render memory remain
separate gates; this probe does not qualify a large Blender run.

## Verification, preservation and next action

[13 focused tests](amd-brick-resampled-colour-evidence/focused-tests.txt),
[independent candidate review](amd-brick-resampled-colour-evidence/independent-candidate-review.json),
[all measured cohorts, source/array hashes and resource samples](amd-brick-resampled-colour-evidence/execution-results.json),
[independent outcome review](amd-brick-resampled-colour-evidence/independent-outcome.json)
and [preservation AFTER](amd-brick-resampled-colour-evidence/preservation-after.json)
are public source-safe evidence. One independently owned saved-data verifier
reopened all arrays, replayed the frozen draws and reproduced all 112 metric rows.
Its separate F64 field oracle at the existing 25,967 vertex positions, 2,048
holdout positions and corrected worst original query differed by at
most **2.85719e-7**, below the unchanged **2e-6** oracle tolerance. It introduced
no new query locations or colour candidate. Private arrays and receipts remain at
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-resampled-colour-ad619e5\feasibility`.
The compact `subset_linear_COLOR0.npy` is bound to `unique-original-vertex-ids.npy`;
it is not a complete full-model colour table. Original material/geometry, previous
accepted GLBs, all 100 protected files, 42 frozen trees/1,132 files/19 reports,
environments, AMD backups, original caches and quarantine remain unchanged.
Checkpoint preservation read headers only, zero weight payload. Models/images
and raw geometry/material arrays stay private.

**Next action:** separately approve one full-model source-resampled vertex-colour
GLB export, with unchanged geometry, fixed material factors and independently
verified output. This bounded result is not exhaustive surface/visual equivalence;
full export/import admission still applies. Stop here without starting the export.
