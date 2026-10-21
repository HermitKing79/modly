# Full-model vertex-colour export — BLOCKED at material suitability

Approved base `1012a6c04976085c5cf8d4c6911d9010d0e8cc59`; isolated branch
`codex/amd-brick-full-colour-glb`.

The single CPU phase-one measurement completed and released naturally. **The
full-model colour approximation failed its unchanged quality gates.** No large
export admission, GLB writer, Blender import or render was attempted. No retries,
alternative representation, neural work, geometry changes or new bake followed.

## Actual full-model comparison

Authenticated all original **1,957,891 vertices / 4,173,584 triangles**, mesh/PBR
hashes, full coordinate-row correspondence and prior nominated source IDs.
The unchanged eight-neighbour sampler used original raw F32 material features.
The candidate used the qualified IEC decode → F32 linear vertex RGB → barycentric
interpolation → IEC encode contract. Comparison targets were clipped only after
source-spatial sampling. No colours or constants were fitted.

**56,910 frozen queries**, 16 separately gated regions, seed 20261022: three
corners, centroid and two seeded interior positions per selected face. All pools,
face IDs and query positions were saved before the first error query. Every
region also gates its corners, centroids and interiors separately. All populations
met their declared counts; zero unsupported reference samples.

| Region | Queries | Encoded RGB RMSE | p95 absolute | Maximum | Decision |
|---|---:|---:|---:|---:|---|
| Whole-surface area sample |6,144|0.008063|0.014362|0.135029|PASS|
| All 781 nominated brick faces |4,686|0.006394|0.013241|0.055553|PASS|
| Nominated leaf/vine surface |3,072|0.006590|0.014620|0.066873|PASS|
| Rear X depth quartile |3,072|0.009269|0.017469|0.122478|PASS|
| Rear-half negative-X normals |3,072|0.008902|0.018805|0.092227|PASS|
| Natural boundary incidence |3,072|0.007096|0.015370|0.071416|PASS|
| Non-manifold edge incidence |3,072|0.006372|0.013788|0.098481|PASS|
| High input RGB variation |6,144|0.019817|0.043969|**0.156474**|**BLOCKED**|
| Octant 0 |3,072|0.008779|0.016695|0.081141|PASS|
| Octant 1 |3,072|0.005598|0.011617|0.071493|PASS|
| Octant 2 |3,072|0.009312|0.018717|0.097647|PASS|
| Octant 3 |3,072|0.006393|0.012748|0.090240|PASS|
| Octant 4 |3,072|0.007229|0.014310|0.083771|PASS|
| Octant 5 |3,072|0.007786|0.014949|0.081524|PASS|
| Octant 6 |3,072|0.010942|0.018648|0.131919|PASS|
| Octant 7 |3,072|0.010562|0.019538|0.109188|PASS|

Limits remain **0.02 RMSE / 0.05 p95 / 0.15 maximum**. High-variation corners
also fail all three: **0.022842914 / 0.052302825 / 0.156474287**. Its centroid and
interior cohorts pass; their success does not override its corner/max failures.

Fixed metallic `0.00011082693761409814`, roughness `0.999995986777544` and actual
OPAQUE alpha 1 pass separately in every region/subcohort. Worst scalar RMSEs:
**0.000173724 / 0.000091058 / 0.000099976**, below 0.02. OPAQUE remains an
approximation of this measured nearly opaque field, without arbitrary cutouts.

## Located limitation and next action

The high-variation pool is the input-only top ceil(F/100) = **41,736** original
faces by clipped encoded corner-RGB span. It contributed 1,024 area-weighted
unique faces, with stable source-ID tie rules; no observed error drove selection.

The maximum occurs at saved query **29,335**, original face **2,045,193**, corner
1, original vertex **985,998**, red channel. One sampled location exceeds 0.15.
Its accepted +X camera projection is approximately **(476.24, 613.99)** in the
1024² preview; projection alone does not establish visibility or semantic class.
Private `worst-location.json` retains the source point and compared values.
Independent original-ID readback places it in geometric octant 1, outside both
nominated patches. A separate eight-neighbour F64 tuple oracle at this existing
location differs from the stored source sample by **3.12e-8**, below 2e-6; no new
query locations or SparseSampler calls were used for this verification.

These sharp spatial-colour transitions need source-spatial RGB capture to meet
the declared fidelity requirement. The measured restriction concerns the fixed
direct-row vertex-colour candidate; it does not prove every possible vertex fit
fails. The other 15 sampled regions passed; this is not exhaustive full-surface
or complete brick/foliage segmentation evidence. Rear labels are geometric,
without a visibility claim.

**Single next action:** review the located high-variation surface and separately
approve a genuine spatial PBR bake there before revisiting a complete export.
Do not re-export unchanged vertex colours or relax the failed gates.

## Measured resources, evidence and preservation

Worker **4.922 seconds**, supervised owned execution **6.016 seconds**. Peak
process **556,781,568 B (530.99 MiB)**; peak Job **557,608,960 B (531.78 MiB)**.
Lowest sampled available RAM **17,729,110,016 B** (12 GiB floor), destination
free **113,543,884,800 B** (2 GiB floor). Process and Job caps stayed 6 GiB with
the original 180-second deadline. Exit 0, naturally empty Job, no forced cleanup,
and all watched input identities unchanged. CPU execution PASS records completed
measurement only; the mission remains **BLOCKED**.

[15 focused tests](amd-brick-full-colour-glb-evidence/focused-tests.txt),
[independent candidate review](amd-brick-full-colour-glb-evidence/independent-candidate-review.json),
[per-region results and private-array hashes](amd-brick-full-colour-glb-evidence/execution-results.json),
[independent outcome](amd-brick-full-colour-glb-evidence/independent-outcome.json)
and [preservation AFTER](amd-brick-full-colour-glb-evidence/preservation-after.json)
are source-safe evidence.

Private evidence root:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-full-colour-glb-1012a6c`.
`suitability\result.json`, `frozen-sampling.json`, original face nominations and
query/reference/prediction/support arrays remain local. No new model/image/GLB
artifact is published. Before/after audit preserves all 100 protected files,
40 frozen trees / 1,072 files / 18 reports, selected environments, AMD backups,
historical caches and quarantine. Checkpoint checks read headers only, zero
weight payload. Existing accepted GLBs and original generated assets are intact.
