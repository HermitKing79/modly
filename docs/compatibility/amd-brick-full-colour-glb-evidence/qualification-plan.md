# Full-model vertex-colour feasibility — phase one first

Approved base 1012a6c04976085c5cf8d4c6911d9010d0e8cc59; isolated branch
codex/amd-brick-full-colour-glb. No large export or Blender run until material
suitability passes. A completed CPU measurement is not overall mission PASS.

Authenticate original 1,957,891 F32 vertices, 4,173,584 I32 triangles, complete
six-channel learned field and original mapping receipt. Verify every geometry
and material sparse coordinate row in batches. Reuse source-pinned IEC sRGB to
F32 linear COLOR_0 conversion and the unchanged sparse eight-neighbour sampler:
raw field values are sampled first; clip only derived comparison targets.

All 16 original-face pools and selected query positions are frozen before any
material-error query. Seed 20261022, PCG64 independent SeedSequence streams per
cohort. Area-weighted unique faces, sorted original IDs, underfill recorded.

| Input-derived pool | Maximum selected faces |
|---|---:|
| Entire surface |1,024|
| Previously nominated original brick patch |all 781|
| Previously nominated leaf/vine patch |512 of 4,531|
| Rear X depth quartile, inclusive ties |512|
| Negative X normal and below median X depth |512|
| Original incidence-one boundary |512|
| Original incidence-greater-than-two edges |512|
| Top ceil(F/100) input corner-RGB spans |1,024|
| Eight original AABB octants |512 each|

Brick/foliage are authenticated visual nominations, not full semantic labels.
Rear/backfacing are geometric predicates, not visibility or occlusion proofs.
High variation uses original encoded RGB only, never observed error or adaptive
refilling. Exact edge incidence is recomputed from original indices in memory,
with no historical cache write or trust in an unpinned cache.

Six queries per selected face: all three original corners, centroid, two seeded
strictly interior uniform-area barycentric points. Maximum 56,910 total queries,
hard sampler bound 65,536. No source faces are modified or clipped.

For each region separately, and its corners/centroids/interiors separately:
encoded RGB RMSE <=0.02, p95 absolute <=0.05, maximum <=0.15. Each scalar RMSE
<=0.02, using fixed metallic 0.00011082693761409814, roughness 0.999995986777544
and actual OPAQUE alpha 1. No factor/colour refitting, pooled rescue, unsupported
reference removal or changed gate. Empty required cohorts or zero reference
support block qualification and remain visible in results.

Largest topology edge array: 3*4,173,584*8 =100,166,016 bytes. Face gathers use
32,768 rows. Conservative planned NumPy data/temporary budget 1.5 GiB including
sorting/statistics/field/index/linear-colour work, below the unchanged enforced
6 GiB process AND owned-Job ceiling. This is a projection; record actual peaks.
New private evidence needs less than 64 MiB unless qualified full linear colours
are saved (additional23.5 MB). No large GLB memory admission is asserted here.

Reuse unchanged qualified owned supervisor and Windows resource monitor through
a fixed Python-only adapter: exact binary GO after Job attachment, 180-second
owned deadline, 12 GiB live available RAM and 2 GiB destination-free floors,
exclusive suitability output/marker, natural empty Job and source preservation.
The shared internal stage token is named export, but this worker performs only
material measurement; no GLB writer or Blender stage is exposed by this adapter.

Focused synthetic tests and independent source review precede the one real CPU
measurement. Save selected source IDs, queries, reference/prediction/support
arrays and hashes privately, with byte-exact readback. Publish text measurements
only. If any required material region fails, classify the mission BLOCKED and
stop before export admission, export, import or rendering. If all pass, separately
complete reviewed full-export admission within this same authorised mission.
