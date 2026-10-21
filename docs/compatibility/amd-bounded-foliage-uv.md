# Bounded foliage selection PASS; UV feasibility BLOCKED

Base `64d05aa0d40073778f316e788d5ebe595b44495f`; isolated branch
`codex/amd-bounded-foliage-uv`. Exactly one real CPU selection and one unchanged
dominant-normal-axis chart candidate. No alternate ROI, density search, bake,
export, Blender, WSL, model or GPU execution.

## Original surface selection

The authenticated +X EV2 image was visually nominated **before triangle counting**:
ROI **[346,393]–[393,436]**, inside the previous [332,384]–[400,448] ROI.
The byte-pinned original selector retained complete nearest-visible original
faces plus one original-edge neighbour ring inside that same ROI. No triangle
clipping, support truncation, repair, deletion or coordinate changes.

**4,531 triangles / 3,177 vertices** pass the unchanged 6,000-face input bound.
1,668 nearest-visible complete faces seed the ring. All original face/vertex IDs,
oriented corner bits, multiplicity and genuine material-row mapping are retained.
Full original geometry/material coordinate arrays match; saved selected arrays
were byte-exactly reopened. The private overlay shows orange leaf and adjacent
curved-vine **portions**, not a complete closed botanical part. The manually
nominated leaf/vine windows have 685/728 and 478/480 projected-footprint pixels;
these are coverage proxies, not semantic or nearest-depth segmentations.

There are **83 edge-connected / 34 vertex-connected face components**. Selected
edge incidences: 2,461 single, 5,229 double, 219 greater-than-two. Comparing selected
edges with the complete original mesh distinguishes **2,308 extraction-induced
boundaries** from 153 original boundary edges; zero unmatched edges. Extraction
boundaries include visibility/ring selection as well as the rectangular crop.
Fragmentation is measured, not repaired or inferred to be intentional foliage.

## Exact UV admission failure

Dominant projection axes contain 2,072 / 778 / 1,681 original faces. Fixed projected
density is 7,094.487202282228 texels/world-unit, with the unchanged 4,096 actual
tangent-density floor. The source-pinned chart construction uses 6,399 clipping
tests of 250,000 allowed and produces **1,127 islands**, exceeding 256.

The prescribed 32-aligned LOD5 owner rectangles require **6,929,408 texels**:
**6.6084×** the 1,048,576-texel 1024² atlas. Even their 64×64 minimum alone requires
4,616,192 texels (4.4023× capacity). No rectangle individually exceeds 1024;
the aggregate required area and island count are decisive for this candidate.
Projected triangle area itself is 248,448.63 texels², illustrating the cost of
this fragmented chart partition and its mip ownership, not just surface area.

Admission stops **before shelf packing, final UVs, overlap/raster/texel-coverage
validation or final F64/F32 LOD testing**. There are no qualified final UVs or
foliage material-quality results. The unchanged analytical filter contract covers
continuous LOD0–5 including half levels, but cannot be certified for an atlas
that does not fit. This does **not** prove every possible foliage unwrap fails.

## Execution and verification

Worker 6.172s; supervisor elapsed 7.281s. Peak process commit **244,514,816B
(233.1875MiB)** under 6 GiB / 180s. Seven live samples all pass: minimum available
RAM **20,087,750,656B** (floor 12 GiB), destination free **105,519,439,872B**
(floor 2 GiB). Exit 0, naturally empty owned Job, no forced cleanup; all 60 watched
source/input identities unchanged. The experiment verdict remains BLOCKED.

[25 focused CPU tests](amd-bounded-foliage-uv-evidence/focused-tests.txt),
[exact candidate review](amd-bounded-foliage-uv-evidence/independent-candidate-review.json)
and [actual numeric evidence](amd-bounded-foliage-uv-evidence/execution-results.json)
are source-safe. [Independent outcome verification](amd-bounded-foliage-uv-evidence/independent-outcome.json)
confirms the saved geometry/material correspondence and exact capacity blocker.
Private output root:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-bounded-foliage-64d05aa\run`.
`selection.png`: 355,657B, SHA256
`bc9d90f854b9440a0f3781e1fefcbdb206f4d617a160de4c0596ed188e8b9489`.
Original selected arrays, source IDs, direct material rows and detailed receipts
remain private there; no generated geometry or images enter publicGit.

[Preservation AFTER](amd-bounded-foliage-uv-evidence/preservation-after.json)
matches the BEFORE scope: 100 protected files, 32 frozen trees / 883 files / 14 reports,
original caches, prior quarantine, selected environment/AMD edits/backups and
source closures. 1,367 full hashes; checkpoint identity checks are explicitly
stat/header-only (zero weight-payload reads), and three old disposable fixture
copies retain labelled stat-only checks. No new historical bytecode entries.

**Single cheapest next action:** separately qualify one connected-chart UV
unwrap candidate for this exact saved 4,531-face subset, targeting fewer charts under the
same density, 1024² and LOD5 gates. Reuse this selection; do not rerun image
selection or bake an unqualified atlas. Stop for review.
