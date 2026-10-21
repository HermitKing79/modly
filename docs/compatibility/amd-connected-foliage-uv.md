# Connected-chart foliage feasibility — BLOCKED before unwrap

Base `bad7ba4e9e8c3b14539d02a0c3930080891e0c56`; isolated branch
`codex/amd-connected-foliage-uv`. One bounded CPU capacity worker, no ROI
reselection, Blender launch, UV solve, bake, export, model or GPU execution.

## Authentication and cheapest capacity test

Reused the exact saved **4,531 original triangles / 3,177 vertices**. All source
IDs, oriented corner positions, winding, multiplicity and selected learned PBR
rows were checked against unchanged originals. The complete original geometry
and material coordinate arrays match, not merely their row counts.

Total original triangle area is **0.005950395632805763 world²**. Requiring both
UV Jacobian singular values to be at least 4,096 implies at least
**99,831.072817 UV texels²** (9.52% of 1024²), counting every original face.
There are 83 source-edge components, 219 non-manifold edges and 2,308 previously
verified extraction-induced boundaries. Surface area alone does not rule out
an atlas; the former 1,127-chart / 6,929,408-texel result is not a universal bound.
Independent lower bounds are combined with **max**, not added twice.

## One proposed connected-chart method and actual stop

The selected method was installed Blender CONFORMAL, with explicit deterministic
disk cuts. Only incidence-two original edges whose two faces traverse the edge
in opposite directions are eligible for continuous gluing. A maximum-length
dual spanning forest retains those connections; every other adjacency is cut.
Disposable corner duplication preserves every original oriented triangle bit.
Each resulting chart is validated as a disk with one boundary and Euler characteristic 1.

The real selection contains **1,076 orientation-inconsistent incidence-two
edges**, in addition to its 219 non-manifold edges. There are 4,153 eligible edges;
the forest retains 3,716 and cuts 437 cycle edges without changing the eligible
graph's connected-component count. This produces **815 disk charts**, including
330 single-triangle charts; largest chart 1,516 triangles. Seam-only duplication
uses 6,161 vertices and preserves all 4,531 original oriented triangles.

**815 exceeds the unchanged 256-chart limit.** Under this candidate's conservative
32-aligned owner/content-offset policy, even 64×64 per chart requires
**3,338,240 texels — 3.1836× atlas capacity**, before actual conformal distortion,
rectangle extents or packing waste. This is a policy-conditional minimum, not a
globally optimal mip-padding theorem. The worker stops here, before Blender.
No final UVs, occupied texels, actual distortion, overlaps, raster coverage,
padding waste or LOD filtering results exist. Planned solver/packing code was
not developed further after this refusal.

For **edge-connected disk charts** with affine nondegenerate positive-oriented
UV triangles and exact no-overlap, there is a stronger conditional obstruction:
same-directed shared edges cannot be glued continuously without local overlap.
The 219 non-manifold edges have 674 face-edge occurrences. Even allowing optimistic
opposite-direction face pairing, they can add at most
`sum(floor(k/2)) <= sum(k-2) = 674 - 2*219 = 236` joining adjacencies.
Thus at least **815−236 = 579 edge-connected charts** remain, exceeding 256.
This does not cover pinched vertex-only charts, grouping disconnected charts
under one owner ID, mirrored/negative UV faces or discontinuous edge mappings.
The exact theorem also must not be confused with a numerical overlap-tolerance
test. The nominated 815-chart refusal stands independently of that broader proof.

## Measurements, verification and preservation

Capacity worker **1.078s**; supervisor elapsed **2.485s**. Peak process commit
**130,895,872B (124.832MiB)** under 6 GiB / 180s. Both live samples pass: minimum
Windows available RAM **19,042,652,160B** (12 GiB floor), destination free
**104,905,805,824B** (2 GiB floor). Exit 0, naturally empty owned Job, no forced
cleanup; all 67 watched source/input identities unchanged.

[Eight focused CPU tests](amd-connected-foliage-uv-evidence/forest-tests.txt),
[analytic review](amd-connected-foliage-uv-evidence/independent-capacity-review.json),
[exact capacity candidate review](amd-connected-foliage-uv-evidence/capacity-candidate-review.json)
and [actual measurements](amd-connected-foliage-uv-evidence/execution-results.json)
are public source-safe evidence. [Independent actual readback](amd-connected-foliage-uv-evidence/independent-capacity-outcome.json)
and the [conditional chart-count proof](amd-connected-foliage-uv-evidence/independent-chart-lower-bound.json)
verify the saved candidate and its exact limitations. Private disk-chart arrays and raw receipts:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-connected-foliage-bad7ba4\capacity`.
No private geometry, material values or images are published.

[Preservation AFTER](amd-connected-foliage-uv-evidence/preservation-after.json)
matches 100 protected files, 34 frozen trees / 920 files / 15 reports, selected installed
environment identities, AMD edits/backups, old caches and quarantine. 1,405 fresh
full hashes; checkpoint checks are bounded stat/header-only, zero weight payload.
Three old disposable fixture copies retain explicitly labelled stat-only checks.

**Cheapest next action:** separately qualify one multi-atlas/material-group
representation of these same saved disk charts. The 815-chart candidate needs
at least **four groups** at 256 charts per atlas; actual UV and packing capacity
remain unproven. Do not repeat single-atlas nominations or begin baking.
