# Source-sampled vertex colours — one bounded CPU feasibility probe

Approved base ad619e54b80aef70d832b488adab41ddf69779d8; isolated branch
codex/amd-brick-resampled-colour. No export, Blender, UV, bake, geometry edit,
model/native/GPU/WSL work or changed historical source.

Authenticate the full original mesh/geometry/six-channel material field, complete
coordinate-row lineage, previous fixed 56,910 queries and all 24 saved arrays.
Verify all original vertex positions bit-exactly using the pinned dual-grid
formula, in 32,768-row batches. Use stored original vertices for every query.

Cell-centre control samples at grid coordinates+.5 must equal direct material
rows. Actual vertex samples use the unchanged qualified sampler at original F32
world positions, including partial-support renormalisation. Reproduce all old
direct-row predictions, every old corner target and the exact worst case at
face 2045193 / vertex 985998 / saved query 29335. Measure displacement and RGB
differences; expected corner agreement is a cause/mapping check, not independent
surface validation. If this contract fails, stop, without retries.

Before any new source query or corrected prediction, freeze a position holdout:
64 area-weighted unique faces per previously frozen region, 16 regions, seed
20261023 with independent PCG64 SeedSequence([seed,region_index,0]) selection
and stream1 interiors. Two strictly interior uniform-area positions per face;
2,048 total. Selected source IDs sorted ascending. This is a new-position
holdout on previously fixed faces, not face-disjoint validation. Reject exact
final F32 world collisions with any old point or other holdout point; no redraw.
No error-driven selection, colour fitting, sample deletion or gate adjustment.

Sample once each sorted unique ORIGINAL vertex on the old selected face union:
at most 28,455 vertices. Every holdout face is from that frozen union. Keep an
explicit original triangle-corner-to-colour-row map; no full-size placeholder
colour table. Retain all six raw sample channels and support evidence privately.
Clip derived encoded RGB only after sampling, apply original IEC sRGB decode,
round to F32 linear COLOR_0, then original linear barycentric interpolation and
IEC encoding. No material-value fitting or decoding sparse RGB before sampling.

Reevaluate all 16 old regions, separately all/corners/centroids/interiors, using
the pinned unchanged decision implementation. Independently evaluate every new
holdout region, separately all/interior0/interior1. RGB RMSE <=.02, p95 absolute
<=.05, maximum <=.15; each scalar RMSE <=.02. Unchanged fixed metallic
0.00011082693761409814, roughness0.999995986777544, OPAQUE alpha1. Any unsupported
required source sample or missing cohort blocks; all samples remain in evidence.

One original sampler index, three bounded calls <=65,536 each: N actual vertex
samples, N cell-centre control samples, 2,048 new holdout reference samples.
Total new source evaluations <=2*28,455+2,048 =58,958; original input row cap2M.
Working inputs/indexes/geometry/control/reference/prediction tables are projected
under1 GiB; record actual peaks. Reuse unchanged process AND Job6GiB ceiling,
180-second owned deadline, live12GiB available RAM and2GiB destination-free floor,
binary GO, exclusive paths, monitoring and natural empty Job acceptance.

Focused synthetic checks and independent source review precede one real worker.
Save/readback all private NPY arrays and source identities; independently verify
the saved result without generating another candidate or query population.
PASS only if old and holdout gates all pass plus preservation/containment pass.
Quality failure is NO-GO for this fixed candidate. Missing causal/runtime proof
is BLOCKED. A completed measurement PASS must never imply feasibility PASS.
If PASS, give projected full-resampling CPU/memory/GLB cost and recommend a
separately approved full export. If NO-GO, preserve the failure and identify a
targeted genuine spatial bake as the next candidate, without implementing it.
Publish source-safe text/code/hashes only; private arrays/assets remain local.
