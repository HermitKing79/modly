# Closed-solid generation benchmark: prepared, execution HOLD

**Reference/ground truth PASS. Conditional handoff prepared. Separate human
approval is required before the one neural attempt.** No model, WSL, GPU,
conditioning forward, occupancy, shape or decode ran during preparation.

## One deterministic source

The explicit procedural polyhedron has nominal dimensions **1.00 × .65 × .80**
in normalized units, centred at the origin, and planar chamfers offset **.04**.
Six main faces, twelve chamfer strips and eight corner triangles yield
**24 vertices / 44 triangles**. No lid, holes, handles, attachments, damage,
text or decorations. No modifiers, smoothing, repair or borrowed asset geometry.

Independent PLY/NPY readback matches the generator exactly. It is one closed
convex component, Euler characteristic **2**, with zero boundary, non-manifold,
two-face winding, duplicate or degenerate triangles. Signed volume is
**.5125013251040768** after F32 storage, agreeing with the analytic chamfer-volume
formula within F32 rounding. Exact rational intersection tests of **all 946
triangle pairs** found no contacts beyond shared edges/vertices.

Private root:
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-closed-solid-benchmark-20261029-v1`.

| Artifact | SHA-256 |
| --- | --- |
| `source/mesh.ply` (1,080 bytes) | `871b4112924d97ff9049f45d1496bc840c95cb538a3c8086e3245d8c677a3f05` |
| `reference.png` (1024 RGBA, 479,201 bytes) | `fddf6f7de10a548271c3d80b2b813e3be1191f1636ae5f51e08d548c07117c8a` |
| `conditioning-512.png` | `cd6a3daa0aa5e053c71bde96db2ec2541817fa76dcd7e01110c9fa9e2c7e12dd` |
| Prepared input contract | `c5eee9373aeb8be783184bf3cf47b29fc48b50780bf4602c8640d1eeabb16edc` |

The ground-truth mesh is an oracle, **never an AI output or fallback**. Both
older misleadingly named `Pistol_*.glb` files are excluded.

## Reference and actual conditioning

One orthographic three-quarter view: camera **[2.4,-3.2,2.5]**, target origin,
scale **1.7**, +Z up. Blender **5.2.1**, Cycles **CPU**, four threads, 32 samples,
render seed 20261029, Standard/None colour transform, exposure0/gamma1. Matte
linear RGB .28, metallic0, roughness1; two neutral area lights. Full settings
and camera matrix are in `render.json`. Actual imported coordinates/triangles
and the identity object transform match before/after rendering.

The source has genuine semantic alpha. The qualified existing method makes
**black-premultiplied RGB**, then the existing extractor resizes to512 using
LANCZOS. No background-removal model is constructed. Source and 512 derivative
are the same view, not multi-view conditioning. The 512 pixel SHA-256 is
`b1e56471f02403045a7bfe6908cf24f1e385279bbcacf259247f7588d07631b3`.

The exact inherited crop is **[148,148,876,876]**, 728-square. Its tight
rounding removes the extreme outer rows at the top/bottom tips. This visible
limitation is retained, not hidden by changing preprocessing. Candidate renders
will use the same full camera and this frozen crop; they will not be refitted
or cropped according to their own silhouette. Chamfers and all three main
visible planes remain distinguishable in the actual512 image.

## Exact working generation route

Zombie D's preserved pipeline and decoder receipts both passed at
**`bc7160c7439a5d0ea9653e166620b6d535c498ce`**. All nine reused routing,
authentication, CPU-owner and occupancy/shape/decode/extraction files match that
working commit after newline normalization. The new adapter loads the D stage
implementation in a separate module namespace, binds Box paths/authority and
uses a fresh process. It does not isolate shared generic modules magically:
installing another adapter in the same process is explicitly rejected.

The proposal retains seed **20261013**, grid **512**, guidance **7.5**, the
qualified 12-step samplers, checkpoints, precision and AMD backends. One fresh
DINO forward; 22 occupancy-sampling forwards and one occupancy decode; 21 shape
forwards; one geometry decode. No A/C/D conditioning, occupancy or latent reuse.
Large-convolution tiling stays enabled, final-support override stays absent.

Admission remains three fresh **GlobalMemoryStatusEx ≥22 GiB** readings four
seconds apart, plus final22GiB checks before reservations including the final
decoder marker. **24 GiB is preferred only; live host floor remains12GiB.**
Original disk, GPU admission/monitoring, 13GiB initial free VRAM, 12GiB allocator
ceiling, Shape-SLat3GiB live free VRAM, deadlines and owned cleanup remain.
Support limits stay4,430 initial /1,000,000 intermediate /1,842,968 final rows,
with32,768 decoder tiles. Unexpected support/resource needs stop the run.

## Conditional one-shot handoff

Launcher:
`C:/Users/wattz/Documents/ChatGPT/modly/proof/scripts/compatibility/Invoke-ClosedSolidBenchmark.ps1`.

Default execution starts no process and reads no RAM. Live execution requires
all three explicit arguments: `-Run -ApproveOneNeuralAttempt -ExpectedCommit`
with the exact verified preparation commit supplied on delivery. The approval
switch is an explicit human acknowledgement, not proof of chat approval; do
not invoke it until separate approval. Use normal non-administrator PowerShell
after closing Codex, ChatGPT Desktop and browsers. No applications are closed
by the script; no installation, download or environment change is performed.

Fresh destinations:

- `artifacts/amd-closed-solid-generation-v1`: admissions, permits and result.
- `proof/docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/closed-solid-box-v1`:
  only two prepared occupancy input files exist now; future neural stages are
  exclusive. Guest decode run-ID is `run-closed-solid-box-v1` (guest absence
  remains a live check; WSL was not inspected during preparation).
- `artifacts/amd-closed-solid-raw-v1`: untouched extraction and CPU comparisons.

Any occupied output, changed source/hash/commit, failed gate or stage stops
later work. No automatic retry, fallback destination or raised limit.

## What the future experiment will establish

The CPU tail reuses complete current-input/ten-tensor authentication, then the
unchanged raw FDG extractor and exact PLY/NPY readback. It records components,
boundary/non-manifold edges, winding, duplicates, degeneracy, bounds, dimensions,
ratios and referenced-vertex Euler. Raw assets are checked before/after metrics.

One matched raw CPU view, silhouette overlay/contact sheet and IoU compare
against the actual512 reference. Shape distances use4,096 deterministic
area-stratified surface-point samples each way. They are **pose-sensitive
sampled point-cloud distances**, not exact surface/Hausdorff guarantees. No
registration, scaling, fitting or geometry movement occurs; single-image
pose/scale ambiguity must be distinguished from structural failure. Full AI
self-intersections and vertex-link manifoldness remain unassessed.

Missing parts, unintended openings/branches or inconsistent winding falsify
the intended clean-solid result. Clean counters alone do not qualify the
shape or texture readiness. Completion means evidence available for review,
not automatic geometry approval. No postprocessing, UV/PBR bake or texture step
follows. A CPU-tail resource failure preserves raw neural evidence and stops;
it does not authorise another neural attempt.

## Preparation verification and limits

The single reference render took **13.265s**, peak committed memory
**505,536,512 bytes**. Independent qualification took **1.312s /82,489,344 bytes**.
Minimum sampled available host RAM was **17,510,830,080 bytes**, above the
unchanged12GiB CPU floor. Both owned Jobs emptied naturally. Each stage retained
6GiB/180s process bounds and2GiB free disk; host changes are not attributed solely
to these workers. Future real raw extraction/analysis/rendering remains unrun.

The published authority pins eight source/image/qualification artifacts and
seven preparation receipts/exact executed-source archives. Reference creation
finished before CPU-tail additions; archived execution sources and unchanged
reference-core AST checks preserve that distinction without another render.
All **32** focused mocked/synthetic tests and no-execution PowerShell checks pass.
Independent review covers source, authority, ownership, route and limits.

Original assets, selected environment records, protected identities, historical
evidence and existing Git heads are preserved. Only safe code/text is published;
all PNG/PLY/NPY and raw telemetry remain local. No neural marker is reserved.
Preservation passed for3,346 historical hashes, all100 protected identities and
88 retained records,36 prior-phase inputs,69 scoped originals/reference/evidence
files and184 pre-existing local refs. No whole-installation byte audit is claimed.
**Stop here: neural execution remains HOLD pending separate approval.**
