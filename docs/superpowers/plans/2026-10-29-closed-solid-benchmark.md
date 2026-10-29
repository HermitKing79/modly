# Closed-solid benchmark preparation plan

> Execute inline with the executing-plans workflow; independent final review.

**Goal:** One deterministic CPU reference/ground truth and one conditional,
fixed-seed 512 neural handoff; no neural execution during preparation.

**Design:** Explicit convex chamfered cuboid: dimensions 1/.65/.8, offset .04,
24 vertices, 44 oriented triangles. One orthographic three-quarter CPU Cycles
view, matte grey, real semantic alpha. Freeze the qualified alpha-only crop and
512 LANCZOS RGB input (black background). Ground truth cannot satisfy AI gates.

**Files:** `amd_closed_solid_benchmark.py` (fixed specification/geometry),
`amd_closed_solid_render.py` (owned CPU reference or same-camera raw view),
`amd_closed_solid_cpu.py` (existing CPU ownership + qualification/extraction/
measurement), `amd_closed_solid_handoff.py` and `amd_closed_solid_stages.py`
(case authority/route adapters), one PowerShell entrypoint, focused tests,
published reference contract and concise report. Raw assets remain local.

**Reuse:** D memory/admission helpers; private-instance D stage adapter;
original occupancy/shape/decode and ten-tensor authenticator; original CPU
Job supervisor, raw extractor, topology counter and exact intersection helper.
Do not alter those existing files or install dependencies.

## Steps and checks

- [x] Test-first deterministic geometry: 24/44, Euler2, one closed component,
  opposite directed incidences, positive volume, supporting-plane convexity.
- [x] Build one reference under existing 6GiB/180s/12GiB-host/2GiB-disk CPU
  limits, four threads; independently read PLY/NPY and all triangle pairs.
- [x] Reproduce actual CPU image preparation and exact 512 pixels, crop and
  source/contract hashes. Stage only two fresh private occupancy input files.
- [x] Test-first conditional handoff: no-work default; three22GiB readings4s,
  final22GiB gates,24preferred,12live; exact source/commit; exclusive paths;
  stage-stop/no-retry; unchanged qualified neural budgets and decode bounds.
- [x] Route original stages in a fresh process only. Pin every reused/new
  source. Use a unique decode run-ID and CPU evidence root. Never install D/C
  and Box adapters together. Explicit memory-reader binding in each worker.
- [x] Future CPU tail authenticates ten tensors and current-image provenance,
  extracts untouched raw mesh, measures topology/winding, renders one matched
  view with frozen GT framing, measures silhouette and sampled shape distance.
  Report unregistered pose sensitivity and unassessed full-AI intersections;
  never promote geometry from counters alone or automatically postprocess.
- [x] Mock/regression tests and no-execution PowerShell dry-run; independent
  review, preservation verification and isolated publication. Execution HOLD
  pending separate human approval; no attempt marker is created now.

No speculative repair, remesh, texture/PBR step, model import, WSL or GPU work.
Retain all existing neural RAM/VRAM/disk/allocator/deadline/support/cleanup gates.
