# AMD sparse-decoder forward timeout investigation

Date: 2026-10-12. Base: `7192c236fb0aa7559b71740a4995e8623769b821`.
Branch: `codex/amd-forward-timeout-investigation`.

**Classification: PARTIAL.** Native compilation is a credible contributor to the
timeout. The exact stalled operation and successful decoder execution remain
unproved. This investigation ran no GPU operation, AI import, model load or
inference. It adds narrowly scoped diagnostics, without enabling another attempt.

## What the existing attempt proves

The [published report](amd-sparse-decoder-forward-guard-correction.md),
[raw log](amd-sparse-decoder-forward-correction-evidence/run/worker.stdout.txt)
and [process result](amd-sparse-decoder-forward-correction-evidence/run/results.json)
show strict CPU loading, placement and input transfer completed before
`single-forward`. The **30-second budget covered GPU metadata, placement, input,
forward and release**, rather than 30 seconds of measured forward execution.
The supervisor received no final result and forcibly terminated root PID 3232
and child PID 2896. Exit code 1 came from termination; it is not evidence of a
native crash. Stderr was empty and all three historical Windows observations
were healthy. No output, final call count, forward timing or release/VRAM result
was saved. The raw `NOT ATTEMPTED` placeholder is contradicted by the durable
stage log; the historical evidence is preserved unchanged.

## Compiler artifacts, inspected without execution

The [read-only audit](amd-forward-timeout-investigation-evidence/cache-audit.json)
SHA-verified all **524 files / 669,452,910 bytes** against the historical manifest,
both before and after inspection. SQLite was opened with `mode=ro`, `immutable=1`
and `query_only`; two small compressed blobs were decoded as inert bytes.

| Evidence | Observation | Limit |
| --- | --- | --- |
| COMGR directory: 516 files | 515 serialized entries plus an empty timestamp; 514 ELF payloads and one LLVM bitcode payload; captured compiler logs empty | Not 516 convolutions or forwards |
| ELF headers | AMDGPU machine 224; flags `0x4e` identify gfx1201 | Does not prove executable validity |
| Entry modification times | 18:52:45.285653–18:53:12.185375 UTC; span **26.899722 seconds** | Wall-clock file activity, not measured compiler CPU time |
| MIOpen `gfx1201_32.ukdb`, 86,016 bytes | SQLite check `ok`; two FP32 gfx1201 records: `naive_conv.cpp.obj` and `MIOpenIm3d2Col.cpp.obj` | No per-operation association or full warm-cache proof |
| Temporary `comgr-2896-516-d750b7` | PID matches killed child; CCOB input and nonempty gfx1201 ELF output; three empty outputs and two empty MIOpen lock files | Child executable, command line and CPU counters were not captured |

AMD documents that a MIOpen kernel-cache miss can require compilation, increasing
first-use startup time. COMGR's source describes the inspected length-prefixed
payload/log serialization and local LLVM cache. These facts plus the compilation
artifacts make cold convolution setup **plausible**, but cannot distinguish
compilation, solver selection, GPU work or a subsequent stall in this attempt.
The decoder's initial convolution is FP32, consistent with the two database
records; FP32 also occurs later, so attribution to the first convolution is not
proved. See [MIOpen cache documentation](https://rocmdocs.amd.com/projects/MIOpen/en/latest/conceptual/cache.html),
[COMGR cache source](https://raw.githubusercontent.com/ROCm/llvm-project/amd-staging/amd/comgr/src/comgr-cache.cpp)
and [LLVM architecture flags](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/include/llvm/BinaryFormat/ELF.h).
These sources explain formats/mechanisms; they are not a byte-matched source
verification of the installed runtime.

## Cache policy and the small instrumentation change

`scratch_environment()` redirects COMGR, MIOpen user/cache, HIP and other caches
into a fresh exclusive scratch directory. Each attempt therefore starts with
empty **private user caches**; system caches need not be cold. `benchmark=False`
and the Triton prohibition do not establish that MIOpen avoids compilation.

A future reuse should use **hash-verified writable copies** of the COMGR entries
and MIOpen kernel database in fresh scratch, with unchanged runtime, native
libraries, source/checkpoint identity, gfx1201 target and deterministic settings.
Never point the runtime at historical directories. Exclude temporary products,
lock files and incomplete outputs. Let native cache-key checks accept or reject
entries; do not alter their keys. Copying or executing caches was **not** done
here. A structurally valid cache is not guaranteed compatible or complete.

The harness now emits durable host entry/return/error markers for the root,
Conv3d, residual and upsampling modules, major decoder sequences, dispatched
convolutions and the existing final synchronization. Markers include elapsed
times and input metadata. Returns mean **host return**, not synchronized GPU
completion. Existing VRAM queries are durably emitted; no per-layer GPU query,
synchronization, extra forward or computation allowance was added.

The supervisor samples owned-process identities and cumulative/delta CPU time
every two seconds using already-installed host `psutil 7.2.2`. OS sampling runs
in a daemon with bounded startup, freshness and shutdown. Journal writes run in
a separate daemon with deadline-bounded durable acknowledgments; a stall/error
triggers owned-worker cleanup and prevents PASS. CPU authorization is rechecked
against its deadline immediately before sending the GPU permit. Ordinary root
exit is handled; root PID reuse and permission failures fail closed. Cache-audit output
is confined to this investigation's evidence directory.

The original **45 / 30 / 90-second** limits, transfer guards, one-forward permit,
Triton prohibition and three Windows observer gates remain. The old canonical
run directory/attempt marker remain consumed. No new GPU execution path, cache
copy policy or longer deadline was activated.

## Verification and preservation

All **48 focused CPU tests** pass: 13 diagnostics, 13 forward/supervisor,
13 transfer and 9 loading-contract tests. New negative cases first failed, then
passed after correction: stalled/failed durable writes, late CPU authorization,
root-exit race and historical-output rejection. Only a disposable sleeping CPU
process was launched for process-counter testing. See
[final CPU results](amd-forward-timeout-investigation-evidence/cpu-review-green.json)
and [independent review](amd-forward-timeout-investigation-evidence/review.md).
No broader suite or successful GPU/IPC proof was repeated.

[Preservation evidence](amd-forward-timeout-investigation-evidence/preservation.json)
records byte-identical installed dependency inventories, virtual-environment
and model/cache metadata, historical reports/logs/caches, preserved AMD source
HEAD and `nodes.py`/backup hashes, and unchanged historical local/remote branch
heads. Production workers and both installed applications were untouched.

## One recommended next experiment — separate approval required

Authorize **one instrumented forward of the same strictly loaded sparse decoder
and existing `[1,8,1,1,1]` float32 fixture**, using verified private cache copies.
A **120-second GPU-stage ceiling**, including placement, forward and release,
is defensible because cache timestamps span 26.9 seconds, close to the old
30-second GPU-stage allowance;
it is not a prediction of completion. Retain a 45-second CPU gate, a 180-second
worker ceiling, bounded cleanup, no warm-up and no retry. Publish that exact
harness before execution into a new exclusive attempt/evidence directory.

Before authorization, require healthy Windows monitoring and verified runtime,
source/checkpoint/cache identities. For the longer window, require bounded
passive Windows event/controller checks at most ten seconds apart, in addition
to the implemented CPU heartbeat; current three-gate observation alone is not
continuous monitoring. Stop on unhealthy/stale monitoring, guard failure,
native error/reset, timeout or abnormal exit. Record the last acknowledged
operation, owned child CPU activity, available/allocated/reserved/peak VRAM,
validated `[1,1,4,4,4]` finite float32 output and owned-tree exit/cleanup.
No other model, CuMesh, extraction, texturing or GLB operation is included.

This is the shortest useful next execution proof, not a readiness claim. It
cannot resolve full generation or the earlier CuMesh 719 failure. **No experiment
was run or authorized by this investigation.**
