# Instrumented AMD sparse-decoder forward

Date: 2026-10-12. Base: `9fe6ae24f683aecd14f1eb970fd76b9d5088f9db`.
Branch: `codex/amd-instrumented-sparse-decoder-forward`.

**Classification: PARTIAL.** The one authorized attempt stopped at GPU-state
logging with a Python argument collision. Strict CPU loading and the GPU identity
query succeeded, but **zero decoder forwards, placements or transfers occurred**.
The earlier timeout/compilation hypothesis remains untested. No retry was made.

## Preparation published before execution

The reviewed preparation was committed and verified on GitHub at
[`6792284bea97f48fa97af7a5bf3bdb08aebca1cd`](https://github.com/HermitKing79/modly/commit/6792284bea97f48fa97af7a5bf3bdb08aebca1cd)
before the separately authorized single attempt. The supervisor verified all
19 executed dependencies against that commit. The [timeout investigation](amd-forward-timeout-investigation.md)
and [two](amd-sparse-decoder-forward.md) [earlier attempts](amd-sparse-decoder-forward-guard-correction.md)
remain unchanged. Production workers/mock routes are untouched.

The same strict local sparse decoder, 74 tensor entries / 147,585,412 payload
bytes, deterministic float32 `[1,8,1,1,1]` input and expected finite float32
`[1,1,4,4,4]` occupancy-logit output are retained. One CPU checkpoint load and
one root forward are allowed. No warm-up, other model, Triton compilation,
CuMesh, mesh extraction, GLB, download or installation is admitted.

## Conservative cache gate

[Cache eligibility](amd-instrumented-forward-evidence/cache-plan.json) verifies
the historical 524-file manifest and both virtual-environment/source metadata
against the prior snapshot. COMGR outer lengths and complete ELF header/program/
section-table file bounds are inspected as inert bytes; machine 224 / flags
`0x4e` require gfx1201. Eight direct-ELF entries are selected. The 506 entries
with unexplained inner framing and the bitcode entry are excluded. Empty
timestamp/lock files and all temporary products are excluded.

The MIOpen database is opened immutable/read-only, checked with SQLite, and both
historical records' identities, bounded bzip2 streams, decoded SHA256, target and
ELF bounds are verified. Its two gfx1201 FP32 kernels are retained as one complete
database file. All **nine** selected files have byte-identical writable copies
in fresh ignored workspace; historical files are never runtime cache paths.

The installed COMGR, HIP, HIPRTC and MIOpen DLL bytes/sizes match the SHA256
attestations in package RECORDs whose hashes match the historical ROCm 10
inventory. Library attestations and private-cache hashes are checked again before
launch. Native cache keys and deterministic settings remain unchanged. This
establishes a defensible provenance/format gate, not proof that cached code is
functional, will be hit, or forms a fully warm cache. Native loaders retain
their own compatibility/key checks. See [COMGR serialization](https://raw.githubusercontent.com/ROCm/llvm-project/amd-staging/amd/comgr/src/comgr-cache.cpp)
and [cache key inputs](https://raw.githubusercontent.com/ROCm/llvm-project/amd-staging/amd/comgr/src/comgr-cache-command.cpp).

## Diagnostics and containment

The published host module/convolution entry/return markers retain single-use
guards. UTC, absolute monotonic and relative times permit correlation with
owned-process CPU/executable samples and post-run cache mtimes. Host return is
distinct from the existing final GPU synchronization. Existing VRAM queries are
durably emitted; no per-layer GPU sampling or extra synchronization is added.

Windows controller/System-event checks run on a separate daemon every five
seconds, each subprocess capped at four seconds. Readiness follows a durable
capture; unhealthy/late/stale results prevent authorization or PASS. The GPU
supervisor polls at most every 0.5 seconds. Durable writer acknowledgments remain
bounded; logging cannot suspend the supervisor deadline indefinitely.

Review identified that the inherited reaper could miss a compiler child after
root exit. A private, unnamed Windows Job Object now contains the worker and
its descendants before the worker receives its preparation handshake. The
non-inheritable handle has kill-on-close and no breakaway permission. Accounting
must show zero active processes; error cleanup terminates only this job, bounded
to five seconds, then the original root reaper verifies the root. Forced or
unverified cleanup prevents PASS. See [Microsoft job ownership semantics](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

Worker CPU preparation is capped at **45 seconds**; GPU metadata, placement,
forward and release at **120 seconds**; overall worker supervision at **180
seconds**. A new exclusive attempt marker/canonical run directory prevent retries
or evidence replacement. All executed harness dependencies and the cache plan
must match the exact published commit.

## Gate 1 verification

[65 focused CPU tests](amd-instrumented-forward-evidence/cpu-owned-job-final.json)
pass: 17 cache/monitor/ownership, 13 diagnostics, 13 forward/supervisor, 13
transfer and 9 loading-contract tests. Negative cases include uncertain/truncated
ELF, changed library/RECORD, corrupt database, unsafe paths/copies, observer
failure/staleness, logging bounds, late GPU authorization and cleanup failure.
A real disposable CPU parent/child verifies containment of a child surviving
root exit; no AI package or model is loaded. Broad suites and old successful
GPU/IPC proofs are excluded.

[Independent read-only review](amd-instrumented-forward-evidence/pre-attempt-review.md)
confirmed no remaining critical/important finding after the containment fix,
and readiness for the one authorized attempt after exact publication verification.

The independent [passive Windows probe](amd-instrumented-forward-evidence/windows-preflight.json)
passed in 1.984 seconds: RX 9070 XT status OK, unchanged driver
`32.0.31041.1004`, valid System log, no matching reset/error events and no Torch
import. Before inventories and historical/source/Git hashes are saved.

## Exact stopped result

The [exclusive attempt](amd-instrumented-forward-evidence/attempt-started.json),
[raw result](amd-instrumented-forward-evidence/run/results.json),
[durable trace](amd-instrumented-forward-evidence/run/trace.jsonl) and
[interpretation](amd-instrumented-forward-evidence/attempt-interpretation.json)
show the following:

| Operation | Actual evidence |
| --- | --- |
| Strict CPU loading | Once, 1.734s, all 74 entries / 147,585,412 payload bytes matched; no missing/unexpected keys; GPU uninitialized at CPU gate |
| CPU memory after load | Working set 642,920,448 bytes; private commit 1,085,849,600 bytes |
| Runtime | Python 3.12.10; Torch 2.13.0+rocm10.0.0; HIP 7.15.26333; CUDA null; intended AMD venv |
| GPU identity query | Returned AMD Radeon RX 9070 XT, gfx1201, total 17,095,983,104 bytes |
| Last durable operation | `GPU-metadata` stage; identity returned in final record before the logging exception |
| Failure | `TypeError: Trace.emit() got multiple values for argument 'name'` |
| Model placement / transfers / forwards | Placement never reached; input/output transfers 0; root and permit calls 0 |
| Output / VRAM / measured release | Unavailable; no output or GPU tensor allocation/placement operation reached; no post-error GPU query |
| Worker / containment | Exit 1, 9.562s; no timeout, forced cleanup or native crash; job active count 0 and owned-tree/root exit verified |
| Observer | Before launch, CPU gate, periodic readiness and after exit healthy; same driver, controller OK, no matching reset/error events |

The exception is at `trace.emit('state','gpu',**record['gpu'])`: the record
contains `name`, colliding with the method's positional operation-name argument.
A pure Python reproduction emits no event and matches the exact error without
importing Torch. The 65 passing preflight tests did not cover that complete named
GPU-state payload. This was a diagnostic coverage gap, **not a native computation
failure**. The executed harness is preserved unchanged; no post-failure fix,
second attempt or GPU operation was performed.

[Independent stopped-result review](amd-instrumented-forward-evidence/post-attempt-review.md)
confirmed the failed boundary, zero forwards, healthy monitoring, preservation
and factual limits of the PARTIAL classification.

Only the initial periodic Windows capture was needed: the GPU-authorized interval
ended before the next five-second poll. All capture/writer threads stopped with
no error. The sole CPU-process sample was taken before GPU metadata, showing the
venv launcher and Python child, with 3.6875 CPU seconds in the latter and no
compiler executable. There is no forward delay or coincident compilation to
analyze. All nine private cache seed files remained byte-identical, with **zero
new scratch files**; see the [inert manifest](amd-instrumented-forward-evidence/scratch-side-effects.json).
Cache hits/functionality and the prior timeout cause are not established.

## Preservation and next step

[Preservation](amd-instrumented-forward-evidence/preservation.json) verifies
byte-identical dependency inventory, both raw compressed virtual-environment
and model/cache metadata, all historical evidence/caches/reports, historical
local/remote branch heads and existing AMD source HEAD/`nodes.py`/backup hashes.
Checkpoint before/after records are identical. Both installed environments and
production mock behavior remain unchanged; no installation, rebuild or download
occurred.

The smallest justified next step is **CPU-only correction of the GPU diagnostic
payload**, using a nested `gpu` field rather than unpacking it, plus a regression
covering the complete real metadata dictionary and reserved diagnostic keys.
Then obtain independent review and separate approval for a new exclusive
single-forward attempt. No extra model operation is justified until the first
decoder forward completes. The eventual DINO/flow conditioning and geometry
path, texturing and earlier CuMesh 719 failure remain unproved. Work stops here.
