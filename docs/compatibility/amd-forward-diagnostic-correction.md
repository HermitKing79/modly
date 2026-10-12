# CPU-only AMD forward diagnostic correction

Date: 2026-10-12. Base: `18ef0677a0acbab1eab987ac69842027401af9ca`.
Branch: `codex/amd-forward-diagnostic-correction`.

**PASS for this CPU-only logging correction.** The complete saved GPU identity
serializes under `gpu`, passes the real durable journal and retains all values.
No real weights loaded, AI runtime imported, GPU initialized or forward attempted.
The [previous attempt](amd-instrumented-sparse-decoder-forward.md) remains PARTIAL;
its evidence, consumed attempt marker and historical cache files are unchanged.

## Correction and complete call-site audit

The actual worker expression is now
`trace.emit('state','gpu',gpu=record['gpu'])`. Metadata is never unpacked into the
operation-name argument. The regression reads the full saved identity, rather
than a reduced fixture: AMD Radeon RX 9070 XT, `gfx1201`, 17,095,983,104 total bytes.

[All 15 production trace expressions](amd-forward-diagnostic-correction-evidence/call-site-audit.json)
were inspected: ten worker expressions, one process observer, one Windows
observer and three internal `Trace.call()` emissions. Runtime, memory, release,
tensor, process and Windows payload producers were inspected as well. Their
current keys do not conflict; nested process `name` fields remain valid.

`Trace.emit()` rejects metadata containing its record-owned fields (`event`,
`name`, `origin`, `seq`, `elapsed_seconds`, `monotonic_seconds`, `timestamp_utc`)
and method argument names `self`/`call`. `Trace.call()` additionally rejects
`duration_seconds`, `completion`, `error_type` and `error` before executing its
wrapped operation. Positional-only control arguments let conflicts produce an
explicit reserved-field error. Nested data may retain these names.

Both emission and journal ingestion validate finite JSON values, string keys,
clock/UTC identity and existing record/line bounds. Lossy key coercion, duplicate
JSON keys, nonfinite values (including exponent overflow), unsupported objects,
excessive nesting and oversized lines fail clearly. Invalid payloads do not
advance the trace sequence or reach the sender. Existing durable-write bounds
and host-return-only semantics remain intact.

CPU inspection caught an additional boundary in the candidate validator:
`torch.__version__` is `TorchVersion(str)`. An exact built-in-string check would
reject the runtime identity before any forward. The installed declaration was
read without importing Torch, the rejection was reproduced in
[a regression](amd-forward-diagnostic-correction-evidence/runtime-type-red.json),
and string subclasses now retain their text through validation and serialization.
See [source hashes](amd-forward-diagnostic-correction-evidence/runtime-type-inspection.json).

## Focused CPU verification and review

[Final focused verification](amd-forward-diagnostic-correction-evidence/review-replay.json):
**52/52 tests pass** (9 payload/lifecycle, 13 diagnostics, 13 forward/supervisor,
17 cache/observer/ownership). Each test group runs in an isolated host Python
3.12.10 subprocess with a 15-second bound and an import blocker for Torch,
Triton, TRELLIS.2 and safetensors. No broad application suite or earlier GPU
proof was repeated.

The [initial red run](amd-forward-diagnostic-correction-evidence/red.json)
reproduced the exact GPU argument collision and demonstrated reserved-field and
malformed-record gaps before the correction. `green.json` records the initial
51 passing tests; `reviewed-candidate.json` includes the version-subclass test.
An evidence-runner parsing error on supervisor stdout was corrected before the
first complete green capture; it was not a worker or native failure.

The replay compiles and executes the actual production logging expressions,
with synthetic callbacks and tensor descriptions. Its 16 durable records cover
runtime/GPU identity, placement/forward/release memory, module and convolution
entry/return, synchronization host return, process CPU data, Windows observer
data and completion. It executes the actual production shutdown assignments
against the real journal: `diagnostic_journal_stopped=true`, error null, writer
thread stopped. It does not fabricate a worker-termination event or claim that
an actual GPU worker was launched. Existing focused tests separately exercise
owned CPU-process termination and fail-closed supervisor cleanup.

[Independent read-only review](amd-forward-diagnostic-correction-evidence/independent-review.md)
audited the correction and replay, then independently reran all 52 CPU tests
with the AI import blocker. No remaining critical/important finding was reported.

## Preservation and limits

[Before/after checks](amd-forward-diagnostic-correction-evidence/preservation.json)
confirm byte-identical dependency inventories, compressed metadata for both
installed environments and model/cache directories, selected source/native
hashes, all 685 historical evidence/report/cache files and historical local/
remote branch heads. AMD source HEAD and the existing modified `nodes.py` and
backup remain unchanged. Main remains `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.

Only two production harness files changed. Cache eligibility/copy procedures,
GPU authorization, Windows monitoring, job containment, single-forward/transfer
guards, model loading policy and 45/120/180-second deadlines are unchanged.
Production mock routes and installed applications were untouched.

No additional predictable pre-forward diagnostic failure remains identified
from this CPU audit and replay. This does not prove native execution, allocator
behavior, compilation or the entire live worker lifecycle. The prior timeout
cause remains unresolved. A future forward requires separate approval and new
exclusive evidence; this task stops at reviewed correction publication.
