# AMD sparse-decoder forward guard correction

Date: 2026-10-12. Base: `0f82092515849041ee7df313fef5d2bad2a7ab98`.
Branch: `codex/amd-sparse-decoder-forward-guard-correction`.

**Classification: PARTIAL.** Gate 1 passed. The single fresh attempt passed the
corrected input-transfer boundary and reached the forward stage, but did not
produce a result within the authorized GPU-stage deadline. No retry was performed.

## Correction and evidence

The [first attempt](amd-sparse-decoder-forward.md) remains PARTIAL: its decoder
placement completed, but `cpu_input.to('cuda:0')` produced
`aten.to.dtype_layout`, which the placement guard rejected. Root decoder and
forward call counts were both zero. This demonstrated a harness boundary failure,
not a native decoder computation failure.

The installed PyTorch 2.13 schemas confirm that `to.dtype_layout` accepts optional
dtype/layout/device/pinning, nonblocking, copy and memory-format arguments;
`_to_copy` has the same optional conversion fields except `copy`, and always
returns a copied tensor. The CPU characterization under production nesting
(dispatch mode outside inference mode) shows the original input `.to()` surrogate
and output `.cpu()` both use `to.dtype_layout`; explicit `_to_copy` uses only
`aten._to_copy.default` with a device argument and preserves shape, stride, float32
dtype and values. GPU initialization remained false. The original AMD-target CPU
probe stopped at the blocked initializer with no dispatch record; its actual GPU
arguments were **not** captured. See
[schema and surrogate evidence](amd-sparse-decoder-forward-correction-evidence/transfer-schema-production-nesting.json).

The smaller correction uses the existing permitted `_to_copy` operation for both
copies, adding no operator allowance. Single-use permits bind exact source
identity, lifecycle phase, shape, strides, contiguous strided layout and float32:
CPU `[1,8,1,1,1]` to `cuda:0`, then verified decoder `[1,1,4,4,4]` from `cuda:0`
to CPU. No dtype/layout/nonblocking/pinning/memory-format arguments are permitted.
Each permit is consumed before native redispatch and copied metadata is checked.
HIP runtime/GPU identity gates continue to establish that `cuda:0` is AMD.

The 39 [focused CPU tests](amd-sparse-decoder-forward-correction-evidence/cpu-tests-reviewed.json)
pass: 13 transfer, 12 forward/supervisor, 9 loading contract and 5 synthetic strict
loader tests. Negative cases cover source/device/shape/dtype/layout/stride,
arguments, operations, phase, duplicate transfers and classification. No real
weights or GPU allocations were used. Broad application suites were excluded as
requested.

## Single-attempt safety

The corrected harness retains exact local checkpoint/full payload validation,
one strict CPU load, one root decoder call, no warm-up, network and alternative
checkpoint prohibitions, and disabled Triton compilation/JIT execution. It uses
a separate evidence directory and an exclusive attempt marker. Original
45-second CPU, 30-second forward/release and 90-second supervisor bounds remain;
the observer must pass before launch, at the CPU gate and after exit. Any failure
ends the attempt, with owned-process reaping and no automatic retry.

All historical branch heads, original report/evidence (including local scratch),
user-modified AMD sources, environment/package inventories and model-cache trees
are captured for preservation checks. Production worker/mock routes are untouched.

## Execution result

The reviewed harness was committed and published **before execution** at
`1bfb7ba3091dd314a03a138f479f301ee7243548`; Git remote and GitHub commit checks
matched, and all 14 executed dependencies matched that snapshot.

The [exclusive fresh attempt](amd-sparse-decoder-forward-correction-evidence/attempt-started.json)
loaded the same genuine decoder on CPU with strict matching of all 74 entries,
147,585,412 payload bytes, no missing/unexpected keys and no CPU-stage GPU context.
CPU loading took 1.328 seconds; working set after loading was 640,729,088 bytes and
private commit 1,083,887,616 bytes. The [durable worker log](amd-sparse-decoder-forward-correction-evidence/run/worker.stdout.txt)
then recorded GPU metadata, decoder placement, input transfer and `single-forward`.
By the reviewed control flow, that last stage requires placement, the exact input
copy, synchronization and copied input metadata checks to have completed.

No `FORWARD_RESULT` arrived. The supervisor's queue wait expired (`Empty()`) under
the unchanged **30-second GPU-stage budget**, which includes placement, forward
and release. It terminated worker PID 3232 and child PID 2896, recorded exit code
**1**, forced cleanup, verified worker exit and completed both log readers. This
exit code resulted from timeout termination; it is not a captured native crash.
Stderr was empty. No guard exception, native exception or validated output was
captured. See [raw result](amd-sparse-decoder-forward-correction-evidence/run/results.json)
and [interpretation](amd-sparse-decoder-forward-correction-evidence/attempt-interpretation.json).

The raw supervisor result retains its default `NOT ATTEMPTED` worker placeholder
when no final worker message arrives. The CPU gate and durable stage log prove
that an attempt occurred; the placeholder must not be interpreted as no GPU work.
Root-call counters, output shape/dtype/finiteness/statistics, VRAM snapshots/peaks,
and worker-side resource-release measurements were not durably emitted. They are
**unavailable**, and allocator cleanup cannot be claimed. Owned-process exit is
verified. No second forward, warm-up, expanded test or further GPU query followed.

All three Windows observers were healthy: AMD Radeon RX 9070 XT status OK,
driver `32.0.31041.1004`, valid System log probe and no matching reset/error events.
Checkpoint before/after identity files were byte-identical. Installed AMD/Modly
dependency inventories, full virtual-environment metadata and model/cache
metadata were also byte-identical. All historical local/remote branch heads,
the entire first attempt (including local scratch), AMD source HEAD and user
`nodes.py`/backup hashes were preserved; see [preservation](amd-sparse-decoder-forward-correction-evidence/preservation.json).

The isolated scratch workspace contains 524 files totaling 669,452,910 bytes,
including native ROCm COMGR code-object caches, a MIOpen cache database and
temporary COMGR artifacts. Their [hash manifest](amd-sparse-decoder-forward-correction-evidence/scratch-side-effects.json)
is published; binary caches remain locally preserved and ignored. This is evidence
of native runtime cache activity during the attempt, consistent with first-use
compilation, **not proof of the timeout's cause or successful GPU computation**.
No package, source, checkpoint or production-worker modification was made.

The next justified step is a separately scoped review of this forward-stage
timeout and durable stage/operation diagnostics before considering another GPU
attempt. The earlier CuMesh 719 failure remains unresolved. Full generation,
model-loading readiness of other components, texturing and GLB export are not
established. Work stops at this published bounded result.
