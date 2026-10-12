# AMD sparse-decoder forward guard correction

Date: 2026-10-12. Base: `0f82092515849041ee7df313fef5d2bad2a7ab98`.
Branch: `codex/amd-sparse-decoder-forward-guard-correction`.

Gate 1: CPU correction verified; GPU result pending the independently reviewed,
published harness and healthy Windows observer. No new GPU attempt has occurred
at this documentation stage.

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

Pending. This report will be updated from the separately authorized attempt's
saved results; CPU transfer characterization does not establish AMD forward success.
