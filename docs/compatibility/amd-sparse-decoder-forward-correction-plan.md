# Sparse-decoder transfer correction plan

Scope: correct only the synthetic input/output transfer boundary of the published
single-forward harness. Preserve the first attempt and all application code.

1. Capture Git/environment baselines. Inspect the installed operator schemas and
   characterize `.to()` versus explicit `_to_copy` on small CPU tensors inside
   inference mode, with GPU initialization prohibited. Inspect the original
   CPU-to-AMD call without copying; document if initialization blocks observation
   before dispatch, and distinguish CPU surrogate evidence from the saved failure.
2. Choose the already permitted copy operation if characterization supports it.
   Bind a single-use transfer permit to the exact source object, shape, stride,
   float32 strided layout, source/target devices and lifecycle phase. Reject
   additional arguments and computation. Cover both input and output copies.
3. Write focused CPU positive/negative tests first, implement the narrow fix, and
   run only affected contract/regression tests. Obtain independent read-only review.
4. Publish the reviewed harness before execution. Use a new evidence directory
   and exclusive attempt marker; retain the original 45/30/90-second deadlines,
   Windows observer, strict loader and single-forward/checkpoint permits.
5. Make the one newly authorized attempt if every gate passes. On any failure,
   stop without retry. Capture exit/observer/cleanup and preservation evidence,
   publish the final report and verify remote heads.

Review focus: wrong device/index, dtype/layout/stride changes, extra arguments,
wrong source tensor, wrong phase, repeat transfers, output copy, GPU computation
outside the one forward, and preservation of the first attempt evidence.
