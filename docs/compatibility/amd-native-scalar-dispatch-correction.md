# Native post1 scalar dispatcher correction

**PASS for the CPU-only correction and production-dispatch acceptance. Full native-convolution execution remains UNKNOWN; a GPU retry is currently NO-GO.** No GPU initialization, native GPU execution, checkpoint loading, compilation, installation or download occurred in this task.

Branch `codex/amd-native-scalar-dispatch-correction`, based on verified local/GitHub `534897addbc3dab755a81c312facef13b9ef7a25`. The [previous failed GPU attempt](amd-native-convolution-hook-correction-evidence/run/results.json) and all historical evidence remain unchanged. This corrects its dispatcher boundary; it does not reclassify that attempt or prove native numerical correctness.

## Proven scalar contract

Installed `flex_gemm/kernels/cuda/spconv/neighbor_map.hip` SHA256 is `f5de4b1b1f8ddcf54ccbdb76c7a7ff8759801c3cc312fb0ab8f869d3b78abb92`. At lines 288, 309–310, post1 allocates a contiguous Int32 `[55323]` Boolean neighbor mask, computes an Int32 dimension-0 cumulative sum, selects its last element and reads a zero-dimensional Int32 scalar once. Native mask writes are 0/1 (218–221), and the center is mandatory for every point (66–68). Thus the count lies in `[2049,55323]`.

For the unchanged N2049 fixture—two complete 32×32 planes and point `(2,0,0)`—an independent CPU membership reference establishes **35353**, also derived as `4 × 94² + 8 + 1`. This exact count is required, in addition to the general bounds. Installed PyTorch reports `aten::item(Tensor self) -> Scalar`; installed C++ headers identify the actual item, cumsum and select entry points. [Source/ABI audit](amd-native-scalar-dispatch-correction-evidence/source-audit.json) records paths, hashes, line ranges and limitations.

The production dispatcher admits only the actual installed `aten.item.default` or its existing `_local_scalar_dense` equivalent, sharing one read token, during the source-pinned native post1 scope in the convolution phase. It captures the fresh mask, exact cumsum result and exact last-element view. Tensor identity, original StorageImpl identity, dtype, device, shape, offset, contiguity and arguments are revalidated before the read. Original storage references prevent `.data` substitution from preserving a misleading TensorImpl identity. Foreign, duplicate, malformed, out-of-scope, changed-storage and wrong-value reads fail; the result is checked before C++ can use it for valid-signal allocations. References are released on success/error and policy close.

The native wrapper still requires CUDA:0 neighbor input. CPU device support exists only to test these actual operator/dispatcher contracts without native execution. No general `.item()` permission, detach permission or additional native path was introduced. Existing cache, compilation, helper-build, autotuning, allocation, deadline and monitoring policies remain unchanged.

## Actual verification

| Check | Result |
|---|---|
| [Final installed-runtime CPU worker](amd-native-scalar-dispatch-correction-evidence/cpu-final/results.json) | **31 PASS**: 13 complete real NativePolicy constructor tests plus 18 scalar dispatcher tests; zero errors/failures/callbacks |
| Actual installed operators | Production handler returns 35353; exact native-source ATen overloads also pass through normal TorchDispatchMode, with the genuine CPU operators executing |
| Negative coverage | Foreign/repeated reads; wrong dtype, shape, arguments, phase, source/storage provenance; `.data` replacements; incorrect count including an in-range wrong value; unrelated arithmetic; post2 rejection |
| Guard lifecycle | All 17 named policy seams installed/restored by constructor acceptance; scalar references cleared; no native callback executed |
| Isolation | 120-second supervisor bound; actual 11.875 seconds, exit0, owned Job active0/0, no forced termination, worker exit verified |
| Runtime | Python3.12.10, PyTorch2.13.0+rocm10.0.0, HIP7.15.26333, CUDA metadata null, Triton3.8.0; GPU initialization/native GPU execution/weights loading false |
| [Existing focused regressions](amd-native-scalar-dispatch-correction-evidence/focused-tests.log) | **10 PASS**, 0.676 seconds |
| [Independent read-only review](amd-native-scalar-dispatch-correction-evidence/independent-review.json) | Approved exact candidate hashes; no actionable issues; CPU-only limitations retained |

CPU test-development evidence is preserved separately. RED reproduced the old dispatcher's acceptance of a foreign CPU scalar. Earlier normal-mode tests used Python convenience factories and encountered `aten.detach.default`; inference mode alone did not remove that difference. Final tests use the exact ATen overloads corresponding to installed C++ native source, and pass without adding detach permission. This is not a claim that CPU operator tests execute the native C++ postprocessor or establish its GPU behavior.

The final worker recomputed the same N2049/LOGN11 specialization, tuning/cache key and HIP-helper identity under execution prohibitions. [After preservation](amd-native-scalar-dispatch-correction-evidence/git-after.json) verifies byte-identical installed inventories, unchanged cache/model metadata, **5690 historical files**, main and every prior local/remote branch. AMD source HEAD remains `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`; modified `nodes.py` and `nodes.py.before-profiler-fix` retain their recorded hashes. No new GPU attempt marker or GPU run exists.

## Remaining boundary and one recommended action

The surrounding installed native source uses empty/full, argsort, cumsum, select, view, reduction and dtype-copy operations already covered by the existing allowlist. However, **post2 has another Int32 last-element scalar read at line440**, from a different 34-entry prefix sum for 33 blocks. It is deliberately not admitted by this correction. Source-level auditing cannot prove every future GPU dispatch, and a full convolution retry would meet this known uncorrected boundary.

**One next action, requiring separate approval:** a bounded CPU-only review of native-entry/input/output/allocation instrumentation for this fixed convolution fixture and its two postprocessors, while retaining the independently verified cache/JIT/build/tuning/native-path rejection policies and all execution/resource limits. Decide whether that small boundary-based instrumentation can replace fragile operation-by-operation interception; do not build a general framework or run another GPU attempt merely to encounter the already-known post2 rejection. This task stops after publication.
