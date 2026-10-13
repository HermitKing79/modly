# Native postprocessor admission decision

**NO-GO for a GPU convolution attempt with this instrumentation.** The fixed postprocessor correction passes CPU acceptance, but the genuine Python-source interface replay exposes a remaining dispatcher rejection. No GPU initialization, FlexGEMM/HIP-helper execution, compilation, model loading, download, installation or environment modification occurred. No GPU attempt marker exists.

Branch `codex/amd-native-postprocessor-admission`, based on local/GitHub-verified `493063ace1d303252ed86d3f2c96e1867e4bc13d`. The original failed GPU attempt and all previous evidence remain unchanged.

## Approach decision

| Approach | Finding |
|---|---|
| A: retain dispatch checks and complete the fixed contracts | Smallest candidate: two scalar chains, exact native factories and returned-tensor provenance. Implemented narrowly and CPU-tested. However, genuine Python interfaces do not yet pass; numerical/native readiness is unproved. |
| B: source-pinned native-call boundaries with input/resource/output safeguards | Not implemented. Removing operation checks inside those calls would lose protection before scalar-controlled allocation and against unexpected GPU operations. Equivalent internal safeguards were not demonstrated smaller than the scoped dispatch/allocation contracts. Isolation and hashes alone are insufficient. |

The source audit initially identified only the second scalar as missing from its source-visible operation list. **The actual installed interface test overturned that completeness assumption:** `weight.reshape(1024,27,1024)` dispatches as `aten.reshape.default`, whereas the lower-level test used `aten.view.default`. This difference is a demonstrated CPU harness boundary, not proof of an AMD computation defect or identical GPU dispatch. No reshape or detach permission was added, and no speculative repair loop followed.

## Bounded implemented correction

Installed `neighbor_map.hip` SHA256 remains `f5de4b1b1f8ddcf54ccbdb76c7a7ff8759801c3cc312fb0ab8f869d3b78abb92`. [Complete source/strategy audit](amd-native-postprocessor-admission-evidence/source-strategy-audit.json) records source/binary identities, operations, native entry points and resource calculations.

- Post1 retains one Int32 scalar from its exact fresh `[55323]` mask → cumsum → last-element view, with count **35353** and bounds `[2049,55323]`.
- Post2 has a separate Int32 `[34]` segment-length → cumsum → offset33 scalar chain, exactly once, with bounds `[33,891]` and fixed-fixture count **599**. Canonical sorted Gray masks produce 33 block popcounts: one17, two21 and thirty18. Equal binary sort keys imply equal Gray masks, so ties do not change this total.
- Both scopes reject foreign, repeated, malformed, wrong-value and changed-storage reads. The original TensorImpl and StorageImpl provenance, shape, dtype, device, offset and lifecycle checks remain. Scalar-dependent factories require the verified count before allocation.
- Source-derived factory shape/dtype/count maps reject additional allocations. Post1 permits two Int32 `[2049]`, Int32 `[55323]`, UInt32 `[55323]`, two UInt32 `[35353]` and UInt32 `[28]`. Post2 permits Int32 `[33]`, `[34]` and `[599]`. Factory device/layout/pinning/memory-format arguments are constrained, with numeric-zero pinning rejected before allocation.
- Returned native tuples are checked against captured factory/sort/prefix identities before return. Neighbor output provenance feeds post1; captured post1 Gray/sorted outputs feed post2. Release/close clear retained references. Native inputs still require CUDA:0; CPU support is strictly for blocked-runtime contract tests.
- The convolution phase now applies the existing operation/resource checks to CPU replay too. Native/cache/JIT/build/autotuning/helper/foreign-launch restrictions and GPU limits remain unchanged. The smoke launcher's new CPU-evidence gate rejects this failed full-interface result **before GPU admission or attempt creation**; the acceptance code/result are included in its committed-file manifest.

## Actual verification

| Saved evidence | Actual result |
|---|---|
| [Initial RED](amd-native-postprocessor-admission-evidence/cpu-red/results.json) | Old production dispatcher rejected the genuine post2 scalar; positive expectation599 failed. Constructor13 and existing scalar18 passed. |
| [First acceptance](amd-native-postprocessor-admission-evidence/cpu-acceptance/results.json) | 39/39 PASS: 13 constructor +18 scalar +8 postprocessor checks, 12.172s, exit0. |
| [Strictness RED](amd-native-postprocessor-admission-evidence/cpu-acceptance-reviewed/results.json) | New malformed-factory regression exposed numeric-zero pinning acceptance. One failure; preserved despite the directory's earlier chosen name. |
| [Final lower-level acceptance](amd-native-postprocessor-admission-evidence/cpu-final/results.json) | **41/41 PASS**, including10 postprocessor positive/negative tests, 12.297s, exit0. Both complete postprocessor ATen sequences and the explicit view/empty/reduce/copy tail passed. |
| [Genuine Python-interface probe](amd-native-postprocessor-admission-evidence/cpu-probe/results.json) | **42 tests, one error**, 12.484s, exit1: `Unexpected convolution Torch operation: aten.reshape.default`. It stopped at weight.reshape, before Python split-output creation, sum and conversion. Those later Python interfaces remain untested. |
| [Focused regressions/admission gate](amd-native-postprocessor-admission-evidence/focused-tests.log) | **11 PASS**, 0.662s; includes failure before later admission when full-interface evidence is not passing. |
| [Independent read-only review](amd-native-postprocessor-admission-evidence/independent-review.json) | Approves publication **as NO-GO**, with exact reviewed hashes; no numerical/native compatibility claim. |

All installed-runtime probes used Python3.12.10, PyTorch2.13.0+rocm10.0.0/HIP7.15.26333 and Triton3.8.0. They ran in owned CPU-only workers under a120-second bound; GPU initialization/native GPU execution/weights loading were false. Owned Jobs exited cleanly without forced cleanup. The complete real policy constructor installed/restored its17 named seams, with zero callbacks. Native-written fixture arrays were represented by independent bounded NumPy data solely to exercise genuine CPU ATen host operators and production guards; no native computation was substituted and claimed successful.

The legacy scalar suite's `post2_permitted:false` records its negative **unscoped** read case. The separately scoped positive post2 contract is tested by the new suite and returns599.

## Resource limits and preservation

The unchanged requirements are13GiB available VRAM,12GiB allocator ceiling,1GiB observed PyTorch smoke peak/reserved limit,32MiB factory/result limit and45/120/180-second CPU/GPU/overall execution deadlines. Explicit declared tensors conservatively retained together sum to **91465224 bytes (87.23MiB)**. This excludes internal sorting values/workspaces, allocator overhead and native/runtime scratch; it is not a universal native peak or hard1GiB pre-allocation ceiling. Installed sources omit included native headers, so complete binary internals cannot be reconstructed from the wheel. Existing pinned binary provenance remains necessary.

[Preservation evidence](amd-native-postprocessor-admission-evidence/git-after.json) verifies byte-identical installed inventories, AMD edits/backups, cache/model metadata and historical local/remote refs. **5756 historical files** are hash-verified. Three intentional repository harness edits are explicitly compared with their reviewed replacement hashes; the first after-check rejected those edits until this distinction was made. No installed/native-source differences were excluded.

## Exactly one recommended next experiment

**Separately approve one CPU-only observational trace of the finite genuine Python tail** (`weight.reshape`, Python split-output factory, sum and dtype conversion), recording every actual overload plus argument/result tensor, storage, shape, dtype and alias provenance in the existing isolated runtime. Keep GPU/native execution, compilation and downloads blocked; use one owned worker with a120-second bound and the same fixed fixture sizes. The decision criterion is whether every observed operation can be covered by one small, source-faithful allocation/alias contract; foreign provenance or a need for broad operation permission means NO-GO. This captures the missing interface evidence once, rather than patching one operator and discovering another during GPU retries.

No GPU retry, decoder integration or further instrumentation implementation is authorized by this result. This task stops after publication.
