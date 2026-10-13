# Native-convolution HookChain correction and single attempt

**Overall UNKNOWN. CPU HookChain correction PASS; the one GPU attempt stopped at another harness guard.** No retry or further correction occurred. Native numerical correctness remains untested.

Branch `codex/amd-native-convolution-hook-correction`; base `0e23dfcd4eb7d0d86ee91e081903bc4280016cc9`. Candidate `e31fb4168a534af3bf4b8227abac979425cd134c` was [published and GitHub-verified before execution](amd-native-convolution-hook-correction-evidence/candidate-publication.json). [Exact result/traceback](amd-native-convolution-hook-correction-evidence/run/results.json), [durable trace](amd-native-convolution-hook-correction-evidence/run/trace.jsonl) and [exclusive marker](amd-native-convolution-hook-correction-evidence/attempt-started.json) preserve the sole attempt. The executed code is unchanged afterward.

The source-pinned correction accepts only the five original installed empty `HookChain` instances, with exact list/type/attributes and default `reversed` flags, or the existing supported `None` state. Identities are captured immediately after verified Triton import. Replaced class/runtime defaults, subclasses, foreign/nonempty/malformed chains and compiler/JIT callbacks reject before patches or callback execution. Compilation/cache-policy checks remain unchanged. Installed `knobs.py` SHA256 is `7468fb1fb4740bb9237c57216b04499400825f62a6177da4214153900feda824`.

[Final CPU acceptance](amd-native-convolution-hook-correction-evidence/cpu-acceptance-reviewed/results.json) passed **13 tests**, including the complete real constructor, all17 named guard boundaries and restoration: zero failures/errors/callbacks, GPU/native/weights false,9.407s, exit0, unforced owned Job exit. [Ten focused regression tests](amd-native-convolution-hook-correction-evidence/final-focused-tests.log) also passed. [Independent candidate review](amd-native-convolution-hook-correction-evidence/independent-review.json) verified the final hashes. All CPU test-development evidence is retained: isolated RED reproduced the original hook defect; an earlier negative-test environment-restoration issue used Triton's scope, and an incorrect guessed patch-count assertion was replaced by named guard checks. These were not GPU attempts.

| Single-attempt measurement | Actual observation |
|---|---|
| Runtime/GPU | Python3.12.10, PyTorch2.13.0+rocm10.0.0/HIP7.15.26333, Triton3.8.0; RX9070XT, `gfx1201` |
| CPU gate / hook state | PASS; five verified empty defaults, no callbacks |
| Admission | Host RAM16,227,000,320 bytes; free VRAM16,937,582,592 bytes, above13GiB |
| After synchronized input transfer | Allocated/observed peak60,854,784 bytes; reserved/observed peak79,691,776 bytes |
| Native progress | One convolution entered; neighbor-map call returned to host; `post1` entered and failed |
| Exact blocker | `Unexpected convolution Torch operation: aten.item.default` |
| Later counts | `post2`, outer/inner tuning, JIT, cached binary and Triton launch all0; helper0, cache lookups empty |
| Output/synchronized convolution time | Not produced or measured; no elementwise/coordinate/output validation |
| Process |10.375s, exit1, no timeout; owned Job active0/0, no forced cleanup, exit verified |
| Health/release | All observer queries healthy, controllerOK, no reset events recorded; live-worker zero-allocation release not reached |

This is a **guard failure**, not demonstrated numerical/native incompatibility. Neighbor return is explicitly `host-return-only`, not synchronized GPU completion. The last memory sample precedes native preprocessing and is not a final whole-attempt peak. After failure the worker issued no additional explicit GPU operations; process exit supplied containment, without claiming successful zero-allocation acceptance.

The pinned native source `neighbor_map.hip:310` reads `prefix_sum_neighbor_mask[-1].item<int32_t>()` before valid-signal allocations. Installed `TensorBody.h`/`item_ops.h` identify `aten::item(Tensor self) -> Scalar`. This supports the observed overload's origin, but no C++ traceback frame identifies line310 directly. The unchanged dispatcher admitted `_local_scalar_dense`, not `aten.item.default`. [Independent outcome review](amd-native-convolution-hook-correction-evidence/outcome-review.json) confirms this boundary and the incomplete numerical/release proof.

The N2049/Ci=Co1024 F16 centre-identity fixture, cache/tuning identities, `atol=rtol=0`,13GiB admission,12GiB allocator ceiling,1GiB observed test-memory limit and45/120/180s CPU/GPU/overall limits were preserved. No compilation, helper build, autotuning miss, warmup, AI checkpoint, decoder, subdivision or geometry stage ran. No output artifacts exist; private verified cache copies remain in owned ignored `run/scratch`.

[After preservation](amd-native-convolution-hook-correction-evidence/git-after.json) verifies both byte-identical installed inventories, **5579 historical files**, main and all prior branches, AMD source HEAD `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, modified `nodes.py` and its backup. Historical proof artifacts and the prior failed launch remain intact.

**One smallest next action, requiring separate approval:** CPU-only characterization of the exact installed `aten.item.default` scalar-read dispatch, its genuine int32 scalar provenance and bounded neighbor-count contract, with focused production-dispatch tests. Do not automatically allow arbitrary scalar reads or rerun this experiment. This task stops after publication; no decoder integration or additional GPU execution is authorized.
