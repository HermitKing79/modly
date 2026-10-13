# One genuine AMD native sparse-convolution attempt

**UNKNOWN: the sole launch stopped at a CPU harness guard before GPU authorization.** Native numerical compatibility remains untested. No convolution, model, GPU tensor, compilation, build, warmup or retry ran. The executed harness is unchanged.

Branch `codex/amd-native-convolution-smoke`; base `06dbd15030ade1cdec0205e1bd2dd0d56a3d6602`. Candidate `2554a08790b189d0a33a89ced1b46a7cad53850e` was [GitHub-verified before execution](amd-native-convolution-smoke-evidence/candidate-publication.json). [Result/exception](amd-native-convolution-smoke-evidence/run/results.json), [raw stdout](amd-native-convolution-smoke-evidence/run/worker.stdout.txt) and the [exclusive marker](amd-native-convolution-smoke-evidence/attempt-started.json) preserve the launch.

| Observation | Actual result |
|---|---|
| Last operation | CPU `NativePolicy.__init__`: `ValueError: Foreign native hooks` |
| GPU authorization / convolutions | false / **0** |
| Worker | **6.687s**, exit1, no timeout; Job active0/0, no forced cleanup, exit verified |
| Windows observation | Healthy before/after; periodic GPU monitor never started because authorization was not reached |
| Available host RAM | **16,806,211,584 bytes**; requirement1GiB |
| GPU timing / VRAM / numerical output / GPU teardown | **Not exercised or measured**; no output artifacts |

**Confirmed harness defect:** installed `triton/knobs.py:497–503` uses five `HookChain()` objects with empty default callback lists (`:439–440`). The policy's None-only check rejects those ordinary defaults. This exception proves neither a foreign callback nor an AMD computation defect. Individual live callback lists were not logged. [Independent outcome review](amd-native-convolution-smoke-evidence/outcome-review.json) confirms the mismatch. The initial review missed it; the ten CPU tests and metadata probe did not instantiate the complete production policy against installed knobs. No safety check was relaxed after failure.

The [owned CPU metadata admission](amd-native-convolution-smoke-evidence/cpu-probe/results.json) passed in11.438s without GPU initialization/native execution. Actual N2049 binder, source/options and helper identities match the pinned runtime and kernel key `d5de3d60ef4c1dfce927f5a59a91fdfa35c7de3400fb71acb214abe5dc4fbea5`. Runtime is Python3.12.10, PyTorch2.13.0+rocm10.0.0/HIP7.15.26333 and Triton3.8.0. Matching LOGN11 alone was not used to admit execution.

The unexecuted fixture remains2049 unique int32 coordinates, F16 features `[2049,1024]`, actual zero F16 bias and installed weights `[Co,Kw,Kh,Kd,Ci]=[1024,3,3,3,1024]`, centre diagonal only. Exactly representable features `(((row*17+channel*31)%2049)-1024)/1024` distinguish rows. Acceptance remains comparison of2098176 elements with `atol=rtol=0`, coordinate identity, finiteness and zero-allocation release. Exact tuning requires SPLITK2, B1=64/B2=128/BK=32, warps4/stages2/waves3, IEEE=True, grid `(264,2)`. Kernel/helper loadability and native neighbor correctness were not demonstrated.

Limits remain45s CPU/120s GPU/180s overall,13GiB available-VRAM admission,12GiB allocator ceiling,1GiB host-RAM admission and1GiB observed test-memory bound. Private verified copies only; no lowering, helper rebuild fallback, autotuning miss, warmup or retry. [Ten CPU tests](amd-native-convolution-smoke-evidence/focused-tests-3.log) passed; [reviewed hashes](amd-native-convolution-smoke-evidence/independent-review.json) identify the exact candidate.

[After preservation](amd-native-convolution-smoke-evidence/git-after.json) verifies both byte-identical installed inventories,5536 historical files, all previous branches, main `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`, AMD source HEAD `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, modified `nodes.py` and its backup. Private cached copies remain local under owned ignored `run/scratch`; [hashes and locations](amd-native-convolution-smoke-evidence/run/private-cache.json) are preserved. No installed environment, driver, model or AMD source edit changed.

**One smallest next action, requiring separate approval:** CPU-only correction accepting the exact installed `HookChain` type only with no registered callbacks, rejecting nonempty/foreign hooks, and regression-testing the actual complete production policy constructor with GPU initialization blocked. A fresh exclusive attempt of the same convolution would then require separate execution approval. Stop here; no decoder integration, repair or relaunch occurred.
