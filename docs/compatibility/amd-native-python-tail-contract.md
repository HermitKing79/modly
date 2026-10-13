# Python-tail contract: evidence-persistence blocker

**NO-GO. The complete operator record was not preserved, so no dispatcher permission correction or genuine full-interface acceptance is justified.** One CPU observational attempt ran; it was not repeated. No GPU initialization, FlexGEMM/HIP-helper execution, AI checkpoint loading, native compilation, installation or environment modification occurred.

Branch `codex/amd-native-python-tail-contract`, based on local/GitHub-verified `c9e1a35ce9941753e64536885e59d947cd0a7f1e`. Existing production policy, postprocessor contracts, smoke launcher, cache safeguards and historical failure evidence remain unchanged.

## What ran and what failed

The [owned CPU worker result](amd-native-python-tail-contract-evidence/cpu-probe/results.json) records **12.0 seconds, exit1**, clean verified Job termination, no forced cleanup, GPU/native/weights false. Its complete traceback identifies an attempted write to `cpu-probe/observation.json`, outside the worker's allowed scratch directory. The existing filesystem audit correctly rejected it.

The traceback occurs after `observe()` returned. This supports that the source-body replay completed, but **the local operator/tensor record was neither written nor printed before worker termination**. No `observation.json` exists in the probe or its scratch. It cannot be recovered from stdout/stderr. Consequently there is no defensible exact overload list, allocation/alias record or output-statistics evidence to inspect before changing permissions. The failure is a harness logging mistake, not an observed AMD/native numerical defect. The prior reshape rejection remains unresolved.

The replay extracted the pinned installed Python function/class bodies, preserving their actual `weight.reshape`, Python `torch.empty`, reduction, conversion and `Function.apply`/final-return syntax. It used the unchanged N2049/C1024 F16 fixture and precomputed CPU cache placeholders. The selected GPU-result boundary alone was explicitly simulated with `[reference features, zeros]` in the FP32 split buffer; no native convolution was run. These placeholders are **not verified native neighbor-map outputs** and cannot be reused as inputs to a real kernel experiment.

The source identities checked were `submanifold_conv3d.py` SHA256 `51ac1ac963e46228e7d0f928c6237af5e0677705278ebef73bffdb7b43e69aab` and selected split-K source `21d5efc8f59f312c62fa45241fc13a9e446cd4c13f11e360cf92216d151301bb`. The existing CPU preflight recomputed the same kernel, tuning and HIP-helper identities under native/GPU execution prohibitions. Source-derived expectations remain one weight-storage alias and fresh split/sum/cast buffers of16785408/8392704/4196352 bytes; they are not substituted for missing observations.

## Only correction made

The new observation runner now writes exclusively to owned `scratch/observation.json`. The supervisor can copy it to evidence only after a passing, clean, verified worker/Job exit; it requires the canonical owned source path, a256KiB size limit and an exclusive destination. Missing evidence cannot produce a synthetic record. The incomplete final-acceptance route was removed. The failed capture and its evidence remain unchanged; the corrected recorder was **not rerun**.

[Focused host tests](amd-native-python-tail-contract-evidence/focused-tests.log): **15 PASS**,0.399 seconds—four persistence/security tests plus11 existing focused contract regressions. They cover byte-preserving/exclusive copy, missing data, failed/unclean workers, oversized data and foreign directories. Persistence fixtures are explicitly synthetic and are not replacement operator observations. These tests do not establish runtime-tail acceptance.

[Independent read-only review](amd-native-python-tail-contract-evidence/independent-review.json) approves publication as NO-GO, confirms the record is unavailable, and verifies the correction does not widen dispatcher permissions. Existing13GiB free-VRAM admission,12GiB allocator cap,1GiB observed smoke budget,32MiB factory/result limit,45/120/180-second GPU-experiment deadlines, observer, Job ownership and compiler/build/autotuning/cache-miss rejection remain intact. No GPU attempt marker was created.

[Preservation evidence](amd-native-python-tail-contract-evidence/git-after.json) verifies byte-identical installed inventories, AMD source edits/backups, model/native-cache metadata, **5836 historical files** and prior local/remote branches. No prior repository harness checksum is excluded from this comparison. The new persistence-test scratch is separate from the failed observation.

## One recommended simpler experiment

**Separately approve a bounded cached-leaf convolution experiment**, following a small CPU preflight: derive and independently verify the real fixed-fixture neighbor map, sorted indices, valid-kernel indices and segments on CPU; preallocate the exact buffers; invoke the existing source/cache-pinned selected native kernel once; synchronize and copy its raw split output to CPU for reduction and exact identity comparison. Retain compiler/build/autotuning/cache-miss rejection, the existing memory/timeout/observer/containment/zero-release requirements and no retries.

This avoids the unverified Python wrapper/postprocessing dispatch path. It would prove only the selected cached convolution kernel on this fixture, not native neighbor-map/postprocessor compatibility, the full FlexGEMM call or the shape decoder. The placeholder arrays in this observation runner must not be used. No implementation or GPU execution of that alternative occurred or is authorized by this report.

The task stops after publication. No second observation, operator repair cycle or automatic GPU retry.
