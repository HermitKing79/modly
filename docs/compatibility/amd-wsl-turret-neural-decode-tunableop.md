# TunableOp-corrected WSL turret neural decode — UNKNOWN

Base: `51429f6b76aaad83e49ed504b3a585b7867b03ed`. Branch: `codex/amd-wsl-turret-neural-decode-tunableop`. The independently reviewed execution candidate was published and GitHub-verified before the single attempt at **`638436ec3a084ec9005b1b03ae97af1115871ca0`**.

**UNKNOWN: insufficient initial free VRAM.** The TunableOp startup correction passed in the genuine installed runtime. The worker then stopped at `decode.py:256`, `RuntimeError: GPU identity/free-VRAM admission refused`, before FlexGEMM import, decoder construction, checkpoint loading or any neural computation. This is a resource-admission refusal, not a demonstrated native/AMD incompatibility. No retry or post-attempt correction occurred.

## Exact evidence

| Observation | Actual result |
| --- | --- |
| Three environment controls before dependencies/Torch | All exactly `"0"` |
| Genuine flags immediately after Torch import | enabled=false, tuning=false, record_untuned=false |
| GPU | RX 9070 XT, gfx1201, count 1 |
| Free VRAM | **13,772,447,744 bytes / 12.826591 GiB** |
| Unchanged admission requirement | **13,958,643,712 bytes / 13 GiB** |
| Shortfall | **186,195,968 bytes / 177.570313 MiB** |
| FlexGEMM import / CPU decoder loading | NOT_ATTEMPTED / NOT_ATTEMPTED |
| Genuine forwards / convolutions / subdivisions | None entered |
| Torch tensor operations / native binding calls / policy rejections | 0 / 0 / 0 |
| Torch allocated/reserved peak and release | **0 / 0 bytes** |
| Guest after GO / supervisor / outer duration | 12.075 / 25.562 / 30.053 seconds |
| Worker exit | 1, controlled admission refusal |

All three flags were queried independently without setters, monkeypatching or short-circuit evaluation. Their actual values and environment strings were durably journaled. The repeated pre-GPU tuning check remains in the candidate but **was not reached**. GPU metadata/HIP context initialization and cleanup occurred; there was no GPU tensor computation. The allocator-cap assignment follows the failed gate and was also not reached. No neural output or artifact directory was created. The reason for the device-level VRAM occupancy was not investigated or attributed to a particular application.

Six Windows observations were healthy. Genuine Job accounting was 0 active processes before and after cleanup; no forced termination. Only the named Modly Ubuntu distro was terminated successfully. There was no timeout, reset or native crash.

[Raw worker](amd-wsl-turret-neural-decode-tunableop-evidence/run/guest/results.json), [durable journal](amd-wsl-turret-neural-decode-tunableop-evidence/run/guest/phases.jsonl), [supervisor](amd-wsl-turret-neural-decode-tunableop-evidence/run/results.json), and [verified summary](amd-wsl-turret-neural-decode-tunableop-evidence/verified-result.json) preserve the exact refusal and measurements. The [exclusive marker](amd-wsl-turret-neural-decode-tunableop-evidence/neural-attempt.marker) is retained.

## Narrow change and preservation

Only startup environment assignments, independent all-false/string-zero checks, two durable diagnostics and fresh execution/artifact paths changed. `runtime.py` and `admission.py` are byte-identical to the original reviewed worker; `neural.py` differs only in its artifact destination. Five focused startup tests and marker-free read-only preflight passed. No previous installed-runtime acceptance, native-import proof or checkpoint-loading test was separately repeated.

The genuine full 3,743-row saved turret, 292-entry strict checkpoint contract, selective 216 BF16 / 76 F32 policy, `explicit_gemm`, original learned masks, 40-convolution/four-stage hooks, one-million-row ceiling and all memory/deadline/containment protections are retained. **Their decoder-execution behavior remains untested in this attempt.** The 12 GiB allocator ceiling was not lowered to evade the 13 GiB admission requirement.

Before/after checks matched both Windows environments, native binaries, AMD edits/backups, decoder metadata/config and four turret artifacts. The actual checkpoint was stream-hash verified at `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581` without deserialization. WSL baseline packages/venv, 922 staged payloads, 23 packaging files, 80 native source files and protected ROCm dependencies remain unchanged. All 482 protected historical/bytecode files matched. Main and all 55 prior local/remote branches are preserved. Both distros are stopped; docker-desktop remains default. Independent source and outcome review evidence is preserved in this directory.

**One next action:** make at least 13 GiB of device VRAM available, then obtain separate approval for one fresh genuine decode using this corrected worker and unchanged turret inputs/limits. Keep a margin above the threshold because device availability can vary. No additional synthetic or compatibility milestone is needed for this refusal. Do not automatically restart, retry, lower the gate, extract geometry or texture anything.
