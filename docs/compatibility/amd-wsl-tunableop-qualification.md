# WSL TunableOp policy qualification — PASS

Starting point: `bd8825b6ebd60e1bd26ae6bf689e51afe4c7bc94`, branch `codex/amd-wsl-turret-neural-decode`. This CPU-only qualification is on `codex/amd-wsl-tunableop-qualification`. The execution candidate was published and GitHub-verified at `97b7e5afcb1833eab7b8a35f51615b920c2b8d74`; its `probe.py` SHA-256 is `fa187428e4ec09b8b96b3922a89904b3460ee27319ab859391a47aae35286886`.

## Actual setting and correction

One fresh, monitored Ubuntu worker queried each genuine Torch flag independently:

| Flag | Before controls | After controls |
| --- | --- | --- |
| `is_enabled()` | false | false |
| `tuning_is_enabled()` | true | false |
| `record_untuned_is_enabled()` | false | false |

All three `PYTORCH_TUNABLEOP_ENABLED`, `PYTORCH_TUNABLEOP_TUNING`, and `PYTORCH_TUNABLEOP_RECORD_UNTUNED` settings were absent; no other inherited `PYTORCH_TUNABLEOP_*` setting was present. The worker called genuine `enable(False)`, `tuning_enable(False)`, and `record_untuned_enable(False)` once each. No flag functions or reported values were patched. The environment remained unchanged.

Thus the existing `is_enabled() or tuning_is_enabled()` assertion rejects this fresh runtime's enabled tuning-permission flag even though TunableOp itself is disabled. This proves the reproducible rejection mechanism, not that tuning or benchmarking occurred. The historical decoder attempt did not record its individual flags, so those historical values remain unknown.

The already-preserved installed [source](amd-wsl-turret-neural-decode-evidence/installed-tunable.py.txt), verified live at SHA-256 `b30632498091a445433ef2d9254c9a1b44ab3e329c6656f0beeae7bd65104a39`, states that environment settings override API controls and are cached on first read. There was no inherited override preventing disabling in this worker. No override experiment was performed.

## Execution evidence

The pinned runtime was Python 3.12.3, Torch `2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, and Triton distribution `3.5.1+rocm7.2.1.gita272dfa8` (metadata only; Triton was not imported).

Five focused host tests passed. Independent pre-execution and outcome reviews passed. The sole Torch qualification took **8.039 seconds**; supervision took **20.578 seconds**, within the 60/90/120-second worker/guest/host bounds. All five Windows observations were healthy. Guest/client exit was 0; Job accounting showed 0 active processes before and after cleanup, without forced termination. Only the named Modly Ubuntu distro was terminated successfully.

There were zero intercepted GPU-entry calls, zero tensor operations, zero policy rejections, and no model/FlexGEMM/Triton imports. Torch GPU initialization remained false. Guards and original function identities were restored. The three requested flag getters ran before and after the three genuine setters (once each); no GPU memory API, allocation, GEMM, native FlexGEMM binding or benchmark was used. GPU memory was deliberately not queried. Python instrumentation cannot inspect hidden implementation details inside C bindings; its evidence is the recorded entry/initialization state and source-pinned calls.

[Raw guest result](amd-wsl-tunableop-qualification-evidence/run/guest/results.json), [supervisor result](amd-wsl-tunableop-qualification-evidence/run/results.json), [verified measurements](amd-wsl-tunableop-qualification-evidence/verified-result.json), and [independent review](amd-wsl-tunableop-qualification-evidence/independent-outcome-review.json) preserve the exact observations.

## Preservation and next action

Both Windows environment inventories, native binaries, AMD edits/backup, decoder metadata/configuration, and four saved turret artifacts matched before/after. The WSL baseline venv/packages/dpkg, 922 staged payloads, 23 packaging files, 80 native source files, and protected ROCm dependencies remained unchanged. All 78 protected historical/evidence/previous-bytecode files matched their original hashes. The original 54 local branches remained at their original commits; publication verifies the corresponding remote heads. `docker-desktop` remained stopped and default. The previous decoder marker, candidate and guard were unchanged; no checkpoint was loaded.

The smallest next change is **worker-local configuration**, setting all three environment variables to `0` before importing Torch, then mandatorily querying all three genuine flags and requiring false before proceeding. That pre-import configuration is documented by the installed source but was not executed here. Keep the existing rejection guard and all decoder safety gates.

Read-only review found no other demonstrated nearby pre-forward defect: the prior CPU dtype/source/helper/workspace-binding checks passed; live placement, stream, memory and native execution checks remain untested. Do not preemptively relax them.

Recommend separate approval for exactly one real neural-only decode of the existing full 3,743-row turret using the already-prepared worker with that minimal configuration and runtime verification. No additional compatibility probe is proposed. Stop before any decoder loading or GPU execution in this milestone.
