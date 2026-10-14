# WSL launcher-path correction — PASS

Base: `511e135b7c7cf4fd3e4af30586e36a64524e6e24`. Branch: `codex/amd-wsl-launcher-path-correction`.

**One genuine installed-WSL CPU acceptance completed all 28 checks: 14 accepted CPU/meta routes and 14 rejections before original callbacks. No instrumented GPU-entry function was reached.** The guest exited 0, Windows monitoring stayed healthy, the owned Job emptied naturally, and only the named Modly Ubuntu distribution was terminated. No retries, FlexGEMM import, model loading, neural forward, installation or build occurred.

## Adapter-local correction

The acceptance adapter now resolves and imports `amd_bounded_gpu_proof.py` and `amd_forward_owned_job.py` from the anchored `ROOT/scripts/compatibility` path before invoking the supervisor or reserving a marker. A scoped helper callback serves those actual modules even while `smoke.SCRIPT` selects the separate acceptance guest. Unplanned helper names fail closed.

[Static preflight](amd-wsl-launcher-path-correction-evidence/static-preflight.json) exercised actual helper resolution after guest selection, verified the intended guest command and left the marker absent. [Execution preflight](amd-wsl-launcher-path-correction-evidence/helper-preflight.json) records the same real paths and hashes before the exclusive marker. Independent preflight review found no blocker.

Only adapter helper resolution changed. The original guest acceptance function is AST-identical. The DeviceContext policy, its 16 passing host regressions, shared supervisor and ownership/observer helpers have unchanged hashes. The earlier failed launcher evidence remains intact; this execution uses a new directory and marker.

## Actual acceptance

| Item | Observed result |
|---|---|
| Ubuntu Python | 3.12.3, `/opt/modly-wsl/venv` |
| Torch runtime | `2.9.1+rocm7.2.1.gitff65f5bc` |
| Torch distribution | `2.9.1+rocm7.2.1.lw.gitff65f5bc` |
| HIP / CUDA metadata | `7.2.53211-e1a6bc5663` / null |
| Triton | `3.5.1+rocm7.2.1.gita272dfa8` |
| Runtime checks | 28 completed: 14 positive, 14 negative |
| GPU-entry probes | Empty; `_lazy_init` was never reached |
| PyTorch GPU initialized | False before, after and at final cleanup |
| Constructor identities / guard restoration | Verified unchanged/restored |
| Guest / outer host exit | 0 / 0 |
| Owned Job | Zero active before/after cleanup; forced cleanup false |
| Windows observations | Five healthy; maximum start gap 5.016 s, maximum duration 1.266 s |
| Named distro termination | `Modly-AMD-Ubuntu-24.04`, exit 0 |

The genuine checks covered explicit and implicit CPU/meta creation, the exact prior `with torch.device("cuda"): torch.empty((1,))` case, explicit CUDA destinations, nested/default-device selection, explicit CPU override, Tensor.to/cuda/new_empty routes, CPU empty_like, foreign modes and malformed context metadata. Every negative check recorded zero original callbacks and zero GPU probes. Positive callback controls demonstrated that observation was active. Constructor-cache and frontend restoration assertions passed; excluded FlexGEMM/HIP-helper/model modules remained absent.

The outer command took **21.422 seconds**; the unchanged supervisor measured **20.500 seconds**. Existing limits remained 60 seconds after GO, 90-second Linux timeout with five-second kill grace, and 120-second host supervision. Binary GO, five-second monitored Job grace and Windows health checks were reused unchanged.

[Guest results](amd-wsl-launcher-path-correction-evidence/run/guest/results.json), durable [check journal](amd-wsl-launcher-path-correction-evidence/run/guest/phases.jsonl), [CPU verdict](amd-wsl-launcher-path-correction-evidence/run/CPU-verdict.json) and [independent outcome review](amd-wsl-launcher-path-correction-evidence/independent-outcome-review.json) contain the complete evidence.

The unmodified production GPU supervisor reports **UNKNOWN**, because this CPU worker supplied no vector-add or GPU-memory acceptance result. That raw result is preserved. The separate CPU PASS requires genuine route outcomes, no GPU entry and healthy/unforced containment; no GPU PASS is claimed or historical result relabeled.

## Preservation and next action

[Windows preservation](amd-wsl-launcher-path-correction-evidence/Windows-preservation.json) found zero differences in both AI environments, 12 native binaries, AMD source/edits/backups, four turret artifacts and decoder metadata/header. No checkpoint payload was loaded or re-hashed. [Passive Linux preservation](amd-wsl-launcher-path-correction-evidence/linux-preservation-after.json) confirmed unchanged baseline metadata, package inventory, dpkg and inspected sources. That read-only inventory imported no AI libraries and was not a second acceptance. A final [read-only registration check](amd-wsl-launcher-path-correction-evidence/final-wsl-registration.json) confirms Modly Ubuntu is stopped and `docker-desktop` remains the default; the acceptance recorded successful termination of only Modly Ubuntu. Previous evidence and untracked bytecode were preserved.

**Next task: separately approve the actual contained FlexGEMM native import using the existing staged wheel and reviewed native-risk controls.** Resolve the 18 real bindings and record authorized incidental HIP initialization, synchronization/VRAM and clean exit, with no convolution, kernel, model loading or inference. This milestone qualifies the CPU guard and launcher, not native compatibility or decoder workspace admission. Stop here; no synthetic test or native import follows automatically.
