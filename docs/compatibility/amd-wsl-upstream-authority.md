# WSL supervisor → UpstreamJob authority v1

Base: `7152764593b17c338b907698abb80e2b18542797`. Branch: `codex/amd-wsl-upstream-authority`.

| Boundary | Result |
| --- | --- |
| WSL authority implementation | PASS |
| Focused CPU tests and independent source review | PASS |
| Single native WSL conditioning ticket | PASS |
| Natural Linux and Windows settlement | PASS |
| Exact one-use UpstreamJob acceptance | PASS |
| Windows authority | Previous PASS preserved; regression tests pass |
| Genuine neural inference | NOT RUN |
| Fresh image-to-3D | NOT RUN |

This qualifies one explicitly synthetic, nonrecursive, standard-library CPU conditioning stage. It does not qualify model/configuration evidence, neural stages, complete upstream generation or AMD Generate. The accepted receipt is `WSL_CPU_SUPERVISED`, with `genuine_execution_qualified=false`; no complete shape job was issued.

## Implementation

The existing private invocation registry now supports `bind_wsl` and `accept_wsl`. It binds the original ticket, full product context, independent control nonce, guest command, named distro/UID, sources/configuration/predecessor, original runner/evidence directory, cancellation, deadline and resource policy. Wrong-backend or copied consumers cannot consume the matching ticket's authority.

A WSL-specific completion validator shares native Windows Job observation with the existing Windows validator. Authority requires current authenticated client identity before START, exact broker/leader/boot identity, native executable/argv/RLIMIT observation, authenticated control ordering, successful reaping, empty Linux ownership, successful client/bootstrap exits, private-pipe EOF, empty closed Windows Job and unchanged sources. The writable protocol mirror is used only to time the fixture release; it cannot issue authority. Routine bound heartbeats stop after terminal while exit, cancellation, deadline and resource observation continue.

The original bootstrap and historical readers are unchanged. Two narrow compatibility measures retain their limits: bound WSL receipts use lossless compact JSON under the original 4 MiB reader cap; only the exact identity/hash-pinned System32 `wsl.exe` accepts its original Windows servicing link count. Ordinary single-link source/artifact checks remain strict. Historical Windows and unbound paths retain their behavior.

## Tests and actual execution

[CPU evidence](amd-wsl-upstream-authority-evidence/cpu-tests.json): **117 test methods**, zero failures/errors, including 40 bound-runner scenarios, Windows regressions and negative authority/lifecycle cases. An independent subset of 31 methods, including those 40 scenarios, passed in 74.968 seconds. [Pre-execution review](amd-wsl-upstream-authority-evidence/preexecution-review.json) passed before native launch.

Static preflight passed. Exactly one native stage ran through the supported outside-sandbox override under the previously authenticated Windows user, with administrator role disabled and no RunAs. The host accepted its exact ticket once. Copied authority, copied ticket, worker-authored PASS, consumed-authority replay and repeated invocation were rejected without another worker launch.

| Actual measurement | Value |
| --- | ---: |
| Outer host elapsed | 10.328 s |
| Production runner receipt elapsed | 6.765 s |
| Linux broker stage elapsed | 0.126209 s |
| Outer peak process / Job memory | 25,182,208 / 43,040,768 bytes |
| Stage peak process / Job memory | 9,682,944 / 21,098,496 bytes |
| Sampled Linux aggregate RSS peak | 28,467,200 bytes |
| Initial Windows available RAM | 19,687,133,184 bytes |
| Minimum observed outer available RAM | 17,060,290,560 bytes (15.889 GiB) |
| Minimum observed destination free disk | 135,617,024,000 bytes |

Linux reported normal successful leader reaping and zero owned survivors. Client and bootstrap exited zero; stage and outer Windows Jobs were empty. No forced cleanup, diagnostic errors or cleanup errors occurred. Original 25/90-second deadlines, 6 GiB process/Job and Linux policies, 12 GiB RAM floor and 2 GiB disk floor remained unchanged. [Execution summary](amd-wsl-upstream-authority-evidence/execution-summary.json) binds the private observations by SHA-256; [independent outcome review](amd-wsl-upstream-authority-evidence/independent-outcome-review.json) checks the saved observations rather than relying on PASS labels.

The synthetic output remains private under `artifacts/amd-wsl-upstream-authority-7152764/cpu-once/synthetic-evidence/conditioning/`, SHA-256 `e2c362d04e7ab17580900405332d0e0d6621c5a5f010489646ef37ea5476a893`. It is not a newly generated model.

## Preservation and limits

[Before](amd-wsl-upstream-authority-evidence/preservation-before.json) and [after](amd-wsl-upstream-authority-evidence/preservation-after.json) checks preserve all 100 protected files, accepted assets, historical evidence/caches, environments and main. All 133 pre-existing remote branch heads are retained; publication adds only this branch. Raw process identities, control nonces, traces and synthetic output stay private.

This retains the trusted-host, non-adversarial same-user broker assumptions. Linux RSS is sampled rather than atomically cgroup-limited; Windows Job limits do not cap guest memory. The broker observes the Linux interpreter's native path/arguments, without hashing that interpreter binary. This single nonrecursive proof does not expand prior descendant or cancellation qualifications.

Stop here. A subsequent genuine upstream integration would need separately reviewed model/configuration receipts and execution/cancellation admission; this milestone does not enable AMD Generate or authorise another worker.
