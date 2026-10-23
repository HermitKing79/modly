# Windows fixture readiness timing: prelaunch BLOCKED

Base `0c031befef0483f4ac3bed05fa35c3546166f255`; isolated branch `codex/amd-fixture-readiness-timing`.

**BLOCKED before native control-pair execution.** One diagnostic host invocation stopped while publishing its initial evidence. No owned Job, worker, target/sentinel pair, WSL stage or GPU operation was reached. There was no retry. The earlier failed lifecycle attempts and their BLOCKED verdicts remain unchanged.

## Qualified candidate

The historical 27-file attempt and reviewed timeline were authenticated. A separate diagnostic fixture and supervisor/bootstrap copies preserve the original ownership, nonce, cancellation and release logic. Historical and production files were not edited.

The diagnostic fixture uses an exact 5,000,000,000 ns readiness budget, with the original post-Popen origin, short-circuit readiness condition and inclusive deadline check. It records entry, launch, last-poll state, timeout, readiness write/flush/fsync/link and controlled exit boundaries. Bounded summaries distinguish earlier completed instrumentation writes from their own unmeasured final tail. Supervisor observations and protocol emission have scalar count/total/max timing; small shutdown summaries precede large raw receipts.

The original API interpreter passed **26 mocked CPU tests** and actual static preflight. Independent review passed 41 checks, including native Job/resource AST equivalence, unchanged cancellation/release predicates and normalized fixture logic equivalence. The [pre-execution receipt](amd-fixture-timing-evidence/independent-preexecution-review.json) pins the exact admission, executable, nine source identities and both test files. These checks qualify instrumentation and static admission, not successful native execution or the real filesystem writer.

## What happened

The coordinator invoked the reviewed host once through the sandboxed execution route. The controller returned code 1 after 2.219 seconds including interpreter startup. Its console contains only `{"classification": "BLOCKED"}`. The fresh private `cpu-once` directory contains `attempt.json.writing`; the final `attempt.json` was never published.

The pinned host calls `OUT.mkdir()` and `save(attempt.json, ...)` before entering the block that creates the Job. No worker, transport, stage, protocol or host-result files were produced. This establishes failure during initial attempt publication, before native pair admission.

The original entry-point handler writes its detailed preflight exception only when `OUT` does not exist. Because the directory already existed, this invocation discarded the exception details. A complete staged JSON document does not prove flush, fsync, close or hardlink publication completed. **The exact exception, Windows error code and failing filesystem operation are unknown.** Sandbox involvement is a launch-context fact, not proof of a particular permission or filesystem fault.

The partial staging file, console, launch receipt and all prior evidence are retained privately. Nothing was deleted, reused or overwritten to restart the attempt. [Independent outcome review](amd-fixture-timing-evidence/independent-outcome-review.json) confirms this boundary; the [saved-data classification](amd-fixture-timing-evidence/timing-evidence.json) leaves all unobserved startup measurements null.

The staged record reports initial available RAM **16,960,552,960 bytes** and destination free space **132,114,186,240 bytes**, above the unchanged 12 GiB / 2 GiB floors. These are prelaunch resource labels in an authenticated partial record, not a completed host receipt or live minima. No process/Job peak or owned-memory release measurement exists.

| Requested measurement | Result |
| --- | --- |
| Parent origin to both readiness publications | Unknown; fixtures never launched |
| Descendant launch and readiness-file I/O duration | Unknown; no descendants |
| Parent observation of published readiness files | Not run |
| Native supervisor observation / protocol costs | Not measured |
| Five-second deadline insufficiency | Not established by this invocation |
| Process/filesystem fault or observation-delay attribution | Unresolved |
| Target cancellation, sentinel survival/release, natural settlement | Not run |
| Peak process/Job memory or measured owned-memory release | Not measured; owned Job creation not reached |

The five-second fixture, 25-second stage, 90-second outer, 6 GiB process/Job, 12 GiB RAM and 2 GiB disk limits remain unchanged. No complete lifecycle, WSL or orchestrator PASS is claimed.

## Smallest next action

Before separately approving another native attempt, qualify the tiny evidence-publication operation in the intended execution context and ensure prelaunch exceptions are captured even after directory reservation. Preserve this partial reservation and use a new destination for any future approved run. Do not increase readiness timeouts or infer a startup root cause from this failure. If native timing remains impractical after that gate, simplify or replace the synthetic qualification method instead of repeating it blindly.

Source-safe saved-data outcome and independent review receipts accompany this report. Private process/model data is not published. Preservation before/after receipts cover the 100 protected files, accepted assets, installed environment identities, historical raw evidence and caches; publication must retain all 122 existing remote heads and add only this branch.
