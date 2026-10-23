# Fixture startup timing attribution

Base: `474fbb4f88fae957a007ce71fc640769bd570aae`. Saved-evidence analysis and mocked CPU contracts only. Production supervision, fixture code and deadlines are unchanged. No native lifecycle pair, WSL, GPU or orchestrator was launched.

| Question | Verdict |
| --- | --- |
| Startup-failure attribution | **UNRESOLVED**: the readiness failures are proved; their timing cause is not. |
| Actual descendant startup | **PASS**: both roles have authenticated child/grandchild writer identities and context-bound readiness records. |
| Instrumentation caused the failure | **POSSIBLE**, with measured work volume; no causal timing proof. |

All 27 historical raw files matched the committed file set, sizes and SHA-256 hashes (52,436,620 bytes). Five source identities match both the attempted source and this base. Raw process identities and traces remain private. See [timeline and authentication](amd-fixture-startup-evidence/timeline-evidence.json).

## What the saved clocks establish

The following times are UTC on 2026-10-22, rounded to milliseconds. Process creation is authenticated Windows FILETIME converted to UTC; it is not Python entry. Protocol labels precede serialization/flush. Current filesystem timestamps describe a staged inode, not the later hardlink publication, and were not captured in the historical hash manifest.

| Event | Target | Sentinel |
| --- | --- | --- |
| Runner GO-send label, before pipe write/flush | 23:18:11.769 | 23:18:12.601 |
| Bootstrap GO-received label | 23:18:14.340 | 23:18:15.220 |
| Stage-started label, after launching the stage executable | 23:18:16.020 | 23:18:16.910 |
| Child readiness-writer process created | 23:18:20.504 | 23:18:21.435 |
| Grandchild launch process created, identified by child receipt | 23:18:22.463 | 23:18:23.426 |
| Child-ready staged-inode mtime, supporting metadata | 23:18:24.320 | 23:18:25.293 |
| Grandchild readiness-writer process created | 23:18:24.324 | 23:18:25.297 |
| Grandchild-ready staged-inode mtime, supporting metadata | 23:18:26.349 | 23:18:27.355 |
| Parent entry, child-Popen start/return, five-second origin/expiry, public-file link completion | **Not recorded** | **Not recorded** |
| Bootstrap reports stage child exit code 1 | 23:18:27.311 | 23:18:27.295 |

Launch executables and actual Python writers have distinct authenticated creation identities; treating each launcher as immediate fixture entry would give false timing. The saved records do not contain a complete parent-PID tree. Other-image members consistent with the fixture parents were observed with exit code 1 at monotonic 119796.875 and 119797.781; assigning these members to the parents is an inference, not a recorded role identity.

The pair-admitted trace was monotonic 119782.296; the worker's readiness error was 119797.296, a 15.000-second interval. The first trace precedes thread starts and its own fsync, so this is not the exact worker deadline origin. Target and sentinel stderr independently contain the original parent's `TimeoutError: Fixture descendants did not start`. Neither parent produced its authenticated `ready.json`. Both descendant readiness records exist after the run; their existence alone does not establish when a parent could observe them. Grandchild file metadata later than the worker error supports late activity, with the clock limitations above.

Both inner Jobs required forced cleanup; the intended target-cancel/sentinel-release test was never admitted. The historical **BLOCKED** outcome remains unchanged. The outer Job emptied without forced outer cleanup. Neither inner natural settlement nor complete lifecycle/orchestrator qualification follows from this analysis.

## Five-second contract and observation work

The original parent creates its child first, then sets `until = monotonic() + 5`. It checks both readiness names in the `while` condition and checks the deadline only inside that loop. A timeout therefore proves an earlier condition check found at least one name absent and the following clock read reached the deadline. It does not prove descendant creation failed. Conversely, names becoming visible during a scheduling overshoot can satisfy the next condition after nominal expiry without triggering the inner deadline check.

[Four mocked contract tests](amd-fixture-startup-evidence/focused-tests-final.json) exercised the actual unchanged fixture main: absent markers, only one marker, publication during overshoot, and Popen time excluded from the five-second budget. All passed; process creation, clock, sleep and filesystem were simulated. An initial temporary-directory test harness failed sandbox cleanup; its original failure receipt/log is retained separately and is not evidence about the historical startup. Independent rerun passed all four tests.

[Overhead evidence](amd-fixture-startup-evidence/overhead-evidence.json) establishes:

- 637 target and 572 sentinel snapshots, averaging 26.63 ms apart; largest recurring gaps were 47/46 ms. Host observation had 421 samples, maximum gap 63 ms. This demonstrates service of those observer loops, not service of every fixture or worker thread.
- At least 90,014 kernel32 calls are implied by the pinned successful-snapshot paths, including diagnostic liveness probes; 72,830 precede the worker error. These are source-derived work counts, not profiled call durations.
- Snapshot dictionaries/history accumulate in memory. Large final receipt serialization occurs after the readiness failure and cleanup, so it cannot cause that earlier failure. Approximate post-label-to-return tails of 156/125 ms include serialization, scheduling and return work; they are not disk benchmarks.
- Startup still includes launch JSON, control-pipe JSON, regular-file bootstrap protocol writes, and readiness JSON/flush/fsync/link. Pre-error-stamped protocol records include 27/26 received PINGs. Their write durations and physical-disk costs were not recorded.

Saved UTC/monotonic anchors vary by about 16 ms and are sequential reads, not simultaneous clock calibration. Coarse gaps cannot isolate active CPU time, GIL waits, interpreter initialization, security scanning or fsync latency. No recurring multi-second observer gap is saved, but this does not rule out contention or scheduling delay elsewhere.

## Decision and smallest next measurement

**A — Genuine descendant startup failure:** contradicted as a claim that no descendants ran; successful timely readiness remains unproved. **B — Valid but late startup / too-short fixture deadline:** consistent with the records, but the missing parent origin and publication times prevent a defensible deadline revision. **C — Instrumentation delay:** possible; work is proved and causal cost is unmeasured. **D — Missing timing evidence:** established and currently prevents root-cause attribution.

The smallest justified change for a separately approved experiment is bounded monotonic instrumentation of parent entry, child-Popen begin/return, the five-second origin, the final readiness flags and timeout; and descendant entry, grandchild-Popen begin/return, and readiness write/fsync/link begin/completion. Use one UTC anchor per process, keep process-creation identities distinct, and buffer tiny records without per-poll disk traces. This would distinguish launch time from publication/observation time without weakening ownership or containment.

One further bounded native qualification is **conditionally justified after that instrumentation receives CPU tests and independent review**, with its first purpose being startup attribution. It is not authorised or executed here. No arbitrary timeout increase, observation removal or ownership relaxation is supported. A future fixture-only timing proposal must remain separate from the existing 25-second stage and 90-second outer limits. No present lifecycle PASS is implied.

## Verification and preservation

The [preservation receipts](amd-fixture-startup-evidence/preservation-before.json) cover 100 protected files, accepted master/application inputs, the original profile, selected environment identities and backups: 2,280 unique files / 2,283,191,348 bytes. Historical raw evidence and four existing cache directories are retained. The first preservation hash pass lacked a remote receipt after a sandbox network failure; its interruption is recorded and a completed BEFORE was then captured. No failed check was counted as PASS.

Final preservation and independent outcome receipts accompany this report. Only this report and new source-safe evidence/tests are published. The original fixture, supervisor, private raw records, assets and installed environments are unchanged. All 121 pre-existing remote branch heads, including main, are required unchanged at publication; the only added head is this isolated diagnostic branch.
