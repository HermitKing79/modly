# Verified retired-process identity handling — correction PASS, lifecycle BLOCKED

Base `060d560e00d52d4e7ffbe2a4ef190cc1ccb37007`; branch `codex/amd-retired-process-identity`.

Exactly one CPU-only Windows control pair ran. The narrow observation correction passed source review, 53 focused mocked tests and saved native reuse checks. **The complete lifecycle qualification remains BLOCKED by fixture startup timeouts.** No correction, deadline change or retry followed execution.

| Requested verdict | Result |
|---|---|
| Terminated-process observation correction | **PASS**, scoped to reviewed code, mocked checks and 19 valid Job-scoped retired identities in the saved native observations. |
| Windows target cancellation | **BLOCKED**: setup never reached the admitted cancellation point. |
| Independent sentinel natural completion | **BLOCKED**: intended successful release never occurred. |
| Complete Windows control-pair ownership | **BLOCKED**: both inner stage receipts require forced cleanup. |
| Full AMD orchestrator | **NOT QUALIFIED** |
| WSL ownership | **NOT RUN** |
| Fresh GPU generation | **NOT RUN** |

## Correction and provenance

Only [`_OwnedJob._identity`](../../api/services/amd_stage_process.py) changes production behavior. A valid current process handle and live private Job handle are required. Membership and creation FILETIME are checked on that held handle, followed by a checked `GetExitCodeProcess` call. Native API failures retain their exact errors; unverified handles or membership fail closed. Status 259 never permits cache reuse.

For a positively terminated member, reuse requires the exact PID/creation key, private Job label and handle, and a nonempty image record originating from an earlier successful native `QueryFullProcessImageNameW`. Original query provenance is retained through repeated reuse; current exit evidence and reuse source are recorded separately. Missing, incomplete, foreign or changed identities fail closed. Status 259 still requires a native image query; access denied is not caught or suppressed. The existing advisory liveness field alone authorizes no reuse.

Access masks, Job settings, resource limits, accounting, cancellation acknowledgement, cleanup and natural-settlement rules are unchanged. Historical failure receipts remain unchanged. [Prior evidence authentication](amd-retired-process-evidence/evidence-authentication.json) confirms the previous image-query error 5, successful held-handle membership/creation checks, exit 0 before/after and exact prior image identity; it does not upgrade the older sentinel's unknown failing operation.

## Single actual run

Both original fixtures raised **`TimeoutError: Fixture descendants did not start`**, at line 50 of the unchanged fixture, whose descendant-readiness deadline is five seconds. Neither parent `ready.json` was produced. Child/grandchild readiness files exist after the run; why startup missed the gate is **unproved**. The worker stopped at `Original fixtures not authenticated ready`, before normal target cancellation or sentinel release. Cleanup cancellation is not the approved successful-path test.

Both inner receipts have natural=false, forced=true, returncode=1 and final active=0. The outer Job ended with active=0 and **no outer forced cleanup**. This does not make the inner lifecycle natural.

| Measurement | Actual |
|---|---:|
| Overall elapsed | 24.375 seconds |
| Peak owned process commit | 73,998,336 bytes |
| Peak outer Job commit | 114,720,768 bytes (109.4 MiB) |
| Minimum available Windows RAM | 17,455,673,344 bytes |
| Minimum destination free space | 135,188,844,544 bytes |
| Processes accounted / final outer active | 23 / 0 |

[Independent saved-data verification](amd-retired-process-evidence/independent-outcome-review.json) validates every recorded identity's provenance: 19 unique **Job-scoped** retired keys, not 19 unique physical processes. No invalid reuse, status-259 reuse or native API error was recorded. Eight keys also have separate saved native rows; the eleven outer keys retain successful native-source provenance emitted by the pinned implementation, rather than a separate saved outer observation history. This limitation is explicit.

Limits remain 6 GiB per process and Job, 12 GiB available RAM, 2 GiB destination free, original 25-second stage deadlines and the approved 90-second diagnostic maximum. No WSL, GPU, model, mesh or asset-promotion operation ran.

## Verification and preservation

- [53 focused tests](amd-retired-process-evidence/focused-tests.json) and [independent candidate review](amd-retired-process-evidence/independent-candidate-review.json) cover cache rejection, native failures, Job identity, status 259, cancellation/resource regressions and receipt transport. Native boundaries are mocked in unit tests.
- Static [size projection](amd-retired-process-evidence/receipt-size-projection.json) predicted 27,573,739 raw JSON bytes, exceeding the unchanged 16 MiB JSON reader. The copied harness writes a small verdict bound to the complete raw controls' bytes and streamed SHA-256. Actual raw size was 27,416,752 bytes. Original lifecycle criteria and full observations are retained; no JSON read cap was raised.
- [Execution summary](amd-retired-process-evidence/execution-result.json) and independent review preserve the exact failure and raw hash closure. Private evidence is `artifacts/amd-retired-process-060d560/cpu-once`, outside the proof repository. Raw identities, tracebacks and model assets are not published.
- [Preservation review](amd-retired-process-evidence/preservation-review.json): all 2,280 scoped identities, including 100 protected files, accepted master, selected environments, profile and backups match. All 120 pre-existing remote heads match before publication. Earlier failed raw receipts and historical cache directories remain unchanged. Only two existing source/test files changed; no installed environment or checkpoint was modified.

The next justified action is a separately approved, focused diagnosis of the fixture's five-second descendant-start readiness using the preserved timing evidence before considering another lifecycle run. Do not presume that more time is the fix, change the limit automatically, rerun the pair or begin the full orchestrator.
