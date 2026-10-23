# Outside-sandbox Windows control pair: BLOCKED

Base `91420eec2dee0b8ac61cd30a845b7fee4f89ff23`; isolated branch `codex/amd-windows-lifecycle-outside`.

Exactly one native CPU pair ran through the accepted supported outside-sandbox execution request. The same qualified Windows user matched, with administrator role disabled. Both fixtures failed their unchanged five-second descendant-readiness condition. No retry, writer probe, WSL, GPU, model or full-orchestrator execution followed.

## Separate results

| Qualification | Result |
| --- | --- |
| Both fixtures authenticated ready before cancellation | BLOCKED: neither parent published `ready.json` |
| Target-only cancellation | NOT RUN: controlled cancellation was never admitted |
| Independent sentinel survival during target-only cancellation | NOT RUN |
| Sentinel normal release / natural success | NOT RUN |
| Target and sentinel natural settlement | NOT QUALIFIED: both stage Jobs required forced cleanup |
| Outer Job settlement | Active count zero; no forced **outer** Job termination; worker exited 1 |
| Complete Windows CPU control pair | **BLOCKED** |
| Current startup timing attribution | Established: grandchild fixture entry/publication occurred after the parent deadlines |
| Exact cause of interpreter-entry delay / earlier historical failures | UNRESOLVED |
| WSL ownership and fresh AMD generation | NOT RUN |

The final stage receipts say `terminal=cancelled`, `natural=false`, `forced=true`, `windows_settled=true`. These are failure-cleanup results, not successful authenticated target cancellation. Both cancellation acknowledgements were absent. Cleanup sent authenticated cancellation to both contexts; bootstrap stderr records the missing acknowledgement as `FileNotFoundError`, errno 2. That is secondary to readiness failure, rather than a captured native Job/identity WinError 5. The worker never recorded pre-cancellation readiness, survival observations or a sentinel release. An empty outer Job does not convert the inner forced exits into natural completion.

## What the timing actually establishes

All times below are relative to each parent's original post-Popen timer origin. The same 5,000,000,000 ns budget and inclusive timeout check were preserved.

| Event | Target parent | Sentinel parent |
| --- | ---: | ---: |
| Child fixture script entry | 1.922 s | 1.985 s |
| Child begins grandchild Popen | 1.984 s | 2.047 s |
| Child completes grandchild Popen | 3.875 s | 3.922 s |
| Child readiness publication completes | 3.875 s | 3.938 s |
| Parent last poll / timeout decision | 5.000 s | 5.000 s |
| Grandchild fixture script entry | 5.890 s | 5.985 s |
| Grandchild readiness publication completes | 5.953 s | 6.063 s |

At both timeout decisions the saved poll explicitly reports child-ready true and grandchild-ready false. Thus this invocation did not merely overlook an already-published grandchild marker. The descendants did run and their hardlink publications succeeded, but the grandchild fixture code entered too late. Script entry is not OS process birth: the observations do not distinguish interpreter initialization, launcher costs, scheduling or shared-machine contention as the underlying cause.

The six bounded fixture sidecars preserve parent entry, descendant Popen boundaries, polling, publications and timeout decisions. Nanosecond units do not create nanosecond accuracy: this runtime advertises GetTickCount64's 15.625 ms clock resolution. Late shutdown receipt serialization cannot explain an earlier readiness timeout; observer wall-time summaries alone do not prove the instrumentation had no scheduling effect.

## Preparation, resources and cleanup

The timing fixture, bootstrap and diagnostic supervisor were copied byte-for-byte. The corrected host changed only experiment paths, base/branch, independent error destination and authenticated source-pin map. Existing production and historical sources were untouched. Admission binds 13 dependencies, including the wrapper and focused tests. Initial and independent final-byte runs each passed 32 focused mocked CPU tests; one actual static preflight passed without reserving the attempt.

The accepted tool request used `sandbox_permissions=require_escalated`, meaning unsandboxed execution rather than Windows administrator elevation. The same-shell user/token guard and direct original API invocation were recorded privately. No RunAs, permission or security-setting change occurred. This mode evidence does not constitute an independent child-token inspection.

Actual host runtime was **24.609 s**, below 90 s. Across 1,649 saved health observations, minimum available Windows RAM was **16,508,915,712 bytes (15.375 GiB)**, above 12 GiB, and minimum destination free space was **131,745,546,240 bytes**, above 2 GiB. Outer Job peak aggregate memory was **124,989,440 bytes (119.199 MiB)**; peak process memory was **75,309,056 bytes (71.820 MiB)**, below the unchanged 6 GiB caps. Both stage runtimes remained below 25 s. Final outer accounting was empty, with no outer cleanup errors or forced outer termination; both inner stage Jobs were forcibly settled. Owned-memory release bytes were not measured; the cleanup evidence is process/Job accounting, not a measured zero-byte allocation claim.

Source, admission and interpreter checks remained unchanged. The [independent outcome review](amd-windows-lifecycle-outside-evidence/independent-outcome-review.json) evaluates actual private receipts, identities, marker contents, stderr, timing and cleanup rather than relying on the host's final status. Source-safe measurements accompany this report; raw process identities, nonces and traces stay local. Preservation receipts retain the original assets, protected100 scope, previous failures, environments and all prior remote heads.

## Recommendation and stop

Prefer replacing this recursive cold-interpreter-startup synthetic qualification strategy with a separately reviewed method that admits/measures startup before testing cancellation of an authenticated descendant graph. Do not rewrite or weaken production ownership checks on this evidence, and do not merely raise the five-second threshold or repeat this pair. The current data identifies the startup boundary but does not establish a production supervisor defect.

No further lifecycle investigation, WSL stage or GPU generation is authorised by this result. Stop with the control pair BLOCKED and all historical verdicts preserved.
