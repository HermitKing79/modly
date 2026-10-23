# Bound Windows terminal/heartbeat correction

**PASS — one synthetic Windows conditioning invocation completed naturally and
its exact UpstreamJob ticket accepted the supervisor authority once.**

Base: `ab8de19f400d688e31aa9c013b1b987282c1ae77`.
Branch: `codex/amd-windows-terminal-heartbeat`.

| Boundary | Verdict |
|---|---|
| Terminal/heartbeat correction | PASS |
| Focused regression tests | PASS, 65 distinct checks; 50 independently rerun |
| Actual bound Windows natural settlement | PASS, one invocation |
| Supervisor-issued authority | PASS, synthetic Windows CPU scope only |
| Exact conditioning ticket acceptance and one-use enforcement | PASS |
| Genuine neural execution | NOT RUN; genuine upstream integration remains NOT QUALIFIED |
| WSL | NOT RUN |
| Fresh AMD generation | NOT RUN; generated dispatch remains closed |

## Correction and CPU proof

The only production change is the bound-path PING condition in
`OwnedStageRunner._run()`: after an authenticated terminal is consumed, routine
heartbeats stop. The process remains under actual polling, cancellation, deadline,
RAM/disk, private protocol EOF/error, source identity, Job settlement and closure
checks. The strict `_WindowsCompletion.control()` validator is unchanged. No
bootstrap, authority, upstream-job, unbound or WSL production code changed.

The new actual-runner mocked regression first reproduced the original
`Bound Windows completion: invalid heartbeat phase` failure. After the correction,
13 new cases plus 25 existing authority/runner tests passed. Another 23 neighboring
ownership/validator checks and four qualification checks passed. The 50 independent
checks overlap these 65; they are not additional distinct tests.

The deterministic test consumes terminal success while `poll()` still returns
`None` and a heartbeat is due, confirms no post-terminal PING, then observes a
separate successful exit and EOF before authority issuance. Refusal cases cover
deadline expiry, cancellation, unexpected/duplicate records, nonzero bootstrap
exit, missing EOF/ownership, forced cleanup, missing/changed sources and
expired/reused authority. Ordinary pre-terminal PING remains exercised.

## One actual execution

The original nonrecursive fixture and coordinator were reused byte-for-byte.
Only fresh evidence paths, branch and base identities changed in the qualification
host/wrapper. Exact source review and static preflight passed before reservation.
The supported outside-sandbox command used the previously authenticated Windows
user with administrator role disabled and no RunAs.

- Trusted child exit **0**, actual bootstrap exit **0**, clean private EOF,
  successful natural terminal, and zero diagnostic errors.
- Stage and outer Jobs ended with **0 active processes**, accounting for **6/9**
  processes respectively. No forced cleanup; Job closure succeeded.
- Exactly **one synthetic conditioning stage** was accepted. Copied authority,
  copied ticket, worker-authored PASS, consumed authority and repeated invocation
  were rejected. The replay checks launched no additional worker.
- Nine PINGs were sent and acknowledged. No post-terminal heartbeat was accepted;
  the validator remained strict. The exact poll/terminal race ordering is proved
  by the deterministic regression, not inferred from the final native status.

| Measured resource | Result |
|---|---:|
| Host / runner receipt / wrapper interval | 11.110 / 6.750 / 13.952 s |
| Authority evidence / coordinator interval | 6.813 / 7.344 s |
| Outer peak Job / process memory | 41,517,056 / 27,131,904 bytes |
| Stage peak Job / process memory | 21,331,968 / 9,748,480 bytes |
| Minimum Windows available RAM | 19,225,452,544 bytes (17.905 GiB) |
| Minimum destination free space | 135,895,773,184 bytes (126.563 GiB) |
| Health readings, including initial admission | 272 |

The 25-second stage, 90-second outer, 6 GiB process/Job, 12 GiB RAM and 2 GiB
disk limits were unchanged. The runner receipt interval is sampled before final
health/source checks and sealing; the later authority and coordinator intervals
include more of those steps. Cached identity observations are not assertions that
every descendant's exit code was sampled: child exit 0 is authenticated through
the pinned bootstrap, while complete Job accounting establishes empty settlement.

## Evidence and preservation

Private raw evidence remains under
`artifacts/amd-windows-terminal-heartbeat-ab8de19/cpu-once` in the outer Modly
workspace. No private process identities, raw controls, model assets or generated
images are published. Source-safe records:

- [Prior failure authentication](amd-windows-terminal-heartbeat-evidence/prior-failure-authentication.json)
- [CPU tests](amd-windows-terminal-heartbeat-evidence/cpu-tests.json)
- [Exact preexecution review](amd-windows-terminal-heartbeat-evidence/preexecution-review.json)
- [Measured execution](amd-windows-terminal-heartbeat-evidence/execution-summary.json)
- [Independent outcome review](amd-windows-terminal-heartbeat-evidence/outcome-review.json)
- [Preservation after execution](amd-windows-terminal-heartbeat-evidence/preservation-after.json)

The previous BLOCKED result, its first PING error and secondary cleanup error
remain unchanged. Preservation covers 100 protected file identities, the same 78
bounded payload hashes, original interpreter, accepted-master metadata and prior
full-hash provenance, historical evidence and caches. Large model/master payloads
were not rehashed. Main and all 131 preceding remote branch heads are retained;
only this isolated correction branch is published.
The preservation receipt's no-native flag describes that read-only checker;
the separate native qualification is explicitly recorded above.

This qualifies the Windows CPU authority boundary, not neural stage semantics or
WSL ownership. Any further execution requires a separate milestone. No retry,
broader orchestration, GPU work or Generate enablement followed this attempt.
