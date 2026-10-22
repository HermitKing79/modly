# Windows sentinel WinError 5 attribution — BLOCKED overall

Base `20e79f4586491e8168e9f29c5dee52142c175e17`; branch `codex/amd-sentinel-winerror-attribution`.

Exactly one Windows-only diagnostic ran. **A new WinError 5 is precisely attributed, but the pair did not settle naturally overall. The original sentinel's exact failing operation remains unproved.** No repair or second execution followed this result.

| Question | Verdict |
|---|---|
| New reproduced native failure attribution | **PASS, scoped:** outer host observer's `QueryFullProcessImageNameW` failed with immediately captured error 5. |
| Original sentinel failure attribution | **UNRESOLVED:** its old receipt lacks a traceback/API/PID record. Similar symptoms cannot establish historical causality. |
| Complete Windows controls | **BLOCKED:** target natural cancellation passed; sentinel natural success and outer natural release did not. |
| Full orchestrator / WSL / GPU | **NOT RUN; previous verdicts unchanged.** |

## What the new evidence proves

The five-frame private traceback follows `host → snapshot → snapshot → _identity → _call`. The native caller is [`_identity`](../../api/services/amd_stage_process.py), line 178 in this candidate, during `host.observe`, before shutdown. The exact failing API is **`QueryFullProcessImageNameW`**; its false return and error **5** were captured before any subsequent liveness probe or cleanup API.

The same held process handle had just passed `IsProcessInJob` and `GetProcessTimes`. Its creation identity matches one sentinel-owned Python process, not a target process. A previously authenticated outer image record for that same identity exists. Raw PID, creation FILETIME, handles, image paths and full traceback remain private. A more specific descendant role is not established.

`GetExitCodeProcess` returned **0 before and after** the image-name query: this was a query against an already terminated, authenticated member. It was a real native API failure, not an inferred exit from an unsuccessful `OpenProcess`. The exact exit instant relative to Job enumeration is not established. Nor does this trace alone prove Windows' internal reason for denying the query. Creation, attachment, membership and time queries had succeeded; this observed failure is not a demonstrated Job creation or cross-job ownership failure.

Both fixtures were authenticated in separate Jobs before cancellation. There are **80** fresh sentinel observations within the recorded target-cancellation interval and a fresh sentinel observation after target settlement. The sentinel was then released. The outer observer error followed that release; host-error cleanup subsequently sent the sentinel a context-bound CANCEL. Its exit 0 without a cancellation acknowledgement cannot count as natural cancellation. Its missing-ack `FileNotFoundError` is a separate cleanup observation, not another WinError 5.

## Measured result

| Measurement | Observed |
|---|---:|
| Overall elapsed | 19.390 seconds |
| Peak owned process commit | 42,532,864 bytes |
| Peak outer Job commit | 113,909,760 bytes (108.6 MiB) |
| Minimum saved available Windows RAM | 18,017,431,552 bytes |
| Minimum saved destination free space | 135,250,321,408 bytes |
| Processes accounted / final active | 23 / 0 |
| Forced outer cleanup | **true** |

Target: cancelled/0, bootstrap/0, natural=true, forced=false, Job empty. Sentinel: cancelled/0, bootstrap/0, natural=false, forced=false, Job empty. Outer containment required forced Job termination. **Zero final processes does not upgrade this to natural settlement.**

Unchanged limits: 6 GiB process and Job ceilings, 12 GiB available-RAM floor, 2 GiB destination-free floor, 25-second original stage deadlines; this diagnostic has a 90-second overall maximum.

## Instrumentation, checks and preservation

- Checked native calls now retain API/category/access mask, immediate error, Python caller, historical versus current-handle identity, before/after exit status, Job label and phase. Full tracebacks and separate cleanup errors are private. Host sends and bootstrap receipts/exits are timestamped.
- Access masks, containment, gates, cancellation acknowledgement and natural-settlement criteria are unchanged. **One stricter diagnostic difference:** failed `OpenProcess` calls now fail closed instead of being silently skipped. Such a newly exposed failure would not itself explain the old surfaced exception. This run failed at the existing image-query boundary instead.
- [35 focused mocked tests](amd-sentinel-winerror-evidence/focused-tests.json) pass. They include stale/missing/misleading errors, API/category attribution, cached-PID ambiguity and primary-versus-cleanup preservation. Mocked contracts are not native execution proof.
- [Independent source review](amd-sentinel-winerror-evidence/independent-candidate-review.json), [execution summary](amd-sentinel-winerror-evidence/execution-result.json) and [independent saved-data review](amd-sentinel-winerror-evidence/independent-outcome-review.json) separate the new attribution from the historical unknown.
- [Preservation review](amd-sentinel-winerror-evidence/preservation-review.json): all 2,280 scoped identities, including 100 protected files and selected environment/app/profile/backups, match; original failed evidence is unchanged. All 119 pre-existing remote heads match before publication. No model/header load, environment change, WSL command or generated asset occurred. Historical cache directories remain untouched.

Private observations: `artifacts/amd-sentinel-winerror-20e79f4/cpu-once`, outside the proof repository. No raw identities, tracebacks, models or images are published.

## Smallest next correction to review separately

Review a retired-process observation rule: after current held-handle membership and creation checks, use a **previously authenticated image for the exact same PID/creation identity** when that held handle demonstrably reports termination. Preserve Job accounting and retired-process evidence. Live, unknown, changed-identity or uncached processes must remain fail-closed. Do not broaden access rights or globally ignore error 5.

This correction is **described, not implemented or qualified here**. Any correction and later Windows lifecycle qualification require separate approval. Do not rerun the full orchestrator or begin AMD generation.
