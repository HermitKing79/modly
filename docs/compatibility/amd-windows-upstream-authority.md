# Windows supervisor → UpstreamJob authority bridge v1

**BLOCKED — the single native integration attempt issued no completion authority.**
Base: `7400e8c8b72e9e0984aa8e9b33619405be17b445`.
Branch: `codex/amd-windows-upstream-authority`.

The synthetic conditioning worker wrote its explicitly labelled output and the
pinned bootstrap reported child exit 0 and natural terminal success. The runner
then sent a due PING using an earlier `poll() == None` observation, after its new
completion validator had already consumed terminal success. The validator raised
`ValueError: Bound Windows completion: invalid heartbeat phase`. Cleanup sent
CANCEL and produced a second validator error. Neither error was a Win32 API
failure. The candidate is preserved unchanged; there was no repair or native retry.

| Boundary | Result |
|---|---|
| Successful Windows authority issuer | **BLOCKED** in native execution |
| Exact ticket/context/command/source binding | **PASS, CPU contracts**; native bound launch observed, successful acceptance not reached |
| One-use and forged/foreign/replayed authority rejection | **PASS, CPU contracts**; native negative checks not reached |
| Natural settlement acceptance | **BLOCKED** by diagnostics; both actual Jobs ended empty without forced cleanup |
| Native integration | **BLOCKED**, one invocation |
| Genuine upstream neural integration | **NOT QUALIFIED** |
| WSL / fresh AMD generation | **NOT RUN / NOT RUN** |

## Implementation and qualification

`UpstreamJob.bind_windows()` binds a detached command and exact pending ticket to
the original runner. A separate ticket nonce preserves the full product context.
Private opaque registries enforce one execution and one acceptance; source,
interpreter, predecessor, output-path, cancellation, resource and deadline checks
remain mandatory. Legacy `HostStageObservation` stays CPU-only and cannot accept
a Windows-bound ticket. Final shape completion still says
`genuine_execution_qualified=false`; the generated dispatcher is unchanged and closed.

Bound invocations use a private bootstrap stdout pipe. Worker stdout/stderr are
separate files; the saved protocol is a diagnostic copy, never authority input.
Native membership, creation/image identities, ordered controls, complete process
accounting, EOF, Job closure and receipt publication must succeed before sealing.
Mutable returned dictionaries cannot mint authority. This trusts the host Python
process and pinned bootstrap; it is not a sandbox against hostile same-user code.

There are **70 distinct focused passing CPU tests**, with **37 independently
rerun** against the frozen candidate. They cover registry and actual-runner flows
with mocked OS boundaries, cancellation, late exit, missing identity, forced
cleanup, forged diagnostic files and historical interfaces. They did not cover
the native terminal-before-bootstrap-exit heartbeat ordering exposed here.

## Actual measurements

The approved supported outside-sandbox command used the same Windows user with
administrator role disabled; no RunAs was used. The original 6 GiB process/Job,
12 GiB RAM and 2 GiB disk floors remained unchanged. The fixture used a 25-second
stage deadline and a 90-second outer deadline; longer production profiles were
not natively qualified.

- Host interval: **11.859 s**; runner: **7.204 s**; wrapper: **14.783 s**.
- Outer peak Job memory: **41,369,600 bytes (39.45 MiB)**; peak process: **26,144,768 bytes**.
- Stage peak Job memory: **21,663,744 bytes**; peak process: **9,846,784 bytes**.
- Minimum available RAM: **19,180,949,504 bytes**; minimum disk: **136,122,306,560 bytes**.
- **269 health readings**, including initial admission. Stage/outer final active
  counts were **0/0**, accounting for **6/9** processes, with no forced cleanup.

The runner receipt remains `failed`, `natural=false`, with `returncode=null`
and no `bootstrap_returncode` field. Child exit 0 comes from the authenticated bootstrap protocol; it must not
be substituted for a missing successful runner receipt. Cached identity rows
are not claims that every process's exit code was measured. No host stage
acceptance receipt exists and no native replay/foreign-ticket probe ran.

Private attempt: `artifacts/amd-windows-upstream-authority-7400e8c/cpu-once` in the
outer Modly workspace. Raw identities, controls, tracebacks and synthetic files
remain private. Published [execution measurements](amd-windows-upstream-authority-evidence/execution-summary.json),
[tests](amd-windows-upstream-authority-evidence/tests.json),
[preexecution review](amd-windows-upstream-authority-evidence/preexecution-review.json)
and [independent outcome review](amd-windows-upstream-authority-evidence/outcome-review.json)
retain hashes and scope without publishing raw process identities.

Preservation covers all 100 protected file identities, 78 bounded payload hashes,
the interpreter, accepted-master metadata and prior full-hash provenance,
historical evidence and caches. Large model/master payloads were not rehashed.
Main and all 130 prior remote heads are retained; only this new branch is published.

## One next step

Separately approve a narrow terminal-wait correction and CPU regression: after
authenticated terminal receipt, stop routine PING while continuing bounded pipe
EOF, actual bootstrap exit, cancellation, health and Job-settlement checks. Test
the observed `poll(None) → terminal → heartbeat due → later exit` ordering and
stuck-bootstrap/cancellation failures before requesting another native attempt.
Do not treat terminal success alone as process settlement or relax the authority
requirements. No further execution is authorised by this report.
