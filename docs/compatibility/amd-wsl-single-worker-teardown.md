# WSL single-worker teardown — UNKNOWN

Base: `codex/amd-wsl-stdin-correction`,
`c7ad0ddb9d5040030cfa6d615c228991a0d63726`.
Continuation: `codex/amd-wsl-single-worker-teardown`.
Exactly one CPU observation and one CPU confirmation; no GPU execution.

**Single-worker clean release was demonstrated. Overall diagnostic qualification
remains UNKNOWN:** the confirmation had one incomplete auxiliary identity query.
Neither launch required forced Job cleanup, and neither demonstrated a persistent
process-survival problem. The previous seven-case result remains unchanged.

## Exact observation

The genuine private Job contained these Windows processes. Parent IDs are sampled
Toolhelp metadata; executable paths, creation times and exit states come from
retained native process handles whose private Job membership was verified.

| PID | Executable | Sampled parent PID | Creation UTC (2026-10-14) | Exit |
| --- | --- | --- | --- | --- |
| 21400 | `C:\Windows\System32\wsl.exe` | 27952 | 09:47:27.568106 | Natural, 0 |
| 4896 | `C:\Program Files\WSL\wsl.exe` | 21400 | 09:47:27.582760 | Natural, 0 |
| 15848 | `C:\Program Files\WSL\wslhost.exe` | 4896 | 09:47:33.921162 | Natural, 0 |
| 10480 | `C:\Windows\System32\conhost.exe` | 15848 | 09:47:33.938552 | Natural, 0 |

`wslhost.exe` and `conhost.exe` exited **3.5601 ms and 6.1773 ms after** the first
Windows client, respectively, using native FILETIME values in one clock domain.
At the client-exit sample the Job was already empty. Its monitored natural grace
therefore completed immediately; named-distro termination was unnecessary for
Windows process release and occurred afterwards. `OwnedJob.finish()` reported
zero active processes, no forced cleanup and verified tree exit. Observation:
**PASS**, 16.39 s, four healthy Windows observations, maximum start gap 5.016 s.

The old two forcibly closed processes were never identified. This observation
supports a normal helper-exit race with immediate cleanup as the best explanation;
it does not retrospectively prove their identities or rule out every alternative.

## Narrow correction and single confirmation

The shared `amd_forward_owned_job.py` is unchanged: its `finish()` immediately
terminates a nonempty Job. Only the experimental WSL supervisor now gives an
already successful client a **maximum five-second monitored graceful wait** before
calling that original method. Persistent members still trigger forced containment
and prevent PASS. Failed Job queries remain sticky failures while healthy cleanup
monitoring continues. Failed observers cannot restore PASS. Host/guest/GPU
deadlines, binary GO, native runtime contract and GPU acceptance are unchanged.

The sole confirmation invoked the actual corrected production supervisor with
the standard-library CPU script, a separate exclusive CPU marker, and read-only
Job instrumentation. The real guest accepted GO and exited 0. The graceful wait
took **0.032 s**, then genuine Job accounting confirmed zero active processes,
unforced cleanup and verified tree exit. Named Ubuntu termination returned 0;
monitoring remained healthy. Total: **11.219 s**.

One auxiliary identity capture returned **`[WinError 87] The parameter is
incorrect.`** Its PID and failing API were not captured, so a raced lookup is
only a hypothesis. The adapter recorded this error without interrupting the
original Job queries or containment. Its strict CPU verdict remains **UNKNOWN**.
No retry or follow-up repair was performed. Separately, raw production UNKNOWN
is expected because the CPU guest supplies no GPU computation or memory evidence;
that GPU acceptance contract was neither bypassed nor simulated in this run.

## Verification, preservation and decision

**27 distinct focused CPU tests passed**, including actual production supervisor
replays for delayed exit, persistent members and failed Job queries. Native
read-only API/ABI checks and the diagnostic-error containment regression passed.
Relevant regressions were demonstrated failing before correction. Independent
review verified the code, tests and complete first observation; final outcome
review preserves the confirmation identity limitation.

Both launches imported neither Torch nor Triton. Linux package inventories and
all 26,617 venv metadata entries match the prior verified environment. Both Windows
AI environments, selected native artifacts, AMD source edits and backup were
independently checked. Driver: **32.0.31041.1004**; WSL kernel:
**6.18.33.2-microsoft-standard-WSL2**. All 112 historical WSL evidence files and
prior local/remote branches are preserved. Both distros are stopped;
`docker-desktop` remains default. The old GPU attempt marker is intact. No package,
driver, kernel, model or installed source was changed.

**No process-survival NO-GO was demonstrated.** Clean single-worker release is
supported by both genuine Job accounting records and the complete first identity
record. Full confirmation identity evidence remains incomplete. Review these
records before separately approving the existing minimal GPU smoke test; this
task does not authorize it. Do not build another process manager or rerun CPU
diagnostics automatically.

Evidence: [complete observation](amd-wsl-single-worker-teardown-evidence/observation/results.json),
[actual supervisor confirmation](amd-wsl-single-worker-teardown-evidence/confirmation/results.json),
[strict CPU verdict](amd-wsl-single-worker-teardown-evidence/confirmation/cpu-confirmation.json),
[focused tests](amd-wsl-single-worker-teardown-evidence/cpu-final.stderr.txt),
[adapter regression](amd-wsl-single-worker-teardown-evidence/adapter-green.stderr.txt),
[unchanged contracts](amd-wsl-single-worker-teardown-evidence/unchanged-execution-contracts.json),
[preservation](amd-wsl-single-worker-teardown-evidence/preservation-after.json).
Read-only API contracts: [private Job PID list](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list),
[process times](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[sampled parent metadata](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w).
