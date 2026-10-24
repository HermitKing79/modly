# Windows subprocess audit repair for the standalone RAM diagnostic

Repair base: `1553e73a7bf27ff2de0f331861595e02b0b9a30e`.
Branch: `codex/amd-zombie-windows-audit-repair`.

## Confirmed cause

The live failure occurred after original-source authentication, before project CPU preparation. A harmless Windows Git-read probe reproduced the exact guard rejection. CPython 3.12.10's Windows `subprocess._execute_child` formats argv with `list2cmdline()` before emitting the `subprocess.Popen` audit event. For `check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, timeout=15)`, the actual event contains `executable=None` and `args='git rev-parse HEAD'`. The old validator required a list/tuple and a named executable; either assumption rejected this legitimate read. The old tests exercised blocked commands, not a successful Git read through the real Windows audit path.

The new Windows regression tests first failed against the original implementation: all five legitimate variants were rejected. The fixed tests execute actual harmless `git rev-parse HEAD` / `git show <commit>:README.md` reads through the installed audit hook. They confirm the actual string payload, the implicit `None` executable, and returned commit/source content. They do not run the diagnostic, query RAM, import project/model modules or stream checkpoints.

## Minimal policy repair

Only subprocess command/identity validation and Git pinning change. On Windows, the validator recognises canonical command strings for `git`, `git.exe`, or the pinned absolute Git path. The permitted tails remain exactly `rev-parse HEAD` or `show <40 lowercase hex characters>:<previously permitted relative path>`. Absolute executable quoting comes from `subprocess.list2cmdline`; arbitrary command lines are never tokenised or interpreted. Extra whitespace, quote variants, options, operators, newlines and traversal remain rejected.

An implicit executable is accepted only on Windows and only after that narrow command match. Both argv[0] and the effective executable must resolve to the Git identity pinned when the guard was installed. Exact checkout working-directory validation is retained. The audit hook stays installed, and the existing memory callback still runs immediately before every allowed subprocess. Shell wrappers, WSL, foreign executables and Git mutations remain denied.

Independent review identified an additional identity issue in the initial candidate: Windows can find an implicit executable in the application, parent-current, system or Windows directories before PATH. Checking `shutil.which('git')` alone does not enforce the pin, and PATHEXT can select a different suffix. See [Microsoft's CreateProcess search rules](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw). The final repair pins a real `git.exe`, rejects any foreign `git.exe` in those pre-PATH directories, confirms PATH's `git.exe` matches the pin, and fails closed if native directory queries fail. Explicit Windows application names must be absolute. Historical implicit Git reads remain supported; no alternate executable is launched to test resolution.

All preparation phases, neural import/call/write restrictions and Git timeouts are unchanged. The PowerShell launcher is unchanged. The established three-reading 22 GiB admission, final 22 GiB pre-neural guard, preferred-only 24 GiB headroom, original 12 GiB live floor, disk floors and deadlines remain intact.

## Verification and preservation

The focused Windows suite covers five successful read variants and twenty-three denied audit events, plus rejection of a non-exe Git pin. Cases include mutation, argument/raw-string injection, shell execution, foreign executable/argv[0], incorrect cwd, WSL, traversal, noncanonical strings, pre-PATH shadows and native query failure. Negative tests retain CPython's actual event formatting while mocking the final Windows CreateProcess boundary, so forbidden commands cannot execute even if validation regresses. Shadow tests additionally simulate native lookup directories without writing into Python or Windows installations. The previous 15 synthetic child tests and 14 PowerShell scenarios are also retained. Launcher and child dry-runs are safe; the real diagnostic remains unexecuted.

The failed diagnostic's nine private evidence files are snapshotted before repair and checked again afterward by size, modification time, file identity and content hash. Raw memory/process evidence remains local. The inherited 100 protected identities, 78 bounded hashes and 88 retained zombie records are checked separately. Existing environment files and historical branches are preserved. See the accompanying repair verification/review receipt for final results; no live preparation success or memory attribution is claimed.

Final independent review of code commit `85c236241f07f009b5483fd6f77f58b261077196` found no remaining Critical, Important or Minor findings and independently reran both Python suites successfully. Author verification additionally passed the 14 PowerShell scenarios, both dry-runs, all 256 original source hashes, unchanged launcher hash and unchanged unrelated preparation/safety/timeout AST nodes. Preservation matched all protected/retained records and all nine failed-run files. The [source-safe verification and review receipt](amd-zombie-windows-audit-repair-verification.json) records the scope; raw private receipts stay local.

## User execution after closing desktop applications

Close Codex Desktop, ChatGPT Desktop and browsers. In a normal, non-administrator PowerShell terminal:

```powershell
$repo = 'C:\Users\wattz\Documents\ChatGPT\modly\proof'
$repair = (git -C $repo rev-parse codex/amd-zombie-windows-audit-repair).Trim()
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\compatibility\Invoke-ZombieRamDiagnostic.ps1" -Run -ExpectedCommit $repair
```

The launcher requires the checkout to match the published repair, performs fresh admission, and creates a unique local evidence directory. The failed run is never reused or overwritten. A successful Git audit regression test is not a completed CPU diagnostic or neural admission. No neural execution follows automatically.
