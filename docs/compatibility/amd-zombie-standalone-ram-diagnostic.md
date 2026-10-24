# Standalone Windows zombie CPU preparation RAM diagnostic

Implementation based on verified checkout `28f4f0163833d59d861404719e6b2cb042083f9e`.
**The real diagnostic was not executed during implementation. No neural attempt is authorised.**

The user reports increased available physical RAM after closing Codex, Brave and unnecessary applications. These observations support application overhead contributing to insufficient headroom. The raw readings remain private. They do not identify each application's exclusive contribution, measure the historical preparation process, or replace fresh admission.

## Run after closing desktop applications

Use the normal Windows user, without administrator elevation. Close Codex Desktop, ChatGPT Desktop and browsers first. Open a normal PowerShell terminal and run the following; it selects the published implementation branch's exact local commit and requires the checkout to match it:

```powershell
$repo = 'C:\Users\wattz\Documents\ChatGPT\modly\proof'
$implementation = (git -C $repo rev-parse codex/amd-zombie-standalone-ram-diagnostic).Trim()
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\compatibility\Invoke-ZombieRamDiagnostic.ps1" -Run -ExpectedCommit $implementation
```

The execution-policy option affects this PowerShell process only. No environment installation or persistent setting change is needed. The launcher uses the original pinned interpreter at `D:\AI\Trellis2AMD\venv\Scripts\python.exe`. Without `-Run`, the launcher only prints its dry-run plan. `-SelfTest` uses synthetic admission values and cannot be combined with `-Run`.

The launcher obtains three new `GlobalMemoryStatusEx.ullAvailPhys` readings with four seconds between them. Every reading must be at least **23,622,320,128 bytes (22 GiB)**. Equality passes. Any failure stops immediately without starting Python or retrying. **24 GiB is preferred only.** The existing 22 GiB final pre-neural guard and original 12 GiB live-host floor remain unchanged in their historical files. This diagnostic uses the stronger 22 GiB throughout its own CPU preparation. It retains the original 50 GiB D: and 2 GiB evidence-volume disk floors and 480-second overall host bound. Git reads retain the existing 15-second timeout.

## What is measured

PowerShell starts an external journal before the Python process exists. It records system-wide available physical RAM, its own memory, the child working set/private commit and OS working-set peak. It requests samples every 200 ms; Windows scheduling and process snapshots can lengthen the actual interval, whose timestamps are retained. Best-effort private snapshots of visible concurrent processes occur at admission, phase boundaries, approximately every two seconds and after exit. Inaccessible processes are explicitly recorded.

Python is launched with `-B -I -S`: no bytecode writes, user/environment path injection, sitecustomize or `.pth` startup hooks. Its stdlib bootstrap is covered by the external observer. After the guards are installed it adds the original environment's site-packages directory without processing startup hooks. Consequently this measures a controlled CPU-only preparation path, not an exact replay of unrestricted Python site startup.

Measured phase boundaries include:

1. Python startup, before project imports, and streaming authentication of 256 unchanged original Python sources.
2. Historical zombie adapter imports, including CPU-library imports.
3. CPU runtime construction: original first complete `source_inputs()` lineage validation, then the same generic run-contract parsing. The generation `Runtime`, guest decoder module and supervisor are not constructed.
4. The original repeated source lineage validation.
5. Source-verifier imports, committed-source checks and preserved control/loader hashes.
6. Full decoder checkpoint SHA-256 streaming in 1 MiB blocks. Only `checkpoint_record()` is compiled from the historical source into a restricted namespace; the decoder module body and entrypoint never run. Hashing reads weights as opaque bytes, without deserializing or loading model weights.
7. Metadata-only loader imports, component/header/configuration construction, source hashes, unloaded-component release and saved dependency-qualification checks. No Torch, model constructor or model code is imported.
8. Child process exit followed by recovery observations immediately and approximately 4, 8 and 12 seconds later.

The CPU subset deliberately excludes guest-module imports and supervisor construction, which contain neural execution authority. It reuses the accepted lineage readers and validation semantics. Nested phase intervals overlap; do not sum their deltas as independent allocations. The original first lineage validation is isolated because historical telemetry began after it.

## Fail-closed boundaries

Before and after every phase the child obtains a fresh physical-RAM measurement and an observer permit. A recorded value below 22 GiB cannot be overridden by a later recovered value. Checkpoint streaming checks between every 1 MiB read. The external observer also checks during work; a low reading or deadline/disk rejection stops the owned diagnostic Job, including any Git descendants. This is reported as forced diagnostic cleanup, not natural exit. No unrelated process or WSL distribution is terminated.

The child blocks neural import roots, generation/model-loading function calls, network actions and filesystem mutations. Writes are restricted to the new private evidence directory. Its only permitted subprocesses are exact bounded `git rev-parse HEAD` and `git show <40-character-commit>:<relative-path>` reads from this checkout. The launcher never executes an existing generation script. The guards are defence in depth for pinned reviewed code, not a general-purpose security sandbox.

## Evidence and interpretation

Each invocation exclusively creates a new directory:

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-standalone-ram-<UTC timestamp>-<UUID>`

`admission.json`, `observer.jsonl`, `phases.jsonl`, `child-result.json`, `summary.json` and `report.txt` contain the timing and memory evidence. `processes-private.jsonl`, stdout and stderr stay in the same local directory. Some files are absent when admission fails before the child starts. The launcher prints the exact directory and outcome; non-completion exits with code 2. There is no automatic retry or neural continuation.

The textual report gives available-RAM, working-set and private-commit changes per completed phase, sampled peaks and post-exit recovery. The observer covers startup before the child can record its own first sample. Concurrent-process snapshots allow comparison with host changes, but do not provide exclusive physical-memory accounting: working sets share pages, private commit is not physical residency, filesystem caching can affect availability, and some host processes are inaccessible. Sampled private-commit peaks can miss short transients. A stage-correlated increase plus matching process changes and exit recovery is stronger evidence than timing alone; residual host changes must remain unattributed when the measurements do not explain them.

No additional RAM threshold has been derived or introduced. Assess any future operating margin from the measured peak decline, available variability, process growth and recovery, with these limits stated. A single successful CPU preparation cannot establish reliability or authorise a later neural attempt. That stage still needs its existing fresh three-reading admission, final 22 GiB pre-neural check and unchanged runtime protections.

## Authoring verification and preservation

Only dry-runs, PowerShell parsing/C# declaration compilation and synthetic tests were executed: 15 Python tests and 14 PowerShell scenarios. The tests exercise exact-22-GiB admission, low first/second readings, phase failures and recovery races, missing observer permits, in-stream aborts, pure checkpoint-function extraction, permitted CPU stage ordering and actual import/call/process/write/network guard rejection against mocked OS boundaries. Launcher tests also cover phase-permit order, disk/deadline failure, owned cleanup, unsafe receipts, bounded Git timeout and incomplete settlement. No real project preparation, checkpoint verification, model import, WSL call or neural reservation occurred.

Independent review of implementation commit `3741629` found two Important launcher defects: initial Git reads were unbounded and outside the evidence block; failed recovery/stderr settlement could retain a successful outcome. The follow-up fix puts the read-only Git operations under a 15-second process bound and owned containment inside the evidence block, and makes settlement errors force a non-success outcome. Synthetic regression cases exercise both fixes. Native Job behavior and actual CPU preparation remain deliberately unexecuted and cannot be claimed as live-validated.

All existing tracked files remain unchanged; only new files are added on `codex/amd-zombie-standalone-ram-diagnostic`. The preservation check compares all 100 protected identities and 78 inherited bounded hashes, nine original failed-run files, five saved shape artifacts and 74 other retained records against authenticated historical receipts, before and after authoring. The already documented Windows volume-ID representation mapping is used solely for identity comparison; sizes, file IDs, timestamps and hashes must still match. This is not a fresh full hash of large protected checkpoints, models or the accepted master.

Source-safe verification, preservation and independent review receipts accompany this report. Raw memory/process evidence and private preservation details are not published. Existing local and remote branch heads must remain unchanged when the one new branch is published.
