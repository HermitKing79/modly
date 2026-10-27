# V3 mutation denial and ownership-read failure

**NO-GO for another operator attempt.** Publish instrumentation and ownership hardening; grant no filesystem exception and issue no new execution handoff. The original audit event/path/origin and kernel cause of the permission failure remain unproven. No WSL, GPU operation, live admission, download or environment change occurred in this investigation.

## Preserved sequence

Case: `amd-pixal3d-operators-v3-kitchen-overlay-20261027T121514Z-615ba592624c45d88cd497d9e34d5a36`. All 27 original files are preserved, including empty stdout/stderr and cleanup logs.

| UTC on 2026-10-27 | Recorded evidence |
|---|---|
| 12:17:06.844562 | Controller records the worker launch/GO phase (before Popen and the pipe write). Controller PID/PGID 303, start ticks 328, exact case nonce. |
| 12:17:07.075992 | Worker begins imports; packaging overlay later reports PASS. |
| 12:17:13.180431 | Device admission begins; saved `gfx1201` result is PASS. |
| 12:17:13.265306 | `Kitchen-HIP-load` begins. |
| Before 12:17:15.552038 | Worker catches `RuntimeError: Destructive mutation forbidden in operator proof`; the exact error timestamp is absent. Its exception handler precedes the recorded release phase. |
| 12:17:15.552038 | Worker enters release. |
| 12:17:15.566933 | Successful resource sample still authenticates worker 438: PGID 303, start ticks 10564, exact nonce, RSS 753,102,848 bytes. |
| Untimestamped failing read | Controller records `PermissionError` for `/proc/438/environ` and takes the forced direct-child teardown branch. No failing-read state/identity recheck was saved. |
| 12:17:18.452533 | Later resource sample contains controller 303 only. |

This is a partial reconstruction from recorded phases and control flow, not a timestamped trace of every kernel transition. The worker mutation error had already been caught before release and the subsequent successful worker sample. The controller's missing failure timestamp prevents exact transition timing or causal attribution. Worker result mtime (12:17:16.433458) is filesystem evidence, not a timestamp of the denied operation.

The controller sets `child_forced_cleanup:true` / `natural_child_exit:false`. No preserved worker wait status proves which signal, if any, caused its final exit. The later external cleanup's `natural_exit:true` means that invocation found the old group empty without signalling it; it does **not** establish natural worker termination. Windows Job cleanup also reports no forced termination. Environment snapshots match exactly. HIP INT8/ConvRot, SDPA, NAF, sparse and conversion numerical operators are NOT_RUN.

## Worker: confirmed policy stop, unknown operation

The original error text comes from the unconditional deny for `os.remove`, `os.rename`, `os.rmdir`, `os.link`, `os.symlink` or `os.truncate`. The old hook saved none of the event, arguments, path, dir-fd, or caller frames. Empty stderr and the bounded child result cannot identify which of those six fired. We cannot classify the historical operation as legitimate scratch activity or an attempted protected-file mutation.

A separate harmless Windows CPU control installed the real audit hook and invoked stdlib `tempfile.gettempdir()` with a fresh owned TMPDIR. It captured and denied the actual `os.remove` event from `_get_default_tempdir`; the probe file remained present. This proves that a normal tempfile probe *can* produce the same generic error under the old blanket policy. It does **not** prove Kitchen caused that operation in the preserved run.

No new operation is permitted. New instrumentation writes event/arguments (including dir-fds), best-effort descriptor targets/identities, UTC/monotonic time, PID/thread, current phase, cwd and bounded caller frames to an exclusive preopened evidence file. It records filesystem denials before raising; canonical-path failures remain denied. Records are serialized, bounded (32 records, 64 KiB each), and flushed. Logging failure also denies the operation. Worker exception frames/timestamps are saved separately. Existing scratch create/write allowances remain; all six mutation families remain denied even inside scratch. V2, TRELLIS, original caches and unrelated writes stay forbidden.

## Controller: unstable observation and incomplete failure harvesting

Original `process_rows()` reads a numeric PID's stat, then environ and status without pinning the directory or rechecking start identity/state. PermissionError propagates and aborts supervision. Exiting-task access is a plausible explanation consistent with the sequence; an active task becoming unreadable is not ruled out. There is no saved evidence of the kernel permission/dumpability condition, so neither explanation is asserted as the cause.

V3 now uses its own reader, leaving historical V2 source unchanged. It pins `/proc/<pid>` with a directory descriptor, binds a process pidfd with flags **0**, and validates PID/PGID/start ticks around nonce/RSS reads. A terminal or disappeared leader is omitted only with process-pidfd proof that the whole process has exited: a zombie leader alone can have surviving native threads. Unknown/active permission failure, handle disagreement, PID reuse, missing RSS or unidentified disappearance stops the scan. Unrelated proc churn before membership can be established may therefore conservatively stop it.

Before any future GO, pidfd availability is probed against the kernel using a self handle, then closed without signalling. External group-member cleanup validates the complete member set and rechecks each pinned identity before signalling retained **per-process pidfds**. There is no numeric PID/PGID signalling fallback for that group cleanup. Direct unreaped Popen-child teardown retains parent/child wait ownership and records uncertain outcomes. Linux documents that [pinned proc descriptors do not retarget reused PIDs](https://docs.kernel.org/filesystems/proc.html), [process pidfds identify an instance](https://man7.org/linux/man-pages/man2/pidfd_open.2.html), and [pidfd signals avoid numeric-PID targeting](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html). A [thread can exit while other process threads continue](https://man7.org/linux/man-pages/man3/pthread_exit.3.html), which is why terminal-leader state is insufficient.

Controller teardown exceptions are recorded without bypassing worker-result harvesting. Worker diagnostics remain separate; they cannot overwrite a controller error or promote its FAIL to PASS. This also corrects the original reporting loss: the saved worker result shows gfx1201 PASS and HIP-load FAIL even though the controller exception bypassed the old normal-result merge.

## Offline verification and stop decision

The focused suite ran **138 cases: 137 passed, one optional unchanged 13-wheel audit skipped**. Thirty-two new audit/proc cases cover all six denials, allowed scratch creation, protected/escaped paths, logging failure, permission races, disappearing processes, duplicate/foreign nonce, PID reuse, whole-process exit versus zombie leader, kernel pidfd refusal, stable signal targeting, teardown failure and controller/worker status separation. The separate real tempfile audit control passed. No Torch/Kitchen/native modules, real Linux proc state or signals were used in those fixtures.

Independent review identified disappearing-target, surviving-thread, kernel-capability and teardown-reporting gaps; each was corrected and tested. Target-kernel behavior is still unverified because WSL was not launched. All original RAM/VRAM/disk/process/deadline/owned-cleanup bounds and the 22 GiB neural gate are retained; the operator profile remains unchanged. Existing assets, 3,346 historical hashes, 100 protected identities, 88 retained records, cached files and old Git heads were checked for preservation.

PowerShell, host, controller and worker operator routes explicitly hold execution. Cleanup remains available with strict ownership verification. No replacement thresholds, scratch-mutation exemption or speculative operator retry are introduced.

**Minimum next action:** review the instrumented code and define a separately authorised, narrowly bounded capture of the actual denied Kitchen operation before deciding whether any precise scratch exception is justified. Do not rerun the numerical GPU proof. No new operator command is qualified by this report.
