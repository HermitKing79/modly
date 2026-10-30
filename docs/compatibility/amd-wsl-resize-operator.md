# Phase 6B — attended native resize handoff

30 October 2026. **Preparation only; execution HOLD.** This implements the
[reviewed runbook](amd-wsl-resize-preparation.md) at
`b909cb17350e0fd00a7c19840fba310edaa28186`. The project lead accepted its
non-cancellable storage critical section. Separate approval to execute is still
required. No WSL, live admission, original VHDX access, resize or decoder ran during this
preparation. The default launcher does nothing.

## One conditional operation

Launcher: `scripts/compatibility/Invoke-ModlyWslResize.ps1`. Use an ordinary
64-bit Windows PowerShell console under the distro-owning Windows account.
Close Codex and unnecessary browsers yourself for RAM headroom; do not start
Docker or other WSL work. All distributions and the utility VM must already be
stopped. The launcher refuses conflicting activity and never shuts it down.
Use stable power, prevent sleep, and keep the console open and attended until
natural settlement. An inherited external Windows Job is refused; there is no
breakaway or elevation fallback.

The final delivery supplies the exact published commit for this template:

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\wattz\Documents\ChatGPT\modly\proof\scripts\compatibility\Invoke-ModlyWslResize.ps1' -Run -ApproveResize -AcceptNativeCriticalSection -ExpectedCommit '<published-commit>'
```

Do not execute before separate approval. `-ExecutionPolicy Bypass` is confined
to this process. No administrator elevation, installation or configuration
change is requested. Permission failure stops the attempt. The accepted local
backup receipts and both private authority files must remain available; this
is a handoff for this authenticated machine, not a portable resizing tool.

Before mutation the runner checks the exact checkout, pinned base Python and
WSL client/service binaries (WSL 2.7.12.0), all 23 accepted backup receipts,
registration, physical volumes and original eight box inputs. It obtains an
exclusive read-only source handle and a read-only backup handle denying writes
and deletion. Fresh complete SHA-256 passes must both equal
`36d8679eaad94fdfa7b0cc12a48c728ee4ec73f83de90d339005295852365c5c`, with
29,581,377,536 bytes and the accepted identities/32 GiB metadata. Hashing uses
4 MiB buffers, the 12 GiB host floor and a 512 MiB observer commit ceiling.
The E: backup remains held read-only through completion. No copy is made.

After hashing, a second cold inventory and three fresh GlobalMemoryStatusEx
readings **each >=22 GiB, four seconds apart**, precede release of the source
handle. A final >=22 GiB reading immediately precedes the sole native command:

```text
C:\Windows\System32\wsl.exe --manage Modly-AMD-Ubuntu-24.04 --resize 68719476736B
```

The fixed new evidence directory is
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-wsl-resize-64g-v1`.
It must be absent. A consumed directory stops execution; nothing is overwritten,
deleted, resumed, retried or redirected. Timestamped RAM/disk observations,
fsynced journal, exclusive attempt receipt and stdout/stderr files exist before
native launch. The owned client PID and natural exit are recorded.

## Preserved limits and critical section

- 24 GiB is preferred only. No replacement threshold is introduced. Earlier
  WSL startup declined by approximately 2.354 GiB while still falling; that
  observation is not a maximum or a promise of adequate headroom.
- >=22 GiB is required across the entire native invocation for PASS, with the
  original 12 GiB emergency floor retained. Polling is about four times per
  second; normal durable samples are bounded to about one per second and every
  observed breach is recorded. Unsampled transients cannot be excluded.
- D: must retain >=50 GiB. Preflight also reserves the full 64 GiB target above
  that floor, conservatively covering growth. C: evidence and E: each require
  >=2 GiB free. Virtual size does not imply equal physical allocation.
- The 1800-second preflight deadline stops before mutation. The native
  1800-second window, observer 512 MiB ceiling and 16 MiB output budget are
  **fault checkpoints**, never cancellation triggers after storage starts.

The native client has a new process group and no kill-on-close diagnostic Job.
RAM/disk/deadline/logging faults or console interrupts latch FAIL; monitoring
continues until natural exit. Output streams stay open. There is no kill,
client timeout, VM shutdown, automatic rollback or second native invocation.
The service VM has no claimed hard memory/time containment. A stuck operation
or continuing output can exceed these budgets; the accepted policy requires
waiting, not forcibly enforcing a cap. Closing the console, sleep, power loss
or external termination cannot be made safe by this launcher.

## Verification and result

Only natural return 0 with no latched faults permits the ordinary-user Linux
capacity query. The runner first checks the same source file identity and exact
64 GiB VHDX capacity read-only. Another >=22 GiB gate precedes the one query
using system Python `-B -I -S`, a nonce and GO/EOF handshake. It cannot launch
children, import model/native packages or write filesystem data. Limits:
512 MiB address space, 10 CPU seconds, 15-second self-alarm, 30-second controller
fault checkpoint. Query exit must be natural. It records the statvfs equivalent
of `df` bytes, root mount/superblock flags, device bytes and peak RSS. Both mount
layers and effective statvfs flags must establish read/write ext4.

PASS additionally requires native resize2fs block count × block size exactly
64 GiB, root device exactly 64 GiB, ordinary-user root available space >=5 GiB,
D: >=50 GiB, unchanged accepted backup, registration and original box inputs,
and clean handle release. Missing/localised native block geometry is incomplete
verification and returns FAIL rather than guessing from usable filesystem size.
No complete guest file-byte/health audit is claimed. No automatic shutdown is
issued after the query; the named distro may remain running.

`final-result.json` is authoritative; missing, partial or non-PASS means no
accepted result. The required text report is flushed before this final receipt.
The JSON is flushed/closed in an exclusive pending file before a no-overwrite
rename publishes it. A staging/publication failure leaves no accepted authority.
Any late cleanup/report fault forces FAIL. Failure preserves the changed disk,
backup and logs, captures settled Windows container state where possible, and
prohibits Linux follow-up after a native fault. Do not rerun. Separately review
partial growth and any rollback prerequisites in the original runbook.

No decoder is prepared or authorised. The existing 3,888-row shape result is
authenticated before/after, not regenerated. All neural RAM/VRAM/allocator,
disk and attempt guards remain unchanged. Successful storage acceptance would
enable a separately reviewed decode-only recovery preparation.

## Offline verification

Focused synthetic tests exercise natural settlement despite faults/interrupts,
exact command routing, inert defaults, approval requirements, source-hash
refusal, breached-sample logging, byte-exact filesystem geometry, capacity-child
setup/EOF failures, effective read-only mounts, checkout line endings, and late
cleanup/report failures. Existing backup/readiness/survey/ownership suites are
also run on small fixtures. No performance or successful live resize is claimed.
Independent review and final preservation/test counts accompany publication.

Final verification: **129/129** scoped offline/synthetic tests passed, with no
skips, plus PowerShell parsing and four inert/invalid-approval dry-runs.
Independent review found no remaining material blocker after regression fixes
for late receipt failures, capacity-child settlement and read-only mount flags.
Preservation passed for 3,346 historical files, 188 protected/retained records,
1,199 box records, 23 accepted and 13 failed backup receipts, all eight box source
pins and 189 pre-existing local refs. The 188 retained records include the
protected identity set; they are not 188 distinct protected identities. No full
environment byte audit or fresh real-disk rehash is claimed during preparation.
