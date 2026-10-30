# Phase 6B — conditional native WSL capacity expansion

30 October 2026. **Preparation PASS; execution HOLD.** The native 32-to-64 GiB
method is technically supported. A separately approved, operator-attended
critical-section policy is required before any command below may run. This is
a reviewed runbook, not an executable resize launcher. No resize, WSL startup/
shutdown, restore, installation or neural operation occurred during preparation.

## Authenticated starting point

The completed cold-backup commit is
`2ec4c3dcdbe387127b16c4ca281eb0d1d76d8fa4`. Both final and worker receipts say
PASS; their embedded verification agrees, all three Jobs exited naturally,
and the independent .NET destination-hash receipt agrees. All three full hashes:

`36d8679eaad94fdfa7b0cc12a48c728ee4ec73f83de90d339005295852365c5c`

Source: `D:\AI\Modly-WSL\Ubuntu24.04\ext4.vhdx`.
Accepted backup: `E:\Modly-Cold-Backup-closed-solid-v2\ext4.vhdx`.
Both files remain **29,581,377,536 bytes**, with the accepted file IDs, timestamps,
volume identities and bounded VHDX tables unchanged. Both virtual capacities
are **34,359,738,368 bytes / 32 GiB**. Source inspection obtained an exclusive
read-only handle; backup inspection used shared-read only. Only container
metadata was read; no mount, filesystem traversal or complete rehash was done.
Current byte equality rests on the completed full-hash evidence plus unchanged
identities/metadata, not a new full-file SHA pass. A provisional E: receipt alone
does not substitute for the final PASS. All 23 saved backup-evidence files remain
unchanged. Original eight closed-box pins, including 3,888-row shape inputs,
reauthenticate; no regeneration is needed.

Fresh Windows-only inspection confirms the registered WSL 2 distro
`Modly-AMD-Ubuntu-24.04` still points to the source above. Installed WSL client
and service file versions are **2.7.12.0**. No VmmemWSL/wslhost process was
observed; WslService itself is running. Registry State=1 means installed, not
proof of stopped state. The latest running-list receipt is the accepted backup's
empty list; a fresh list is mandatory at execution. No WSL command ran here.

Current D: free is **184,708,202,496 bytes / 172.022919 GiB**. E: free is
**454,298,730,496 bytes / 423.098663 GiB** on the separately authenticated HDD.
Filling the entire 64 GiB virtual data capacity from current D: allocation would
leave approximately **135.572723 GiB**, before additional container/NTFS
metadata. This is a budget estimate, not an allocation promise or Linux free
space. Keep the actual 50 GiB D: floor continuously; no second VHDX copy on D:
is required for the proposed growth.

## Supported operation and privileges

Microsoft supports `--manage --resize` in WSL >=2.5. Integer byte units avoid
GB/GiB ambiguity; **68,719,476,736 bytes is exactly 64 GiB**. The managed procedure
invokes filesystem checking/growth as well as container expansion; no custom
VHDX editing, diskpart, extra repair flags or new packages are proposed.
[Microsoft disk-space guidance](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

Use the same Windows account that owns this distro and source. Current owner
and read/write/modify ACL access were confirmed without opening a write handle;
the preparation process is not elevated. The 2.7.12 client resize branch calls
the service without a mandatory elevation step, and the service impersonates
the caller for VHD metadata access. Administrator is expressly mentioned for
the separate manual diskpart route, not universally for managed resizing.
This is source/ACL evidence, not an executed permission test. Stop on access
denial; do not automatically elevate, switch accounts, change ACLs or retry.
[Versioned client](https://github.com/microsoft/WSL/blob/2.7.12/src/windows/common/WslClient.cpp#L880-L886),
[versioned service](https://github.com/microsoft/WSL/blob/2.7.12/src/windows/service/exe/LxssUserSession.cpp#L1614-L1659).

## Approval condition: non-cancellable storage critical section

The service enlarges the VHD **before** `_CreateVm`, then attaches it and requests
filesystem growth. A startup RAM failure can therefore leave a 64 GiB container
with an unexpanded filesystem. A Job around wsl.exe does not own the service VM.
Killing the client, closing its output stream or shutting down WSL can interrupt
or detach from an ongoing filesystem operation. No guaranteed cancellation or
bounded-completion claim is made.
[Native sequencing](https://github.com/microsoft/WSL/blob/2.7.12/src/windows/service/exe/LxssUserSession.cpp#L1627-L1648),
[guest-result wait](https://github.com/microsoft/WSL/blob/2.7.12/src/windows/service/exe/WslCoreVm.cpp#L2218-L2239).

**Separate storage approval must explicitly accept this narrow exception:**
once the native command starts, a RAM/disk/deadline fault latches FAIL and
prohibits further guest or mutating work, while read-only Windows monitoring
and evidence capture continue until natural settlement. Do not automatically terminate the client, close a
kill-on-close Job, terminate a VM/distro, shrink, restore or retry. Keep the
console/output open. A deadline is a failure/escalation checkpoint, not a safe
cancellation guarantee. Numerical thresholds are not lowered. This exception
does not apply to any neural, decoder, survey or other workload.

If immediate termination or an enforceable wall deadline is mandatory for this
storage operation, the native procedure is **NO-GO under that policy**. A wrapper
cannot resolve this conflict. Generic approval to resize is not approval of
this exception. No automatic execution handoff is qualified without it.

## Exact conditional operator handoff — NOT APPROVED FOR EXECUTION

1. **Quiescence:** all distributions must already be stopped and the utility VM
   quiescent. Microsoft's documented procedure starts with a WSL shutdown;
   this preparation authorises none. If unrelated WSL work or a residual VM
   exists, stop and arrange a separately approved quiescent window rather than
   issuing global shutdown. If all are already cold, confirm that state without
   starting Linux. Record a fresh `wsl.exe --list --verbose` / running list.
2. **Before mutation:** explicitly approve the critical-section policy above;
   close Codex/browsers if needed for headroom, arrange stable power and prevent
   Docker/other WSL clients from restarting. Recheck owning account, the same
   2.7.12.0 client/service versions,
   registration/path, source 32 GiB, accepted receipt set and source/backup
   identities/tables. Any intervening source use/change requires review.
   Hold the source exclusive and read-only through hashing and final preflight.
   For current byte assurance, freshly stream SHA-256 from that frozen source
   and backup against the fixed hash above before mutation; keep the backup
   open read-only with write/delete sharing denied through completion. No new
   backup, export or fallback is permitted. Do not hold an exclusive source
   file handle during native resize; release it only after final preflight.
3. **Fresh admission:** three GlobalMemoryStatusEx readings >=22 GiB, four
   seconds apart, then a final >=22 GiB immediately before native launch.
   24 GiB remains preferred only. Recheck D: >=50 GiB, C: >=2 GiB, backup volume
   and observation/evidence space. Start timestamped RAM/free-space/output
   observation first. Because no qualified external VM-ready signal exists,
   require >=22 GiB across the **entire native invocation** for PASS; retain
   the 12 GiB live emergency floor. Every breach is latched even if RAM recovers.
   Observe the existing 1800-second storage window as a failure checkpoint;
   this does not create a safe hard-kill deadline or bound shared-VM memory.
4. **Exactly once, only after those conditions and approval:**

   ```powershell
   & 'C:\Windows\System32\wsl.exe' --manage 'Modly-AMD-Ubuntu-24.04' --resize '68719476736B'
   ```

   Preserve stdout/stderr and exit status in fresh exclusive Windows evidence
   `artifacts/amd-wsl-resize-64g-v1`; never reuse a consumed directory. The
   command is a reviewed reference, not permission to skip preflight or monitoring.
5. **After natural return:** any nonzero status, latched fault, incomplete output
   or ownership/identity uncertainty stops here. Record Windows container state
   read-only; no Linux probe, repair, retry or decoder follows a failed operation.
   Otherwise verify source virtual capacity exactly 68,719,476,736 bytes, same
   file/registration identity, D: >=50 GiB, unchanged accepted E: backup and box
   inputs. Source VHDX contents/mtime are expected to change; do not require its
   old whole-file hash after the authorised resize.
6. **Post-resize Linux capacity check:** only after another final >=22 GiB gate,
   permit one bounded read-only ordinary-user capacity query in the named distro.
   Keep the 22 GiB startup guard and 12 GiB live floor. No directory inventory.
   Record `df -B1 --output=source,fstype,size,avail,target /` and root mount flags,
   root device byte size and the native resize2fs final filesystem block count /
   block size. Run as `modly`, not merely the default root user. Natural process
   exit must be verified; no blanket distro shutdown is authorised. Any new
   distro launch is part of the separately approved operation, never this task.

## Acceptance and containment

PASS requires all of: native return 0 and natural completion; zero latched
faults; VHDX and root block device exactly 64 GiB; filesystem block count times
block size proving 64 GiB growth; root ext4 mounted read/write; ordinary-user
`f_bavail * f_frsize` >=**5,368,709,120 bytes**; actual D: free >=**53,687,091,200
bytes**; unchanged backup, registration and original Windows box inputs.
`df` usable size excludes filesystem overhead and must not be substituted for
exact filesystem block geometry. Native output and post-query must agree; if
the block geometry is unavailable, mark verification INCOMPLETE rather than
guessing. No filesystem health/guest file-byte preservation beyond these checks
is claimed; future decode still authenticates its existing inputs/checkpoints.

FAIL/INCOMPLETE never authorises decode. Preserve the changed disk, backup and
logs. Once natural settlement is established, separately review rollback if
needed: all WSL must be cold, rehash the accepted E: backup, provide D: space for
the retained failed container plus a verified restored candidate and 50 GiB
reserve, and approve a swap retaining the failed file. No automatic restore,
unregister, shrink, deletion or cache clearing. Original checkpoints and all
3,888-row shape provenance remain protected. Neural RAM/VRAM/allocator/disk gates
and the decoder's 5 GiB floor remain unchanged. No decoder is prepared here.

Offline receipt/acceptance fixtures reject wrong hashes, altered identities,
unsafe cleanup, container-only growth, unknown geometry, insufficient space and
faults that later recover. Independent review supports the conditional native
runbook and explicitly withholds approval of any silently relaxed kill policy.
