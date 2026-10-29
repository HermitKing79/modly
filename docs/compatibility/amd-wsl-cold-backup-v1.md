# Phase 6A — prepared cold WSL backup, execution on hold

This prepares one raw cold backup of the registered Modly distro. It has not
copied or fully hashed the real VHDX. Actual execution requires separate approval.
Resize and neural decoding remain NO-GO.

## Fixed routing and prerequisites

- Source: `D:\AI\Modly-WSL\Ubuntu24.04\ext4.vhdx`.
- Distribution: `Modly-AMD-Ubuntu-24.04`; all distributions must be stopped and
  no VmmemWSL/wslhost process may be observed. The runner never stops them.
- Exclusive backup: `E:\Modly-Cold-Backup-closed-solid-v1\ext4.vhdx`.
- Exclusive private evidence: workspace `artifacts/amd-wsl-cold-backup-v1`.
- Retain the local private preparation `authority.json`. Its published SHA-256
  binds the reviewed registration, source identity, distinct D:/E: physical
  drives/volume serials and eight original closed-box input hashes. Missing or
  changed authority, source, registration, hardware or published code stops.
- E: must remain NTFS on the authenticated separate HDD. Admission requires
  `max(31,728,861,184, source length) + 2 GiB + 4 MiB` free. At the reviewed
  source length this is **33,880,539,136 bytes**. This conservatively includes
  the directive's stated minimum plus the reserve and evidence allowance.

Normal PowerShell, existing qualified Windows Python and Windows PowerShell
are sufficient; no admin action, installation, download or WSL guest is needed.
Default invocation is inert. Only explicit `-Run -ApproveColdBackup` with the
exact published commit enables the backup. Do not use those flags before
execution approval. Avoid concurrent disk/WSL/Docker activity and keep power
stable. Closing Codex/browsers is optional for this CPU-only operation; no
application is closed automatically.

## Copy, independent verification and bounds

The controller authenticates Windows inventory using read-only registry/CIM
queries and only `wsl.exe --list --running --quiet`. Inventory helpers wait for
GO/EOF after Windows Job attachment; the worker likewise waits for its owner.
There is no guest-start, shutdown, mount, export, compact, trim, resize or
decoder code path.

The worker opens the original **GENERIC_READ/share-zero/OPEN_EXISTING** and
retains that same handle through every copy/hash/metadata operation. No-delete
ancestor handles prevent path replacement. Reparse paths, hardlinks and
unqualified file/volume identities fail closed. Inventory is checked again
after source exclusivity; actual destination/root handle volume identities must
match the authenticated E: volume before payload writes.

The destination is **CREATE_NEW**. Copying uses a 4 MiB buffer. The destination
is flushed, closed, reopened read-only against its creation identity and held
against mutation during independent verification. The source is reread fully
with Python SHA-256; destination bytes are independently reread by Windows
PowerShell's .NET SHA256 implementation. Copied-byte/source/destination hashes
and lengths must all match. Source identity must remain unchanged; VHDX
signature, bounded region/metadata tables and 32 GiB virtual capacity must match
the original authenticated tables and copied container. This proves faithful
backup bytes, not filesystem health or tested restorability.

Resource bounds are backup-only: **12 GiB live Windows floor**, **512 MiB
aggregate and per-process child Job commit**, **30-minute total wall deadline**,
30-second inventory stages, up to five seconds owned cleanup/reaping grace.
D: retains 50 GiB; C:/E: retain 2 GiB reserves. Parent commit and evidence growth
are bounded and host memory is sampled throughout child execution. Child Jobs
record peak commit, exit status and natural/forced cleanup. Progress is limited
to phase boundaries and at most one checkpoint per five seconds per stream;
source hashing and copying have distinct byte counters. Neither deadline nor
throughput has been measured on the real 27.55 GiB disk; exceeding any bound is
FAIL with retained evidence. Neural 22 GiB gates and every decoder/VRAM safeguard
remain unchanged; they are not exercised by this CPU backup.

## Acceptance and failure handling

Read `artifacts/amd-wsl-cold-backup-v1/final-result.json` and `REPORT.txt` after
the run. **Only top-level `classification: PASS` is overall acceptance.** Require
identical full hashes/lengths, unchanged source and eight box input identities,
matching physical/registration snapshots, closed handles and natural empty Job
cleanup. The E: `BYTE-VERIFICATION.json` is explicitly provisional until these
controller checks pass; its presence alone is not overall acceptance.

An error, deadline, interruption, running distro, resource breach, read failure,
partial copy, hash mismatch or forced/unverified cleanup is FAIL. The used
directories, complete/partial VHDX and logs are retained. There is no deletion,
overwrite, resume, fallback destination or automatic retry. Reinvocation with
consumed paths refuses execution. Source sharing violations never trigger a
shutdown or unlock attempt. A failure before a new evidence root can be created
is reported on stderr; existing evidence is never written over.

## Preparation evidence

Offline tests cover complete byte copying, source identity/payload changes,
read/short-read failure, interruption and retained partials, inadequate space,
hash/length disagreement, authority tampering, redirected paths, actual native
exclusive/read-only/create-new handles, hardlinks, pinned directories, genuine
.NET hashing, real harmless Windows Job cleanup and setup-failure receipts.
A complete worker test uses a 2 MiB synthetic header fixture on C:, with all
inventory/disk observations mocked. It is not a real mounted filesystem or an
execution of the real backup.

Independent review found and corrected an omitted partition field, inventory
pre-attachment race, missing setup-failure cleanup/receipt and actual destination
volume authentication gap. Tests and private preservation receipts accompany
the handoff. Only safe code, tests and this text are published. No source mesh,
model, environment, historical evidence or existing branch head is changed.
After backup acceptance, capacity expansion needs a separate review/approval;
this runner performs no follow-up operation.
