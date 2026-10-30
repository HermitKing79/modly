# Phase 6C — held read-only filesystem capacity check

30 October 2026. **Preparation only; execution HOLD.** The sole purpose is to
measure the existing root filesystem. There is no resize, repair, privileged
fallback, installation, rollback, cleanup/deletion or decoder route.

## What the failed resize establishes

The authenticated d0779b8 attempt ran the single native command to natural
return 0 at 09:32:41.217 UTC, after 3.265 seconds. The minimum observed Windows
available RAM was 24,752,467,968 bytes (23.073 GiB). The saved Windows container
metadata shows exactly 68,719,476,736 bytes / 64 GiB, with the same backing
file ID and volume serial. Source file length remained 29,581,377,536 bytes;
dynamic container length is not virtual capacity.

Stdout is 80 bytes of UTF-16LE:

```text
The operation completed successfully.
```

Stderr is 100 bytes of ASCII:

```text
e2fsck 1.47.0 (5-Feb-2023)
Pass 1: Checking inodes, blocks, and sizes
resize2fs 1.47.0 (5-Feb-2023)
```

The parser strips NUL bytes and decodes the combined output, then requires a
line giving final filesystem block count and block size. These files contain
no such line. Its matched geometry set is empty, not 64 GiB, so it raises
`Exact native filesystem block geometry unavailable/different`. This is a
missing-output assumption, not demonstrated failed growth or a demonstrated
encoding bug. The service did not expose the expected geometry in this run;
the evidence does not explain why. No CAPACITY_START event or Linux result
exists. Original FAIL is preserved and is not rewritten as PASS.

Both fresh complete disk hashes before that resize matched the accepted cold
backup. The new preparation authenticates all 18 saved resize files, 23 accepted
backup receipts and eight box input pins. It does not access the actual disks
or launch WSL. The future handoff rechecks the existing source identity/64 GiB
headers, backup identity/32 GiB headers and receipt provenance read-only.
It does not unnecessarily rehash both 29.6 GB disks for a metadata query.

## Smallest measurement path

`Invoke-ModlyPostResizeCapacity.ps1` authenticates the base Python and exact
published checkout. The controller reuses the qualified ordinary-user
capacity-query controller, observer and read-only owned inventory helper.
The original resize code and probe remain unchanged. The small Phase 6C guest
fork retains the original audit, read, process-start and writable-mount logic;
offline AST equivalence tests bind those four functions to the qualified source.

One invocation as **modly**, using system Python `-B -I -S`, records:

- Root source, filesystem type, mount and superblock flags, effective statvfs
  read-only flag and root major/minor identity.
- statvfs total/free/available block counts, fragment and block sizes,
  usable capacity and ordinary-user available bytes.
- Root device capacity from the existing sysfs method, peak RSS, nonce,
  PID/start ticks and natural process exit.
- Optionally, two 1,024-byte read-only primary ext4 superblock reads at device
  offset 1,024, only with the existing ordinary-user permissions. The open
  rejects symlinks, validates the block-device handle against the root device,
  and verifies stable identity/geometry. Missing file or permission denial is
  explicit UNAVAILABLE. No sudo, ownership/mode changes or alternate devices.

Readable raw geometry must have ext4 magic, a valid block size/count within the
device, correct 64-bit feature handling and valid CRC32C when metadata checksums
are enabled. Corrupt, redirected, inconsistent or unsupported metadata fails
closed; it is not silently converted to permission-unavailable.
This is a bounded metadata check, not fsck or a full filesystem-health audit.
[Ext4 superblock layout](https://www.kernel.org/doc/html/latest/filesystems/ext4/super.html),
[kernel checksum implementation](https://github.com/torvalds/linux/blob/v6.12/fs/ext4/super.c#L257-L283).

**Usable size is not exact superblock geometry.** Linux ext4 statfs subtracts
filesystem overhead from total blocks; the raw block-count × block-size
measurement is recorded separately. More than 32 GiB usable capacity proves
growth beyond the old capacity but does not prove exact 64 GiB filesystem size.
[Kernel statfs implementation](https://github.com/torvalds/linux/blob/v6.12/fs/ext4/super.c#L6390-L6421).

## Admission and bounds

All distros and utility VM must already be stopped, with no conflicting Docker
activity. No automatic shutdown is allowed. Run under the registered distro's
owning account in a normal 64-bit Windows PowerShell console, outside an
inherited external Job. Close Codex and unnecessary browsers yourself after
separate approval. Retain stable power; keep the console open until cleanup.

The future invocation uses a new exclusive directory:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-wsl-post-resize-capacity-v1`.
An occupied path stops; no fallback, overwrite, resume or retry is provided.
The private local authority files and original evidence must remain available.

Before the query: three fresh GlobalMemoryStatusEx readings each **>=22 GiB**,
four seconds apart, then final >=22 GiB before WSL launch and before guest GO.
The observer retains >=22 GiB throughout the query for acceptance and the
original **12 GiB** emergency floor. **24 GiB stays preferred only.** Prior
successful resize RAM is not a guarantee of a new startup. No neural or storage
threshold is lowered.

The reused observer limits its private commit to 512 MiB, keeps D: >=50 GiB and
C: evidence/E: >=2 GiB, and saves timestamped Windows observations before
startup. Metadata preflight has a 120-second deadline; read-only inventory
helpers retain their 512 MiB owned Jobs and 30-second bounds. The guest has
512 MiB address-space, 10 CPU-second and 15-second alarm limits; the reused
capacity controller latches its 30-second fault checkpoint and requires natural
child exit. Any observed breach withholds GO/closes input and prevents PASS.
There is no blanket VM/distro termination. The shared WSL VM is not claimed to
have a hard process-memory bound; a blocked client can delay natural cleanup.

The accepted E: backup is held read-only with writes/deletion denied through
the query and checked unchanged. Source directories stay pinned; the backing
file ID and original eight Windows box inputs are checked again afterward.
The probe issues no Linux or VHDX writes. Normal WSL startup/read-write root
mount can itself update filesystem journals; byte immutability of the running
VHDX is not claimed. This task performs no such startup.

## Decisions after the separately approved measurement

| Evidence | Classification |
|---|---|
| Valid readable superblock exactly 64 GiB, device64, ext4 rw | FULL_64_GIB_CONFIRMED |
| Exact filesystem >32 and <64 GiB | PARTIAL_EXPANSION_CONFIRMED; review, no follow-up |
| Exact filesystem still32 GiB | UNCHANGED_32_GIB; review, no follow-up |
| Device64 and valid statvfs, raw permission denied/missing | Exact geometry UNAVAILABLE; report usable/free bytes separately |
| Wrong device/mount, corruption, inconsistent measurements or resource fault | FAIL; no follow-up |

The **decoder disk gate** is independently PASS only if ordinary-user available
root bytes are >=5,368,709,120. PASS_SPACE_GATE requires valid measurements,
device64, ext4 rw, that free-space gate and all preservation/resource checks.
Known partial/unchanged/below-original exact capacity returns NO_GO_CAPACITY
even if there is enough free space. Unavailable exact geometry can return
PASS_SPACE_GATE, but never FULL_64_GIB_CONFIRMED or acceptance of the old
runbook's exact-geometry requirement. That result supports only separately
reviewing decode-only preparation, not running a decoder.

`final-result.json` is authoritative after durable pending-file publication.
`linux-capacity.json` contains measurements and `REPORT.txt` gives the short
result. Missing/incomplete final evidence is not acceptance. Do not retry after
FAIL, NO_GO or incomplete measurement. Preserve everything for review.

## Held operator command

The final delivery provides the exact published commit for this template.
**Separate execution approval is required.**

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\wattz\Documents\ChatGPT\modly\proof\scripts\compatibility\Invoke-ModlyPostResizeCapacity.ps1' -Run -ApproveCapacityCheck -ExpectedCommit '<published-commit>'
```

Default invocation is inert. No elevation, new dependency or environment change
is required. No second resize, capacity change or neural operation is possible
through this entry point. The 3,888-row shape result stays preserved for a later
separately authorised recovery.

Verification: 147 scoped offline/synthetic tests passed with zero skips,
including 18 new focused tests, plus PowerShell parsing and four inert/approval
dry-runs. Independent review found no material blocker. All 18 resize receipts,
23 accepted backup receipts, 3,346 historical files, 188 protected/retained
records, 1,199 box records, eight box pins and 190 pre-existing local refs passed
preservation checks. The original runtime sources are unchanged. No full
environment byte audit or current E: backup verification is claimed.
