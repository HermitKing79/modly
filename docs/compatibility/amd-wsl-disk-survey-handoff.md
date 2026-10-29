# Phase 5 — one read-only disk survey, prepared; execution HOLD

**Preparation only. Separate approval is required before execution.** No live
admission, WSL command, guest survey, storage change or neural operation ran
during preparation. This handoff cannot run a decoder, model, installer or repair.

The old Phase 5 contains a failed admission, not an executed survey worker.
This minimal new owner/metadata worker reuses the qualified Windows memory
reader, original three-reading admission, Windows Job containment, Windows
allocation reader and current pidfd/nonce/group/start-identity cleanup helpers.
Those helpers and all existing model/environment code are unchanged. Offline
qualification is not proof of target-kernel execution; pidfd prerequisites are
tested in the guest before host GO and fail closed if unavailable.

## Exact scope and isolation

Only `Modly-AMD-Ubuntu-24.04`, user `modly`, existing `/usr/bin/python3 -B -I -S`.
No new guest directory, installed package, cache or guest-local evidence file.
New exclusive Windows destination:

`artifacts/amd-closed-solid-disk-survey-standalone-v2`

It must be absent. Any occupied destination stops; no retry or fallback path.
The failed `amd-closed-solid-disk-survey-20261029-v1` and preceding audit remain
unchanged. Existing shapes/occupancy are read-only and hash-authenticated before
and after; the original **3,888-row** job remains at upstream commit
`853a9a28da970beeb53685f967de059bb90c7364`, source-job SHA-256
`498948bb64a88e1f869a2c01ed0f38c51346ae347bc1775aa263f1c8a69f7ea5`.

Measurements:

- Root, home and D: statvfs capacity, used/free/ordinary-user available blocks,
  reserved/unavailable difference and inode counts; mountinfo device, source,
  filesystem and mount flags; root device's sysfs identity/size.
- Descriptor-pinned, no-follow, same-device metadata traversal only under
  `/home/modly`, `/opt`, `/usr`, `/var/cache`, `/var/log`, `/tmp`. Mounted
  subtrees (including same-device bind mounts) are excluded. No model payloads
  are read. Record largest64 directory aggregates and separate retained
  `run-*/geometry,scratch` size/uid/gid/mode/device/inode rows, bounded64 with
  omissions explicit. Hardlinked inodes counted once. Nested aggregates overlap;
  their sum is not root filesystem usage. Permission gaps are not zero usage.
- Fixed backing VHDX `D:/AI/Modly-WSL/Ubuntu24.04/ext4.vhdx`: bounded metadata
  virtual capacity, logical container file length and **separate Windows-reported
  allocated storage** via existing GetCompressedFileSizeW. Allocated storage is
  not virtual capacity, guest free space, a forecast of future allocation or
  physical NAND sectors. Record current D: and output-volume free space.
- Time, guest peak RSS, Windows client peak commit, host memory/disk monitoring,
  owned exit, independent guest group cleanup and input preservation receipts.

## Unchanged guards and bounded survey

Three fresh GlobalMemoryStatusEx readings **>=22 GiB**, four seconds apart;
final22 before startup, >=22 throughout startup and immediately before GO.
**24 GiB is preferred only**. After GO, retain **12 GiB** live host floor,
**50 GiB D:** and **2 GiB output volume** floors. The original **5 GiB Linux
decoder floor is unchanged**: this diagnostic reports that value, including
failure, without requiring the scarce disk space merely to inspect metadata.

No desktop applications are closed automatically. Live mode requires Codex and
ChatGPT Desktop absent; close browsers as well for headroom. A read-only WSL
running inventory refuses an already-running named distro; no distro is stopped
or launched to make room. The inventory starts no distribution.

Survey budget: **120 seconds** from WSL survey-client start, **512 MiB** Windows
client commit / guest address-space ceiling (guest CPU120s, wall115s), one million
metadata entries, depth64. Result JSON <=768 KiB; combined survey stdout/stderr
hard capped128 KiB; auxiliary captures8 KiB each and control receipts8 KiB each,
keeping captured diagnostic data below the reviewed1MiB budget. Host sampling
logs are separate, bounded by stage time and polling frequency. These are caps,
not measured workload consumption. No raised limits or retries.

Both WSL clients and auxiliary clients are owned Windows Jobs; no GO before
attachment and nonce-bound guest readiness. Current pidfd protections validate
ownership before any guest signal. An unattached Windows child can be terminated
only through its retained Popen handle. Guest cleanup has its own15s self-limit,
20s host limit. Additional cleanup grace is separate from the120s survey:
EOF wait5s, each Job drain5s, each reap5s (up to10s for an unattached handle
fallback), each capture-reader drain2s. These are individual bounds, not a
claim that total wall time including cleanup is120s. Host live resources cover auxiliary clients too;
cleanup records health failure while prioritising bounded owned-process cleanup.
Any forced/unverified cleanup fails the result.

No whole-distro termination, shutdown, global WSL setting, numeric guest-group
signal, unrelated-process termination, deletion, resize, trim or installation.
The VM may remain running after the owned processes exit. WSL housekeeping and
filesystem access times may change naturally; no byte-unchanged VHDX claim is
made for the future live run. Original model/environment contents are not edited.

Permission gaps return `SURVEY_INCOMPLETE`; deadline/entry/identity/resource
failures return FAIL with available partial facts. Neither authorises a retry,
cleanup of data, resizing, or decoder execution. Consult `REPORT.txt`,
`summary.json`, `result.json`, `host-result.json`, and cleanup/resource receipts
after reopening Codex. A failed startup may legitimately have no `result.json`.

## Conditional handoff

Normal non-administrator Windows PowerShell, **only after separate approval**
and after closing desktop applications. Use the exact published commit supplied
with delivery:

```powershell
& 'C:\Users\wattz\Documents\ChatGPT\modly\proof\scripts\compatibility\Invoke-ModlyDiskSurvey.ps1' -Run -ApproveReadOnlySurvey -ExpectedCommit '<verified-commit-from-delivery>'
```

Default invocation starts no child and queries no RAM or WSL. Missing approval
fails before Python starts. The launcher checks the qualified Windows Python,
and Python verifies the exact current commit and the source closure before work.
No downloads or setup prerequisites beyond the existing qualified installation.

Preparation verification: **59 passing tests** (36 focused mocked/synthetic,
23 unchanged pidfd regressions); actual PowerShell/Python no-work dry runs and
missing-approval stop. Independent review approves preparation-only publication;
preservation settles before publication. Raw evidence
remains local; only source/tests/text are published. No fresh Linux disk numbers
are claimed. Stop after preparation/publication; the next decision is approval
of this single survey, not a storage change or neural recovery.
