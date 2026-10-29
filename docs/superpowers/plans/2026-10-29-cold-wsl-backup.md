# Verified cold WSL backup implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implementation and offline testing only, followed by independent review.

**Goal:** Prepare one fail-closed, Windows-only raw VHDX backup; do not execute it during preparation.

**Architecture:** A dry-by-default PowerShell entry authenticates the interpreter. A Python controller authenticates its published sources and private hardware authority, checks Windows-only inventory and resources, and owns a bounded child Job. The worker retains exclusive source and pinned directory handles through streaming, source hashing, independent .NET destination hashing and metadata readback. Failure preserves all partial files.

**Tech stack:** Existing Python stdlib, Windows file/Job APIs, Windows PowerShell/CIM/.NET; no installs or guest commands.

**Spec:** Phase 6A user directive and accepted local `artifacts/amd-wsl-capacity-readiness-20261029-v1/REPORT.md`.

## Constraints and decisions

- Source is exactly `D:\AI\Modly-WSL\Ubuntu24.04\ext4.vhdx`; fixed fresh E: backup and C: evidence roots, no alternate paths or retries.
- Private authority pins registry, physical drives, volume serials and original source identity; publish its digest only.
- Source `GENERIC_READ`, share mode zero, existing-only. Destination create-new; no compact/export/mount/resize, deletion or source writes.
- CPU-only backup uses unchanged 12 GiB live floor, 50 GiB D: and 2 GiB output reserves; neural 22 GiB gates are untouched and do not run here.
- New backup-only bound: 30 minutes total; 512 MiB child-Job aggregate/private ceilings; 4 MiB transfer buffer. No existing survey/decoder deadline is changed. Failure may leave a retained incomplete copy.
- E: admission requires at least 31,728,861,184 bytes **plus** 2 GiB reserve and 4 MiB evidence allowance, or the current source-size-based requirement if larger; this conservatively follows the directive's wording.
- Existing branch heads stay unchanged. Work on a fresh isolated branch in the existing checkout to retain authenticated absolute artifact routing; do not install dependencies or create a redundant checkout.

## Review focus

- A stopped-state snapshot alone does not freeze the source: retain exclusive source handle and no-delete ancestor handles.
- Detect redirected paths, hardlinks, changed file identity, bad storage mappings and running/lingering WSL VM activity.
- Never equate copied bytes with an independently verified backup; flush and independently rehash all destination bytes.
- Timeout/interruption/read errors retain the consumed directory and partial file; no overwrite, resume, retry or unrelated cleanup.
- Dry-run and missing approval perform no storage/resource query or child launch; no route may start/stop WSL or prepare decoding.

## Tasks

- [ ] Write failing offline tests for streaming/identity/space/interruption/hash disagreements, native exclusive handles, unsafe paths, routing and guard/cleanup failures; record RED.
- [ ] Implement `amd_wsl_cold_backup.py`, read-only `amd_wsl_cold_backup_inspect.ps1`, and `Invoke-ModlyColdBackup.ps1`; keep existing files unchanged.
- [ ] Run synthetic Windows I/O and offline safety suites, PowerShell parse/default dry-runs and missing-approval checks. Never invoke the real run mode.
- [ ] Independently review and address material findings with regression tests.
- [ ] Reverify original box inputs, historical/protected records and old Git refs. Publish code/tests/safe text on the isolated branch, verify the remote commit, and provide a conditional command only.
