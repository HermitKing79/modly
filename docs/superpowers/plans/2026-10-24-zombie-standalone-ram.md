# Standalone zombie RAM diagnostic implementation plan

**Goal:** Deliver a Windows CPU-only preparation measurement tool for the user to run with desktop applications closed. Do not run the real diagnostic during authoring.

**Architecture:** A PowerShell observer performs fresh GlobalMemoryStatusEx admission and supervises a guarded Python child from interpreter startup through exit and recovery. The child reuses the unchanged zombie lineage reader, source verifier and metadata-only component. It extracts only the pure checkpoint authentication function from the historical decoder, without importing or executing that decoder. CPU runtime construction reproduces lineage and run-contract preparation without creating a generation Runtime or importing the guest entrypoint.

**Tech stack:** Existing Windows PowerShell/.NET, original D: API Python 3.12, Python stdlib, existing CPU validation dependencies. No installation or environment changes.

**Authority/spec:** User's detailed standalone-runner requirements in this conversation, including explicit authorisation to implement and publish; these supersede extra design/plan confirmation workflows. Author-selected implementation: native inline development, final independent review. Branch isolation is used as requested; existing source files and historical branches remain intact.

## Constraints

- 22 * 1024**3 available physical bytes in three readings four seconds apart; 24 GiB is preferred only. Stop on any failed reading; never retry automatically.
- Continuous 200 ms observer checks plus immediate child phase checks; below 22 GiB stops the owned child and further preparation. Preserve the original 12 GiB runtime floor, 50 GiB D: floor and 2 GiB evidence-volume floor.
- Overall diagnostic deadline 480 seconds, existing Git read bound 15 seconds, checkpoint streaming in 1 MiB blocks. Recovery observations do not prepare anything.
- No WSL, model weights loading, neural imports, marker reservation or inference. No historical source/evidence writes. No generation script is an executable child entrypoint.
- Unique evidence directories outside the repository; raw process identities/memory evidence stay local. Dry-run and mocked tests only during authoring.

## Review focus

1. A failing RAM read must stop before another phase or marker can run, including an in-phase decline.
2. Child import/call/subprocess/write guards must fail closed, including indirect imports and historical methods.
3. Interpreter startup, import, constructor lineage, repeated lineage, source checks, checkpoint hashing and metadata need separate measurements.
4. Timeouts, malformed protocol, child exceptions and missing result must produce a failure with durable evidence, never success or a retry.
5. Working-set deltas, host availability and concurrent process observations must not be presented as exclusive physical-memory accounting.

## Task 1: guarded CPU preparation child

- [ ] Add focused failing unittest cases for admission/boundary gates, checkpoint extraction, forbidden imports/calls/processes/writes and preparation order with synthetic backends.
- [ ] Implement `scripts/compatibility/amd_zombie_ram_diagnostic.py`: stdlib import-safe helpers, restricted CPU backend, source authentication and durable phase records.
- [ ] Run the mocked suite to green without importing project/model modules or reading real checkpoints.

## Task 2: standalone observer

- [ ] Add failing static/protocol checks for the PowerShell launcher and execute safe mocked integration cases.
- [ ] Implement `scripts/compatibility/Invoke-ZombieRamDiagnostic.ps1`: dry-run by default; explicit `-Run` for the user's later invocation. External telemetry precedes Python startup; fresh gates, deadlines, owned-child-only termination, recovery and process snapshots.
- [ ] Verify PowerShell syntax and dry-run only. Never invoke `-Run` during preparation.

## Task 3: preservation, report and publication

- [ ] Compare the inherited 100 protected identities/78 bounded hashes before/after; verify only added tracked files and all preexisting branch refs unchanged.
- [ ] Obtain independent code review, address material findings with regression tests, rerun focused checks.
- [ ] Add textual report, source-safe test/review/preservation receipts and exact post-close command. Keep raw evidence local.
- [ ] Publish only the isolated new branch, compare all preexisting remote heads and verify the published commit. Stop without a real diagnostic.
