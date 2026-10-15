# Guided turret texture decode: resource diagnostics

Base `5f89b0a9390716c119af2b6325f6b3db8425c568`; branch `codex/amd-turret-texture-resource-ready`.

**BLOCKED before any inference marker or WSL launch.** The requested five-minute Windows RAM window never reached the unchanged 20 GiB prelaunch requirement. No model load, native import or GPU attempt occurred; no genuine texture field was produced.

## Exact readiness evidence

The instruction was received at 08:12:35 UTC on 2026-10-15. The first measurement was at 08:14:36.377 UTC, after 121.377 seconds. Thirty-seven Windows `GlobalMemoryStatusEx` readings followed at approximately five-second intervals, ending at 08:17:35.001 UTC. Every reading was below **21,474,836,480 bytes (20 GiB)**.

- Minimum/final available RAM: **20,155,133,952 bytes (18.77 GiB)**.
- Maximum: **20,467,212,288 bytes (19.06 GiB)**.
- Final shortfall: **1,319,702,528 bytes (1.23 GiB)**.
- D: free after the window: **75,137,884,160 bytes**, above its unchanged 53,687,091,200-byte floor. This disk reading is separate from the RAM window.

[All timestamped RAM measurements](amd-turret-texture-resource-evidence/ram-readiness-samples.jsonl) | [Readiness decision](amd-turret-texture-resource-evidence/ram-readiness.json) | [Result summary](amd-turret-texture-resource-evidence/result-summary.json).

## Narrow diagnostic correction

The preserved-copy launcher now appends, flushes and fsyncs an actual host-resource record **before** raising: available Windows RAM, D: free, both runtime floors, each failed condition, UTC timestamp, admission result and execution-stage hint. An incomplete concurrently appended guest phase is labelled unavailable rather than preventing the resource measurement. The existing runtime floors remain **12 GiB RAM / 50 GiB D: free**. The additional 20 GiB prelaunch check is retained immediately before marker creation.

Four focused CPU regressions passed: RAM-only failure, disk-only failure, both failures, and equality at both runtime floors. They exercise the real candidate `host_admission()` and real durable journal using synthetic measurement providers, then reopen the record after the exception. Red-before-fix and final green logs are preserved. An initial sandbox denied temporary-file cleanup; the same focused cases ran successfully with normal Windows temporary-file access. No 71-test suite, WSL acceptance, GPU test or unrelated suite was repeated.

Only the host diagnostics, exclusive scratch/evidence paths and references to the existing CPU qualification changed. The eight copied numerical/writer/ownership files are byte-identical to the qualified versions. The prior genuine WSL writer acceptance remains preserved; it was not rerun or relabelled as a new acceptance. The original empty, Windows-owner-attested permanent `field` leaf is retained. The new candidate has not received runtime qualification because the mandatory RAM gate blocked it.

## Preservation and disposition

Fresh stream hashes matched 91 protected artifact/source/backup/native-binary records. The original texture latent (`5506c4d0199bef64423b41b8bed560e2b7def9d29c5bb1c3072c0ea31ee1ddc4`), complete geometry/guides (`dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22`) and genuine decoder checkpoint (`97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c`) are unchanged. The original 28-file qualified source manifest matches. No installed environments, permissions, models, mesh, AMD edits, drivers, WSL configuration or historical evidence were modified. No application was closed automatically. No comprehensive new Linux inventory is claimed: WSL was never launched.

[Preservation checks](amd-turret-texture-resource-evidence/preservation.json) | [Independent review](amd-turret-texture-resource-evidence/independent-review.json).

The permanent output leaf remains empty:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-owner-067f668\field`

There is no learned six-channel artifact or new output SHA-256. No cleanup was required because no production worker or owned GPU process started. All historical branches and previous attempt markers remain preserved.

**Next action requiring approval:** free additional host memory and repeat the three-reading 20 GiB admission before one fresh genuine decode. Keep all runtime, memory, precision, ownership and containment requirements unchanged. No fallback, retry or further texture processing was performed.
