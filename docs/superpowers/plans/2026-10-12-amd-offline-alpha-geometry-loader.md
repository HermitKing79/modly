# Offline Alpha-Only Geometry Loader Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans and test-driven-development; use a read-only whole-branch review before publication.

**Goal:** Prepare the genuine local geometry pipeline through Modly IPC without loading real weights or producing an asset.

**Architecture:** Add an opt-in geometry-preparation runtime mode and IPC action, retaining the existing mock default. Reuse the inspected AMD geometry factory: its base creates only None model slots and its subclass configures samplers while deferring DINO/RMBG. Validate local configuration/checkpoint headers first; attach lazy CPU-only strict component loaders. Move previously tested metadata/offline guard code into reusable application modules rather than depend on proof scripts.

**Tech Stack:** Existing Python 3.12.10 / PyTorch 2.13.0+rocm10.0.0; Modly ExtensionProcess and runner; preserved AMD TRELLIS.2 source; safetensors.

**Spec:** User's Offline Alpha-Only Geometry Loader request; docs/compatibility/amd-sparse-decoder-loading-proof.md and amd-alpha-headless-import.md provide the prior boundaries.

## Global Constraints

- No real checkpoint load/placement, forward pass, generated GLB, downloads, installations or native rebuilds.
- Preserve installed environments/source edits, production mock behavior and all previous branches.
- Explicit local paths, alpha-only masks and strict state dictionaries; no remote fallback.
- Real import-time metadata initialization may occur only with the established healthy observer and no tensor allocations.
- Targeted tests only; stop if upstream needs substantial reconstruction.

## Review Focus

- Relative/remote/path-escaping references must fail before factory execution.
- Missing or changed assets must fail without leaving prepared state.
- Persistent buffer omissions/mismatch and remaining metadata tensors must fail lazy loading.
- Preparation must not relax ordinary mock-worker identity/guards.
- Cancellation/timeouts must terminate and reap the owned worker.

## Task 1: Local assets and lazy strict loading

Files: application metadata helper, experimental model_assets.py and strict_loader.py; targeted tests.

- [x] Write tests for bounded metadata, local references, complete parameter/buffer loading and mismatch cleanup; observe failure.
- [x] Factor existing parser; implement geometry/DINO asset validation and StrictLocalComponent.load_cpu()/unload() without import-time torch/model loads.
- [x] Verify small synthetic checkpoints only in the existing AMD interpreter, with GPU initialization blocked.

## Task 2: Real preparation through existing IPC

Files: amd_runtime.py, runner.py, extension_process.py, generator.py, geometry_preparation.py and preparation guards; targeted tests.

- [x] Write tests for explicit mode/identity, diagnostics IPC and fail-closed cancellation/timeouts; observe failure.
- [x] Reuse the local upstream factory with headless interfaces and deferred model slots; bind local maps, DINO and absent RMBG.
- [x] Return structured readiness/requirements from prepare_geometry(); no asset path or GLB.
- [x] Retain mock defaults; provide bounded shutdown and session cleanup.

## Task 3: Verification and publication

- [x] Run one observed application IPC preparation scenario: valid alpha, opaque rejection, recovery, unload/reload and shutdown; exercise cancellation without additional real loading.
- [x] Run focused regressions and compare passive before/after inventories/source hashes.
- [x] Review and write the report; independently verify saved evidence after interruption without rerunning GPU/IPC tests.
- [ ] Commit normally and push only codex/amd-offline-alpha-geometry-loader; verify remote refs/content at delivery.
- [x] Recommend one separately approved deterministic sparse-decoder forward as the next model-operation proof; stop here.

## Recovery ledger

The interrupted checkout and all 34 existing modified/untracked files were
preserved in `../alpha-loader-recovery-backup-2026-10-12` before documentation
edits. Saved results independently confirm 67 targeted test passes and real
application IPC preparation with no real weights/GLB. Fresh passive inventories
reconcile only directory allocation-size reporting differences; no installed
file/source/dependency changes are detected. Read-only recovery review found
no publication blocker. Publication verification is recorded after push, outside
the commit snapshot; no further implementation or GPU operation is authorized.
