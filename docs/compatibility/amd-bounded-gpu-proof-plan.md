# Bounded GPU Compatibility Proof Plan

> **For agentic workers:** Execute this test-only plan in the current session. No adapter implementation is authorized.

**Goal:** Test metadata initialization, then one minimal CuMesh boundary-component operation if imports pass.

**Architecture:** System Python supervises separate existing-AMD-venv processes. Reuse the verified SDK/process/network/write guard, redirect all temporary/cache paths to workspace scratch, and permit GPU metadata while blocking tensor operations during Gate 1. Gate 2 is a separate single-shot process with staged diagnostics.

**Tech stack:** Existing Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, ROCm-native packages; stdlib supervision and unittest checks.

**Spec:** User's “Modly AMD — Bounded GPU Compatibility Proof” request in this chat.

## Constraints and review focus

- Preserve main, both previous proof branches, both installed environments, and modified AMD source.
- No installations, builds, downloads, models, inference, assets, or automatic retries.
- Import-side effects, unexpected tensor allocation, compilation, subprocesses and escaping caches must fail closed.
- Stop at the first failed import, timeout, crash, driver reset or native error.
- A fixture PASS cannot prove hole filling, texture baking, large-mesh stability, or complete generation.

## Tasks

- [x] Verify clean starting branch/HEAD/remotes, read both reports and evidence, inspect import-time source.
- [x] Add `scripts/compatibility/test_amd_bounded_gpu.py` checks for visible device selection, Gate 2 rejection without Gate 1 PASS, bounded termination, and rejection of malformed/incorrect fixture results.
- [x] Add `scripts/compatibility/amd_bounded_gpu_proof.py` supervisor/worker. Run the guard checks before native work. Keep the previous harness unchanged.
- [x] Preserve the interrupted zero-worker PARTIAL run; reconcile directory allocation-size observations; repair and independently validate stock PowerShell monitoring before any GPU resume.
- [x] Snapshot dependencies, native/source hashes and complete venv file metadata; capture passive Windows GPU/driver metadata/events. Run two Gate 1 import workers once, each with a 30-second timeout. Result: PASS.
- [x] Only after Gate 1 PASS, inspect CuMesh implementation/prerequisites and provenance. Define one fixed fixture and independently derive its expected boundary component count/IDs.
- [x] Run Gate 2 once in a disposable process with a 30-second timeout and per-stage synchronization/diagnostics. Result: PASS; no retry.
- [x] Compare inventories and source Git state; write `docs/compatibility/amd-bounded-gpu-proof.md` and exact evidence.
- [x] Review the final scope/claims. Independent read-only review approved the scoped PASS report for publication.

Publication uses only `codex/amd-bounded-gpu-proof`. The final chat response records the verified published commit after the commit containing this plan; no merge or adapter implementation is authorized.
