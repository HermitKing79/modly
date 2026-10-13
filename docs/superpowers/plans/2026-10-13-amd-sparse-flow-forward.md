# AMD sparse-flow forward implementation plan

> **For agentic workers:** Use superpowers:executing-plans to execute the reviewed tasks inline. Steps are tracked below.

**Goal:** One strictly loaded, image-conditioned sparse-flow forward, without sampling.

**Architecture:** Keep production workers unchanged. Use a scoped checkpoint-storage adapter around the unchanged StrictLocalComponent and the genuine pinned dense flow class; restore its mixed runtime dtype contract. Reuse the owned Job, observer, diagnostics and synchronized BLAS teardown supervisor.

**Tech Stack:** Existing Python 3.12.10, PyTorch 2.13 ROCm10, safetensors; CPU-only metadata and synthetic tests.

**Spec:** User's First Image-Conditioned Sparse-Flow Forward request; source/config/feature identities recorded in the compatibility evidence.

## Global constraints

- One forward only; no warm-up, retry, sampler, other models, installation or download.
- Preserve source edits, installed environments, main, prior branches and historical evidence.
- Publish and verify the exact reviewed execution harness before GPU authorization.
- Strict all640 checkpoint payloads. Only220 explicitly named reversible BF16→FP32 widening exceptions. Procedural complex64 RoPE must be genuine and validated.
- Zero allocated/reserved bytes and verified owned-process exit remain mandatory.

## Review focus

- Generated buffer absent from checkpoint: an explicit named nonpersistent policy, with genuine deterministic initialization and value checks.
- Mixed precision: reject blanket conversion, wrong keys/shapes/dtypes and nonreversible widening.
- Source helper creates timestep CPU frequencies: authorize only that exact source copy within the timestep module.
- GPU operations/transfers and single module invocation: reject unexpected devices, lifecycle phases and counts.
- Error/timeout cleanup: no GPU recovery or second call; owned tree termination and durable evidence.

### Task1: CPU contract

Files: scripts/compatibility/amd_sparse_flow_contract.py, amd_sparse_flow_local.py, amd_sparse_flow_probe.py and test_amd_sparse_flow.py.

- [ ] Write and run failing focused tests for exact640/220 key schema, buffer policy, single invocation and output acceptance.
- [ ] Implement weights-free genuine meta construction and source-derived CPU RoPE validation; validate audited headers and saved features.
- [ ] Test scoped storage/widening interface with small local synthetic BF16 checkpoints only.
- [ ] Record actual full model meta manifest, feature integrity and no GPU initialization/real weights loading.

### Task2: bounded execution candidate

Files: scripts/compatibility/amd_sparse_flow_forward.py and narrow supervisor deadline-profile extension if needed.

- [ ] Implement explicit transfers, mixed-dtype guard, single root/block/attention counts, durable progress/memory and unchanged workspace cleanup.
- [ ] CPU-replay actual controls and cleanup with synthetic data; retain existing focused diagnostic/monitor regressions.
- [ ] Independent read-only review and focused verification; record reviewed deadlines and memory limit.
- [ ] Commit, push and verify exact harness before consuming exclusive attempt marker.

### Task3: one authorized run and publication

- [ ] Healthy observer; strictly load only flow checkpoint; one forward using saved DINO features.
- [ ] Validate finite FP32 [1,8,16,16,16], synchronised timing, release and process exit, or stop on first failure without retry.
- [ ] Compare preservation snapshots, document PASS/PARTIAL/FAIL, commit/push and verify remote commit. Stop.
