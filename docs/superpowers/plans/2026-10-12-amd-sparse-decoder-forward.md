# Single sparse-decoder forward plan

> **For agentic workers:** Execute inline with executing-plans/TDD; request a read-only safety review before the GPU attempt.

**Goal:** One genuine sparse decoder forward on the RX 9070 XT, then release and stop.

**Architecture:** Reuse StrictLocalComponent, audited checkpoint identity, scratch/cache confinement, passive Windows observer and verified owned-child reaping. A new standalone proof harness uses a CPU-load gate followed by one parent-authorized placement/forward/release phase. Production and experimental application routes remain unchanged.

**Spec:** User's First Real Sparse-Decoder Forward request; geometry-loader and sparse-decoder loading reports.

## Constraints and inspected architecture

- Base 74f63c844e94da5b4a0742e5fe7ada9355ba3c13; new branch codex/amd-sparse-decoder-forward.
- Only audited ss_dec_conv3d_16l8_fp16 JSON/safetensors; no other checkpoints, mesh extraction, downloads, installations or native-extension rebuilds.
- Input [1,8,1,1,1] float32, fixed eight values [-0.875,-0.625,-0.375,-0.125,0.125,0.375,0.625,0.875]. Output [1,1,4,4,4] float32.
- Padded Conv3D and channel layer normalization permit a one-voxel latent. Two convolution/pixel-shuffle stages double each spatial dimension. The torso casts to float16; input/output convolutions retain float32.
- No gradients, no warm-up, exactly one root decoder invocation. Statistics/finite checks run on the copied CPU output.
- Commit and verify GitHub publication of the harness before any real GPU attempt.
- One persistent exclusive attempt marker; existing run evidence cannot be overwritten.
- 45s CPU gate, 30s placement/forward/release gate, 90s overall worker ceiling; observer queries capped at10s. Always reap the owned child and stop on error without retry.

## Task 1: Test and publish the harness

Files: amd_decoder_forward_contract.py, amd_sparse_decoder_forward.py, test_amd_decoder_forward.py; docs/evidence.

- [x] Add pure CPU tests for exact architecture/fixture, strict gate authorization, single-call consumption, valid output/stats, cleanup and fail-closed classification; observe RED.
- [x] Implement contract and worker. Reuse StrictLocalComponent; forbid additional model opens/imports, remote factories, arbitrary subprocesses, network, writes outside scratch and GPU work before authorization.
- [x] Permit only placement operations outside the one forward scope; allow only the actual decoder's module tree during inference. Block Torch/Triton compilation and pickle/other checkpoint loads.
- [x] Add bounded supervisor with durable stage logs, healthy before/gate/after observers, strict result validation and unconditional reaping.
- [x] Run targeted CPU-only tests; review exact safety boundaries; commit/push and verify remote commit before attempt.

## Task 2: One approved attempt and final publication

- [x] Capture passive environment/source/model preservation and checkpoint/source hashes.
- [x] After healthy observer, make the single approved attempt. Strict CPU load and AMD placement succeeded; the input transfer guard rejected aten.to.dtype_layout before forward. No retry.
- [x] Capture errors/process status even on failure; no retry. Compare inventories with directory allocation-size normalization only.
- [ ] Publish concise PASS/PARTIAL/FAIL report/results, verify remote refs and prior branch preservation; stop.

## Review focus

- No forward before complete CPU evidence and a fresh healthy observer.
- A failed invocation still consumes the single-forward permit.
- All exception/timeout branches terminate and reap, preserving last durable stage.
- A valid tensor alone cannot pass if count, cleanup, exit, observer or preservation fails.
- Imported local decoder support files must match the previous successful proof hashes.

## Outcome ledger

Harness was published before the attempt at 550a7ae00d2f9a306f2e7c0661b6767ef94c5614.
26 focused CPU tests passed; read-only review fixes preceded execution. Attempt
marker is consumed. Worker exited 1 and was reaped after 8.438s, observers healthy,
zero forward invocations. All before/after preservation pairs are byte-identical.
PARTIAL is the final evidence classification; no copy-guard relaxation or model
execution retry is authorized. Results publication is verified at delivery.
