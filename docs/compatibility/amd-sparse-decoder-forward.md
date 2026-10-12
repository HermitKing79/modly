# One minimal real AMD sparse-decoder forward

**Pre-attempt publication: GPU execution NOT ATTEMPTED.** This snapshot publishes
the bounded harness, CPU tests and preservation evidence before the single
approved real run. Final classification/results will be added after that attempt.

Branch `codex/amd-sparse-decoder-forward` starts at
`74f63c844e94da5b4a0742e5fe7ada9355ba3c13`. The
[geometry-loader report](amd-offline-alpha-geometry-loader.md) and
[sparse-decoder loading proof](amd-sparse-decoder-loading-proof.md) were read first.
Application workers and production mock behavior are unchanged.

## Inspected operation

Only the genuine local `SparseStructureDecoder` at
`D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16`
is permitted. Its verified JSON specifies latent channels 8, output channels 1,
channels [512,128,32], two residual blocks per stage, two middle blocks and fp16
torso. The complete checkpoint has 74 tensors / 147,585,412 payload bytes.

Padded 3D convolutions and channel layer normalization allow a one-voxel latent;
two convolution/pixel-shuffle stages double each spatial dimension. The smallest
input is therefore **[1,8,1,1,1] float32**, expecting **[1,1,4,4,4] float32** output.
The eight fixed latent values are [-0.875,-0.625,-0.375,-0.125,0.125,0.375,0.625,0.875].
The decoder casts its middle layers to float16 and casts back for the output
convolution. No full-model dtype conversion or architecture patch is applied.
The result is occupancy logits, not a mesh or generated asset.

## Scope and guards

[Harness](../../scripts/compatibility/amd_sparse_decoder_forward.py) reuses the
application `StrictLocalComponent`, audited checkpoint identity, 17 pinned
imported source files, scratch/cache environment, passive Windows observer and
owned-child reaper. CPU loading requires all parameter/persistent-buffer keys,
shapes, dtypes and payload bytes to match. Generic remote/partial factories and
pickle loading are prohibited.

Healthy Windows observations are required before launch and again after complete
CPU loading, before parent authorization of GPU work. Outside the forward scope,
GPU dispatch allows only copies/allocation bookkeeping. During inference, module
calls are restricted to the genuine decoder's own module tree; both a single-use
permit and root call counter prohibit retries and warm-ups. Output finite checks
and statistics run on a copied CPU tensor. Deterministic algorithms are required,
with cuDNN/MIOpen benchmark disabled; unsupported deterministic execution fails.

No other checkpoint, real pipeline class, DINO, flow, shape/texture decoder,
CuMesh, mesh extraction or GLB export is admitted. Network, arbitrary subprocesses
and Python writes outside scratch are blocked. Only the previously verified
read-only SDK discovery command is allowed, capped at five seconds. Torch/Triton
compilation and launches are blocked; internal ROCm runtime cache activity is
recorded separately and is not an installed-extension rebuild.

The exact executed harness/helpers must match the committed, published snapshot.
An exclusive persistent attempt marker and canonical run directory prevent
automatic reruns or evidence replacement. Stage/stdout/stderr lines are flushed
and fsynced. CPU work has a 45-second deadline; placement/forward/release has
30 seconds; the worker ceiling is 90 seconds. Observer queries have 10-second
timeouts. Every exit path verifies child reaping; after an error, no further GPU
work or repair is attempted.

## Pre-attempt verification

[Final CPU tests](amd-sparse-decoder-forward-evidence/cpu-tests-final.json)
record **26 passes**: 12 new pure harness-contract tests, 9 existing strict gate/
reaper tests, and 5 synthetic CPU-only StrictLocalComponent tests in the existing
AMD Python. Tests cover exact fixture/output, failed single-call consumption,
finite statistics, cleanup/classification, unhealthy observers, protocol errors,
lazy package admission and committed-byte binding. They perform no real-model
loading, GPU initialization or GPU forwards.

[Read-only safety review](amd-sparse-decoder-forward-evidence/pre-attempt-review.md)
found two pre-attempt guard issues; both were fixed and reviewed before any real
run. [Passive identity evidence](amd-sparse-decoder-forward-evidence/pre-attempt-identity.json)
confirms the same audited checkpoint/source and records `torch_imported=false`.
Before inventories preserve both installed environments, model/cache metadata,
selected native/source hashes and existing AMD source edits. Main and all previous
branches remain unchanged. No installation, download or native rebuild occurred.

## Pending single attempt

After verifying this harness commit on GitHub, run the existing AMD Python worker
once through the supervisor. Record strict loading, placement, one synchronized
forward, output/statistics, VRAM, weak-reference release, worker exit and Windows
observations. Stop on any error, timeout, reset or native crash without retry.
Even a PASS will establish only this decoder operation on this tiny synthetic
latent, not image conditioning, full geometry generation, textures or CuMesh 719.
