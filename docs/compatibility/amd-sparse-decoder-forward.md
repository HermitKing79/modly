# One minimal real AMD sparse-decoder forward

**PARTIAL — 12 October 2026.** The single approved attempt stopped at the
synthetic input transfer because the harness rejected `aten.to.dtype_layout`.
Strict CPU loading and decoder placement succeeded, but **zero decoder forwards
executed**. This is a test-guard boundary, not evidence of an AMD forward failure.
The consumed attempt was not retried and the guard was not relaxed after failure.

Branch `codex/amd-sparse-decoder-forward` starts at
`74f63c844e94da5b4a0742e5fe7ada9355ba3c13`. The
[geometry-loader report](amd-offline-alpha-geometry-loader.md) and
[sparse-decoder loading proof](amd-sparse-decoder-loading-proof.md) were read first.
Application workers and production mock behavior are unchanged.
The reviewed harness was committed and verified on GitHub at
[`550a7ae00d2f9a306f2e7c0661b6767ef94c5614`](https://github.com/HermitKing79/modly/commit/550a7ae00d2f9a306f2e7c0661b6767ef94c5614)
before any real GPU attempt; the supervisor verified 14 executed local dependency
files against that commit. Final evidence is published in a subsequent commit.

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

## Exact single-attempt result

[Complete result](amd-sparse-decoder-forward-evidence/run/results.json), durable
stdout/stderr and the three Windows observer captures preserve the diagnostic.
Runtime was Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, CUDA null,
from the existing AMD venv. GPU identity was RX 9070 XT / gfx1201.

| Operation | Actual result |
|---|---|
| Strict CPU checkpoint load | Once, 1.797s, all 74 tensors and payload hashes matched, no missing/unexpected entries; GPU uninitialized |
| CPU memory after load | 611.14 MiB working set, 1,033.67 MiB private commit; peaks 751.82 / 1,174.71 MiB |
| GPU baseline | 0 allocated / 0 reserved, 16,152.94 MiB available before placement |
| Decoder placement | Completed in 0.203s; synchronized and all tensor GPU schemas validated |
| Synthetic input transfer | Rejected before execution: `Unapproved GPU operation outside forward: aten.to.dtype_layout` |
| Root/permit forward counts | Both 0; no warm-up or decoder forward |
| Output / finite statistics | Absent; no output tensor was produced |
| Forward/after-placement/after-release VRAM | Not recorded after the rejected copy; no zero-allocation cleanup claim |
| Worker | Exit 1, 8.438s; reaped, output readers finished, no timeout or forced cleanup |
| Observers | All three healthy, controller OK, driver 32.0.31041.1004, no queried reset/error events |

The error occurred at `latent = cpu_input.to('cuda:0')` within inference mode.
The transfer whitelist reused from the earlier placement proof permits
`aten._to_copy`, copies/allocation bookkeeping and detach/alias operations; this
input-copy dispatch used the additional `aten.to.dtype_layout` overload. The
guard raised before calling the operation and before consuming the forward
permit. The stage label remains `decoder-placement`; the traceback pinpoints
the subsequent input transfer. No native crash, error 719, reset or unsupported
decoder math was observed. CPU tests did not cover this actual GPU dispatch path.

On the error path, owned references were dropped and the worker exited. No
further model computation or explicit GPU memory query was issued to prove allocator or global VRAM
recovery after failure; process termination is verified, measured tensor-release
acceptance remains unfulfilled. A PASS is therefore explicitly rejected.

SDK discovery ran once, returned the previously known missing-devel error in
1.656s, and was handled during import. No package repair occurred. Six native
runtime cache artifacts / 148,712 bytes appeared only in workspace scratch
([inventory](amd-sparse-decoder-forward-evidence/scratch-side-effects.json)); they
remain local and ignored. This is not a claim of zero internal driver/cache
activity. No installed GPU library was rebuilt and no GLB was exported.

## Preservation and next step

[Before/after preservation](amd-sparse-decoder-forward-evidence/preservation.json)
verifies byte-identical environment/dependency records and raw compressed file/
model/cache metadata, selected native/source hashes and all previous branch heads. Checkpoint
before/after records are identical. The existing `M nodes.py` and untracked
`nodes.py.before-profiler-fix` remain intact. Both installed environments, model
weights and production mock routes were preserved. No installations, downloads,
source edits or additional checkpoints occurred.

The next prerequisite is to inspect and narrowly admit the exact input-copy
overload, with transfer-only CPU regression coverage and another pre-published
harness. A **fresh single-forward attempt requires separate approval**. No retry
or speculative guard change was made in this milestone. After that forward
passes, the shortest image-to-geometry path is offline alpha/DINO conditioning
followed by the local 512 sparse-flow/decoder stages, each under separately
approved model-load/execution scopes. Real generation, texturing and CuMesh 719
remain unresolved. Stop here.
