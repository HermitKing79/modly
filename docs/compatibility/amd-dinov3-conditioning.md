# First real AMD DINOv3 image conditioning

2026-10-12. Branch `codex/amd-dinov3-conditioning-proof`, based on
`bb6988d46cf49343208723bd591eef2ec42bfd53`. Execution harness
`846decff797517280ac9b8b426ed9bcdd4d9490e` was independently reviewed, pushed and
verified on GitHub before the exclusive attempt marker was created.

**PARTIAL.** The genuine DINOv3 component completed exactly one GPU conditioning
pass and validated the expected output. Strict live-worker cleanup failed:
79,691,776 bytes (76 MiB) remained allocated and reserved. The model/output were
collected and the coordinate cache was cleared, but the remaining allocation's
owner is unresolved. The harness stopped with handled exit 1; no retry or
post-error GPU recovery was attempted. The owned process tree exited normally,
with no forced termination, timeout, recorded native error or driver reset.
This is a release-validation failure, not a model-state or computation failure.

## Offline model and CPU preparation

The local asset is
`D:/AI/Trellis2AMD/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m`, using
its genuine `config.json` and `model.safetensors`. The previous DINO audit matches
all metadata. Checkpoint SHA-256:
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
It contains 415 F32 persistent entries, 303,129,600 parameters and 1,212,518,400
tensor bytes. The explicit DINOv3ViTConfig/DINOv3ViTModel constructor uses local
JSON only; no `from_pretrained()` or remote identifier is accepted.

The [owned CPU/meta probe](amd-dinov3-conditioning-evidence/cpu-metadata/results.json)
matched every checkpoint key, shape and dtype in 23.437s without weight loading,
forwards or GPU initialization. The unchanged StrictLocalComponent subsequently
loaded the real checkpoint once, CPU-only, with strict/assign loading, no missing
or unexpected keys, and byte-for-byte payload verification of all 415 entries.
Loading took 2.391s. Peak CPU working set was 3,001,597,952 bytes and private
commit 4,793,417,728 bytes. All persistent entries remained F32; no quantization
or precision conversion was used.

The sole nonpersistent buffer, `rope_embeddings.inv_freq` (F32 [16]), is created
by the genuine CPU RoPE constructor, then explicitly verified on CUDA:0 after
placement. The constructor initializes no real model weights. The genuine
patch-coordinate LRU closure is verified, starts empty and is cleared after
execution; its GPU tensor is included in resource accounting.

The original DinoV3FeatureExtractor receives the strict-loaded model through
its inspected constructor assignments. The unmodified, hash-pinned upstream
`preprocess_image` and `get_cond` bodies are bound without importing unrelated
geometry/native modules. Existing alpha validation is reused; RMBG is absent.
Production mock and preparation routes are unchanged.

## One execution and measurements

Runtime: existing Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333,
CUDA build metadata null; transformers 4.56.0, torchvision 0.28.0+rocm10.0.0,
safetensors 0.8.0. GPU: AMD Radeon RX 9070 XT, gfx1201, 17,095,983,104 reported
total bytes. Runtime/module paths, seven pinned source hashes and full strict
state manifests are saved in the [evidence](amd-dinov3-conditioning-evidence/run/results.json).

The [deterministic input](amd-dinov3-conditioning-evidence/input.png) is 64x64
RGBA with a transparent border and opaque gradient foreground. PNG SHA-256:
`9abc7b8f89e5b001e863fc3ef2fbd6813b2e83f63c0a6b0a45203c2ef1edfd8d`.
Genuine alpha preprocessing produces a 46x46 RGB crop; the real extractor uses
its 512x512 path, F32 [1,3,512,512] normalized image and SDPA backend.

Exactly one extractor/features invocation, embeddings/RoPE call and each of 24
transformer layers completed. Counters record one patch convolution and 24
SDPA calls. One image copy, one output copy and the two separately counted
F32 normalization-constant copies are recorded. Scoped copy wrappers permit
only these exact objects/devices/phases, without changing normalization math.
The second same-device image `.cuda()` is a verified no-op.

Output: **[1,1029,1024], float32, CUDA:0, 1,053,696 finite values**. This retains
1024 patches, one CLS and four register tokens, with 1024 conditioning channels.
CPU validation reported min -29.3398266, max 24.2708645, mean -1.9984e-9 and
standard deviation 1.0. Output hash
`94080c671ad938d567ebdc87cbe21aabcd61ae0b153595d3af8a3d6c69aa90bc`
is worker-reported; full values were not saved for independent recomputation.
Inference mode was enabled, gradients disabled, with no CPU model fallback.

Placement and synchronization: 0.312s. Conditioning including final GPU
synchronization: **29.703s**. Overall worker supervision: 44.640s, within the
120s CPU / 180s GPU / 360s overall bounds. Durable module entry/host-return
markers are distinct from the final synchronization marker; these are not
per-kernel timings. Embeddings host return took 25.813s, first layer 3.125s and
final synchronization 0.281s.

| Memory sample | Allocated bytes | Reserved bytes | Available VRAM bytes |
|---|---:|---:|---:|
| Before placement | 0 | 0 | 16,937,582,592 |
| After placement | 1,213,567,488 | 1,226,833,920 | 15,699,476,480 |
| After conditioning | 1,297,482,240 | 1,371,537,408 | 15,408,168,960 |
| Release check | **79,691,776** | **79,691,776** | 16,695,820,288 |

Peak allocated: 1,343,300,096 bytes; peak reserved: 1,371,537,408 bytes. Release
weak references confirm model/output collection and coordinate-cache occupancy
1→0. Zero live-worker allocation was not achieved. The reused validator's error
label, “Decoder release incomplete”, refers to this DINO cleanup check; no sparse
decoder ran. No GPU memory query was made after process exit, so zero post-exit
VRAM is not claimed.

Fresh private scratch contains 518 COMGR files, three MIOpen cache/database
files, two temporary files and the input PNG: 524 files, 670,346,156 bytes.
[Their inert hashes](amd-dinov3-conditioning-evidence/scratch-side-effects.json)
are preserved; no historical cache was executed, changed or deleted. Sampled
host CPU activity during embeddings totals 19.078 CPU-seconds. Fresh compilation
products and this activity are consistent with first-use compilation, but no
distinct compiler executable was observed. Its contribution cannot be isolated
from other native work.

## Tests, monitoring and preservation

[65/65 focused CPU tests](amd-dinov3-conditioning-evidence/cpu-reviewed-candidate.json)
passed in bounded isolated subprocesses with AI imports blocked, including
actual production transfer/dispatch/module functions, supervisor protocol,
malformed/duplicate rejection and the 52 prior diagnostic/observer/ownership
regressions. Independent review ran all 13 new fixtures and found no remaining
pre-attempt blocker. [Post-attempt review](amd-dinov3-conditioning-evidence/post-attempt-review.md)
and the saved-evidence analysis confirm PARTIAL without relaxing cleanup.

All seven periodic Windows samples were healthy, maximum sample gap 5.047s;
fresh before/CPU-gate/after-exit checks were healthy. Job attachment preceded
preparation, owned-tree exit was verified with zero active processes, and both
journals, monitors and output readers stopped without errors.

[Preservation](amd-dinov3-conditioning-evidence/preservation.json) confirms both
environment inventories and compressed file metadata, model/HF/Torch cache
trees, existing AMD `nodes.py` edits/backup, 1,571 historical files and all prior
local/remote branch heads unchanged. Main remains
`f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`. No downloads, installations,
native-library rebuilds, other checkpoints, geometry, texture or GLB operation
occurred. Ignored scratch/cache artifacts remain in the local evidence directory.

## Smallest next proof

The falsifiable blocker is the **76 MiB active allocation after model/output
and RoPE-cache release**. Its owner is not established; do not call it either a
known runtime workspace or a persistent leak. The next separately approved
experiment should repeat only this deterministic conditioning fixture with
source-reviewed allocator-history/snapshot diagnostics at release, identifying
the exact remaining blocks and owners. Keep the same one-pass, model, precision,
observer, Job and 120/180/360s limits; do not lower the zero-memory criterion or
load a flow model. No such experiment was implemented or run here.

Conditioning computation is demonstrated; full cleanup, quality/parity, sparse
generation, pipeline reliability and the historical CuMesh 719 issue remain
unproved. Stop after publishing this result.
