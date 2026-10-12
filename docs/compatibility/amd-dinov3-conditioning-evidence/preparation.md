# One DINOv3 conditioning attempt

Base `bb6988d46cf49343208723bd591eef2ec42bfd53`; branch
`codex/amd-dinov3-conditioning-proof`. This authorizes one image-conditioning
pass only. No flow/decoder/mesh/texturing operation or production route changes.

The explicit local asset is
`D:/AI/Trellis2AMD/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m`.
The audited configuration and checkpoint metadata match: 415 F32 persistent
entries, 303,129,600 parameters, 1,212,518,400 tensor bytes. Complete checkpoint
SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
The weights-free owned CPU/meta probe matches every key/shape/dtype and exits
cleanly in 23.437s, without a GPU context, model-file tensor reads or forwards.

StrictLocalComponent remains unchanged. DINOv3ViTConfig(**verified_local_json)
and DINOv3ViTModel(config) construct on meta; the real CPU
DINOv3ViTRopePositionEmbedding initializer creates the sole nonpersistent
F32 [16] inv_freq buffer. No from_pretrained factory is used. The runtime's
patch-coordinate LRU closure is verified empty and explicitly cleared after
the pass; cached GPU tensors count toward cleanup.

The genuine DinoV3FeatureExtractor receives the strict-loaded local model via
its four inspected constructor assignments. Only the unmodified, hash-pinned
preprocess_image/get_cond method bodies are bound from the actual pipeline;
unrelated full-pipeline/native geometry imports are excluded. Established
alpha validation and actual crop/compositing are used. A deterministic 64x64
RGBA fixture has a transparent border and a 48x48 opaque gradient foreground;
the upstream crop is 46x46 RGB, resized by the genuine extractor to 512x512.
The output contract is F32 CUDA:0 [1,1029,1024]: 1024 patch + 1 CLS + 4 register
tokens. Negative conditioning is explicitly excluded; only one extractor call,
one feature call, embeddings, RoPE and each of the 24 layers may execute.

Two narrow copy wrappers preserve the real feature/Normalize computation:
the first image .cuda() uses the known explicit aten._to_copy(device=cuda:0)
path; the second, same-device .cuda() is a verified no-op. Normalize's exact
F32 mean and std are each copied once, in order, only inside that module. This
avoids granting an unverified constructor-wide/device-transfer allowance.
CPU/meta constructor inspection produced no dispatch events; it is not claimed
as evidence of a native GPU constructor. The actual scoped wrappers and guard
are replayed with CPU fixtures, including negative device/phase/dtype/argument/
duplicate cases. No float16/quantization/CPU model fallback is allowed.

65 focused CPU tests pass with AI imports blocked, including the production
supervisor handshake, actual dispatch/module/transfer functions, durable
journals and all 52 existing diagnostic/observer/ownership regressions.

The original supervisor is reused with a minimal explicit proof profile.
CPU 120s, GPU placement/conditioning/release 180s, overall worker 360s; Job
Object attachment precedes preparation, GPU authorization requires a strict
CPU load and fresh healthy observers, Windows sampling starts every 5s and
becomes fail-closed if unhealthy/stale. CPU/child activity and entry/host-return
markers persist durably; GPU synchronization is a distinct recorded operation.
On failure, terminate only the owned tree and stop without GPU recovery/retry.

All cache/home/temp paths are fresh owned scratch. No old artifact is executed
or assumed eligible for this different workload. Triton compile/JIT launch,
external installers/subprocesses/network/downloads and other checkpoints are
blocked; only verified read-only rocm-sdk discovery is allowed.

The model uses about 1.13 GiB of F32 tensor payload; strict CPU validation needs
clones and byte comparisons. Require at least four payload sizes plus 1 GiB
available host RAM and payload plus 2 GiB available VRAM before placement.
No-gradient/inference mode avoids activation retention. These are preflight
margins, not measured peak claims. Capture actual CPU/GPU peaks and release.

Publish and independently verify the exact reviewed harness commit before the
exclusive attempt marker is created. No second conditioning attempt is allowed.
