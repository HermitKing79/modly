# Genuine guided turret texture decoder

Base `c9e4d3b3555fc3f35a998053592e13c476b60463`; branch `codex/amd-turret-texture-decoder`.

**Prepared; real decode not yet attempted.** The single authorized run is conditional on the reviewed candidate being published and all fresh admission gates passing. No earlier image/model/geometry stage will run.

The original texture decoder has 284 F16 checkpoint tensors, 32 input / six output channels and `pred_subdiv=False`. Strict loading preserves 208 original F16 body parameters and restores only 76 source-required F32 norms/projections with byte-exact F16 round trips. No BF16 model conversion. Real weights were not loaded during CPU qualification.

All four saved BF16 guide tensors are preserved byte-for-byte. Masks use `> 0`; the live parent coordinate order, original guide order and both C2S expansions at each stage are checked. Supports are 3743 -> 17015 -> 79169 -> 378891 -> 1842968 rows. The final texture coordinates must exactly equal the original shape support, including order.

All 40 genuine convolutions remain. The final explicit-GEMM convolution retains the full global neighbor map and original F16 mathematics, processing 32768 output rows per tile (57 tiles, tail 7960). No support truncation, resolution change, subdivision alteration or mesh operation.

## Qualification and safeguards

72 focused CPU tests: **71 passed, one Windows symlink-privilege skip**. These cover strict texture schema/precision, adjacent strict-loader regressions, actual input reader and original source C2S topology, live guided hooks, F16/F32 row-tiling equivalence, six-channel writer/reader and launcher/receipt preflight. The permanent outside-Git writer probe passed actual file creation, fsync, SHA-256 and independent readback. Independent static review found no open blockers. The exact read-only launcher preflight passed before an attempt marker, WSL launch or Torch import.

A helper briefly started the unrelated application test suite; it was stopped, is not claimed as passing, and resulted in no unrelated fixes. The visible Volume Decoding progress was an existing literal test fixture. The approved real decoder attempt remains unused at preparation time.

Admission is unchanged: minimum 13 GiB free VRAM and 12 GiB available host RAM; 12 GiB Torch allocator ceiling; 2 GiB device-free reserve plus 1 GiB opaque-workspace headroom; original padded sparse-allocation admission and the reviewed exact final-support exception. Native/CPU/GPU deadlines are 60/120/240 seconds; guest/host deadlines 450/480 seconds. Binary GO, Windows Job containment, periodic Windows observation, five-second natural Job grace and termination of only the named Modly Ubuntu distro remain in force. No retry.

Expected output is finite raw F32 `[1842968,6]`, the original unclamped F32 `raw * 0.5 + 0.5`, and original I32 `[1842968,4]` coordinates. Channel order is red, green, blue, metallic, roughness, alpha. The actual writer and independent reader must verify all tensors and provenance while the worker lives and again after clean exit. Zero allocated/reserved Torch memory, collected model/tensors, healthy observation and unforced process release are mandatory.

Permanent output leaf, already prepared and write-tested:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-decoder-c9e4d3b\field`

[CPU qualification](amd-turret-texture-decoder-evidence/qualification.json) · [input identities](amd-turret-texture-decoder-evidence/input-qualification.json) · [precision/schema](amd-turret-texture-decoder-evidence/texture-cpu-contract.json) · [persistence](amd-turret-texture-decoder-evidence/persistence-preflight.json) · [read-only preflight](amd-turret-texture-decoder-evidence/preflight.json) · [independent review](amd-turret-texture-decoder-evidence/independent-preexecution-review.json).

No learned field, coloured mesh, UV texture, baking or textured GLB is claimed by preparation. A successful field would justify a separately approved CPU learned-RGB preview on a separate copy of the existing mesh, retaining every original vertex and triangle.
