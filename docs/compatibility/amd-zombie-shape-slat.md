# Genuine zombie Shape-SLat generation

PASS: one genuine Windows AMD shape-sampling sequence completed and all five
artifacts were independently reopened after natural process settlement.

| Boundary | Verdict |
| --- | --- |
| Zombie image and prior neural artifacts | PASS; exact separate completed roots reauthenticated |
| Pooled shape support and input admission | PASS; 1,000 unique I32 tokens, unchanged 1..4,430 cap |
| Genuine Shape-SLat computation | PASS; one sequence, 12 Euler steps, 21 model forwards (12 positive / 9 negative) |
| Five-artifact numerical/provenance readback | PASS; finite contiguous F32 [1000,32], exact coordinates and byte-exact CPU F32 denormalisation |
| Natural settlement and owned GPU memory release | PASS; healthy monitoring, natural empty Job, measured Torch 0/0 bytes |
| Detailed geometry and full image-to-3D | NOT RUN |

No DINO, sparse sampling or occupancy rerun occurred. WSL, detailed geometry,
materials, meshes, GLB and product Generate were not invoked. The running image
preview was left untouched. This numerical latent proof makes no new visual
geometry-quality claim and does not qualify full generation.

## Lineage and implementation

Approved base: `0d15dd96f31ded71564657e5890f51ed1858ad97`.
Actual new execution: **`35a5b825e3e9128326897a2ce35c87b237fdfa8b`** on
`codex/amd-zombie-shape-slat`. The saved DINO execution remains
`93269412602a338acd9fbde41798901ddb5f707d`; sparse/occupancy remain
`e6b6330ba2f9e51abfb74c726729e0be4ade7cac`. Their original receipts were neither
rewritten nor combined into an invented historical execution.

Original image SHA-256:
`db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e`.
DINO SHA-256: `e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60`.
Sparse latent SHA-256: `ecf57d3a0db0bec99b79712fc3626c5dd224cff03e33c60e0297500cde761930`.
The original logits/mask/coordinate identities and 114 prior receipt hashes are
recorded in the independent [input review](amd-zombie-shape-evidence/input-review.json).
Native logits/mask were reopened and original positive-mask max-pool-by-two
matched an independent floor-divide/unique-coordinate reference.

The resulting 32³ support has D/H/W bounds `[0,12,0]` to `[31,19,31]`.
Its coordinate payload SHA-256 is
`b871475ff86493ca9d023aa7736262a898193e3a1b6833a2b6716f427cf2a603`.
Original order and all pooled cells were preserved; no clipping or resampling.

Only `amd_zombie_shape.py` and two focused test files were added. The adapter
subclasses the existing current-image Runtime for separate-root authentication.
It retains the original factory, sampler, precision, transfer controls,
supervisor, binary authorisation, workspace teardown and artifact writer.
Its constructor does no input I/O before the original real owned-Job handshake.
Two exact immutable historical proof tensors remain readable solely because
the old readers hash their source closure; they are not neural inputs.
The production CPU-only UpstreamJob authority is not used.

The full shape checkpoint was rehashed:
`ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f`.
Config SHA-256: `ddbd7a1d34ce9a8e7af2f6ef9ae686d9ab49f69e62dacb98d8409b511cdbd5e2`.
Original 640 BF16 storage entries and 220 reversible runtime F32 widenings were
retained. Runtime: Python 3.12.10, Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`,
RX 9070 XT / gfx1201, explicit genuine SDPA. No backend fallback or convolution.

Sampling retained seed20261013, sigma1e-5, rescale_t3, guidance7.5,
guidance-rescale0.5, interval[0.6,1.0] and original dino-lock0. Every original
30-block self/cross-attention hierarchy completed 21 times; cache identities and
the complete original step schedule passed validation.

## Qualification and measured resources

Twenty focused CPU tests passed. Independent source review also passed all20,
then exercised the actual input/model-metadata route under the original tensor
read boundary with neural imports prohibited. Review findings added exact
checkpoint-provenance equality, all six CPU input schemas/auxiliary hashes,
paired CPU-gate negative checks, Job-close rejection and signed-zero-sensitive
F32 denormalisation checks before execution. Original neural sources remained
unchanged. See [tests](amd-zombie-shape-evidence/focused-tests.json) and
[pre-execution review](amd-zombie-shape-evidence/pre-execution-review.json).

CPU projections for N1000 were 307,424,000 visible-buffer bytes and
3,078,619,136 bytes using the conservative historical reserved-peak comparison.
These are estimates, not measured GPU peaks or guaranteed bounds. CPU admission
also accounted for the genuine saved input sizes, 16,000 coordinate payload
bytes, 400,512 numeric-output-file bytes and <=65,536 provenance bytes.

| Actual measurement | Value |
| --- | ---: |
| CPU model load | 14.156 s |
| Synchronized sampling | **9.078 s** |
| Supervised worker elapsed | 34.093 s |
| Peak Torch allocated | **2,787,808,256 B** |
| Peak Torch reserved | **2,808,086,528 B** |
| Initial free device memory | 16,937,582,592 B |
| Minimum observed block-free device memory | 13,985,710,080 B |
| Available host RAM at strict load | 16,562,831,360 B |
| Required original strict-load RAM | 11,411,771,648 B |
| Worker peak working set / private commit | 5,677,793,280 / 6,809,354,240 B |
| Owned Torch allocated / reserved after release | **0 / 0 B** |

The unchanged 13 GiB initial free-VRAM gate, 12 GiB allocator ceiling,
3 GiB live block reserve (2 GiB margin + 1 GiB opaque allowance) and
180/240/480-second CPU/GPU/worker-overall deadlines remained active.
Non-elevated outside-sandbox context, original interpreter and all91 committed
source identities were authenticated before launch. Fresh prelaunch available
RAM was 17,023,365,120 bytes; destination free space was 133,898,084,352 bytes.

Both output readers and all journals/observers settled. Three periodic Windows
samples plus before/gate/after captures were healthy; 88 contiguous worker
traces independently show 21 root entries/returns and 12 completed steps.
There was one model load and one BLAS workspace clear. Worker exit was zero,
owned Job active counts were 0 before/after cleanup and no forced cleanup or
timeout occurred. Model and owned tensor collection were verified.

Original standalone Jobs provide ownership containment without aggregate or
per-process RAM byte caps. Continuous whole-host RAM minima and aggregate Job
peaks were not collected. Sampled global GPU usage at release was 310,575,104
bytes; Torch zero is not driver/context zero. No extra post-exit GPU query was
performed. Original SDK-tool discovery still reports the absent development
package; the installed runtime executed successfully and no packages changed.

## Private outputs

Exact common directory (all outputs remain local and Git-excluded):

`C:\Users\wattz\Documents\ChatGPT\modly\proof\artifacts\amd-zombie-shape-0d15dd9\shape\run\scratch`

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `shape-latent-raw.npy` | 128,128 | `dc67484797a60e3e1c4ff1941bb5c29977707d2c0444d0c63c0df43087a91405` |
| `shape-latent-denormalized.npy` | 128,128 | `18778e295e7d9405ca2e80b8fde64dacc790696c361f6418436d2e498149b2b3` |
| `shape-latent-coordinates.npy` | 16,128 | `c432b04ec70c6200fadf8f5f62ce995514e24bf4194b154ac30a76148a7bb4a2` |
| `shape-initial-noise.npy` | 128,128 | `bcc80a25b7f1e2fe713499edcf056df956d77da0f4425d8c97b2eee80ebe2147` |
| `shape-latent-provenance.json` | 18,748 | `1484e9efdd60cee9b86f3d8bf07eac6a1bce318c5960fb5f00cac435580a5ab4` |

Raw/denormalized/noise are finite contiguous F32 `[1000,32]`; coordinates are
I32 `[1000,4]` and exactly equal the authenticated original pooled support.
Initial-noise payload SHA-256:
`493775467c0beab6c72aac9955f7f537ea013571df83d15b422f98488f2f9c6c`.
The original seeded CPU generator and sampler were unchanged. Sidecar contents,
all file/payload hashes, source-image/checkpoint/old-and-new commit lineage,
and byte-exact original CPU F32 `raw*std+mean` were independently verified.

The host reopened outputs after the real worker exit; a separate CPU process
reopened all5 again. An independent reviewer separately authenticated all91
Git source blobs, full checkpoint, actual input roots, tensors, trace/gate/Job
evidence and numerical readback: [outcome PASS](amd-zombie-shape-evidence/outcome-review.json).
The actual result receipt SHA-256 is
`913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf`.
See [measured results](amd-zombie-shape-evidence/measured-results.json).

## Preservation and stop

[Preservation after](amd-zombie-shape-evidence/preservation-after.json) equals
before: 100 protected identities, 78 bounded content hashes, interpreter,
accepted-master metadata and saved full-hash provenance, historical caches,
original image/DINO and all prior zombie source/receipts/arrays/previews remain
unchanged. Large historical payloads/caches were not unnecessarily bulk hashed;
the receipt records the precise bounded scope. Only this new branch is pushed;
the fresh139-head remote baseline is compared again at final publication.

User images, neural tensors, generated assets and raw process identities stay
private. No installed environment, accepted asset or historical source was
modified. Next justified step is separately authorised detailed geometry
decoding from this saved latent and exact coordinates; it has not been run.
Stop reached after the single Shape-SLat milestone.
