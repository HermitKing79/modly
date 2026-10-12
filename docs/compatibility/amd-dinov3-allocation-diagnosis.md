# DINOv3 allocation diagnosis and saved conditioning features

2026-10-12. Branch `codex/amd-dinov3-allocation-diagnosis`, based on
`dfc01cb2f233e2d2cbabfcec6a27ac4c42f23668`. Independently reviewed execution
harness `827bd6d00c549a28c435b4602b8f4bdc6a2eb8a5` was published and verified
on GitHub before the exclusive single attempt.

**PARTIAL:** Real image conditioning succeeded and its CPU feature checkpoint
passes independent integrity verification. Complete live-worker GPU release
failed: one active 76 MiB allocation remains. Its origin and exact workspace-size
match strongly support a retained BLASLt workspace; native ownership is not
proved. No second attempt or workspace-clearing operation was performed.

## Conditioning and checkpoint

Reuse the unchanged genuine DINOv3ViTModel architecture, strict local loader and
alpha-only pipeline/extractor methods. The checkpoint is
`D:\AI\Trellis2AMD\ComfyUI\models\facebook\dinov3-vitl16-pretrain-lvd1689m\model.safetensors`,
SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
Its 415 F32 persistent state entries loaded strictly once, with no missing or
unexpected entries; parameter payload is 1,212,518,400 bytes. The genuine
nonpersistent RoPE buffer also remained F32. No remote factory was used.

Runtime: existing Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333,
CUDA build metadata null, transformers 4.56.0 and safetensors 0.8.0. Device:
RX 9070 XT, gfx1201, 17,095,983,104 reported total bytes. Exact module paths,
state manifests, source identities and operation counters are in
[results.json](amd-dinov3-allocation-evidence/run/results.json).

The same [64x64 transparent RGBA fixture](amd-dinov3-allocation-evidence/input.png)
has PNG SHA-256 `9abc7b8f89e5b001e863fc3ef2fbd6813b2e83f63c0a6b0a45203c2ef1edfd8d`.
Alpha-only preprocessing produces the genuine 46x46 RGB crop, then the
512-resolution conditioning path. Exactly one conditioning/extractor/features
call and each of 24 transformer layers completed, with one patch convolution
and 24 SDPA operations. No warm-up, other checkpoint or model pathway ran.

Output: **[1,1029,1024], float32, CUDA:0; 1,053,696 finite values**. This includes
1024 patch, one CLS and four register tokens with 1024 conditioning channels.
Inference mode was enabled; gradients were disabled. CPU loading took 2.407s,
placement/synchronization 0.328s and conditioning including final GPU
synchronization **29.000s**. Overall supervised worker time was 44.703s.
Host-return markers are distinct from GPU completion markers.

The published [conditioning.safetensors](amd-dinov3-allocation-evidence/conditioning.safetensors)
contains only `cond`, F32 [1,1029,1024]: 4,214,784 tensor bytes, 1,008 header
bytes, 4,215,800 total bytes. File SHA-256:
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`.
Tensor payload SHA-256:
`d94e87d27429936956616a7a1c09a1fd5e0e27337b5716486c271881497dc7ca`.
Metadata preserves checkpoint/config hashes, source image pixel/PNG hashes,
extractor/pipeline source hashes, runtime, harness commit, alpha-only mode,
resolution and width. Exclusive write, fsync, bounded header/payload checks and
genuine CPU safetensors readback passed in the worker and independently after
exit. The independent CPU analysis imports no Torch/AI framework.

Recomputed min/max: -29.3398247 / 24.2708645; float64 mean -1.72562e-9 and
standard deviation 0.999999986. The payload hash differs from the previous
worker-reported output hash; bitwise repeatability or numerical parity is not
claimed. The full current tensor is now preserved for future approved tests.

## What remains allocated

| Memory point | Allocated bytes | Reserved bytes | Available bytes |
|---|---:|---:|---:|
| Before placement | 0 | 0 | 16,937,582,592 |
| After placement | 1,213,567,488 | 1,226,833,920 | 15,699,476,480 |
| After conditioning | 1,297,482,240 | 1,371,537,408 | 15,408,168,960 |
| Release | **79,691,776** | **79,691,776** | 16,695,820,288 |

Peak allocated/reserved: 1,343,300,096 / 1,371,537,408 bytes. Existing cleanup
collected the model/output weak references and cleared the genuine RoPE cache
1→0. No obvious retained model/output closure, hook or cache reference was
identified in the reviewed code. The weak-only registry observed 2,288 GPU
tensor wrappers; none was alive at snapshot time. This establishes collection
of observed wrappers, not every possible native or unobserved owner.

The [105,837-byte allocator snapshot](amd-dinov3-allocation-evidence/allocator-release.json),
SHA-256 `c592a8064afcd28439565d87cc2a7fae73ae252e575cd89e8645cab819a431e6`,
reconciles exactly with allocated/reserved counters: **one segment, one
active_allocated block, requested/allocated size 79,691,776**. Awaiting-free and
inactive bytes are zero. This is an active allocation, not merely unused
allocator reservation or delayed freeing.

Its saved Python origin is `torch.nn.Linear.forward` → DINO attention
`query_states = self.q_proj(hidden_states)`, inside conditioning. No observed
Python owner matches its storage address. Read-only native settings report
BLAS workspace 33,554,432 bytes and **BLASLt workspace 79,691,776 bytes**, exactly
matching the sole surviving block. The installed source documents persistent
BLAS workspaces and exposes separate handle/stream workspace maps and a cache
clearing binding; [hash-pinned source excerpts](amd-dinov3-allocation-evidence/allocation-source-context.json)
preserve those findings.

**Inference:** a persistent ROCm BLASLt workspace is the best-supported cause.
The snapshot does not expose a native owner identifier, and this Windows build
warned that C++ stack capture is unsupported. Python origins, exact byte match
and absent observed wrappers do not prove the current native owner. Neither a
generic tensor leak nor confirmed BLASLt ownership is claimed. History was
bounded to 512 events, with current allocation origins retained independently;
snapshot/weak-registry bounds were respected, totals verified and history
disabled. No workspace size/backend/cleanup policy was changed.

The unchanged zero allocated/reserved validator raised `ValueError: Decoder
release incomplete`; its legacy label refers to DINO cleanup here. The last
stage label is `allocator-snapshot`, but snapshot capture succeeded: the precise
failure is the subsequent release validator. Worker exit code **1** is a
handled cleanup-acceptance failure, not a forward guard failure, native crash,
timeout or driver reset. No post-exit GPU memory query was made.

## Verification and preservation

[76 focused CPU regressions](amd-dinov3-allocation-evidence/pre-attempt-review.md)
passed: 65 existing plus 11 new, including actual logging/audit functions,
installed allocator-wrapper replay, weak non-retention, bounds, malformed state
rejection and genuine safetensors integrity/corruption/path/dtype handling.
Independent review covered the complete candidate before publication.
[Saved-evidence verification](amd-dinov3-allocation-evidence/saved-evidence-analysis.json)
reconciles the real output, snapshot, counters, monitoring and preservation.

All seven periodic Windows checks were healthy, maximum gap 5.047s; before,
CPU-gate and after-exit checks also passed. Job attachment preceded preparation;
owned-tree exit was verified with zero active processes, no forced cleanup and
no observer/journal errors. Limits remained CPU120/GPU180/overall360 seconds.

Both installed environment inventories and compressed metadata are unchanged;
local model/HF/Torch cache trees are unchanged. Existing AMD source edits and
backup, all prior local/remote branch heads, main and 2,141 historical evidence
files are preserved. Main remains `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.
Fresh scratch retains 526 files / 674,667,788 bytes with an
[inert hash manifest](amd-dinov3-allocation-evidence/scratch-side-effects.json).
Only bounded features, input and snapshot copies are published; historical and
fresh native caches remain untouched locally. No production worker behavior,
installation, weights, drivers or AMD source files were changed.

## Smallest concrete correction to test next

Review the exact ROCm build's native `_cuda_clearCublasWorkspaces` binding for
BLASLt-map coverage, then, under separate approval, add one scoped call after
the existing synchronized model/output/cache release and before `empty_cache`.
Test the same one-pass fixture once with the same observer/Job/deadlines and
reconciled snapshot. Acceptance remains zero active/allocated/reserved bytes,
unchanged shape/dtype/finite-output contract and verified exit. This is a falsifiable
workspace-cleanup correction; if the block survives, that result rejects the
proposed correction rather than relaxing acceptance or retrying. No such call
or further inference was executed here. Stop before sparse-flow progression.
