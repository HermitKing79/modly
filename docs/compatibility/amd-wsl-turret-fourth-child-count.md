# Genuine fourth-stage child count — PARTIAL

Base: `5a7bf53d0ed53cce8daf52eb2dab90dba9036fe6`. Branch: `codex/amd-wsl-turret-fourth-child-count`. Reviewed candidate: `994c4d0f1fe08f3ae4843e382233e488f80d801e`. Exact harness plus qualified live-readiness evidence were pushed and GitHub-verified before execution at **`eda00a114936c9b59f089f7928b10a3745ffa153`**.

**PARTIAL:** one genuine parent decoder forward again completed 39/40 convolutions. The authentic fourth-stage mask selected **1,842,968 children**, exceeding the unchanged one-million-row cap by **842,968**. The durable record survived `AdmissionStop`. Execution stopped before the first fourth-stage expansion; no complete neural output, mesh or texture was produced. No retry followed.

## Decisive observation

| Fourth-stage field | Recorded value |
| --- | --- |
| Original site / invocation | `blocks.3.4.updown` / 1 |
| Parent rows / selected children | 378,891 / **1,842,968** |
| Selected children per parent | Mean 4.8641113143; architectural maximum 8 |
| Original mask | `torch.bool [378891,8]` |
| Feature / skip / output widths | 512 / 128 / 64 |
| Allocated / reserved / device-free bytes at count | 1,594,233,856 / 6,129,975,296 / 8,342,953,984 |
| Zero children / over row cap | False / **True** |

The genuine C2S source hash is `f36819339e48a89ed368f45feb0987309de23a4fe1d7d43eb347ab30a0b9f09c`; the mask-producing block source is `c9ce51db70bb1374c8bdc858129f6ff1170af1447e5210d434b7ebb289dbd2ce`. Module names, original paths, channel widths and memory are preserved in the [durable journal](amd-wsl-turret-fourth-child-count-evidence/run/guest/phases.jsonl).

Only the pre-admission logging changed. It reuses the existing single mask reduction, journals through the original flush/fsync path, then calls the unchanged `c2s_plan()`. Masks, learned arithmetic, both original C2S expansions per stage, 40 convolution hooks and memory admission remain intact. Completed support is **3,743 → 17,015 → 79,169 → 378,891**; the last returned convolution is `blocks.3.4.conv1`.

## CPU-only estimate of the remaining convolution

The original uncached `blocks.3.4.conv2` uses BF16, 3×3×3, Ci=Co=64 and `explicit_gemm`. Applying its pinned source and existing admission formula to the actual child count yields:

| Visible requirement | Bytes |
| --- | ---: |
| Full im2col, allocated unconditionally | **6,369,297,408** (5.932 GiB) |
| Gathered features, conservative upper bound | 6,369,297,408 |
| Native neighbour map / hash tables | 199,040,544 / 29,487,488 |
| Long-index intermediates / boolean mask | 796,162,176 / 49,760,136 |
| Output / existing block-intermediate allowance | 235,899,904 / 2,358,999,040 |
| Visible total / with existing 25% padding | **16,407,944,104 / 20,509,930,130** |

Using the measured **pre-expansion** reserved memory plus the existing 1 GiB opaque headroom gives an allocator admission estimate of **27,713,647,250 bytes (25.810 GiB)**, above the unchanged 12 GiB ceiling. The device-free formula also fails with its existing 2 GiB reserve. [Complete offline calculation](amd-wsl-turret-fourth-child-count-evidence/fourth-convolution-offline-estimate.json).

These are conservative planning bounds, not observed allocations or an OOM. Actual child coordinates, valid-neighbour density and post-expansion retained memory do not exist in this run. Raising `MAX_ROWS` alone cannot satisfy the current unchunked admission formula. A source-faithful chunked final explicit-GEMM operation merits **separate CPU design and memory review**, including C2S/global neighbour/index allocations and numerical equivalence, before any implementation or further GPU approval. No chunking was implemented or authorised here.

## Execution, checks and preservation

The complete original F32 `[3743,32]` turret latent and I32 `[3743,4]` coordinates retained their hashes/order. Actual admission had **14,631,796,736 bytes (13.627 GiB)** free VRAM. All three TunableOp settings were `"0"` and genuine flags false. The original FlexGEMM binary qualified; all 292 checkpoint entries loaded strictly, followed by the reviewed 216 BF16 / 76 F32 runtime configuration. Strict load/widen took 10.270 s.

Exactly one parent invocation and four genuine native neighbour maps occurred. Peak Torch allocated/reserved bytes were **5,588,655,616 / 7,291,797,504**. Worker/supervisor duration: **30.982 / 45.656 s**. The last convolution returned at 7.138 s after GPU-stage entry; **no complete-forward synchronized runtime or final-output finiteness result exists**. Cleanup synchronization succeeded, workspace clearing occurred once, references were collected and allocated/reserved bytes reached **0 / 0**. Ten Windows observations were healthy. Controlled worker exit was 1; Job accounting was empty with no forced cleanup; named Ubuntu termination succeeded.

Three focused CPU count/journal tests and two offline-formula tests passed. Independent source and outcome reviews found no material blocker or altered safety gate. [Verified result](amd-wsl-turret-fourth-child-count-evidence/verified-result.json) links the raw failure to this count. This is a support/resource-admission limitation, not demonstrated native incompatibility, numerical failure, timeout or driver reset.

Before/after audits matched both Windows environments, AMD source edits/backups, four turret assets, checkpoint/config identity, baseline WSL packages/venv, 922 staged payloads, 23 packaging files, 80 native source files and protected ROCm dependencies. All **575 protected historical/bytecode files**, main and **57 previous local and remote branches** remain unchanged. Docker remains default; both distributions are stopped. No packages, binaries, drivers or production worker changed.

Stop here. The missing count is resolved; completing neural output now requires separately approved memory work. No additional forward, row-cap increase, extraction or texturing is authorised.
