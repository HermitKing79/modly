# AMD image-conditioned sparse-flow resume — PARTIAL

2026-10-13. Branch `codex/amd-sparse-flow-forward-resume`, based on
`53c971134763ba4583803262439b5e0592f7e6d1`. **One genuine sparse-flow forward
completed successfully on RX9070XT**, with finite output and complete allocator
cleanup. Overall classification remains **PARTIAL** because the published
harness did not validate/count the actual timestep-frequency transfer overload.
No second execution occurred. Raw results, safety criteria and harness are unchanged.

## Admission, identity and publication

[Read-only initial admission](amd-sparse-flow-resume-evidence/first-admission.json)
recorded **19,599,089,664 available host bytes** against the unchanged
**11,411,181,632-byte** requirement. Torch was not imported. Later passive
preflight also passed, as did a fresh
[pre-execution admission](amd-sparse-flow-resume-evidence/pre-execution-admission.json).
The original worker independently rechecked its RAM margin after runtime import.
No Hyper-V/VM operation was performed; the user's shutdown resolved the earlier
resource restriction.

The exact reviewed entry point was committed, pushed and fetched from GitHub
before execution:
[`9505b9f95023cec8edf85556ec2a61c7d79f4db0`](https://github.com/HermitKing79/modly/commit/9505b9f95023cec8edf85556ec2a61c7d79f4db0).
All 38 execution dependencies matched publication. Only the new evidence/scratch
location and bootstrap changed; original forward, loader, contract, supervisor,
source pins and CPU/meta proof are byte-identical to their previous review.
The previous consumed attempt and its evidence remain untouched.

Runtime: Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333,
CUDA build metadata `None`; GPU AMD Radeon RX 9070 XT/gfx1201,
17,095,983,104 total device bytes. Torch path is the existing AMD venv.
The unchanged source/config and checkpoint identities are described in the
[previous report](amd-sparse-flow-forward.md). Sparse-flow checkpoint SHA256:
`ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`.
Saved DINO feature SHA256:
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`.
No DINO model was instantiated or its features regenerated.

## What actually completed

[Raw result](amd-sparse-flow-resume-evidence/run/results.json),
[durable trace](amd-sparse-flow-resume-evidence/run/trace.jsonl) and
[passive result verification](amd-sparse-flow-resume-evidence/saved-result-analysis.json)
record one CPU checkpoint load with all 640 keys/shapes/storage dtypes and raw
payload bytes validated. All 640 parameters are finite; the exact 220 permitted
BF16→FP32 widenings passed reversible payload-hash checks. Genuine procedural
RoPE was validated, then placed on GPU with the unchanged mixed runtime schema.

Inputs: deterministic F32 `[1,8,16,16,16]`, timestep F32 `[500]`, and verified saved
F32 DINO `[1,1029,1024]`. Genuine source explicitly casts the transformer inputs
to BF16 and restores output F32. Gradients/autocast were off and inference mode on.
Exactly one root forward, each of 30 blocks, 30 self-attention and 30 cross-attention
calls executed; 60 SDPA calls were observed. No warm-up or sampler loop.

Output: **F32 `[1,8,16,16,16]`, CUDA:0 (ROCm), all 32,768 values finite**.
Min/max −3.03426981/4.31456423, mean 0.24628739, stddev 1.04816949.
Worker-reported output SHA256:
`6f9431c4bad8c90379d856555dcaa921115e2af7aa9a31c8f4f537adc47de56a`.
No tensor artifact, geometry or numerical-reference parity claim is made.

CPU strict load 10.781s; placement/synchronization 0.563s; **forward including
synchronization 4.250s**; supervised worker 28.844s. Trace records root host
return followed by explicit GPU synchronization; host return alone was not
treated as GPU completion. Last completed operation was successful cleanup,
followed by stage `complete`. Worker exit 0; owned Job empty, no forced cleanup,
timeout, native error or driver reset.

| Memory point | Allocated bytes | Reserved bytes | Available bytes |
|---|---:|---:|---:|
| Before placement | 0 | 0 | 16,937,582,592 |
| Before forward | 2,657,260,544 | 2,661,285,888 | 14,267,121,664 |
| After synchronized forward | 2,737,083,392 | 3,028,287,488 | 13,793,296,384 |
| After teardown | **0** | **0** | **16,815,292,416** |

Peak allocated/reserved: **2,911,063,040 / 3,028,287,488 bytes**. Model and all
owned tensor weak references were collected. The approved workspace binding
ran once after synchronization, followed by allocator-cache release and another
synchronization. Zero PyTorch bytes does not mean zero global driver usage;
280,690,688 global used bytes remained in the release sample.

All 180/240/480s deadlines, 12GiB allocator cap and 13GiB available-VRAM gate were
retained. Two periodic Windows samples were healthy, maximum gap 5.032s;
before/CPU-gate/after-exit observations also passed. All monitors, readers and
journals stopped cleanly. Four owned-tree CPU samples saw only Python processes;
no compiler child was observed. This does not exclude native work performed
inside the worker. Fresh private scratch contains 8 files/60,017,416 bytes with an
[inert manifest](amd-sparse-flow-resume-evidence/scratch-side-effects.json).
Historical caches were not reused, modified or executed.

## Why overall PARTIAL remains

The sole failed classifier predicate is `transfers.frequency == 1`; the record
contains 0. The pinned timestep source creates CPU float32 frequencies and calls
`.to(device=t.device)`. The observed histogram contains **one
`aten.to.dtype_layout`**, while the special source-identity/shape/value/timestep/
once allowance only handles `aten._to_copy.default`.

The observed overload followed the general authorised-forward ATen path. Its
association with the source frequency copy is strongly supported by source and
operation counts, but that runtime source/value check was **not recorded or
enforced by the special branch**. This is an incomplete transfer guard/accounting
proof, not a demonstrated native computation failure. We have not relabelled raw
PARTIAL as PASS or accepted a synthetic counter substitution.
[Independent post-attempt review](amd-sparse-flow-resume-evidence/post-attempt-review.json)
confirms successful computation/release and this exact unresolved predicate.

## Verification, preservation and next step

22 focused CPU regressions passed with AI imports blocked; independent review
separately passed the 4 new admission/isolation tests and checked the exact
candidate. No repeated native compatibility or DINO/decoder proof was run.
Installed dependency inventories and raw metadata structures are unchanged,
with no meaningful or directory-size differences. All 3,375 historical evidence
files, original AMD source edits/backup, model/feature/native hashes, main and
prior local/remote branches are preserved. Main remains
`f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.
No installation, download, source/environment repair, other real model, mesh,
CuMesh, texture or GLB operation occurred.

The smallest correction is **CPU-only**: validate/count the actual
`aten.to.dtype_layout` timestep-frequency copy with the same narrow CPU F32 `[128]`
source, exact values, CUDA:0 destination, lifecycle and single-use policy.
Test the genuine production guard under inference mode, including negative
arguments, then independently review it. Do not merely set the counter to 1 or
lower the acceptance rule. That correction requires its own authorised work;
this task stops after publication. Model computation is now demonstrated, but
no sampling or additional GPU execution is automatically approved.
