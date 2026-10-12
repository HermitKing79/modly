# AMD BLASLt workspace teardown

2026-10-13 (Europe/London). Branch `codex/amd-blaslt-workspace-teardown`, based
on `62409edb5545857012116cb64fc2335acd938f54`. Exact reviewed execution harness
`bb9421207171ac591a66f43ef8d09f8a9bda25c4` was published and independently
verified on GitHub before the exclusive single attempt.

**PASS for this bounded cleanup proof.** One genuine DINOv3 conditioning pass
completed, the approved workspace-clear binding returned once, and the live
worker measured **zero allocated and reserved PyTorch bytes** after teardown.
Model/output collection, RoPE-cache release and normal owned-tree exit were
verified. No retry, other model or production-worker change occurred.

## Installed binding evidence and correction

[Independent static PE review](amd-blaslt-teardown-evidence/binary-review.json)
and [bounded raw disassembly](amd-blaslt-teardown-evidence/binary-review-evidence.txt)
trace the installed Python method table's METH_NOARGS callback through its
torch_hip.dll import to `clearCublasWorkspaces`. The native function visits two
distinct BLAS and BLASLt maps, locks each exclusively, erases its tree, resets
its links/size and unlocks. Erased nodes destroy their c10::DataPtr values.
The reviewed entry does not create handles, query devices or change workspace
sizes. This matches the relevant [upstream v2.13.0 implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L122);
version/git metadata alone was not used as proof of binary equivalence.

Native binary SHA-256 identities:

- `torch_hip.dll`: `28ea13f7de8a0480081d44fc27ac693419ec29a96782afccc4b212e472600119`
- `torch_python.dll`: `f710dc3a3ebb7863fe987d717242c329f66acc73d81578d545e1e99a9bf98171`

The [CPU-only exposure check](amd-blaslt-teardown-evidence/cpu-binding.json)
verified the genuine builtin/module/name and loaded DLL paths under the AMD
venv, without invoking cleanup, initializing the GPU or loading weights.
Fourteen installed source/binary files are hash-pinned. Startup requires the
correct WORKSPACE_DIR before consuming the attempt marker. Preliminary CPU
smoke errors (missing workspace variable, then correctly rejected outside-scratch
write) were fixed using explicit workspace/stdout-parent recording; all failed
captures are preserved and safeguards remain intact.

Only the new proof profile enables the private call. Teardown releases all
existing known model/output/pipeline/extractor/bound-method references, clears
the genuine RoPE cache, runs GC and synchronizes completed GPU work. It then
calls `_cuda_clearCublasWorkspaces()` **once**, followed by the established
`empty_cache()` and synchronization. The call clears both process-wide maps,
so it is confined to this exclusive owned worker with no concurrent BLAS work.
Native/synchronization failures propagate without GPU recovery or retry.
The original zero-allocation criterion and inference contract are unchanged.

## One execution

Existing Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, CUDA build
metadata null; RX 9070 XT / gfx1201 / 17,095,983,104 reported total bytes.
Only the previously audited DINOv3ViTModel checkpoint loaded, strictly once:
415 F32 persistent entries, no missing/unexpected state, 1,212,518,400 parameter
bytes. Checkpoint SHA-256 remains
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
The same transparent 64x64 RGBA fixture, 46x46 alpha-only crop and genuine
512-resolution conditioning path were used. Input PNG hash remains
`9abc7b8f89e5b001e863fc3ef2fbd6813b2e83f63c0a6b0a45203c2ef1edfd8d`.

[Results and exact counters](amd-blaslt-teardown-evidence/run/results.json)
record one extractor/features/conditioning call and each of 24 layers, one
patch convolution, 24 SDPA operations, one input/output copy and the original
normalization-copy permits. Output: **[1,1029,1024], float32, CUDA:0**, 1024
conditioning channels and 1,053,696 finite values, with inference mode enabled
and gradients disabled. Worker-reported min/max: -29.3398209 / 24.2708645;
mean -1.73050e-9, standard deviation 1.0. Current output hash is worker-reported;
no current tensor artifact or bitwise/parity claim is made.

CPU load: 2.422s; placement/synchronization: 0.297s; conditioning including GPU
synchronization: **26.594s**; supervised worker: **43.093s**. Clear trace has
exactly one entry/host-return pair, seq 77/78. Its measured duration is below
the Windows clock resolution; the accepted memory sample follows the reviewed
post-clear synchronization, rather than treating host return as GPU completion.

| Memory point | Allocated bytes | Reserved bytes | Available bytes |
|---|---:|---:|---:|
| Before placement | 0 | 0 | 16,937,582,592 |
| After placement | 1,213,567,488 | 1,226,833,920 | 15,699,476,480 |
| After conditioning | 1,297,482,240 | 1,371,537,408 | 15,408,168,960 |
| After teardown | **0** | **0** | **16,775,512,064** |

Peak allocated/reserved: 1,343,300,096 / 1,371,537,408 bytes. Model/output weak
references were collected; RoPE occupancy 1→0; all 2,288 observed tensor
wrappers were gone. History was disabled. No snapshot was captured because
allocated/reserved counters were zero, as required. Exit code **0**, owned Job
empty, no timeout, forced termination, native error or driver event.

This demonstrates that the workspace teardown correction works on this fixture
and supports the persistent BLASLt explanation for the prior exact-size block.
Clearing both maps does not separately identify which map owned that historical
block. Zero PyTorch allocator bytes is not zero total device/process memory:
sampled global usage at release remained 320,471,040 bytes. No post-exit VRAM
query or broader GPU reliability proof was performed.

## Tests, preservation and limits

All **86 focused CPU regressions** pass (65 existing control/diagnostic, 11
allocation and 10 new teardown tests). New tests replay the actual worker
teardown/logging code and cover ordering, clear/sync failures, single invocation,
unchanged inference/zero-release policy, conditional snapshot, history shutdown,
default-mode preservation and pre-attempt workspace rejection. Independent
[final candidate review](amd-blaslt-teardown-evidence/pre-attempt-review-final-independent.json)
passed 10/10 new tests. Its separate allocation replay passed 8/11; three
existing checkpoint fixtures hit sandbox temporary-directory permissions.
Their failed capture is preserved; the primary scoped run passed all 11.
No environment or unrelated test repair was made.

[Post-attempt review](amd-blaslt-teardown-evidence/post-attempt-review.json)
and [passive saved-result verification](amd-blaslt-teardown-evidence/saved-result-analysis.json)
confirm the result. All six periodic Windows samples were healthy, maximum
gap 5.031s; fresh before/CPU-gate/after-exit checks passed. Job attachment
preceded preparation. Observers/journals/readers stopped cleanly. Limits remain
CPU120/GPU180/overall360 seconds.

Installed dependency inventories and normalized file metadata are unchanged.
Raw venv gzip hashes differ: [every structural difference](amd-blaslt-teardown-evidence/metadata-reconciliation.json)
is directory st_size only (3,668 entries); paths, mtimes, modes and all file
entries are identical. The pre-existing inventory policy normalizes directory
allocation size; no acceptance rule was changed. The cause of this raw metadata
representation difference is not established. Model/HF/Torch raw metadata is
byte-identical. Native/source/checkpoint hashes and existing AMD edits/backup
remain unchanged. Main, prior local/remote branches and all 2,718 historical
evidence files are preserved. Main remains
`f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.

The verified [saved DINO features](amd-dinov3-allocation-evidence/conditioning.safetensors)
are untouched, file SHA-256
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`.
No new feature checkpoint was generated or republished. Fresh private scratch
retains 524 files / 670,346,148 bytes with an [inert manifest](amd-blaslt-teardown-evidence/scratch-side-effects.json);
historical caches were not modified or executed. No installations, downloads,
other model execution, CuMesh, extraction, texturing or GLB output occurred.

## Shortest next milestone

Under separate approval, strictly load only the audited
`ss_flow_img_dit_1_3B_64_bf16` sparse-structure flow checkpoint and execute one
architecture-valid image-conditioned flow forward using the existing saved DINO
features. First establish its latent/timestep/conditioning shapes and explicit
F32-feature/BF16-model precision contract; do not silently cast or substitute
weights. Keep bounded monitoring, one invocation and zero-release acceptance.
This advances toward the sparse-structure sampler without rerunning DINO. A
sampler loop and subsequent decoder combination require their own approval;
full geometry, texturing and the previous CuMesh 719 failure remain unproved.
Stop after this published teardown proof.
