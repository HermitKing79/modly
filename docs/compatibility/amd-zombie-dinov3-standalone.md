# Zombie DINOv3 standalone proof

**PASS — one genuine Windows-native conditioning pass.** Based on
`b8afdd58c3d01af264b92af5f4406fc8096dfb6d`; execution harness
[`93269412602a338acd9fbde41798901ddb5f707d`](https://github.com/HermitKing79/modly/commit/93269412602a338acd9fbde41798901ddb5f707d)
was published and GitHub-verified before the exclusive attempt.

| Boundary | Verdict |
|---|---|
| Exact validated zombie input | PASS |
| CPU admission and focused contracts | PASS |
| Genuine DINO neural forward | PASS |
| Fresh artifact and independent post-exit numerical readback | PASS |
| Windows Job settlement and GPU teardown | PASS |
| Production job integration | NOT QUALIFIED |
| Full image-to-3D | NOT RUN |

WSL, occupancy, shape, texture, GLB and product Generate were not executed or
enabled. This does not change the preceding production-integration BLOCKED result.

## Exact input and implementation

The preserved preview snapshot is the original 642,414-byte, 1216×1294 RGBA PNG:
SHA-256 `db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e`;
decoded-pixel SHA-256
`e958193cad52cb3b15efd69410c3727853df61fd35310468d1b4540284ea2fb5`.
Archived eligibility receipts were hash-authenticated. Original alpha-only
preprocessing produced a 980×980 RGB crop, pixel SHA-256
`655ef5f8144975213ba5ce41e684473b9c0690f4b661561b74282c7ba949f92e`.
The worker copied the exact original PNG bytes exclusively into fresh scratch;
there was no replacement image or re-encoding. The running preview was untouched.

The [experiment adapter](../../scripts/compatibility/amd_zombie_dinov3.py) reuses
the unmodified standalone DINO worker, teardown profile and original owned Job
supervisor. Job attachment precedes preparation. Only the real strict-load
CPU-ready event and healthy observers permit the original binary GPU authorization.
No production runner, completion authority or three-stage occupancy entrypoint is
used. Scratch is under the proof checkout to satisfy the unchanged offline guard;
the exact private leaf is locally excluded from Git.

Installed Python 3.12.10, Torch 2.13.0+rocm10.0.0, HIP 7.15.26333 and
RX 9070 XT/gfx1201 were verified. All 415 original F32 checkpoint entries loaded
strictly once, with no missing/unexpected keys. Full checkpoint SHA-256 remains
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`;
config SHA-256 `135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5`.
The independently checked safetensors header domain is the eight-byte length
prefix plus padded JSON header, SHA-256
`3495b172735ff88c94b28a3257614f51764abfd7879b1498ec71abf855f259d3`.

## Actual execution and persisted output

One conditioning/extractor/features call, all 24 transformer layers, one patch
convolution and 24 SDPA operations completed. Original F32 normalization, one
input/output transfer, evaluation/inference mode and disabled gradients were
retained. The fresh output has 1,053,696 finite F32 values, shape **[1,1029,1024]**.

Private Windows artifact:

```text
C:\Users\wattz\Documents\ChatGPT\modly\proof\artifacts\amd-zombie-dinov3-b8afdd5\dino\run\scratch\conditioning.safetensors
```

File: **4,215,800 bytes**, SHA-256
`e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60`.
Tensor payload SHA-256:
`da13f1a6415db96cba44aaa807c94265995afbb13ef7e69c66911ede2bc6ee0e`.
An independent post-exit process reopened the actual container and tensor,
verified all provenance against trusted input/model/source/commit pins, and
recomputed finite-value, hash and numerical statistics. It imported no neural
framework. Private raw images, features, traces and process identities remain local.

| Actual measurement | Result |
|---|---:|
| Strict CPU checkpoint load | 2.781 s |
| Placement and synchronization | 0.422 s |
| Conditioning including synchronization | **36.407 s** |
| Supervised worker | **80.297 s** |
| Peak Torch allocated / reserved | **1,343,300,096 / 1,371,537,408 bytes** |
| Peak worker private commit / working set | 4,793,794,560 / 3,002,593,280 bytes |
| Measured Torch allocation / reservation after teardown | **0 / 0 bytes** |

Initial device-free memory was 16,937,582,592 bytes. CPU preflight measured
17,536,045,056 available host bytes against the original 5,923,815,424-byte
admission requirement; C: had 135,439,704,064 free bytes. The original initial
device requirement remains 3,360,002,048 bytes. CPU120/GPU180/overall360 deadlines
and original baseline/monitoring checks were retained. The legacy DINO host has
no process/aggregate Job byte cap or allocator-fraction ceiling; later decoder
limits were neither substituted nor claimed. No continuous host-RAM minimum or
aggregate Job-memory peak was measured.

Nine periodic Windows observations were healthy; maximum saved gap 5.375 s,
maximum query 1.500 s. Before/gate/after-exit observations also passed. The approved
BLAS workspace clear ran once; model/output collection, RoPE cache 1→0 and allocator
history shutdown passed. Owned Job active processes were zero before and after
cleanup, exit code zero, with no forced cleanup or timeout. Global device usage
at the release sample was still 320,471,040 bytes: Torch 0/0 is not zero device
usage. No post-exit VRAM query is claimed. The historical Windows allocator
C++-stack-context warning remains a diagnostic limitation, not a failed forward.

## Verification and preservation

[20 focused CPU tests](amd-zombie-dinov3-evidence/focused-tests.json) passed in
4.242 s, including a genuine CPU safetensors round-trip and rejection of corrupt,
nonfinite, foreign, aliased, hardlinked or inconsistent inputs/results. The
independent reviewer passed 19 tests; the primary normal-user run covered the
real hardlink test. The initial missing-adapter RED, a test-only historical-path
assertion error, sandbox hardlink refusal and command/deadline rejection RED
captures are preserved privately. No neural retry occurred.

[Pre-execution review](amd-zombie-dinov3-evidence/pre-execution-review.json),
[independent numerical result](amd-zombie-dinov3-evidence/result.json),
[measured resources](amd-zombie-dinov3-evidence/measured-resources.json) and
[preservation](amd-zombie-dinov3-evidence/preservation-after.json) bind the evidence.
All 100 protected identities and 78 bounded content hashes, accepted master
metadata/full-hash provenance, 20 historical cache files and 65 intervening proof
files were preserved. Large protected assets retain metadata and prior full-hash
provenance; this is not a fresh bulk rehash of every installed environment.
The DINO checkpoint, seven installed source pins and 14 reviewed binding files
were freshly checked. No dependency, checkpoint, original asset or historical
policy changed. Main and all 137 pre-existing remote heads are checked separately
in the publication receipt. Only this isolated experimental branch is published.

Stop at conditioning. This artifact is genuine new zombie conditioning evidence;
it does not qualify a product job or a complete image-to-3D generation.
