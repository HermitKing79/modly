# AMD image-conditioned sparse-flow forward — PARTIAL

2026-10-13. Branch `codex/amd-sparse-flow-forward`, based on
`065bc79b8c2db97eea3892c34481dbe21e6a118d`. The single approved attempt
**stopped before checkpoint loading and GPU authorization**, because available
host RAM did not meet the strict-loading safety margin. There was **no sparse-flow
forward, GPU placement or retry**. This is a resource guard failure; no native
computation failure or checkpoint incompatibility was demonstrated.

## Verified contract and candidate

Pinned AMD source `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`; genuine
`SparseStructureFlowModel`, configuration resolution16, width1536, 30blocks,
12heads, MLP8192, shared modulation, cross image attention and RoPE. The “64”
checkpoint filename does not override the actual resolution16 configuration.
[Source/configuration pins](amd-sparse-flow-evidence/source-contract.json) and
[owned CPU/meta proof](amd-sparse-flow-evidence/cpu-metadata-final-candidate/results.json)
match all640 parameter keys/shapes and show no real weights loading or GPU
initialization.

Local checkpoint:
`D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS.2-4B\ckpts\ss_flow_img_dit_1_3B_64_bf16.safetensors`.
SHA256 `ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`;
1,292,179,976 parameters /2,584,359,952 stored payload bytes, all640 entries BF16.
The full file was hashed only after weights-free architecture matching.

The source runtime requires420 BF16 and220 FP32 parameters. FP32 exceptions
are the ten outer timestep/modulation/input/output parameters plus, per block,
modulation, norm2 weight/bias and self/cross q/k RMS gamma. This proof-only
adapter uses the **unchanged StrictLocalComponent** to validate all640 stored
keys/shapes/dtypes and exact payload bytes, then widens only those220 named
parameters. Each widening must be finite and BF16-roundtrip hash-identical.
Runtime parameter budget2,619,359,264 bytes. The genuine checkpoint's tensor
values have **not** passed this loading contract yet: the RAM guard stopped it.
The interface was exercised only with tiny synthetic local checkpoints.

Generated `rope_phases`, absent from disk, is explicitly nonpersistent only in
this proof's factory. Its genuine CPU construction equals an independent
source-derived reference: complex64 `[4096,64]`, 2,097,152 bytes, SHA256
`31612f9475a55a4ba9d03f9450b99841b92e4ef4a7fd0985b9e93b3243c090f6`.
No unexplained missing learned parameters or zero/meta buffers are accepted.
Global strict policy and installed source remain unchanged.

Saved DINO features remain SHA256
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`.
The original provenance, F32 `[1,1029,1024]` payload hash and all finite values
were verified without DINO execution or regeneration. Planned inputs are
F32 x`[1,8,16,16,16]` with `(arange32768%257-128)/128`, timestepF32`[500]`,
and those saved features. Source explicitly casts hidden state/modulation/cond
to BF16, uses FP32 norm/RMS/RoPE intermediates and returns F32`[1,8,16,16,16]`.
Output and numerical correctness remain unproved.

[Reviewed preparation](amd-sparse-flow-evidence/preparation.md) sets genuine
`sdpa`, disabled autocast/gradients and inference mode. CPU-to-meta inspection
proved `Module.to` and frequency-copy schemas; only unchanged dtype/strided
layout defaults are allowed for placement. Internal casts are F32↔BF16, with
complex64 preserved. Upstream downloading/strict=False factory is unused.
The inert pinned pipeline package initializer is permitted; pipeline submodules
and unrelated native components are denied.

## Single attempt and stop reason

Exact execution harness was published and independently fetched from GitHub
**before** running:
[`90c656e5d85da78194c3ebe5262aa5d87ecd35ce`](https://github.com/HermitKing79/modly/commit/90c656e5d85da78194c3ebe5262aa5d87ecd35ce).
All35 declared execution dependencies matched that commit. The exclusive
[attempt marker](amd-sparse-flow-evidence/attempt-started.json) remains consumed.

[Raw result](amd-sparse-flow-evidence/run/results.json) and
[durable trace](amd-sparse-flow-evidence/run/trace.jsonl) record:

| Observation | Actual result |
|---|---|
| Stop operation | Host available-RAM check before `strict-CPU-load` |
| Error | `RuntimeError: Insufficient host strict-load margin` |
| Required available RAM | **11,411,181,632 bytes**, approximately10.63GiB |
| Real checkpoint loads / forward calls | **0 / 0** |
| GPU authorization / transfers / operations | **false / all0 / none** |
| Last durable marker | Runtime identity, seq2; stage`runtime-import` |
| Worker duration / exit | **8.515s / controlled exit1** |
| Timeout / crash / forced cleanup | None |
| Owned Job and output/journal shutdown | Clean, verified |
| Windows before/after observation | Healthy; RX9070XT statusOK, no driver events |
| Output, VRAM and workspace clearing | Not reached; no measurements or clear call |

The margin is the published guard `4 × checkpoint payload +1GiB`; it was not
lowered after rejection. Exact available bytes at the failed check were not
logged. A [later passive sample](amd-sparse-flow-evidence/host-memory-after.json)
at07:17:44UTC showed5,957,574,656 available out of34,270,023,680 total bytes.
That contextual sample is **not** substituted for a gate-time measurement.

CPU/GPU/overall bounds remain180/240/480 seconds. GPU work would require13GiB
available VRAM, a12GiB per-process PyTorch allocator cap, periodic5s Windows
checks, owned-tree CPU activity collection, one full model invocation and
synchronized single-call BLAS teardown followed by zero allocated/reserved
bytes. GPU monitors and these measurements were not started because the CPU
gate never authorized GPU work. No native cache compilation/reuse occurred.
No GPU initialization is supported by the binding/runtime records and the
subsequent forbidden lazy-initialization/CPU dispatch path; no post-exit GPU
query was run merely to demonstrate this absence.

## Tests, review and preservation

[Focused verification](amd-sparse-flow-evidence/cpu-focused-candidate.json)
contains **73 passing CPU regressions**:18new and55existing. The preparation
note's69 count was an arithmetic error;73 is the sum of the saved test results.
Tests replay actual production dispatch, module calls, supervisor admission/
rejection, diagnostics, monitoring and teardown with synthetic data. AI imports
were blocked in these host tests. The separate AMD CPU/meta probe passed in
7.578s with exit0 and an empty owned Job; its only loaded tensors were the
approved saved features and tiny synthetic checkpoint. Earlier CPU harness
failures/copy-schema observations are preserved in their own directories.

[Independent candidate review](amd-sparse-flow-evidence/independent-contract-review.json)
passed and separately ran all18new tests with AI imports blocked.
[Passive result analysis](amd-sparse-flow-evidence/saved-result-analysis.json)
verifies the exact stop path and zero invocation counts. No acceptance criteria
were relaxed, no second execution occurred and the execution harness is unchanged.

Before/after installed dependency inventories are equal. Both installed
environment metadata trees, native/source pins, model/HF/Torch caches, saved
features and checkpoint hashes are preserved. All3,302 historical compatibility
files, AMD `nodes.py` modifications and its backup are unchanged. Main and all
prior local/remote branches are preserved; main remains
`f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.
[Every raw metadata difference](amd-sparse-flow-evidence/metadata-reconciliation.json)
is directory st_size only (2,103 entries), with identical paths/mtimes/modes and
all file entries. Existing normalization excludes directory allocation size;
its policy was not changed. The cause of that raw representation difference
is not established. No installation, download, driver change, source edit,
production worker change, additional real model loading or geometry work occurred.

## Cheapest falsifiable next step

Under separate approval, move the **unchanged** RAM requirement into a CPU-only
admission check before consuming an attempt marker, logging available/required
bytes. Ensure at least11,411,181,632 available bytes after runtime imports,
then authorize one fresh run of this same sparse-flow contract. Closing unused
memory-consuming applications may be sufficient; no package/environment repair
or reduced strictness is supported by this result. Do not weaken the margin
without a separately reviewed peak-memory argument.

No native computation or sampling conclusion is available. A successful forward
would justify reviewing a separately bounded source-faithful sparse-structure
sampling proof using these same features; this task stopped before that point.
