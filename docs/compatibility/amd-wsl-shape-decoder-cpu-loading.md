# Genuine WSL shape-decoder CPU loading — UNKNOWN

Base: `687eb2b7d6088bae0690541c93b5b8591a07a5b4`. Branch: `codex/amd-wsl-shape-decoder-cpu-loading`. The reviewed harness was pushed and GitHub-verified at `04aeb78d9e1f50ad8625fd3496abdf43601daca1` before execution; SHA-256 `00a7b3f4fafea5b172a9a3fe75779f6a67f280e43e235eae06c3f9550ba44ea6`.

**The single attempt stopped on a harness schema error before importing Torch or FlexGEMM, constructing the decoder, or loading weights. No genuine model-readiness conclusion is possible. No correction or retry followed.**

## Exact failure

At [loading.py line 170](amd-wsl-shape-decoder-cpu-loading-evidence/loading.py), the worker reads `windows-preservation-before.json['decoder_checkpoint']`. The actual key is `decoder_checkpoint_metadata_and_header_only`. This produced `KeyError: 'decoder_checkpoint'`; the complete traceback is preserved in [guest results](amd-wsl-shape-decoder-cpu-loading-evidence/run/guest/results.json).

Both `native_import` and `decoder_cpu` are `NOT_ATTEMPTED`. The worker accepted binary GO, verified staged wheel payloads and saved turret arrays/source hashes, then failed at checkpoint metadata lookup. It imported staged NumPy for saved-input validation. Torch import, policy installation, HIP admission, native import, meta construction and `ShapeDecoderComponent.load_cpu()` were never reached.

The static preflight and independent pre-execution review missed this consumed-field mismatch. [Independent outcome review](amd-wsl-shape-decoder-cpu-loading-evidence/independent-outcome-review.json) explicitly acknowledges that review omission. The executed candidate and original records remain unchanged.

| Measurement | Actual result |
|---|---|
| Guest / supervisor / outer duration | 1.402 / 10.484 / 10.708 seconds |
| Guest / outer exit | 1 / 1 |
| Host minimum available RAM | 14,590,124,032 bytes |
| Windows monitoring | Three healthy observations |
| Owned Windows Job | Empty naturally after 0.016-second grace; active before/after cleanup 0/0; no forced cleanup |
| Named Ubuntu termination | Exit 0; only Modly-AMD-Ubuntu-24.04 |
| Construction/loading, CPU peak, parameter verification | Not reached; unmeasured |
| GPU identity/memory, native qualification | Not reached; unmeasured in this attempt |

[Supervisor results](amd-wsl-shape-decoder-cpu-loading-evidence/run/results.json), durable journals, exclusive marker, stdout/stderr and [outer capture](amd-wsl-shape-decoder-cpu-loading-evidence/host-capture.json) preserve the single attempt. Missing GPU measurements are not reported as zero. No forward, GPU model placement, native binding invocation or geometry extraction occurred.

## Preparation and preservation

The unchanged `ShapeDecoderComponent.load_cpu()` and `StrictLocalComponent` retain the intended contract: genuine parent on meta; strict keys/shapes/dtypes/bytes for all 292 original F16 entries; exactly 76 lossless F32 widenings; remaining 216 F16; CPU/eval/grad-disabled and zero buffers. This contract was not exercised.

[Resource preflight](amd-wsl-shape-decoder-cpu-loading-evidence/checkpoint-and-resource-preflight.json) streamed the original 948,490,494-byte checkpoint without deserialization. SHA-256 is `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`; config SHA-256 is `5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209`. A passive post-attempt full hash also matches. No weight copy is published.

RAM admission was 6,972,719,480 bytes (6.49 GiB), including conservative clone/temporary allowance and 3 GiB reserve. Native/CPU/guest/authorized-worker/host deadlines stayed 60/120/240/220/270 seconds. The 13 GiB free-VRAM gate and 12 GiB allocator ceiling were retained but not reached. No successful guard suite or earlier GPU proof was rerun.

Windows [after preservation](amd-wsl-shape-decoder-cpu-loading-evidence/windows-preservation-after.json) matches both environments, 12 selected binaries, AMD edits/backups, four turret artifacts and decoder metadata/header. Linux [after preservation](amd-wsl-shape-decoder-cpu-loading-evidence/linux-preservation-after.json) matches baseline venv/packages/dpkg/source, approved library metadata and all 922 staged payloads. Passive checks import no AI/native packages. Modly Ubuntu finishes stopped; Docker/default registration remains preserved.

[Final preservation verification](amd-wsl-shape-decoder-cpu-loading-evidence/verification-after.json) confirms 207 historical files, five control sources, both loaders and four preexisting bytecode files unchanged. All 49 prior GitHub branch tips, including main, match the fresh remote-before snapshot. The inherited refs manifest contains one older launcher-branch tip; this is documented and reconciled against the fresh pre-task GitHub snapshot and pre-task reflog. Historical evidence is preserved. Existing untracked bytecode stays outside the commit.

**Smallest next action:** separately authorize correcting the checkpoint-record key and a standard-library schema preflight that exercises the exact guest-consumed fields before reserving a new marker. Then authorize one fresh genuine strict CPU load using the unchanged loaders. This failure justifies no native compatibility repair or GPU neural execution. Stop here.
