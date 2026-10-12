# AMD sparse decoder: first real checkpoint-loading proof

**PASS for this bounded milestone — 12 October 2026.** One genuine local sparse
decoder loaded completely on CPU, transferred to the RX 9070 XT through ROCm,
and released. The owned worker exited normally. No model forward pass,
inference, geometry decoding, texturing or additional checkpoint load occurred.
This does **not** establish full pipeline compatibility or resolve CuMesh error 719.

Development branch `codex/amd-sparse-decoder-loading-proof` starts at
`4ce67676d9a0cd4e9dc12f63cbc4763e639d4101` on the preserved
`codex/amd-alpha-headless-import` branch. The [alpha/headless report](amd-alpha-headless-import.md)
and its earlier compatibility evidence were reviewed first. Only standalone
proof tools, focused tests and documentation are added; application and
production mock-only behavior are unchanged.

## Gate 1: exact offline identity and complete CPU load

The only selected prefix was
`D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16`.
Its `.json` and `.safetensors` match the [previous audited identity](amd-offline-model-audit-evidence/final-assets/ss_dec_conv3d_16l8_fp16.metadata.json):
`SparseStructureDecoder`, output channels 1, latent channels 8, channels
`[512,128,32]`, two residual blocks and two middle blocks, `use_fp16=true`.
There are **74 tensors / 73,671,201 parameters**, 36 F16 tensors and 38 F32
tensors, with **147,585,412 payload bytes** in a 147,591,972-byte file.
These are actual parameter bytes, not the scaffold's metadata-only fixtures.

[Before/after checkpoint records](amd-sparse-decoder-loading-evidence/run/checkpoint-before.json)
retain configuration, header, tensor descriptors, source identity and full-file
SHA-256 `1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a`.
The full SHA is a current before/after baseline; the earlier audit verified
metadata, not an independently authenticated full payload hash.

The worker imports the actual `trellis2.models.sparse_structure_vae.SparseStructureDecoder`
from the preserved AMD checkout at `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`.
This class uses dense Conv3D modules; importing it does not construct the full
pipeline, image encoder, RMBG or CuMesh. The generic factory's download fallback
and `strict=False` loader are never called.

The real architecture is constructed on the metadata device, then the selected
safetensors file is loaded **once** on CPU and cloned into physical CPU storage.
`load_state_dict(strict=True, assign=True)` returned no missing or unexpected
keys. Every loaded key, shape, dtype and device matched; each tensor's CPU bytes
hashed identically to its corresponding checkpoint payload range. GPU context
state remained **uninitialized throughout Gate 1**.

CPU load, construction and byte validation took **3.203 s**. Working set changed
from **391.26 to 612.62 MiB**, with **753.29 MiB peak**; private commit changed
from **801.23 to 1,035.10 MiB**, with **1,176.13 MiB peak**. These are Windows
process measurements, including interpreter/library overhead.

## Gate 2: ROCm placement and release

[Complete result and process evidence](amd-sparse-decoder-loading-evidence/run/results.json)
records Python **3.12.10**, PyTorch **2.13.0+rocm10.0.0**, HIP **7.15.26333**,
CUDA **null**, safetensors **0.8.0** and NumPy **2.5.3**. The interpreter and
PyTorch resolve under `D:\AI\Trellis2AMD\venv`. The device is **AMD Radeon
RX 9070 XT / gfx1201**, reporting 17,095,983,104 total bytes. No CPU/CUDA fallback
or replacement binaries were used.

Only after strict CPU evidence and a fresh healthy Windows observer did the
parent authorize `model.to(device="cuda:0")`. All 74 tensors then validated on
that device with their original shapes/dtypes. Transfer plus synchronization
took **0.062 s**.

| Measurement (MiB) | Before placement | After placement | After release |
|---|---:|---:|---:|
| PyTorch allocated | 0 | 145.76 | 0 |
| PyTorch reserved | 0 | 166.00 | 0 |
| Available device memory | 16,152.94 | 15,980.19 | 16,140.19 |
| Sampled total device usage | 151.06 | 323.81 | 163.81 |

PyTorch peaks were **145.76 MiB allocated / 166.00 MiB reserved**. Global usage
was sampled at those three boundaries; 323.81 MiB is the largest sample,
not a continuously measured global peak. Release dropped the model reference,
ran garbage collection, synchronized and emptied the allocator cache. The weak
reference confirmed collection. Available memory remained 12.75 MiB below its
baseline while the worker's GPU context still existed; zero PyTorch allocation
does not prove recovery of every driver/context byte. The worker then exited
with **code 0**, reaping verified, no timeout or forced cleanup, in **9.813 s**.

The [before-launch observer](amd-sparse-decoder-loading-evidence/run/windows-before.json),
[Gate 1 observer](amd-sparse-decoder-loading-evidence/run/windows-gate-one.json)
and [post-exit observer](amd-sparse-decoder-loading-evidence/run/windows-after-exit.json)
all passed: explicit Windows PowerShell Diagnostics import and System-log probe,
controller status OK, driver `32.0.31041.1004`, and no matching event IDs
4101/17/18/41 during the attempt. No native error, reset, memory failure or
abnormal termination was observed. There was **one controlled attempt**, without retry.

## Guards, tests and preservation

The [harness](../../scripts/compatibility/amd_sparse_decoder_loading.py) enforces
45 seconds for CPU work, 30 seconds for placement/release and a 90-second worker
supervision ceiling; each observer has a 10-second timeout. Every exit path reaps
the owned worker, with bounded tree termination on errors and fail-closed
classification. Model invocation/forward and pickle loading are blocked; GPU
dispatch allows only copies, allocation and bookkeeping. Network, other model
files, arbitrary subprocesses and writes outside workspace scratch are blocked.
The sole permitted discovery command is the previously hash-verified read-only
`rocm-sdk path --root`, capped at five seconds. It ran **once**, taking **1.390 s**
and exiting **1** because the pinned `rocm[devel]` package is absent. Triton's
inspected discovery code catches that failure; the decoder imports, loads and
transfers still passed. The complete SDK error is retained in the result record.
No installation was attempted; availability of future Triton compilation tools
is not established by this proof.
These guards constrain the inspected source, not arbitrary hostile native code.

[Nine focused tests passed](amd-sparse-decoder-loading-evidence/tests/contracts.json):
identity, strict tensor schemas, gate authorization, copy-only GPU operation
selection, release, classification and fake-child error/cleanup failure cases.
They perform no checkpoint loading or GPU work. Read-only review identified a
parent exception-cleanup gap before the real attempt; unconditional verified
reaping and two regression tests fixed it. No application-suite rerun or unrelated
repair was needed for these standalone additions.

[Preservation evidence](amd-sparse-decoder-loading-evidence/preservation.json)
shows byte-identical inventories and raw compressed manifests before/after:
**94,687 AMD venv entries**, **61,313 installed Modly venv entries**, all
inventoried model/Hugging Face/Torch cache entries, selected package/native/source
hashes, the complete decoder hash and all 17 imported AMD source-file hashes.
The existing modified `nodes.py` and untracked `nodes.py.before-profiler-fix`
remain unchanged. No packages, weights, installed extensions or drivers were
installed, upgraded, replaced or rebuilt.

[Six cache artifacts / 148,712 bytes](amd-sparse-decoder-loading-evidence/scratch-side-effects.json)
appeared in the configured workspace AMD COMGR cache directory, with ELF/LLVM
payload prefixes. They remain local and ignored by Git. This native runtime
cache side effect means this proof makes **no claim of zero internal native
compilation or kernel activity**. No model computation or installed-library
rebuild was requested or observed.

## Smallest next milestone

Implement an explicit **offline alpha-only geometry loader** in the experimental
adapter: bind the audited local safetensors/config map, omit the background-model
factory, keep image-encoder loading deferred, require strict per-model loads and
provide bounded teardown/error reporting. Keep production routes mock-only.
Any newly loaded checkpoint needs a separately approved loading/memory proof;
this decoder's small footprint does not predict combined pipeline VRAM.
Texturing's eager DINO/BiRefNet factory, real forward operations, full generation,
texturing and the previous CuMesh 719 hole-filling crash remain unvalidated.
**Stop here: no further model loading or inference is authorized by this proof.**
