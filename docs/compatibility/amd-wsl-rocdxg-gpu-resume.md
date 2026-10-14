# WSL2 ROCm GPU smoke — PASS

2026-10-14. Base: `codex/amd-wsl-single-worker-teardown`,
`3b8ee660664c66016e0129b29dd7d048c59cccad`.
Continuation: `codex/amd-wsl-rocdxg-gpu-resume`.

**One genuine RX 9070 XT GPU addition passed every numerical, monitoring,
containment and release requirement.** No retry, model loading, installation,
native extension build or TRELLIS execution occurred.

## Exact runtime and result

Official Ubuntu 24.04.5 LTS, `Modly-AMD-Ubuntu-24.04`, Python 3.12.3,
`/opt/modly-wsl/venv`. ROCDXG 1.2.0 and ROCm 7.2.1 userspace are the existing
installation. Actual Torch runtime:
`2.9.1+rocm7.2.1.gitff65f5bc`; distribution:
`2.9.1+rocm7.2.1.lw.gitff65f5bc`; full source commit
`ff65f5bc672795c5e5033900ea0a0c4f8566c8cf`; HIP:
`7.2.53211-e1a6bc5663`; CUDA version absent. GPU initialization was false before
the authorized runtime calls. One RX 9070 XT was detected as **gfx1201**.

| Measurement | Actual result |
| --- | --- |
| Operation | Exactly one deterministic 256-element F32 vector addition |
| Numerical comparison | All 256 elements exact; maximum absolute error 0; finite |
| Available VRAM at admission | 14,334,255,104 bytes (13.35 GiB), above 13 GiB |
| Allocator ceiling | 12 GiB |
| Synchronized addition time | 0.227005624 s |
| GPU stage / whole supervision | 6.067107102 s / 21.187 s |
| Peak allocated / reserved | 3,072 bytes / 2,097,152 bytes; below 64 MiB |
| Before and after release | Allocated 0; reserved 0 |
| Owned tensors | All three weak references collected |
| Linux worker exit | 0 |
| Windows Job | Active 0 before/after cleanup; tree exit verified; no forced cleanup |
| Windows observations | Five healthy; maximum start gap 5.015 s |
| Named Ubuntu termination | Success; no other distro targeted |

Independent Python arithmetic scaled by eighths verified every output element.
The journals contain one addition entry and return, with synchronization before
validation. The NumPy-bridge warning is preserved: NumPy is absent, and this
tensor-list fixture does not use it. It did not cause CPU fallback or invalidate
the result.

## Preparation, execution identity and preservation

Only the exclusive marker destination changed in the reviewed execution harness.
Binary GO, five-second monitored Job grace, shared Job accounting, 60-second GPU,
90-second guest and 120-second host deadlines were retained. No auxiliary
process-identity instrumentation entered the production run. Earlier WinError 87
and failed-attempt evidence remain unchanged.

The exact candidate was committed, pushed and GitHub-verified before execution:
`40a139eebbc5d74584786ee764415a5b8cc31b57`. Its SHA-256,
`c10dc0afc08be765bfcfb1ddba6acd1aba22ebdb214da0c82bbdb9e12ae58cc2`,
matches the exclusive marker, supervisor and genuine Linux worker. Twenty-two
focused host-only tests passed; independent review repeated them and verified
the final outcome without further GPU execution.

The CPU-only package audit checked 24,940 hashed Python files. Seven Triton
native binaries disagree with the vendor wheel's own RECORD values, but all seven
match its actual payload exactly. The retained original AMD wheel SHA-256 also
matches the earlier download manifest. This is a demonstrated wheel-metadata
discrepancy; no binary was replaced or repaired. Dpkg OS timer/configuration and
permission findings are preserved; no native runtime payload discrepancy was
found. Linux package inventories and all 26,617 venv metadata entries match
before/after execution.

Windows driver **32.0.31041.1004**, WSL 2.7.12.0 and kernel
**6.18.33.2-microsoft-standard-WSL2** remain unchanged. This local PASS tests that
combination; it is not a claim of equivalence with AMD's older kernel matrix.
Both Windows AI environments, selected native binaries, AMD source edits/backup,
saved turret latent/coordinates/conditioning/provenance and all 157 historical
WSL evidence files were independently verified. Main and prior branches are
preserved. Both distros are stopped; Docker remains default. All earlier markers
and the new consumed marker are retained.

## Smallest next milestone

Separately approve a **minimal isolated WSL neural-decoder dependency installation
and qualification** for the preserved F32 `[3743,32]` turret latent and int32
`[3743,4]` coordinates. Admit only the pinned genuine decoder code and dependencies
it actually needs, including NumPy/safetensors and a compatible Linux sparse
backend; establish ABI, strict loading, precision and pre-allocation subdivision
contracts before considering a real decode. Stop if compatible artifacts are
unavailable or substantial rebuilding is required. No broad setup script or
runtime replacement is justified by this fixture.

The [pinned Linux reference](amd-linux-route-comparison.md) uses a different
PyTorch 2.13/ROCm 7.2.4 stack, explicit GEMM and mixed decoder precision. Those
components are not installed or qualified here. This PASS establishes basic WSL
GPU execution only; genuine neural decoding, extraction, texturing and GLB export
remain separate approval boundaries. Stop after publication.

Evidence: [host](amd-wsl-rocdxg-gpu-resume-evidence/run/results.json),
[real GPU worker](amd-wsl-rocdxg-gpu-resume-evidence/run/guest/results.json),
[independent readback](amd-wsl-rocdxg-gpu-resume-evidence/independent-readable-outcome.json),
[execution identity](amd-wsl-rocdxg-gpu-resume-evidence/execution-harness.json),
[native payload integrity](amd-wsl-rocdxg-gpu-resume-evidence/triton-payload-integrity.json),
[Linux preservation](amd-wsl-rocdxg-gpu-resume-evidence/linux-preservation-after.json).
