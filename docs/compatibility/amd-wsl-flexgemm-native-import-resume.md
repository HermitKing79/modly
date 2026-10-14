# Fresh WSL FlexGEMM native import â€” PASS

Base: `f921f3d0cbc43ad2a4c6d150ec2186fb6c6a3893`. Branch: `codex/amd-wsl-flexgemm-native-import-resume`. The exact candidate was GitHub-verified at `bfbd6007b2b99a4760698266f9ce1a1b0a45cc8c` before execution; SHA-256 `4dacdfdd348438078561fd92999d1db58a8ca6495d8f115c8812ee1ce63eb15a`.

**One genuine full-package/native import succeeded. The loaded Linux HIP extension matches the expected path and SHA-256, all 18 expected native bindings resolved as callable, GPU synchronization completed, monitoring remained healthy, and owned resources/processes released cleanly. No binding was invoked and no computation or model loading followed.**

## Actual binary and runtime

Loaded package: `/home/modly/native-decoder-preparation-6dd94a85/site-packages/flex_gemm/__init__.py`.

Loaded extension: `/home/modly/native-decoder-preparation-6dd94a85/site-packages/flex_gemm/kernels/cuda.cpython-312-x86_64-linux-gnu.so`.

Native SHA-256: `91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01`. Original wheel SHA-256: `cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285`. The upstream logical module name is `cuda`; the genuine compiled implementation is HIP, with no CUDA binary substitution.

| Measurement | Observed result |
|---|---|
| Ubuntu / Python | Modly-AMD-Ubuntu-24.04 / 3.12.3 |
| Torch runtime | `2.9.1+rocm7.2.1.gitff65f5bc` |
| Torch distribution | `2.9.1+rocm7.2.1.lw.gitff65f5bc` |
| ROCm / ROCDXG | 7.2.1 / 1.2.0 package identities preserved |
| HIP / Triton | `7.2.53211-e1a6bc5663` / `3.5.1+rocm7.2.1.gita272dfa8` |
| GPU | AMD Radeon RX 9070 XT, `gfx1201`, one device |
| Device free VRAM before / after import / after release | 14,003,527,680 bytes at all three observations, above 13 GiB |
| Torch allocator ceiling | 12,884,901,888 bytes (12 GiB) |
| Torch allocated/reserved before / import / peak / release | Zero bytes at every observation |
| PyTorch GPU initialization flag | False before admission; true during metadata admission and after import |
| Import host return / synchronized completion | 0.235 / 0.267 seconds |
| Authorized guest stage / supervisor / outer command | 6.281 / 20.484 / 20.831 seconds |
| Native bindings | All 18 callable; zero invocations |
| Policy / Triton helper | Zero rejected/tensor/native calls; driver/helper remained inactive |
| Guest / outer exit | 0 / 0 |
| Windows monitoring | Five healthy observations |
| Owned Job | Naturally empty, no forced cleanup; configured five-second grace observed empty at 0 s |
| Named distro stop | Only Modly-AMD-Ubuntu-24.04, exit 0 |

[Guest results](amd-wsl-flexgemm-native-import-resume-evidence/run/guest/results.json) contain every binding name, exact paths, runtime identity and memory observations. [Durable journal](amd-wsl-flexgemm-native-import-resume-evidence/run/guest/phases.jsonl) distinguishes import entry, host return, synchronized completion and release. [Independent outcome review](amd-wsl-flexgemm-native-import-resume-evidence/independent-outcome-review.json) and [verified summary](amd-wsl-flexgemm-native-import-resume-evidence/verified-result-summary.json) confirm the native PASS.

The unchanged GPU supervisor still reports raw UNKNOWN/exit 1 with its exact absent-vector-result diagnostic. That record is preserved. The narrow adapter separately verifies import and all containment evidence; it does not relabel the raw GPU result or run a vector calculation.

## Scope and preservation

The candidate differs from the old import worker only in the expected policy hash and fresh paths. The accepted policy remains `8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd`. Both saved guard acceptances (28 DeviceContext / 19 cleanup checks) were preserved and not rerun. The genuine `import flex_gemm` and `from flex_gemm.kernels import cuda` executed once in one process; the latter retrieved the module loaded by the former. No phase transition into CPU loading or decoder construction exists.

Binary GO, Job ownership, periodic Windows observations, guest timeout and named-distro termination were unchanged. Limits remained 60-second import stage, 90-second guest timeout with five-second kill grace, 120-second host deadline, 13 GiB free-VRAM admission and 12 GiB Torch cap. Fresh scratch/marker/output were exclusive; no retry, build, installation, download, benchmark, autotuning, kernel, convolution, neural forward or geometry operation occurred.

[Windows after preservation](amd-wsl-flexgemm-native-import-resume-evidence/windows-preservation-after.json) matched both environments, 12 selected binaries, AMD edits/backups, turret artifacts and decoder metadata/header. No checkpoint payload was loaded or rehashed. Passive Linux [before](amd-wsl-flexgemm-native-import-resume-evidence/linux-preservation-before.json)/[after](amd-wsl-flexgemm-native-import-resume-evidence/linux-preservation-after.json) matched baseline package/dpkg/source identities, approved library metadata and all 922 staged payloads. These passive reads performed no extra native import or GPU work. Modly Ubuntu finishes stopped; Docker/default registration is preserved. All 173 historical evidence files, five control sources and four preexisting bytecode files hash-match.

PASS proves genuine binary loadability and binding availability in this runtime. Native numerical computation, decoder compatibility and dynamically expanding workspace admission remain unproved. Opaque ELF/HIP constructor internals and direct native allocation peaks are not fully instrumented; the Torch allocator cap applies to Torch-owned allocations.

**Smallest next step:** separately approve bounded strict CPU loading of the single genuine `shape_dec_next_dc_f16c32_fp16` checkpoint using the existing `ShapeDecoderComponent.load_cpu()` and `StrictLocalComponent`. Construct the genuine parent decoder without O-Voxel extraction; verify all 292 entries and original F16 values, restoring only the already-audited 76 F32 parameters with exact F16 round trips while the other 216 remain F16. Measure CPU RAM/time and clean release. Reuse the full saved 3,743-row turret unchanged; no GPU placement or forward is implied. Do not automatically begin loading or decoding. Stop here.
