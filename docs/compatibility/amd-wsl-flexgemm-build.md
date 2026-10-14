# WSL FlexGEMM build — PASS for build/static scope

Base `a233a753afbcf9789ffe13aec26d0f393b2a0e4f`; continuation branch
`codex/amd-wsl-flexgemm-build`. One toolchain installation and one native build.
**The genuine pinned Linux extension compiled successfully and passed static
qualification. Native loading, full symbol relocation and decoder execution
remain UNKNOWN.** The user's explicitly permitted static stopping point was used.
No GPU initialization/computation, checkpoint tensor loading, convolution or
decoder forward was performed. The wheel is preserved, **not installed** into
the working Python environment.

## Resource admission and actual installation

The exact LLVM archive SHA-256 matched
`1cbad039d161a03f9c4df81193fdb81d55d9a79e400ce8ce26ee55488512e0bd`.
A streamed inspection of its real data tar found **2,121,369,507 regular-file
bytes**, conservatively 2,123,792,384 filesystem bytes. No compression-ratio
estimate was used. [Raw archive/control evidence](amd-wsl-flexgemm-build-evidence/llvm-footprint.json).

Fresh simulation exactly matched **59 additions and two approved upgrades**.
All 61 payload sizes and hashes were verified before a pinned, `--no-download`,
`--no-remove` installation. Total APT downloads: **626,410,386 bytes**; pinned
source archive: **463,566 bytes**. No other upgrade, removal, repair or retry.

| Component | Installed version |
| --- | --- |
| HIP development/runtime | 7.2.53211.70201-81~24.04 |
| hipcc package | 1.1.1.70201-81~24.04 |
| ROCm LLVM | 22.0.0.26084.70201-81~24.04 |
| Device libraries | 1.0.0.70201-81~24.04 |
| GCC/G++ | 13.3.0-6ubuntu2~24.04.1 |
| Ninja | 1.11.1-2 |

The only upgrades were `libexpat1` 2.6.1-2ubuntu0.4 → 2.6.1-2ubuntu0.6 and
`libpciaccess0` 0.17-3ubuntu0.24.04.2 → 0.17-3ubuntu0.24.04.3. The complete
[before/after versions and payload manifest](amd-wsl-flexgemm-build-evidence/toolchain-result.json)
includes Python, HSA, BLAS and sparse development headers. LLVM/headers were
added; the working Torch/Triton/ROCm runtime/ROCDXG stack was not replaced.

Measured package-owned non-directory allocation is **2,447,974,400 bytes** for
all 61 packages, including replacement files; directories/dpkg metadata and
exact net-new consumption are not conflated with this figure.
[Actual allocation readback](amd-wsl-flexgemm-build-evidence/installed-footprint.json).
Admission separately reserved all archives, a full additional extracted-size
allowance for installation temporaries, 3 GiB build scratch, 512 MiB growth reserve
and 256 MiB metadata allowance. One compiler job, 8 GiB compiler-tree RSS stop,
128 MiB individual build-file limit, and a 2,400-second build deadline were used.

Final readbacks: Ubuntu **9,812,627,456 bytes (9.14 GiB)** free; D:
**76,188,581,888 bytes (70.96 GiB)** free. Both exceed the 5/50 GiB floors.
The 32 GiB virtual disk was not enlarged. Build peak/minimum measurements are
sampled at 0.5 seconds in Linux and one second on Windows, not continuous maxima.

## One pinned, offline build

Source: [JeffreyXiang/FlexGEMM@6dd94a859c26ee8246888502eada3dd8ad85532e](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e).
[Source hashes/compiler identities](amd-wsl-flexgemm-build-evidence/source-and-tools.json)
match the independently audited setup, build metadata and binding source.
The existing Python 3.12.3 / Torch 2.9.1 / ROCm 7.2.1 environment supplied the
headers/backend. Actual Torch C++11 ABI is **1**, explicitly present in compiler
arguments. Setuptools 84's existing wheel backend sufficed; no Python package
was installed or upgraded.

Linux scratch `/home/modly/flexgemm-build-6dd94a85` owns source, redirected HOME,
caches, temporary files and outputs. `BUILD_TARGET=rocm`, `GPU_ARCHS=gfx1201`,
`PYTORCH_ROCM_ARCH=gfx1201`, `MAX_JOBS=1`, and `PIP_NO_INDEX=1` were fixed.
Python networking and GPU-query/initialization entries were blocked. Native
compiler subprocesses were the inspected local build tools; this was not a
general network sandbox for those subprocesses. No manual source patch was made;
Torch's normal HIPIFY generated files inside the disposable source tree.

All seven translation units and one link completed in **212.043 seconds**,
exit **0**, with no second invocation. Sampled peak scratch **45,718,652 bytes**;
compiler-tree RSS **2,512,457,728 bytes (2.34 GiB)**; host available RAM remained
at least **13,139,378,176 bytes (12.24 GiB)**. The Linux worker and owned Windows
Job exited naturally; no forced Job cleanup. Only the named Modly distro was
terminated afterward. [Build](amd-wsl-flexgemm-build-evidence/build-result.json),
[containment](amd-wsl-flexgemm-build-evidence/build-host-result.json),
[compiler commands](amd-wsl-flexgemm-build-evidence/compiler-commands.jsonl).

## Artifact and static acceptance

Artifact: [flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl](amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl),
**3,231,402 bytes**, SHA-256
`cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285`.
Original Linux location:
`/home/modly/flexgemm-build-6dd94a85/artifacts/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl`.

CRC and every wheel RECORD hash/size passed. The 11,568,136-byte native module is
ELF64 x86-64, CPython 3.12, exports `PyInit_cuda`, contains `gfx1201` metadata,
and matches all 18 source binding names statically. Its upstream logical module
name is `cuda`; the generated code and actual linked runtime are HIP.
Torch/c10, `libamdhip64.so.7` and host libraries resolve in the approved runtime;
no missing shared libraries or NVIDIA runtime substitution was found.
[Static acceptance and library trace](amd-wsl-flexgemm-build-evidence/static-acceptance.json).

Plain loader tracing establishes library availability, not complete relocation
or loadability. Available evidence does not prove HIP registration constructors
avoid HSA/GPU initialization during `dlopen`. Therefore the extension and full
FlexGEMM package were **not imported**, and no native binding was called.
This artifact is statically qualified, not a demonstrated neural decoder runtime.
Bit-for-bit reproducibility was not tested by a second build.

## Preservation and next action

The baseline Linux Python tree/package identities match exactly; ROCDXG 1.2.0,
HIP and HSA package payload verification passed. Both Windows environments,
native binaries, AMD source edits/backup and all saved turret input hashes match.
The checkpoint was hashed once before work and checked by metadata afterward.
Windows driver `32.0.31041.1004`, WSL kernel/default distro and historical evidence
were preserved. `docker-desktop` was untouched; both distros finish stopped.
[Windows preservation](amd-wsl-flexgemm-build-evidence/windows-preservation-after.json).

The smallest next milestone is **minimal genuine decoder preparation using this
wheel and the full saved 3,743-row turret**, with only audited Python dependencies,
native import authorization, strict original F16/F32 loading and source-faithful
subdivision admission. If those pass, request one bounded neural-only forward.
NumPy/safetensors and decoder import/loading/resource contracts remain unqualified;
no BF16 patch or relaxed memory rule was applied. No additional synthetic
convolution experiment is recommended. Stop here for separate approval.
