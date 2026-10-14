# WSL neural shape-decoder preparation — UNKNOWN

Base `67a9e4ac02bf90b6702b0592216b3093acd48059`; branch
`codex/amd-wsl-neural-decoder-preparation`. CPU-only. **Installed additions: none;
package download/installation bytes: 0. No source build, GPU initialization,
checkpoint tensor loading or neural forward occurred.** Upstream source/metadata
and bounded package-control byte ranges were inspected read-only.

**Stop point:** no compatible prebuilt FlexGEMM binary was established, and the
existing Ubuntu distro has no C/C++ compiler, HIP compiler, Python development
header or required ROCm development headers. The permitted conditional source
build therefore cannot start. Readiness is UNKNOWN, rather than a demonstrated
native incompatibility. Installing a substantial toolchain needs separate approval
under the user's explicit stop condition.

## Native dependency admission

The [pinned Linux reference](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/scripts/setup-rocm.sh)
clones unpinned `JeffreyXiang/FlexGEMM` main. Its successful native build revision
cannot be recovered from that reference commit. The separately audited upstream
revision is `6dd94a859c26ee8246888502eada3dd8ad85532e`.
[Its setup](https://github.com/JeffreyXiang/FlexGEMM/blob/6dd94a859c26ee8246888502eada3dd8ad85532e/setup.py)
builds seven native translation units with PyTorch `CUDAExtension`, C++17/OpenMP
and HIP target `gfx1201`. On this ROCm Torch wheel that links Torch/c10 Python,
CPU and HIP libraries plus `amdhip64`: CPython 3.12, Torch C++/libstdc++ ABI and
HIP compatibility all matter. A build must target this existing Torch 2.9.1 /
ROCm 7.2.1 stack; its actual C++ ABI bit was not inferred from version numbers.

[Explicit GEMM](https://github.com/JeffreyXiang/FlexGEMM/blob/6dd94a859c26ee8246888502eada3dd8ad85532e/flex_gemm/ops/spconv/submanifold_conv3d.py)
still invokes the native neighbour-map function before `torch.mm/addmm`.
It does not remove the extension requirement. The separately compiled Triton
host HIP helper also needs an ordinary compiler and `Python.h`; its cache is not
a substitute for the Torch extension.

[Official release metadata](https://api.github.com/repos/JeffreyXiang/FlexGEMM/releases)
provides no v1.0.0/v1.1.0 wheel assets; the canonical PyPI name has no package
listing. The known dmonkman CPython 3.12 Linux `+rocm10.0` wheel is 3,685,157 bytes,
Git blob `6ed1a2e7f92e73686e7b414c0799eeedf3254274` at commit
`3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`. It remains **unqualified**, not proved
incompatible, for the current stack. It was neither downloaded nor loaded.
No Windows `.pyd` or Triton cache was substituted.

## Exact missing development-package plan

Read-only local APT simulation, using [AMD 7.2.1 Ubuntu 24.04 metadata](https://repo.radeon.com/rocm/apt/7.2.1/dists/noble/main/binary-amd64/Packages),
proposes **59 new packages and two upgrades**, totaling **626,410,386 archive
bytes**. Known installed-size subtotal is **319,531,008 bytes plus LLVM**, whose
installed footprint is absent from both package-index and bounded control-header
metadata. This is a prospective plan, not an installation or disk-space approval.

| Core package | Version | Archive bytes | Installed KiB |
| --- | --- | ---: | ---: |
| hip-dev | 7.2.53211.70201-81~24.04 | 372,798 | 2,823 |
| hipcc | 1.1.1.70201-81~24.04 | 212,662 | 525 |
| rocm-llvm | 22.0.0.26084.70201-81~24.04 | 544,593,292 | Unavailable |
| rocm-device-libs | 1.0.0.70201-81~24.04 | 728,304 | 3,405 |

The complete [version/hash/dependency manifest](amd-wsl-neural-decoder-preparation-evidence/linux-toolchain-plan.json)
also includes Python development files, build-essential/Ninja and required
hipBLAS/hipBLASLt/hipSOLVER/hipSPARSE/hipCUB/rocThrust headers. It proposes
`libexpat1` 2.6.1-2ubuntu0.4 → 2.6.1-2ubuntu0.6 and `libpciaccess0`
0.17-3ubuntu0.24.04.2 → 0.17-3ubuntu0.24.04.3. Neither upgrade occurred.
These development dependencies and unknown expanded/compiler-build footprint
prevent admission as a small Python-only addition. NumPy and safetensors were
also left uninstalled because the native route has not passed admission.

## Genuine decoder and strict precision contract

The original `SparseUnetVaeDecoder.forward(decoder, x, return_subs=True)` can
exclude the FDG child's O-Voxel/Mesh conversion. Preserve evaluation, low-VRAM
mode, input spatial-cache clearing, `guide_subs=None`, learned masks and coordinate
row order. The parent requires sparse basic/linear/conv/spatial, norm and utility
modules; CuMesh, extraction, texturing and rendering are outside this neural path.
Real Linux imports remain unproved with the missing native dependency.

The genuine local 948,490,494-byte checkpoint matches
`e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`.
Its bounded header and all 292 names/shapes/F16 entries match the prior audit.
The prior genuine Windows metadata constructor expects **216 F16 / 76 F32**
entries, zero persistent buffers and 948,598,492 runtime parameter bytes.
That proof does not establish Linux Torch 2.9.1 constructor compatibility.

The unchanged strict loader rejects those 76 raw-F16/F32-target differences and
requires NumPy/safetensors. A future component-scoped loader must validate the
complete original F16 state and bytes, widen only the audited 76 keys, and prove
byte-exact F16 round trips. Missing/unexpected/shape-incompatible entries must fail.
No global loader relaxation or remote fallback is appropriate.

**Proposed baseline:** original F16 body, F32 norms/projections/input/output,
explicit GEMM and TF32 disabled. BF16 necessity is not demonstrated. The reference's
BF16 patches change arithmetic; F16→BF16 is generally lossy (1.0009765625 becomes
1.0 in the recorded CPU example), so they cannot satisfy the original byte-exact
contract automatically. No precision policy was applied to the installed runtime.

## Resources and later admission

Final D: free **79,531,442,176 bytes (74.07 GiB)**, above the 50 GiB floor;
guest filesystem total **33,501,757,440 bytes**, available **13,053,116,416 bytes
(12.16 GiB)**, within the 32 GiB VHD limit. Host available RAM at admission was
**17.35 GiB**; final guest available RAM was **15,935,713,280 bytes (14.84 GiB)**.
The last approved WSL GPU measurement was **14,334,255,104 free bytes (13.35 GiB)**;
it was not refreshed here because GPU initialization is not authorized.

There are 40 sparse convolutions and four learned subdivisions. Source-permitted
support grows **3,743 → 29,944 → 239,552 → 1,916,416 → 15,331,328**. The prior
visible final estimate is **13,826,914,124 bytes**, already above the unchanged
12 GiB allocator ceiling before hidden workspace and retained tensors. It is a
worst-case projection, not an observed OOM, and the old masked-SplitK estimates
do not attest the new explicit-GEMM implementation.

The inherited proposed one-million-token/stage, 256 MiB output and 1 GiB workspace
headroom contracts remain **unintegrated and unqualified for Linux**. Nine CPU
planner regressions passed, including overflow/worst-growth rejection before a
callback; no convolution was exercised. Future admission must precede every native
convolution and both C2S allocations, count retained caches/subdivision tensors,
and bound explicit native workspace against fresh reserved/free memory. Reject
unsafe or empty support without clipping, changing logits, supplying masks or
substituting latent rows. Preserve the full saved turret inputs.

A later real-forward proposal should retain owned Job/guest timeout/named-distro
containment, healthy periodic observation, explicit CPU/GPU/overall deadlines,
inference mode, synchronized finite output and zero allocated/reserved teardown.
BLAS/native-workspace cleanup and decoder workspace bounds still need qualification;
the vector-addition PASS does not establish them.

## Decision and cheapest next step

**UNKNOWN for decoder dependency readiness.** No native incompatibility or F16
computation failure was demonstrated. The working WSL Torch/ROCm stack remains
intact; no install or build can proceed under the current conditional approval.

The single next step is separate approval for a **targeted HIP/C++ development
toolchain admission and one pinned FlexGEMM build against the current stack**.
Before installation, resolve LLVM/build disk headroom and the two proposed OS
upgrades; target `gfx1201` explicitly, without replacing Torch/ROCm or executing
the reference's broad setup script. After native admission, qualify the strict
component loader and full-turret pre-allocation contracts before requesting a
genuine neural-only forward. No synthetic convolution milestone is proposed.

The two Windows environment inventories, native binaries, AMD modifications/backup
and original model/input identities match. Linux venv metadata, all package rows,
tools, headers and OS identity match. The final probe initially compared in-memory
package tuples against JSON lists and reported a false difference; the raw failure
is preserved with a [structural reconciliation](amd-wsl-neural-decoder-preparation-evidence/linux-preservation-reconciliation.json).
Free disk space varied across WSL startup; whole-filesystem byte identity is not
claimed. Driver `32.0.31041.1004`, kernel, 32 GiB VHD setting and default
`docker-desktop` registration were preserved. Both distros are stopped.
No GPU run or model tensor loading occurred. Stop after publication.

Evidence: [installed tool/header inventory](amd-wsl-neural-decoder-preparation-evidence/linux-readiness.json),
[decoder contract readback](amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json),
[CPU regressions](amd-wsl-neural-decoder-preparation-evidence/inherited-admission-tests.stderr.txt),
[preservation baseline](amd-wsl-neural-decoder-preparation-evidence/preservation-before.json),
[native source assessment](amd-wsl-neural-decoder-preparation-evidence/native-dependency-assessment.json),
[final Linux inventory](amd-wsl-neural-decoder-preparation-evidence/linux-preservation-after.json),
[final Windows environment preservation](amd-wsl-neural-decoder-preparation-evidence/windows-preservation.json),
[host status](amd-wsl-neural-decoder-preparation-evidence/host-after.json).
