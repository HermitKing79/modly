# WSL genuine FlexGEMM native import â€” UNKNOWN

Base: `9d439e0f1044804456021649a512d14d38bea7e6`. Branch: `codex/amd-wsl-flexgemm-native-import`. Exact candidate was GitHub-verified at `83f38e0995fc2e7e0eb8d55954672536397fa757` before execution; its unchanged SHA-256 is `c8a05f8a2121f2a527a6a9871d8a51b6d11b897228da57930ea01443577d04ae`.

**One authorized genuine `import flex_gemm` attempt stopped at a filesystem-policy admission error before its native ELF loaded. This is not evidence of AMD binary/runtime incompatibility. No retry or policy repair followed.**

## Exact blocker

FlexGEMM imports its Triton wrappers before `kernels/__init__.py` reaches `from . import cuda`. That wrapper import reaches FileLock 4.1.1's `_probe_link_follow_symlinks()`. Its owned `tempfile.TemporaryDirectory()` cleanup uses CPython's descriptor-safe `shutil.rmtree`, which calls `os.unlink(entry.name, dir_fd=topfd)`. The unchanged policy rejects nondefault directory descriptors:

`ImportPolicyError: filesystem_write: Relative directory descriptor is not admitted`

The [full guest traceback](amd-wsl-flexgemm-native-import-evidence/run/guest/results.json) and [durable journal](amd-wsl-flexgemm-native-import-evidence/run/guest/phases.jsonl) preserve the failure. The later `from flex_gemm.kernels import cuda`, native `.so` loading/path verification and 18 callable-binding resolution were **not reached**. Import synchronization/peak-memory records were not captured; they are not inferred. Static review identified FileLock's temporary probe but missed its fd-relative cleanup requirement.

## Actual execution and release

| Item | Observed result |
|---|---|
| Python / Torch | 3.12.3 / `2.9.1+rocm7.2.1.gitff65f5bc` |
| Torch distribution | `2.9.1+rocm7.2.1.lw.gitff65f5bc` |
| Triton / HIP | `3.5.1+rocm7.2.1.gita272dfa8` / `7.2.53211-e1a6bc5663` |
| GPU | AMD Radeon RX 9070 XT, `gfx1201`, one device |
| Available VRAM before / after release | 14,233,980,928 bytes (13.256 GiB), unchanged |
| Torch allocator ceiling | 12,884,901,888 bytes (12 GiB) |
| Torch allocated/reserved before / release | 0 / 0 bytes at both observations |
| Incidental HIP initialization | Yes, during GPU metadata admission |
| Native bindings / tensor computation | None invoked; policy allowed zero tensor operations |
| Worker GO-to-cleanup time | 6.069 seconds |
| Supervisor / outer command time | 20.500 / 20.841 seconds |
| Guest / host exit | 1 / 1, preserved UNKNOWN failure |
| Windows monitoring | Five healthy observations; no reported GPU reset/error |
| Owned Windows Job | Zero active before/after cleanup, no forced termination |
| Named distro termination | Only `Modly-AMD-Ubuntu-24.04`, exit 0 |

The existing binary GO, Job ownership, periodic health checks and named-distro termination were reused. The configured five-second success-only natural-exit grace was not entered on this error exit; the Job was already empty when cleanup began. The unchanged limits remained 60 seconds after GO, 90-second guest timeout with five-second kill grace and 120-second host deadline. The GPU supervisor's raw vector-add contract and the adapter's separate native verdict both remain UNKNOWN. The native ELF constructor risk is still untested; the Torch allocator cap would not bound direct native HIP allocations.

## Identity and preservation

The staged target is `/home/modly/native-decoder-preparation-6dd94a85/site-packages`. Original wheel SHA-256: `cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285`. Staged binary `flex_gemm/kernels/cuda.cpython-312-x86_64-linux-gnu.so` SHA-256: `91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01`. These are static/staged identities, not a successfully loaded binary.

[Passive Linux before](amd-wsl-flexgemm-native-import-evidence/linux-preservation-before.json) and [after](amd-wsl-flexgemm-native-import-evidence/linux-preservation-after.json) verified all 922 staged payloads, unchanged baseline venv/package/dpkg/source identities, protected ROCm payloads and approved library paths/metadata. [Windows after audit](amd-wsl-flexgemm-native-import-evidence/windows-preservation-after.json) matched both environments, 12 native binaries, AMD edits/backups, saved turret artifacts and decoder metadata/header; no checkpoint payload was loaded or rehashed. The passive after-inventory imported no AI/native libraries; afterward only Modly Ubuntu was stopped, with Docker still the default. Historical reports/markers and existing untracked bytecode remain intact. No installation, build, model loading, convolution or decoder operation occurred.

[Independent preflight](amd-wsl-flexgemm-native-import-evidence/independent-preflight-review.json), [outcome review](amd-wsl-flexgemm-native-import-evidence/independent-outcome-review.json), [verified summary](amd-wsl-flexgemm-native-import-evidence/verified-result-summary.json) and [host capture](amd-wsl-flexgemm-native-import-evidence/host-capture.json) document the exact one-attempt boundary.

**One next action:** separately approve a CPU-only correction for fd-relative cleanup confined to verified owned temporary directories, with a genuine `TemporaryDirectory`/`shutil.rmtree` reproducer and negative foreign-descriptor/traversal cases. Do not broadly allow directory descriptors or bypass FileLock's probe. If accepted, a fresh contained FlexGEMM import still requires separate approval. Strict shape-decoder checkpoint loading and the full saved 3,743-row neural forward remain deferred. Stop here.
