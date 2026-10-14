# Isolated Ubuntu WSL2 ROCm compatibility

2026-10-14. Base `1794c909a8cab0f042c0d38ec088cc8b4935f363`; development branch `codex/amd-wsl-rocdxg-smoke`. **UNKNOWN for basic WSL GPU execution: isolated installation and CPU import succeeded, but the one supervised attempt stopped at the guest authorization gate before Torch import.** No TRELLIS installation or model execution is included.

## Installed candidate and preservation

Official Ubuntu 24.04.5 LTS is registered as `Modly-AMD-Ubuntu-24.04`, with a new dynamically allocated, maximum-32-GB VHD at `D:\AI\Modly-WSL\Ubuntu24.04\ext4.vhdx`. Microsoft's distribution manifest identifies the Canonical image; the full 388,975,696-byte image SHA256 matched before installation. Existing `docker-desktop` remains the default. Windows driver, kernel, features and both Windows AI environments were not modified.

Host: Windows 11 Pro build 26200.9711; RX 9070 XT; driver `32.0.31041.1004`, mapped by [AMD GPUOpen](https://gpuopen.com/version-table/) to Adrenalin 26.8.1. WSL is 2.7.12.0; the unchanged guest kernel is `6.18.33.2-microsoft-standard-WSL2`. This differs from the older AMD matrix's kernel 5.15 and is not claimed to be that exact validated configuration.

Linux candidate: Python 3.12.3; official ROCDXG 1.2.0; ROCm 7.2.1 prebuilt userspace libraries; AMD Torch wheel `2.9.1+rocm7.2.1.lw.gitff65f5bc`; Triton `3.5.1+rocm7.2.1.gita272dfa8`. `/dev/dxg` and Microsoft `libdxcore.so` exist and the unprivileged `modly` user has device access. [ROCDXG compatibility/quickstart](https://github.com/ROCm/librocdxg) and [AMD's framework wheel instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) support this candidate route; actual GPU execution remains the local proof boundary.

All packages are admitted prebuilt binaries. No Linux GPU driver, DKMS, LLVM compiler toolchain, SDK or native extension build was installed/run. Broad ROCm metapackages were avoided. Signed APT metadata, exact resolved versions, download/installed sizes and Python wheel hashes are preserved in [evidence](amd-wsl-rocdxg-evidence/).

## CPU preparation findings

- Independent preservation checks matched both Windows inventories, 12 selected native binary hashes, 169 native Python source hashes, AMD edits/backups and all five saved turret decoder assets.
- Actual Torch wheel ELF dependency inspection selected only required runtime/math libraries. Pinning AMD `rocminfo` prevented selection of an older Ubuntu HSA stack. Required system header dependencies are listed explicitly; they are not a compiler or Linux kernel installation.
- First CPU import identified omitted `libgomp.so.1`. One complete 55-file ELF closure found this as the sole unresolved dependency. Official Ubuntu `libgomp1` (148,062-byte download) was admitted and installed; no speculative library repair loop occurred.
- AMD's genuine runtime version omits the packaging-only `.lw` suffix. Installed `torch/version.py` exactly matches the hash-verified original wheel. The execution guard pins both runtime and distribution versions, full source commit and HIP `7.2.53211-e1a6bc5663`; CUDA is absent. Corrected CPU import completed with zero GPU API calls and PyTorch GPU initialization false before/after. NumPy is intentionally absent; the tensor-list test does not require it.
- Review corrected startup `LD_LIBRARY_PATH` delivery and a coarse-clock observer boundary. Eighteen focused CPU tests pass, including actual supervisor success/failure replays and exact runtime-identity rejection. Independent review approved the resulting bounded candidate.
- Before execution: VHD physical allocation 18,921,553,920 bytes; retained payload/requirements 6,452,645,001 bytes; D: free 79,728,582,656 bytes; guest free 13,272,559,616 bytes. Exact package upper sums and actual physical allocation are distinguished. The earlier 1,769,996,288-byte VHD observation occurred during APT downloading, not in a dedicated pre-APT baseline file.

## Single execution contract

One F32 addition of two deterministic 256-element vectors, with all elements checked against independent exact Python arithmetic and a CPU Torch reference. No models, GEMM/convolution, Triton launch or TRELLIS stage. Require RX 9070 XT/gfx1201, 13 GiB free VRAM, 12 GiB allocator ceiling and a 64 MiB measured Torch-memory smoke limit.

The reviewed harness uses an exclusive attempt marker, healthy Windows observations before authorisation and periodically at no more than ten-second start gaps, a Windows Job for the WSL client, a 90-second guest timeout and termination of only the owned distro. GPU stage maximum 60 seconds; host maximum 120 seconds. A Windows Job alone is explicitly not claimed to own Linux processes. PASS requires exact finite results, synchronization, zero allocated/reserved Torch bytes, collected owned tensors, clean client exit, verified named-distro termination and healthy post-exit observation. No retry is authorised.

## Outcome and next step

The exact reviewed execution candidate was committed and GitHub-verified as `ec7acede847231fa95da2427b9b408a1f0431c36` before the attempt. Its SHA256 `6a34ffc0f799f1900d65d2a860b3933afb4dae8ab07ab111f9a3adc367ff24f2` agrees with the exclusive marker, host and guest records. [Execution identity](amd-wsl-rocdxg-evidence/execution-harness.json), [host result](amd-wsl-rocdxg-evidence/run/results.json), [guest result](amd-wsl-rocdxg-evidence/run/guest/results.json).

The host attached its Job and logged `go-sent`. The guest rejected `await_go(sys.stdin)` with `Host authorization absent or refused; no Torch import or GPU work`. It recorded only `gate-wait` and `worker-finished`, `gate_authorized=false`, and no GPU-stage start. Exact received stdin was not recorded, so neither EOF nor newline conversion is established as the cause. **This is a harness/transport boundary failure, not evidence of a ROCm numerical or native GPU defect.** The dedicated distro's GPU visibility, numerical result and VRAM/allocator cleanup remain unmeasured.

The worker and supervisor exited with status 1. Supervision took 11.172 seconds. Three healthy Windows observations started five seconds apart, including one after termination. The client Job had zero remaining processes with no forced Job cleanup; termination of only `Modly-AMD-Ubuntu-24.04` succeeded. No retry occurred. Both distros are now stopped; Docker remains default. [Independent-readable integrity summary](amd-wsl-rocdxg-evidence/independent-readable-outcome.json), [post-host preservation](amd-wsl-rocdxg-evidence/host-after.json).

Retained download payloads plus the 1,106-byte requirements file total **6,452,645,001 bytes (6.01 GiB)**. This includes Ubuntu, prebuilt ROCm/OpenMP packages, AMD Torch/Triton, and admitted binary Python dependencies. APT index/metadata and transient pip-resolution traffic are additional; total network bandwidth was not metered. Final VHD physical allocation is **18,921,553,920 bytes (17.62 GiB)**; D: retains **79,728,582,656 bytes (74.25 GiB)** free. [Measured disk footprint](amd-wsl-rocdxg-evidence/disk-after.json).

**One recommended next action:** separately approve a CPU-only replay of the actual WSL stdin route with the same startup environment, timeout and Python flags, recording the host-written bytes and guest-received bytes/line. This is the smallest missing observation; do not infer a fix or consume another GPU attempt before it is resolved. The exclusive failed-attempt marker remains intact.

After that boundary and a separately authorised basic GPU proof pass, the decoder-related milestone is installation/qualification of only the pinned genuine neural-decoder dependencies, followed by one strict offline decode of the preserved turret latent/coordinates, stopping before extraction. There is no basis to approve that execution yet. Current WSL/PyTorch 2.9.1/ROCm 7.2.1 differs from the [Linux reference](amd-linux-route-comparison.md)'s native Linux/PyTorch 2.13/ROCm 7.2.4, explicit-GEMM and mixed BF16/FP32 decoder. FlexGEMM, TRELLIS, O-Voxel and CuMesh have not been installed in this distro; no decoder precision policy, learned subdivision admission or neural computation is proved here.

Rollback is documented in the [plan](../superpowers/plans/2026-10-14-amd-wsl-rocdxg-smoke.md). No distro deletion or global WSL shutdown is performed automatically.

Final preservation: independent post-audit matched both Windows venv metadata/distributions, all12 selected native binary hashes,169 native Python sources, AMD modified file/backup hashes and all five turret assets. Direct Git verified the unchanged AMD source HEAD/status. Historical local and GitHub refs match the pre-installation baseline; main is `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`. Only newly added WSL proof/report/test paths differ from the starting commit. [Post-audit](amd-wsl-rocdxg-evidence/independent-preservation-after.json), [Git preservation](amd-wsl-rocdxg-evidence/git-preservation-after.json), [independent outcome review](amd-wsl-rocdxg-evidence/outcome-review.json).
