# Isolated WSL ROCm smoke-test Implementation Plan

> **For agentic workers:** Execute inline with verification and one independent review before the GPU attempt. Steps use checkbox syntax.

**Goal:** Install one isolated official Ubuntu 24.04 WSL2 distribution and prove one small deterministic PyTorch GPU calculation, without touching either Windows AI environment.

**Architecture:** A new distro, `Modly-AMD-Ubuntu-24.04`, stores its dynamic VHD exclusively under `D:\AI\Modly-WSL\Ubuntu24.04`. Use AMD's prebuilt userspace ROCDXG/ROCm libraries and a dedicated Linux Python venv. Reuse the existing Windows observer and Job Object, supplemented by a guest timeout and distro-specific termination on failure; a Windows Job alone cannot contain Linux guest processes.

**Tech stack:** Existing Windows 11/Adrenalin 26.8.1/WSL2; official Ubuntu 24.04/Python 3.12; ROCDXG 1.2.0; ROCm 7.2.1 userspace; AMD Torch 2.9.1+rocm7.2.1.lw.gitff65f5bc and Triton 3.5.1+rocm7.2.1.gita272dfa8 prebuilt wheels.

**Spec:** User-approved four-phase isolated WSL setup and one minimal GPU test, 2026-10-14. No TRELLIS installation or execution.

## Global constraints

- Start at `1794c909a8cab0f042c0d38ec088cc8b4935f363`; publish only `codex/amd-wsl-rocdxg-smoke`.
- No Windows driver, kernel, feature, reboot, global `.wslconfig`, Windows Python/ROCm, model, AMD source or Docker changes.
- Preserve the existing default distro `docker-desktop` and its registration. Always specify the new distro explicitly.
- No Linux GPU driver/DKMS, source builds, TRELLIS/reference setup script, checkpoints or AI inference.
- Existing kernel 6.18.33.2-2 is different from the older AMD matrix's 5.15 row; this is a bounded compatibility test, not a claim of identical certification.
- Stop if unavailable packages, compilation, reboot, feature/kernel/driver changes or uncertain package provenance are required.
- D: initially has 105,102,913,536 bytes free. New dynamic VHD maximum 32 GB; retain at least 50 GiB host free. Stop if this milestone's physical growth exceeds 20 GiB or downloads exceed 8 GiB.
- One GPU attempt only; exclusive marker. Guest deadline 90 seconds (+5-second kill grace), GPU-stage deadline 60 seconds, host overall 120 seconds. Healthy Windows observations at most 10 seconds apart while GPU work is allowed.
- GPU admission: correct RX 9070 XT/gfx1201, at least 13 GiB available VRAM; 12 GiB allocator ceiling and 64 MiB measured allocated/reserved smoke-test limit. No retries or warmup.

## Review focus

- Ubuntu installation must not update WSL or Windows features; installation error stops the task.
- Apt simulation and pip resolution must reject drivers, DKMS, compilers/SDKs, source archives and unexpected runtime versions.
- Linux worker must await host authorisation after Windows Job attachment; no GPU work before healthy monitoring.
- Windows Job containment must be supplemented by guest timeout and termination of only the owned distro if process exit is unverified.
- Numerical result, memory release, process exit and driver health all remain mandatory; a harness failure is UNKNOWN, not a native defect.

## Tasks and verification

1. [x] Verify clean exact Git base, remote base/main, host driver, WSL registrations, virtualization and disk capacity. Verify current official driver mapping and ROCDXG route.
2. [x] Save exclusive host/package/preservation evidence under `docs/compatibility/amd-wsl-rocdxg-evidence`. Independently compare both Windows inventories and AMD edits/backups.
3. [x] Download Microsoft's currently registered official Canonical image `https://releases.ubuntu.com/24.04.5/ubuntu-24.04.5-wsl-amd64.wsl` (388,975,696 bytes; SHA256 `bb415d824822c4b878125729af451a5d18fb13d1cf5cbed9a7393ad64ac6039e`) exclusively into the owned D: download directory. Verify the complete hash before `wsl --install --from-file <verified-image> --name Modly-AMD-Ubuntu-24.04 --location D:\AI\Modly-WSL\Ubuntu24.04 --version 2 --vhd-size 32GB --no-launch`. This fixes image provenance and avoids the installer's implicit network choices. Verify default registration unchanged before guest commands.
4. [x] Explicitly launch only the new distro. Record `/etc/os-release`, `uname`, Python, `/dev/dxg`, `libdxcore.so`, disk allocation and package baseline. Do not call rocminfo or Torch GPU APIs.
5. [x] Configure official signed AMD ROCm 7.2.1 userspace apt repository only inside the new distro. Inspect actual wheel ELF dependencies first. Simulate direct runtime/math/MIOpen packages and record exact versions, download/installed sizes before installation. Do not install `rocm-hip-runtime` or `rocm-ml-libraries` metapackages: both pull the LLVM toolchain. The direct candidate is `hip-runtime-amd=7.2.53211.70201-81~24.04` with a 73,362,592-byte AMD dependency closure; math packages are admitted only when required by actual ELF dependencies. Reject any GPU driver/DKMS, compiler/SDK or excessive footprint. Install only admitted prebuilt packages plus hash-verified ROCDXG 1.2.0 (`3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e`).
6. [x] Record wheel metadata, provenance, dependencies and actual/uncompressed download sizes. Create Linux `/opt/modly-wsl/venv`; install AMD's exact Torch/Triton wheels and only binary Python runtime dependencies. Save package manifest and CPU-only import identity with GPU initialisation forbidden.
7. [x] Add a small Linux worker and Windows supervisor in `scripts/compatibility/amd_wsl_rocdxg_smoke.py`, reusing existing containment/observer helpers. Focused CPU tests must verify exclusive attempt, launch gate, deadline/health handling, exact fixture and cleanup contract. Obtain independent review; commit/push exact candidate before the GPU test.
8. [ ] Run one deterministic F32 vector addition (256 elements) on the GPU, compare every result exactly to independently computed CPU reference, synchronize, capture memory/timing and release. No other GPU arithmetic or model work. Require zero allocated/reserved Torch bytes, collected owned tensors, clean guest and host exits, and healthy post-exit observer.
9. [ ] Independently review outcome, compare preservation/default/driver/kernel state, report actual download and disk footprint, publish report/evidence and verify remote commit. Stop.

## Exact rollback plan (documented; no automatic deletion)

Keep evidence outside the distro. If a setup/test gate fails, stop and preserve the new distro for inspection; do not repair or retry automatically. To remove this newly created setup later, first verify its exact registered name/path and export any desired data. Then terminate/unregister **only** `Modly-AMD-Ubuntu-24.04`. Remove only the confirmed owned `D:\AI\Modly-WSL\Ubuntu24.04` remnants after checking the absolute path. This destroys the new distro and is not performed in this milestone. Neither Docker nor either Windows venv is a rollback target. No global WSL shutdown/update, driver rollback or Windows reboot is part of the plan.

## Decisions

- The user already authorised conditional execution; the plan proceeds through objective gates without another approval prompt.
- Existing dedicated `proof` checkout and a new branch provide the requested isolation; no redundant repository/worktree or broad application suite is needed.
- Prefer ROCm 7.2.1/PyTorch 2.9.1 for this test; the Linux TRELLIS reference's Torch 2.13/ROCm 7.2.4 is explicitly deferred.

## Execution record

Preparation steps1-7 completed; exact reviewed harness published as `ec7acede847231fa95da2427b9b408a1f0431c36`. The one reserved attempt stopped at the Linux stdin authorization gate before Torch import; task8 numerical/memory acceptance was not reached. Result UNKNOWN. No retry or guest restart followed. Post-audit independently confirmed both Windows environments, AMD edits/backups, saved turret assets, driver/kernel/default distro and all historical refs preserved. Both distros are stopped. Final report/evidence publication is the remaining task9.
