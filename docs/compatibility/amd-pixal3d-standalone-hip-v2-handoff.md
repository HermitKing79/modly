# Pixal3D standalone model-free HIP proof v2

**Prepared for one deliberate user run. Actual WSL memory feasibility and HIP compatibility remain UNTESTED. No model generation is authorized.**

V1 commit `4bfbd25afb69a535e5338081117018969ff808b8` staged the approved dependencies, then withheld guest GO when the final22 GiB check failed. First recorded crossing:09:19:40.855473 UTC on26 October2026,21.992882 GiB. The final gate recorded21.281940 GiB. Approximately2.077 GiB declined during WSL startup, but no VM-process counters existed to prove exclusive attribution. The second attempt refused the occupied root. Both failed evidence directories remain unchanged.

## Run independently

Use a normal, non-administrator PowerShell window. Keep this checkout on the exact published v2 commit and use that commit in the final handover command. Close Codex Desktop, ChatGPT Desktop and RAM-heavy browsers manually. Do not start WSL manually, change global WSL settings, delete old roots or retry the v1 command. No internet/download preparation is required for v2.

Launcher: `scripts/compatibility/Invoke-Pixal3DHipProofV2.ps1`. Without `-Run`, it prints a no-execution dry-run and exits without admission readings, WSL or file creation. Live invocation requires `-Run -ExpectedCommit <exact-published-v2-commit>`.

The existing Windows Python3.12.10, Git, qualified `Modly-AMD-Ubuntu-24.04` distro/user `modly`, Python3.12.3 and ROCm7.2.1/ROCDXG remain prerequisites. The launcher verifies its pinned interpreter, committed critical source and contracts before work. It refuses active Codex/ChatGPT desktop processes and never closes applications.

## Cache and isolation

- Immutable cache: `D:\AI\Modly-Pixal3D-Proof-v1`. All13 approved wheels and three source files were independently size/SHA-256 verified during preparation: **2,032,962,149 bytes**. Runtime verification repeats during the future run. Missing, changed, redirected or hardlinked payloads stop; there is no downloader or alternate dependency resolver.
- New Windows root: `D:\AI\Modly-Pixal3D-Proof-v2`, for a small owner receipt only. Wheel archives are never copied here. This root was absent at preparation and must still be absent at live preflight.
- New guest root: `/home/modly/pixal3d-native-proof-v2`, with independent `.venv`. Guest exclusivity was deliberately not queried in Codex. Its future controller verifies the path is unused and records readiness before any environment creation/GO. An occupied guest root stops without deletion or reuse.
- New exclusive evidence: local `artifacts/amd-pixal3d-standalone-v2-<UTC>-<nonce>/`. The console prints its `REPORT.txt`. All failures retain their evidence and used roots. No automatic retries or overwrite/resume path exists.

The host holds read-only Windows cache-file handles denying write/delete while the guest runs. Explicit audit policies deny cache mutations in the host/provision child; pip reads local archives directly using `--no-index --no-deps --no-cache-dir --no-compile --only-binary=:all: --require-hashes`. Archives are unpacked only into the new venv; approximately4,459,761,446 bytes of wheel contents, subject to unchanged disk checks. Three small source files are copied only into the isolated guest tree. Before/after cache hashes and size/mtime/file IDs must match. No existing environment, driver, Torch installation, model checkpoint or asset is changed. No kernel is compiled.

## Stages and guards

1. **Windows admission:** unchanged qualified reader and three fresh available-RAM readings >=22 GiB, four seconds apart, plus final22 GiB.24 GiB remains preferred only. First failure stops without further admission readings or WSL.
2. **Cache verification:** owned, memory/deadline-bounded Windows child verifies the existing cache and HIP ELF/gfx1201 manifest, writes metadata only. No downloads or archive copying.
3. **WSL startup feasibility:** immediately before starting the named controller and throughout its startup, record timestamped host RAM and read-only Windows WSL candidate process counters. Every startup sample must remain >=22 GiB; first lower sample withholds GO and stops. This additional startup check does not lower/replace any original gate.
4. **Provisioning:** guest records Linux `/proc/meminfo`, its own process RSS/size and exclusive-path readiness. A final fresh22 GiB read follows process inventory immediately before GO. Only then create the separate venv and install the exact13 wheels. Preserve original environment critical-file identities. No global cache flushing; only the existing owned-new-tree cache hint is retained, with recovery explicitly not guaranteed.
5. **HIP operators:** another owned controller/handshake and final22 GiB gate precede the unchanged model-free checks: genuine gfx1201, prebuilt HIP extension, INT8/ConvRot, PyTorch GPU math SDPA, NAF, sparse gather/GEMM and tiny geometry conversion. Independent numerical references, finite values, native-call witnesses, device checks and GPU-event timing reject unsupported fallback. No flash-attention or full Pixal3D/model-fit claim follows from a pass.

Existing limits remain:12 GiB Windows live floor; D:50 GiB free; guest/artifact2 GiB free;20 GiB VHD growth;6 GiB owned process/guest RSS;180-second payload/provision bounds;120-second startup/CPU-ready and CPU phases;360-second operator overall;180-second GPU-release phases;13 GiB initial free VRAM,3 GiB live free VRAM,12 GiB allocator cap. Deadlines/failures never trigger increased limits, another attempt or a fallback kernel build.

Cleanup closes the GO pipe, verifies nonce/PGID/start-time owned Linux groups, and separately reaps owned Windows clients/Jobs. A client wait timeout cannot bypass Linux cleanup. Failed Job attachment retains the bounded direct cleanup of that exact owned client. No distro shutdown, unrelated process termination or automatic application closing occurs. Host/VM candidate observations are recorded after owned process exit; the shared WSL VM may remain resident.

## Inspect the result

`REPORT.txt` and `summary.json` separately show **Windows admission**, **WSL startup feasibility**, **isolated provisioning** and **HIP operators**, with cache verification as an extra stage. Unexecuted stages are `NOT_RUN`; an attempted second startup failure cannot be hidden behind an earlier pass. Every operator has its own status. If startup fails, report observed available RAM and withhold provisioning; the result demonstrates insufficient headroom in that session, not a permanent impossibility for32 GB hardware.

Evidence includes admission and host-memory journals, `startup-memory.jsonl`, cache/HIP-wheel identities, `provision/` and `operators/` ownership/readiness, Linux memory, child/results, resources and cleanup receipts. Windows process access failures remain `UNAVAILABLE`, not fabricated zero usage. VM candidate counters can cover shared distros/pages; `wsl.exe` counters describe only the Windows client. Host deltas alone are not exclusive WSL attribution. No VM-total or physical memory release is inferred from client exit.

Preparation ran only blocked-action mock/CPU tests, synthetic file/lease checks and actual no-work PowerShell/Python defaults. No live RAM admission, WSL, package installation, runtime/model downloads, Torch import or GPU execution occurred. A model-free PASS permits only preparing a separately reviewed future single-image handoff. This runner cannot continue into generation.
