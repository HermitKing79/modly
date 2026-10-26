# Pixal3D model-free operator policy: proposed v3 handoff

**GO for a separately authorised, single monitored compatibility proof; execution is NOT authorised by this preparation. Actual HIP fit remains unknown. Neural generation remains subject to its unchanged22 GiB admission/final gates.**

## What v2 established

Authenticated source: commit `681fcc962d4ca904291cc3b7bfbdf28757350ba6`, private case `amd-pixal3d-standalone-v2-20261026T185014Z-8bad81fb19a146eeb36505e309f15fa3`.

| Measurement,26 October2026 UTC | Available Windows RAM |
|---|---:|
| Before provisioning controller,18:50:37.147392 |24.401775 GiB|
| Final provisioning GO |22.203743 GiB|
| Lowest sampled during provisioning |17.855251 GiB|
| After natural provisioning exit,18:53:13.272726 |21.182652 GiB|
| Operator prelaunch refusal,18:53:13.287992 |21.184074 GiB|

The operator refusal happened **before Popen**, not during a second controller launch: no Job attachment, no GO, zero Windows-client peak and no operator ownership/result files. There is no second-startup or HIP peak measurement. VmmemWSL existed but memory queries were unavailable; neither its complete footprint nor exclusive attribution can be recovered from these logs. Small Windows-client counters are not VM totals.

Provisioning succeeded:13 exact pinned wheel versions/hashes, independent site-packages, successful pip exit/report validation,16 unchanged cache identities and unchanged original-environment critical hashes. Linux child OS peak RSS was366,882,816 bytes (349.887 MiB), while the sampled process-tree maximum was211,148,800 bytes (201.367 MiB). This demonstrates that polling can miss a peak. The child and owned group exited naturally; no HIP operation ran. Existing evidence proves installation success, not current unchanged guest bytes. V3 therefore re-authenticates before GO without launching WSL during preparation.

## Why propose a separate profile

The22 GiB value came from the full single-image/neural contract (`amd-pixal3d-native-single-image-proof-contract.json`, guards), then was deliberately preserved in v1/v2. There is no measured22 GiB requirement for these synthetic operators. Their fixed fixtures use64x256 linear inputs, a256x256 Hadamard reference,4-token SDPA, a tiny NAF neighborhood,8 sparse voxels and4 conversion cells. These are not a512-resolution image/model workload. Torch/ROCm/kernel initialisation remains unmeasured; the small fixtures do not prove a low total peak.

| Proposed model-free-only rule | Derivation / difference from v2 |
|---|---|
| Three fresh prelaunch readings,4 seconds apart, plus final before controller: **21,687,472,128 bytes (~20.198032 GiB)** |18 GiB budget plus the entire observed cold-start host decline below. Replaces22 only in this operator-only entry point.|
| Startup and final ready/GO minimum: **18 GiB (19,327,352,832 bytes)** |Existing12 GiB host floor + existing6 GiB sampled Linux RSS budget. No invented operator peak.|
| Cold-start allowance: **2,360,119,296 bytes (~2.198032 GiB)** |26,201,206,784 minus23,841,087,488 bytes from the v2 prelaunch/final-GO receipts. A single observed host delta, not a worst-case bound or exclusive WSL attribution.|
| Full-neural admission/final22 GiB;24 GiB preferred only |Unchanged in every existing contract and launcher. No neural route exists in v3.|
|12 GiB live-host floor;6 GiB process/RSS stops; disk/VRAM/allocator/deadlines/owned cleanup |Unchanged. These are monitored stops, not a hard total-VM physical-memory envelope.|

At the recorded21.184074 GiB refusal, this proposed prelaunch budget would have had0.986042 GiB additional headroom. That is a historical comparison, not fresh admission. The6 GiB Linux stop is polled at0.25 s; Windows process commit limits cover clients, not the VM. Concurrent activity, transient peaks and native allocations remain risks. There is no guarantee that a32 GB host will pass or that monitoring can prevent every floor overshoot. This qualifies one bounded data-acquisition attempt, not production safety or full Pixal3D generation.

## Prepared operator-only route

Entry point: `scripts/compatibility/Invoke-Pixal3DOperatorsV3.ps1`. Default invocation is a no-work dry-run. Live execution additionally requires an exact published commit and an explicit `-ApproveOperatorProfileV3` switch after separate user authorisation. Preparation does not supply that authorisation.

- Read-only existing environment: `/home/modly/pixal3d-native-proof-v2/.venv`. No pip/venv creation, installation, download, dependency resolver or repair route is exposed.
- Fresh exclusive Windows marker root: `D:\AI\Modly-Pixal3D-Operators-v3`; fresh Linux scratch root: `/home/modly/pixal3d-operator-proof-v3`. Refuse occupied roots. Linux exclusivity can only be checked by the future controller and has not been queried in Codex.
- Fresh evidence: `artifacts/amd-pixal3d-operators-v3-<UTC>-<nonce>/`. Preserve failed roots and logs; no retries, deletion or overwrite.
- Before GO, authenticate successful v2 receipts, strict pyvenv configuration, copied qualified interpreter, all13 wheel manifests and installed file contents, import inventory, pinned sources and metadata. Unknown executable files, directory aliases and extra startup hooks stop. Existing pip bootstrap files are explicitly excluded from operator imports. Python bytecode writing remains disabled. Before/after environment file identities must match.
- All writable HOME/tmp/HIP/COMGR/cache paths are outside the reused environment. Audit policies deny environment/cache mutation. After streaming authentication reads, a read-only per-file Linux cache-residency hint may be requested; no fsync, original-file rewrite, global cache clearing or guaranteed physical recovery is claimed.
- Original numerical fixtures, tolerances, native HIP counters, device/ELF identities and timing checks remain unchanged. Only versioned imports, scratch-write scope and an additional installer-import prohibition differ in the operator worker.

Retained limits: D:50 GiB and guest/artifact2 GiB free;20 GiB VHD growth;6 GiB Windows client commit and sampled Linux RSS stops;120-second authentication/startup/CPU phases;180-second GPU-release phases;360-second overall controller/operator bound;13 GiB initial free VRAM,3 GiB live free VRAM and12 GiB allocator cap. No global WSL settings, unrelated process/distro termination, elevated execution or increased limits.

The report separates operator-profile admission, current v2-environment authentication, WSL/controller readiness and each actual HIP operation. Failure before Popen explicitly records `client_started:false`. Cleanup retains independent nonce/PGID/start-time Linux checks and Windows Job/client reaping. Natural process exit does not imply shared VM memory reclamation.

**Next action:** review and separately authorise this exact operator-only resource profile. Do not rerun v2. V3 remains unexecuted; no model weights, inference or automatic generation continuation are included.
