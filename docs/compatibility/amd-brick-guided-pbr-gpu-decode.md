# First brick guided PBR GPU decode — prelaunch admission

**BLOCKED — NO GPU FORWARD.** Approved base `6f0c51eae8d5bf00da7d42078ac6bd32dd19ae75` was verified locally and against GitHub. Outcome branch: `codex/amd-brick-guided-pbr-gpu-decode`. The CPU-qualified F16 implementation is unchanged.

Three fresh readings used the existing preparation adapter's `read_resources()` and Windows `GlobalMemoryStatusEx`, three seconds apart on 2026-10-16 UTC. All were below the unchanged **18.5 GiB / 19,864,223,744-byte** prelaunch requirement:

| UTC | Available Windows RAM, bytes | GiB |
|---|---:|---:|
| 07:44:02.199 | 19,734,081,536 | 18.379 |
| 07:44:05.199 | 19,659,632,640 | 18.309 |
| 07:44:08.200 | 19,704,377,344 | 18.351 |

D: free space was **74,269,659,136 bytes**, above the unchanged **53,687,091,200-byte / 50 GiB** floor. The complete measurements, timestamps, stage and requirements survive in [the admission receipt](amd-brick-guided-pbr-gpu-evidence/early-host-admission.json). Its raw local SHA-256 is `c4560058a81b098158a59402e70b55cc471ef5af7ef7f5b47865148cf3666b18`.

The mandatory stop occurred before supervisor integration, fresh production output attestation, WSL launch, runtime/native qualification, model loading or an inference marker. **Zero forwards were attempted.** No GPU memory was queried, allocated or measured, and no new material field exists. Destination and WSL/VRAM gates remain untested; CPU projections are not reported as actual GPU memory.

All 16 qualified source/test hashes matched the published CPU closure. No decoder, guard, precision, memory cap or deadline was changed, and no test suite was repeated. No application was closed, setting changed, package installed or automatic retry performed. The original texture latent, geometry/guides, mesh, source edits/backups and bounded environment inventory remain protected; [preservation evidence](amd-brick-guided-pbr-gpu-evidence/preservation.json) records its exact verification scope. Original untracked cache directories are retained.

Only source-safe textual outcome evidence is published. Main, the CPU qualification branch and all other pre-existing remote heads are retained. [Independent outcome review](amd-brick-guided-pbr-gpu-evidence/independent-outcome-review.json) is separate from the existing CPU integration qualification.

**Next action:** obtain available Windows RAM above 18.5 GiB with operating headroom, then request a fresh admission check and the single controlled forward. Its concrete supervisor composition and all original ownership, runtime, VRAM, memory, timeout, persistence and natural-cleanup checks must still pass. This milestone authorizes no retry and makes no numerical or visual material-quality claim.
