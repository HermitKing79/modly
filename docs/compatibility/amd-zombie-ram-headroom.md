# Zombie geometry RAM-headroom recovery — NO-GO: INSUFFICIENT_HEADROOM

Base: `35b3c70de7aa61c514a4e0410af246bab880ba9d`.
Isolated branch: `codex/amd-zombie-ram-headroom`.

The additional prelaunch target was **22 GiB available Windows physical RAM**
(23,622,320,128 bytes) in all three readings. `GlobalMemoryStatusEx.ullAvailPhys`
returned the following under the normal, non-elevated Windows user context:

| UTC on 2026-10-24 | Available bytes | GiB | Target |
| --- | ---: | ---: | --- |
| 01:44:02.892561 | 19,937,619,968 | 18.568 | FAIL |
| 01:44:06.892770 | 19,815,133,184 | 18.454 | FAIL |
| 01:44:10.892924 | 19,926,953,984 | 18.558 | FAIL |

The readings span 122,486,784 bytes (0.114 GiB), with four seconds between
samples. Even the highest was 3,684,700,160 bytes below the target. Installed
physical RAM was 34,270,023,680 bytes; this was not substituted for availability.
Committed memory, pagefile capacity and VRAM were not used for host admission.

The required response is **NO-GO**, so no WSL launch, model load, fresh neural
attempt marker, decoder forward or new run-path preparation followed. The
original 12 GiB live floor and all GPU/support/deadline contracts are unchanged.
No application was closed, process killed, cache cleared, machine rebooted or
Windows/WSL setting modified.

## Read-only consumer observations

Individual working sets included ChatGPT (756,695,040 bytes), Codex
(740,798,464 bytes), and Brave (645,816,320 bytes), with multiple additional
ChatGPT and Brave processes. Windows memory compression and Defender also had
significant working sets. These snapshots do not establish exclusive physical
ownership or explain the previous run's decline. Shared pages mean working
sets must not be summed as an exact physical-memory accounting. Raw process
identities remain private; the published evidence omits PIDs.

The previous observed decline of 6,700,421,120 bytes (6.24 GiB) remains a planning
observation, not a guarantee of future usage. No new consumption profile was
measured because execution was not admitted.

## Stage and preservation status

| Required result | This continuation |
| --- | --- |
| Available Windows RAM before launch | Three readings above; launch rejected |
| Minimum RAM during neural execution | NOT MEASURED: no execution |
| Source/latent lineage | Authenticated by the linked independent review |
| Fresh full checkpoint admission | NOT RUN; prior verified identity retained |
| Native neural attempt performed | NO |
| Convolutions/subdivisions completed | None; learned child populations unknown |
| VRAM peaks / GPU teardown | NOT MEASURED |
| Natural/forced owned-worker settlement | NOT APPLICABLE: no worker launched |
| Ten-tensor artifact readback | NOT RUN |
| Conditional CPU mesh / inspection images | NOT RUN |
| Overall generation | NOT QUALIFIED |

The previous consumed marker, all nine failed-attempt evidence files, exact RAM
rejection and raw classifications remain unchanged. Existing zombie artifacts,
100 protected identities and historical source/environment observations are
preserved within the inherited audit scope. No decoder, tiling or policy source
was changed. Private data remain local.

- [Actual memory measurements and evidence hashes](amd-zombie-ram-headroom-evidence/memory-readiness.json)
- [Executed read-only measurement source](amd-zombie-ram-headroom-evidence/memory_readiness.py)
- [Independent lineage, preservation and outcome review](amd-zombie-ram-headroom-evidence/preservation-lineage-review.json)
- [Remote branch preservation](amd-zombie-ram-headroom-evidence/remote-preservation.json)

Private evidence directory:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-ram-headroom-35b3c70`.

Next action: make at least 22 GiB of Windows physical RAM available, then request
a fresh readiness assessment. This report does not authorise another launch or
substitute the prelaunch target for the continuous 12 GiB live floor.
