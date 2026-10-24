# Zombie RAM-headroom recheck — NO-GO: INSUFFICIENT_HEADROOM

Approved decoder base: `35b3c70de7aa61c514a4e0410af246bab880ba9d`.
Parent: `740edf52eeac436732556e10add51216150f6d1c`, preserving the previous
no-go report. Decoder, API and application source are unchanged from the
approved base. New branch: `codex/amd-zombie-ram-headroom-recheck`.

The unchanged, previously qualified Windows memory reader was reused with only
its private evidence directory rebound. Three `GlobalMemoryStatusEx` available
**physical** RAM readings, four seconds apart, all failed the 22 GiB
(23,622,320,128-byte) prelaunch requirement:

| UTC on 2026-10-24 | Available bytes | GiB |
| --- | ---: | ---: |
| 09:31:29.043321 | 20,732,739,584 | 19.309 |
| 09:31:33.043844 | 20,708,589,568 | 19.286 |
| 09:31:37.044398 | 20,710,785,024 | 19.288 |

The readings span 24,150,016 bytes. The shortfall is 2.691–2.714 GiB. Installed
RAM, commit, pagefile and VRAM were not substituted for available physical RAM.
The earlier 6,700,421,120-byte startup decline remains an observation, not a
future usage guarantee.

Read-only observations found individual Codex and ChatGPT working sets of
756,699,136 and 707,051,520 bytes respectively, with additional ChatGPT
processes. OS memory compression and Defender also had significant working
sets. These may share pages and are not exclusive physical-memory accounting.
Raw process identities remain private. No process was closed or terminated,
cache cleared, machine rebooted, setting changed or limit relaxed.

| Required outcome | This recheck |
| --- | --- |
| Source/latent admission | Fresh retained-file hash checks; prior full qualification reused |
| Fresh checkpoint/runtime admission | NOT RUN after RAM rejection; prior checkpoint identity retained |
| Native neural attempt performed | NO; no fresh marker or WSL/model launch |
| Convolutions/subdivisions | None; learned child populations unknown |
| Runtime minimum RAM, VRAM peaks, GPU teardown | NOT MEASURED |
| Natural/forced owned-worker settlement | NOT APPLICABLE; no worker launched |
| Ten-tensor artifact / CPU mesh / inspection images | NOT RUN |
| Overall generation | NOT QUALIFIED |

The failed neural attempt and earlier headroom assessment are preserved without
overwrite. All 100 protected identities and previous remote branch heads are
retained. No decoder or production source changes were made.

- [Fresh measurements and hashes](amd-zombie-ram-recheck-evidence/memory-readiness.json)
- [Reused measurement source](amd-zombie-ram-headroom-evidence/memory_readiness.py)
- [Independent preservation and readiness review](amd-zombie-ram-recheck-evidence/preservation-lineage-review.json)
- [Remote preservation](amd-zombie-ram-recheck-evidence/remote-preservation.json)

Private evidence:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-ram-recheck-740edf5`.

Next action: make approximately 2.7 GiB more physical RAM available and request
a fresh readiness assessment. No further assessment loop or neural attempt was
performed after this rejection.
