# Zombie preflight RAM diagnosis

**Attribution: UNRESOLVED. Diagnostic preflight: NOT RUN (inadequate baseline).**

Starting revision: `3f7af197c25349af95bc97fe70dd99c509b22b8a`. No decoder, admission policy or installed environment was changed. This milestone consumed no neural attempt, marker, Windows Job or WSL launch.

## Fresh admission

The unchanged qualified reader measured `GlobalMemoryStatusEx.ullAvailPhys` three times, four seconds apart, in the supported normal-user outside-sandbox context:

| UTC, 2026-10-24 | Available physical bytes |
| --- | ---: |
| 10:56:53.908728 | 20,933,840,896 |
| 10:56:57.909847 | 20,930,056,192 |
| 10:57:01.910257 | 20,937,019,392 |

Minimum availability was **19.493 GiB**. All readings failed the unchanged **22 GiB** requirement (23,622,320,128 bytes) and the preferred **24 GiB** operational headroom. The span was only 6,963,200 bytes; stable readings do not make insufficient availability safe. As instructed, no full diagnostic preflight followed. The raw reader's `NO-GO: INSUFFICIENT_HEADROOM` is an admission result, not proof of decoder incompatibility.

The user's Task Manager screenshot shows 1,720.7 MB in the ChatGPT/Codex group and 461.9 MB for a separate `codex`: approximately **2.2 GB of displayed application footprint**. This materially occupies RAM headroom. A snapshot cannot establish whether it grew during the earlier preflight; summed process memory is not exclusive physical-memory accounting. The image and raw process identities remain private.

## What the historical evidence proves

The authenticated prior attempt's last initial reading was 23,799,910,400 bytes at 10:20:46.998316 UTC. Its first preflight host reading, 711.387 seconds later, was already 22,027,182,080 bytes. That interval lost 1,772,728,320 bytes but lacks a process or phase memory timeline.

Importantly, that first host reading occurs **after Python imports and runtime construction**, including the first complete `source_inputs()` lineage validation. The subsequent recorded preflight repeats source validation, verifies committed sources and streams the decoder checkpoint. Availability then **increased by 58,392,576 bytes**, reaching 22,085,574,656 bytes at the final pre-marker check. The final 22 GiB check failed by 1,536,745,472 bytes before marker publication.

Therefore the old evidence does not demonstrate a RAM decline during the later recorded checkpoint-verification segment. It cannot distinguish early imports/constructor work, concurrent applications, Windows caching or other system activity. The app screenshot does not resolve that timing gap.

Static inspection found streaming checkpoint hashes (1 MiB blocks or `hashlib.file_digest`), bounded checkpoint headers and metadata-only loader construction. Reachable allocating work includes image preprocessing, saved conditioning/occupancy readers, bounded JSON and per-file Git/source comparisons. No whole-checkpoint `read_bytes()` allocation was found in this CPU path. These are source findings, **not measured RAM attribution**. No model weights were loaded.

## Verification and preservation

Independent static analysis authenticated the historical RAM receipts and source sequence. One bounded preservation collection matched the prior baseline: **100 protected identities / 78 bounded hashes**, nine original failed files, five saved shape inputs and the retained prelaunch evidence. Full checkpoint rehashing and broad qualification were not repeated after inadequate admission. Current source hashes, arithmetic and report consistency are independently checked in the linked evidence.

Preflight working-set/private-commit peaks, phase telemetry, concurrent memory changes and recovery after preflight exit were **not measured**, because the diagnostic worker was not admitted. All prior failures remain unchanged. Current remote-head preservation is recorded separately; private assets and screenshots are excluded from publication.

Evidence: [baseline](amd-zombie-preflight-ram-evidence/baseline.json), [historical source/timeline review](amd-zombie-preflight-ram-evidence/prior-evidence-analysis.json), [bounded preservation](amd-zombie-preflight-ram-evidence/preservation-before.json), [independent outcome review](amd-zombie-preflight-ram-evidence/outcome-review.json), [remote preservation](amd-zombie-preflight-ram-evidence/remote-preservation.json).

**Next action:** obtain adequate operating headroom, preferably 24 GiB available, then separately authorise one CPU diagnostic measured from before imports and runtime construction through process exit. Do not infer that closing this app would recover its displayed total, or that it explains the earlier decline. Neural execution remains separately gated.
