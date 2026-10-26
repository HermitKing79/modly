# Pixal3D Standalone HIP Proof v2 Implementation Plan

> Execute inline with testing, then independent whole-branch review. User's detailed v2 decision is the approved scope; no live execution in Codex.

**Goal:** One user-invoked, cache-only model-free HIP proof with a separately reported WSL startup feasibility gate.

**Architecture:** Add versioned entry points and policy; preserve all v1 files. Reuse the qualified memory reader, admission function, Windows Job and exact operator definitions. Read the v1 payload directory without copying archives, downloading or altering it. New Windows/guest roots are exclusive. Guest exclusivity is checked by the controller before any environment creation; it cannot be checked in this session without violating the WSL exclusion.

**Tech stack:** Existing Windows Python 3.12.10, PowerShell, stdlib ctypes; qualified WSL Python 3.12.3 and unchanged pinned runtime, only during the future deliberate run.

## Constraints and design decisions

- Three fresh >=22 GiB Windows readings four seconds apart; final >=22 GiB before stages. Preferred24 is advisory. Existing12 GiB live floor and all disk/VRAM/process/deadline/cleanup limits unchanged.
- During WSL startup, additionally stop at the first sampled <22 GiB, before GO. No waiting for recovery or automatic retry.
- Observe Windows VM candidate processes read-only; access failures are explicit unavailable observations, never zero usage. A Windows client is not a VM. Linux meminfo/controller RSS recorded at readiness and exit. No complete memory attribution claimed.
- Owned nonce/PGID/start-time cleanup only; no distro termination, global settings, model files, inference or production integration.
- Preparation may hash cached files, write new evidence/code, run mocks and no-work dry-runs. It must not sample live admission, start WSL, install anything or import Torch.

## Tasks

- [x] Snapshot historical refs/evidence and independently hash all16 cache files with before/after metadata; verify new Windows/preparation paths absent.
- [x] RED tests: missing/changed cache cannot download/copy; original guards/pins unchanged; exclusivity; startup crossing blocks GO; unavailable VM observations remain explicit; separate reports and stop routing.
- [x] Implement v2 policy/cache verification, timestamped read-only telemetry, versioned host/guest/operator entry points and PowerShell no-work default.
- [x] GREEN focused regression suite with child/network/live-memory hooks blocked; harmless synthetic fixtures only. Run actual PowerShell no-execution default.
- [x] Independently review cache/isolation/startup guard/cleanup; fix proven issues with regressions. Reverify preservation and stage only safe code/text.
- [ ] Commit/push only the new branch; verify GitHub commit and preservation of old heads. Deliver exact user command, no execution.

## Review focus

Cache mutations/reparse paths; RAM crossing while waiting for ownership; no ownership receipt during early stop; missing/inaccessible VM metrics; partial result JSON; startup/provision/operator statuses on failures. Tests must exercise these boundaries without a real WSL process.

## Closeout

57 focused tests pass (27 new,25 v1,5 original admission). Independent review cleared the final patch after regression-tested cleanup/reporting fixes. Preservation checks cover3,346 historical files,100 protected identities,88 retained records,26 failed-attempt files,16 cache files and174 old local refs. Live results remain NOT_RUN. Final publication SHA/remote-head checks belong to the private publication receipt and user handover, avoiding a self-referential commit hash in this plan.
