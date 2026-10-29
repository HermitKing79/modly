# One-shot WSL disk survey implementation plan

> Execute inline; independent review of the whole change before publication.

**Goal:** Prepare, but do not execute, one named-distro read-only survey.
**Spec:** Current Phase 5 directive and the reviewed local disk audit section 4.
**Approach:** Reuse the existing admission/Job/pidfd/allocation primitives.
Earlier Phase 5 contains only admission, so a small metadata worker is necessary;
there is no previously executed complete survey to relabel as qualified.
No environment, decoder, model, legacy helper or historical evidence changes.

Files: `amd_wsl_disk_survey.py` (Windows owner),
`amd_wsl_disk_survey_guest.py` (stdlib metadata worker/owned cleanup),
`Invoke-ModlyDiskSurvey.ps1`, focused tests and a textual handoff report.

- [ ] Add offline tests; observe missing implementation failure.
- [ ] Implement exact paths, explicit approval/default no-work, immutable-source
  authentication, three22GiB/four-second admission, final22 before WSL and GO,
  12GiB live/50GiB D:/2GiB output floors; no retry.
- [ ] Implement bounded120s/512MiB/1MiB metadata collection. Descriptor-pinned,
  no-follow, same-device and mount-excluding walk; permissions/incomplete scope
  explicit. Guest writes exclusively to fresh Windows receipts.
- [ ] Reuse current pidfd/nonce/group/start identity checks for owned cleanup;
  no numeric group signalling or distro shutdown. Fail closed if unavailable.
- [ ] Test guards, routing, partial measurements, filesystem escapes and cleanup;
  run only mocks/synthetic fixtures and no-execution dry runs.
- [ ] Independently review; verify original shape/evidence and all old refs;
  publish safe source/text on a new branch and verify remote commit.

Review focus: consumed paths, active distro/unknown ownership, permission/PID
races, misleading VHD file-length/allocation labels, incomplete directory counts.
No live admission, WSL command, GPU, installation or actual survey during prep.
