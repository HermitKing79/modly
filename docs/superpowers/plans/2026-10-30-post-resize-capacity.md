# Post-resize capacity verification plan

**Goal:** Prepare one held read-only check of the already enlarged VHDX.
**Spec:** Phase 6C project lead directive; original resize runbook at d0779b8.
**Architecture:** Reuse the qualified observer, inventory and capacity-controller
handshake. Preserve the original probe and resize implementation; fork the small
guest probe solely to add explicit statvfs fields and optional ordinary-user
superblock reads. No resize route, privilege escalation or repair fallback.

- [x] Authenticate saved native return/output/settled container and capture a
  hash manifest without modifying any resize evidence.
- [x] Add failing tests for exact versus usable size, unavailable geometry,
  partial expansion, permission denial, corrupt/redirected device reads and
  prevention of resize dispatch. Implement only the required capacity path.
- [x] Preserve three >=22 GiB readings four seconds apart, final/startup22,
  live12, D50, C/E2 GiB, 512 MiB probe/observer limits, 30-second query checkpoint,
  guest alarm15/CPU10 and fail-closed natural owned process cleanup.
- [x] Distinguish full64 proof, exact unavailable, partial expansion and invalid
  results. A passing 5 GiB space gate does not prove exact filesystem geometry.
- [x] Run offline/synthetic suites, dry-run, independent review and preservation
  checks. Publish safe code/text on an isolated branch and a held exact command.

No live WSL, real VHDX/E: access, resize, decoder, installation or inference.
Known limitation: WSL startup itself can update the mounted filesystem; the
probe issues no writes and cannot promise byte immutability of a running VHDX.
