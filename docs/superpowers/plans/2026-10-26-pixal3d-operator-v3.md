# Pixal3D operator-only v3 preparation plan

User's detailed task authorizes analysis, offline implementation/tests and review only. Live v3 execution needs separate user approval.

## Evidence and policy

V2 provisioned13 exact locked wheels successfully, with natural cleanup and unchanged cache/baseline identities. Its operator preflight failed before Popen at21.184074 GiB: there is no second-launch or HIP peak measurement. Cold-start host availability declined2,360,119,296 bytes; this is an observation, not a worst-case WSL bound. Sampled provisioning RSS201.367 MiB missed the child's349.887 MiB OS peak, illustrating polling limits.

Propose only for this tiny model-free operator proof: ready/startup minimum18 GiB = original12 GiB live floor + existing6 GiB sampled RSS budget. Prelaunch three readings four seconds apart plus final:21,687,472,128 bytes, adding the observed cold-start decline to18 GiB. No threshold is based on an invented operator peak. Keep original neural22 GiB gates and24 GiB advisory policy unchanged, together with all disk/VRAM/process/time/owned-cleanup protections. This proposal does not guarantee a physical-memory envelope or32 GB fit.

## Implementation

- [x] Snapshot v2 evidence, cache identities and historical refs. Pin accepted provisioning receipts in an additive v3 contract.
- [x] RED/GREEN tests for derived policy, failed admission/no retries, operator-only routing, immutable environment and fresh temp paths, authentication and ownership.
- [x] Add versioned PowerShell/host/controller/operator scripts; reuse existing numeric checks unchanged. No provision/download entrypoints. Require explicit future approval switch, exact commit and all gates before operations.
- [x] Implement authentication of installed v2 runtime from trusted cached-wheel manifests, metadata, receipts and critical hashes before GO. Keep all writes/caches in new temporary/evidence trees. Verify before/after environment identities during the future approved run.
- [x] Offline blocked-action tests and real no-execution dry-run; independent whole-patch review; preservation verification and safe-text publication preparation on new branch. Final publication SHA/heads are recorded in the private receipt and user handover.

## Scope and review focus

No WSL, live admission, Torch import, GPU, model files, installation, global settings or original-environment writes in Codex. A runtime PASS would qualify only synthetic operators. Review must distinguish sampled limits from physical guarantees, reject environment writes, and ensure failed Windows-client waits never bypass owned Linux cleanup. Prepared artifacts are pending separate user authorisation.

## Closeout

79 focused tests pass (22 v3,57 retained), including red/green regressions for transitive publication pins, native/startup inventory aliases and virtual-environment configuration. Independent review found no remaining substantive blockers. Both original22 GiB neural requirements and all old source files remain unchanged. V3 is a proposed handoff, not a live authorisation or measured HIP fit.
