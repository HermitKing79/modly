# Minimal native Windows supervisor qualification

Base: `df30cb32e87ab528d9c9961ea0953ec4798f6529`.
Branch: `codex/amd-minimal-windows-supervisor`.

The attached user specification authorizes one nonrecursive CPU pair after source
review. It selects the design and execution sequence. This plan records that
design without introducing a production observation authority.

- [x] Authenticate the actual production runner/bootstrap, prior retired-process
  correction, outside-sandbox writer qualification and failed recursive fixture.
  Capture protected100, prior evidence, cache and remote-head preservation.
- [x] Add `fixture.py` and focused mocked `test_fixture.py`: standard-library
  worker, exact context/nonce readiness, original stdin CANCEL and cooperative
  acknowledgement; sentinel receives an independent exact success-release file.
  The fixture spawns no processes. Its readiness has no five-second subdeadline.
- [x] Add `qualification.py`, `test_qualification.py`, and `outside_command.ps1`.
  Use actual `services.amd_stage_process.OwnedStageRunner` and bootstrap unchanged.
  A contained coordinator runs two independent runner instances; a private outer
  production Job bounds their aggregate. Only transitions and original result
  dictionaries are saved. No extra per-snapshot trace or observer modification.
- [x] Run only focused mock tests and static preflight. Independently review exact
  source hashes, invocation, isolation, cancellation, failure reporting and gates.
- [x] Once approved by source review, invoke the wrapper exactly once through
  supported outside-sandbox approval. Verify the qualified Windows user and
  disabled admin role. Preserve exclusive evidence even on failure. No retry.
- [ ] Independently examine actual native observations, acknowledgements, timing,
  complete result dictionaries and empty Jobs. Publish source-safe report and
  hashes; retain identities/nonces/raw traces privately. Verify GitHub and all
  preceding branch heads, then stop.

Unchanged limits: 6 GiB process AND Job, 12 GiB free Windows RAM, 2 GiB destination
disk, 25-second stages, maximum90-second outer execution. Last-resort forced
containment is mandatory on failure and can never count as natural PASS.

Review focus: a cached identity must match a currently enumerated PID and exact
creation/Job identity; worker readiness alone is insufficient; no sentinel release
before natural target settlement; foreign controls fail closed; diagnostic write
failure cannot replace a primary exception. Synthetic completion contexts are
explicitly test-only and never supplied to UpstreamJob or a product generation.

Decisions: the bootstrap only forwards CANCEL to a Windows child, so independent
sentinel success uses a nonce-bound exclusive file (as in the prior fixture).
Job identity is verified from actual supervisor snapshots, never invented in a
worker file. The host launch token is explicitly distinguished from the native
Job owner/handle. Production sources and observer cadence remain unchanged.
