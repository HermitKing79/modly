# One canonical shape-SLat sampling sequence

Base: `5eb759d186b8f661c954b0a0b680370ff04057cf`. User amended the original22-forward requirement to the canonical21 forwards (12 positive/9 negative) after source-schedule inspection. All other parameters and protections remain unchanged.

Reuse the existing shape worker/strict loader, pinned genuine sampler, Job supervisor, Windows observer, logging and workspace teardown. Add a separate sparse sampling control and bounded artifact writer. The original single-forward route retains its one-call permit.

- [x] Verify initial Git/remote/environment/source/backups and historical artifacts; create dedicated branch.
- [x] CPU-test genuine sampler with synthetic sparse inputs: CFG, exact12/21 schedule, root/block counts, transfer scopes, shared cache, finite layout, normalization and resource collection. Pin source/config/feature/coordinate identities.
- [ ] Review complete candidate independently (passed); commit/push and verify exact harness before execution.
- [ ] Run one sequence only after fresh RAM/13GiB VRAM admission and healthy observation, using180/240/480-second deadlines and12GiB cap. Save raw and canonically denormalized F32 latents and exact coordinates; requirezero allocation release.
- [ ] Independently verify artifacts/counters/preservation, publish, and stop for approval. No decoding or further stages.
