# One shape-SLat 512 forward

Goal: one genuine `SLatFlowModel` call using the preserved turret conditioning and source-faithful 32³ occupancy coordinates. No sampling or downstream models.

- [ ] Save preservation evidence; verify all historical branches/files, environments and AMD source backups.
- [ ] CPU-validate the saved features/native mask and exact upstream pooling/coordinate contract. Audit the local 640-entry BF16 checkpoint and weights-free real architecture, including nonregistered RoPE frequencies and explicit attention backend. Test strict synthetic loading and negative contracts.
- [ ] Reuse the existing supervisor/Job/observer, 180/240/480-second deadlines, 12 GiB allocator cap and zero-byte teardown. Review a narrow one-call worker independently and publish its exact code before execution.
- [ ] Only if every gate passes, run once with deterministic F32 noise, timestep 1000 (the canonical first Euler model timestep), and saved F32 conditioning. Preserve and independently validate the F32 sparse output and coordinates.
- [ ] Recheck preservation, publish measured result, and stop for approval.

Stop before GPU execution on unresolved checkpoint/source/dtype/native-operation or resource contracts. No source patch, remote factory, fallback model, downloads, compilation workaround or retries. Review must check sparse caches/RoPE cleanup, exact transfer identities, actual model/block call counts, and failure containment.
