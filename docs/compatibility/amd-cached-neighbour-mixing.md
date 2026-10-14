# AMD cached non-identity neighbour-mixing proof

CPU preflight GO; single GPU result pending. Starting commit92cdb3fccdf6ccc60f3a241684aafe323c8c34fd. No further synthetic cases are authorized.

Only weights change: diagonal channel taps at v13(center)=1, v14(offset0,0,+1)=1/2 and v4(offset-1,0,0)=-1/4, all other weights0; features, coordinates, bias, native metadata, source, cache, helper, tuning, launch and safety limits retain the previous successful values. Coordinate axes are named by stored index to avoid ambiguous native W/H/D labels.

Independent tuple lookup verifies valid rows2049/1984/1025, boundary misses0/64/1024 and sparse misses0/1/0. There are3,081,216 valid noncentral terms and2,064,894 non-identity output elements. An integer numerator divided by4096 supplies an exact F32 reference, with actual numerator range[-6119,6085]. Each raw SplitK plane is separately predicted using the source partition and must also match exactly. References and final output stayF32; expected F16 rounding cannot hide discrepancies.

13 focused host regressions (7new,6previous identity-route) and13 installed constructor tests pass. Owned CPU acceptance exits0 in13.125s with GPU/native/AI weights false; all11 scratch artifacts read back and independently hash-verified. CPU references were never transferred to GPU. The original native cache/launch policy and supervisor remain unchanged, and the base worker's numerical hooks preserve its original default route.

Unchanged requirements: one launch, no warmup/retry;13GiB freeVRAM,12GiB allocator cap,1GiB current/peak observed smoke budget and45/120/180-second CPU/GPU/overall limits. Periodic healthy Windows observation, Job containment and zero-owned-PyTorch allocation/reservation release remain mandatory. Native allocations are not fully visible in PyTorch metrics; available device memory is separately measured.

Evidence: [CPU numerical/cache audit](amd-cached-neighbour-mixing-evidence/cpu-numerical-audit.json), [installed acceptance](amd-cached-neighbour-mixing-evidence/cpu-acceptance/results.json), [independent candidate review](amd-cached-neighbour-mixing-evidence/independent-review.json), [preservation baseline](amd-cached-neighbour-mixing-evidence/git-before.json). Separate [Linux comparison](amd-linux-route-comparison.md) is read-only and changes no experiment parameters.

The eventual result will establish only this cached kernel's signed two-axis neighbour contributions with diagonal channels. It cannot validate the native neighbour/postprocessor/full-wrapper path,40 decoder convolutions or four dynamic subdivisions. Stop synthetic convolution testing after this single attempt.
