# Brick neural decoder row-tiling recovery

Starting commit: `ebeb48cd9b7a1b8919bd4eff956b65027e96def9`, branch `codex/amd-repeatable-generation-v1`.

**CPU qualification passed; genuine decode not yet attempted.** This is a continuation of the saved new brick shape, not new DINO, occupancy or shape sampling. The earlier 33/40 admission failure and all historical evidence remain unchanged.

The explicit opt-in applies only to original convolutions34–39: `blocks.2.8.conv2`, `blocks.3.0.conv`–`blocks.3.3.conv`, and `blocks.3.4.conv1`. It reuses the established final tiler's complete global neighbour table, source row ordering, pinned weight layout and BF16 mm/addmm arithmetic. Only im2col, gathers, indices and GEMM outputs are row tiled. Full outputs, map/hash allocations, downstream allowances and retained caches remain accounted for. The final64-channel tiler is unchanged.

The authenticated run contract enables this route explicitly; historical default documents/hashes and hook behavior remain unchanged. Native map/cache identity is checked across all six sites. Original40convolutions/eight C2S operations/five native maps, strict292-tensor loader,216BF16/76F32 precision, source/binary pins, no-autotune/build restrictions and cleanup remain mandatory.

At the measured438,684-row support,32768-row tiles give14tiles per added site (last12700rows). Padded pre-operation plans: conv34 **2,207,681,140 bytes**, conv35–38 **2,981,958,400**, conv39 **3,434,552,320**. Live admission remains required before the whole output and every tile:12GiB allocator,2GiB device reserve,1GiB opaque headroom;13GiB initial free VRAM. CPU/GPU/host deadlines stay120/240/480seconds (native60). Fourth learned support remains unknown and must fit the unchanged1,842,968 final-row cap; intermediate cap1M remains.

Root focused CPU verification:27decoder/admission/contract checks,13large-tiling arithmetic/lifecycle/negative checks,17host checks. F32 reference tolerance1e-5; exactly representable BF16 fixtures match exactly. Global cross-tile neighbors, sentinels, bias/no-bias, uneven tile ends, cache/source identity, restoration, unsupported dimensions/sites and insufficient memory are covered. CPU tests do not prove GPU bitwise equivalence or final support fit. A preliminary host test used bundled Python lacking safetensors; the existing AMD interpreter passed without installation or a code workaround.

[Preservation](amd-brick-row-tiling-evidence/preservation-before.json), [CPU measurements/tests](amd-brick-row-tiling-evidence/cpu-qualification.json), [independent pre-execution review](amd-brick-row-tiling-evidence/preexecution-review.json).

One published candidate may execute one genuine decode from the saved F32[4430,32]/I32[4430,4] arrays. Stop on admission, numerical, native, timeout or cleanup failure. No mesh extraction, texture work, GLB or application integration belongs to this milestone.
