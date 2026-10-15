# Brick neural decoder row-tiling recovery — BLOCKED

**The CPU-qualified tiling implementation is published. One genuine attempt was stopped by the unchanged live Windows RAM floor before reaching any new tiled site. No complete neural output was verified or delivered; no retry was made.**

Starting commit: `ebeb48cd9b7a1b8919bd4eff956b65027e96def9`; branch `codex/amd-repeatable-generation-v1`. Exact execution harness published/GitHub-verified before launch: **`e0388f8e15bdc3355de0505459be1f0bf6742051`**. The prior 33/40 decode and every historical result remain unchanged.

## Implemented and CPU-qualified

Only original convolutions 34–39 are added to the row-tiling route: `blocks.2.8.conv2`, `blocks.3.0.conv`–`blocks.3.3.conv`, and `blocks.3.4.conv1`. It reuses `chunked_final.row_chunked_explicit`: full global neighbour indices/source table, original row order, weight layout and BF16 mm/addmm arithmetic. Tiles never become independent supports. Original 64-channel final tiling is byte-unchanged.

Admission includes the complete output, full native map/ratio-2 hash when uncached, retained caches/model/live tensors and original downstream allowances. Only im2col/gather/index/mask/GEMM temporaries become row-bounded. After map/output allocation, only their already-live allowances are removed from subsequent plans. Every tile still passes the original live predicate. The first genuine map and exact cache/coordinate identities are required across all six sites; no extra native calls or autotuning are allowed.

The authenticated `large_convolution_tiling` manifest opt-in and `--tile-large-convolutions` flag default off. Historical canonical documents/hashes and hook defaults remain unchanged. Strict 292-tensor loader, 216 BF16 / 76 F32 configuration, original 40 convolutions/eight C2S/five native maps, source/binary pins and restore/release acceptance are retained.

At 438,684 rows, 32,768-row tiles give 14 tiles per added site (last 12,700 rows). Padded pre-operation plans are:

| Sites | Padded visible plan, bytes |
|---|---:|
| Conv34,128→128, uncached |2,207,681,140|
| Conv35–38,128→128, cached |2,981,958,400|
| Conv39,128→512, cached |3,434,552,320|

Unchanged: 12 GiB allocator, 13 GiB initial free VRAM, 2 GiB device-free reserve, 1 GiB opaque allowance, 1.25 plan padding, 12 GiB live Windows RAM / 50 GiB D: floors, 32,768 maximum tile rows, 1M intermediate/1,842,968 final rows, 60/120/240/480-second native/CPU/GPU/host deadlines and owned containment. Actual fourth learned support remains unknown.

**57 focused CPU checks passed**: 27 decoder/admission/contract, 13 large-tiling arithmetic/lifecycle/security, 17 host. Independent reviewer reran 13+2 checks successfully. Random F32 128→128/512 calculations match the original pinned Linux forward, untiled reference and independent coordinate-derived F64 convolution within 1e-5; exactly representable BF16 fixtures match exactly. Coverage includes cross-tile neighbours, invalid sentinels, uneven tile ends, bias/no-bias, complete rows, cache reuse, source identity, descriptor restoration and negative dimension/site/memory cases. CPU results do not prove GPU bitwise equivalence or support fit. A preliminary bundled-Python host test lacked safetensors; the existing AMD interpreter passed without installation or a workaround.

[CPU receipt](amd-brick-row-tiling-evidence/cpu-qualification.json), [independent pre-execution review](amd-brick-row-tiling-evidence/preexecution-review.json).

## One actual attempt

Reused the authenticated saved F32[4430,32] brick shape and unchanged I32[4430,4] coordinates. Denormalized file SHA-256: `e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808`; coordinate SHA-256: `17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c`. No DINO, occupancy or sampling rerun occurred.

Host preflight verified 87 committed dependency files and the genuine checkpoint SHA-256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`, without model/GPU/WSL execution. The actual worker used the same RX 9070 XT / gfx1201, existing WSL Python 3.12.3 / Torch 2.9.1 / ROCm 7.2.1 and qualified FlexGEMM; all three TunableOp controls/flags remained false in the durable journal. Initial free VRAM was 14,945,849,344 bytes, exceeding 13 GiB.

With `WORKSPACE_DIR` set to this checkout and the existing Windows AMD Python with `-B`, execute-once routing was:

```text
amd_generation_decode.py --shape-evidence docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/shape --evidence docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/row-tiling-ebeb48c/decode --harness-commit e0388f8e15bdc3355de0505459be1f0bf6742051 --run-id run-brick-row-tiling-ebeb48c-v1 --tile-large-convolutions
```

| Measurement | Actual evidence |
|---|---:|
| Final pre-marker available Windows RAM |19,173,322,752bytes|
| Rejected live available RAM,14:20:19.516165UTC |**12,852,568,064bytes**|
| Unchanged required RAM |**12,884,901,888bytes**|
| RAM shortfall |32,333,824bytes|
| D:free at rejection |74,814,922,752bytes; admitted|
| Highest sampled Torch allocated/reserved |1,047,443,968 /1,086,324,736bytes|
| Supervisor elapsed |41.125s|

The worker synchronized model placement, entered one genuine parent forward, returned from the input projection on the host, and returned from the first genuine native neighbour-map call at `blocks.0.0.conv`. The latter is explicitly a **host return, not GPU synchronization**. No convolution-completed, C2S-completed or large-tile event was confirmed. The new tiling code therefore remains **CPU-qualified, GPU-unverified**. True peak VRAM and completed-forward runtime are unavailable.

Windows monitoring remained healthy. Mandatory containment forcibly emptied the Windows Job **4→0**, and named Ubuntu termination returned 0. This was **forced cleanup, not natural settlement**. The final worker report and neural artifact are absent from the host mirror; zero allocated/reserved release was not measured. No successful release is claimed. No native or numerical incompatibility was demonstrated.

Raw supervisor `UNKNOWN` and adapter `FAIL` classifications are preserved unchanged; the milestone verdict is **BLOCKED by live host RAM admission**. The legacy summary's minimum field tracks successful samples only; the exact rejected measurement above comes from the durable resource journal/error.

[Actual result and raw-evidence hashes](amd-brick-row-tiling-evidence/execution-result.json), [exact execution journal](amd-brick-row-tiling-evidence/execution-phases.jsonl), [resource journal](amd-brick-row-tiling-evidence/resource-admission.jsonl), [read-only host preflight](amd-brick-row-tiling-evidence/host-preflight.json), [independent outcome review](amd-brick-row-tiling-evidence/outcome-review.json).

Full local evidence remains under `docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/row-tiling-ebeb48c/decode`; original saved shape remains in `.../occupancy/shape/run/scratch/`. The owned stopped guest scratch is preserved, not restarted for additional inspection. There is no verified new neural output path/hash to deliver.

## Preservation and one next action

[Before](amd-brick-row-tiling-evidence/preservation-before.json) and [after](amd-brick-row-tiling-evidence/preservation-after.json) verify protected files, AMD edits/backups, selected environment/native identities and saved inputs. No package, driver, model, installed source or historical artifact was modified. Generated model data remains outside published Git files. Main and all other historical branch heads are checked at final publication; no merge/force-push.

**Next:** free additional host RAM, then separately authorise one identical decode using the already-published harness and saved shape. Retain the 12 GiB live floor and every GPU/support limit. The observed shortfall does not establish the RAM needed for a complete forward, so extra headroom must be measured rather than assumed. No further implementation, neural attempt, extraction, material work, GLB or Generate integration was performed.
