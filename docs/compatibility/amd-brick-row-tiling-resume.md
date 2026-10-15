# Brick row-tiled neural decode resume — PARTIAL

**One unchanged-harness attempt completed 39/40 convolutions, including all six added tiled sites. It stopped safely before the fourth expansion because the genuine learned mask selected 1,957,891 children, exceeding the unchanged 1,842,968 final-row cap. No complete neural geometry output was produced. No retry or code change occurred.**

Execution revision: GitHub-verified `43a637d51f668c787a2335606a4516a54145ab61`, branch `codex/amd-repeatable-generation-v1`. All qualified decoder/host source hashes match the earlier reviewed candidate. Its 57 focused CPU checks were reused after identity verification; no test suites were unnecessarily repeated.

## Admission and unchanged inputs

Three GlobalMemoryStatusEx readings five seconds apart measured **20,128,989,184**, **20,402,966,528** and **20,469,538,816bytes** available (18.75–19.06GiB), increasing rather than falling. Final pre-marker availability was20,466,962,432bytes. All recorded host RAM/disk checks passed.

Reused the same authenticated 4430-row brick shape and complete provenance. No DINO, occupancy or sampling rerun:

| Input | SHA-256 |
|---|---|
| Denormalized F32[4430,32] |`e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808`|
| I32[4430,4] coordinates |`17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c`|
| Complete 292-entry decoder checkpoint |`e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`|

Actual runtime: WSL Python 3.12.3, Torch 2.9.1+rocm7.2.1.gitff65f5bc, HIP 7.2.53211-e1a6bc5663, RX 9070 XT / gfx1201. Strict CPU loading passed; original 216 BF16 / 76 F32 runtime precision and genuine native identity were retained. All three TunableOp controls/flags stayed false.

Limits remained:13GiB initial free VRAM;12GiB allocator;2GiB device reserve/1GiB opaque headroom;12GiB live Windows RAM/50GiB D: floor;32768maximum tile rows;1Mintermediate/1,842,968final rows; original native/CPU/GPU/host60/120/240/480-second bounds, Windows observer, Job ownership, binary authorisation and named-distro cleanup.

With exact `WORKSPACE_DIR` and the existing Windows AMD Python `-B`, the only execution was:

```text
amd_generation_decode.py --shape-evidence docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/shape --evidence docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/row-tiling-resume-43a637d/decode --harness-commit 43a637d51f668c787a2335606a4516a54145ab61 --run-id run-brick-row-tiling-resume-43a637d-v1 --tile-large-convolutions
```

## Real result

All six ordered128-channel sites completed **14 tiles each: 84 entries and 84 returns**, full 438,684 rows per site, original global indices and cache identity. No cache-release fallback was needed. Both temporary method patches were restored. Original count: **39/40 convolutions**, **6/8 C2S operations**, **4/5 native global maps**, zero final64tiles.

Completed support: **4430→21391→97313→438684**. At `blocks.3.4.updown`, call1, genuine boolean mask shape[438684,8] selected **1,957,891 children**, mean4.463101per parent. Applicable final admission is `min(8*438684,1842968)=1842968`; excess **114,923 rows**. The journal's generic 1M row field describes the intermediate limit; the actual fourth-stage guard uses the final 1,842,968 cap. It rejected before either fourth-stage expansion and before its memory plan could be admitted.

Exact exception: `AdmissionStop: PARTIAL: fourth C2S differs from the proven full support/widths`.

| Measurement | Actual |
|---|---:|
| Initial device free |15,138,869,248bytes|
| Peak Torch allocated/reserved |3,354,263,040 /4,389,339,136bytes|
| Lowest recorded available Windows RAM |13,806,620,672bytes|
| Lowest D:free |74,522,365,952bytes|
| Strict CPU decoder load |11.779s|
| CPU high-water |2,786,889,728bytes|
| GPU-stage elapsed at fourth-count stop |55.047s|
| Worker /supervisor elapsed |81.126 /91.375s|
| Post-release Torch allocated/reserved |**0 /0bytes**|

GPU-stage elapsed includes diagnostics and is not a completed-forward timing. No full output exists for finite-feature/readback acceptance. This is a validated support admission stop, **not a demonstrated numerical/native failure or RAM/VRAM shortage**.

GPU tensors and CPU model/parameters/component were collected. Windows observation remained healthy. The Job was naturally empty, no forced cleanup occurred, worker exit1 recorded the deliberate admission stop, and named-Ubuntu termination returned0. Prior GPU work was synchronized during cleanup. Device-free after release was14,986,571,776bytes; this does not claim zero device-global use.

[Exact result and 11 raw-file hashes](amd-brick-row-tiling-resume-evidence/execution-result.json), [full durable execution journal](amd-brick-row-tiling-resume-evidence/execution-phases.jsonl), [host resource journal](amd-brick-row-tiling-resume-evidence/resource-admission.jsonl), [independent outcome review](amd-brick-row-tiling-resume-evidence/outcome-review.json).

Windows-accessible raw evidence is in `docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/row-tiling-resume-43a637d/decode` relative to the existing proof checkout. Original saved inputs remain in `.../occupancy/shape/run/scratch/`. No new neural-artifact path/hash can be supplied.

## Preservation and next step

[Before](amd-brick-row-tiling-resume-evidence/preservation-before.json)/[after](amd-brick-row-tiling-resume-evidence/preservation-after.json):100protected files, all five saved brick files, both earlier attempts, AMD source edits/backups, archives and selected environment/native identities match. No decoder, dependency, installed environment, driver or source change; no cap relaxation or automatic retry. [RAM samples](amd-brick-row-tiling-resume-evidence/ram-stability.json), [unchanged-harness review](amd-brick-row-tiling-resume-evidence/preexecution-review.json). All 76 preexisting branch heads, including main, are verified at publication.

**Single next step:** separately qualify the measured **1,957,891-row** final C2S, existing64-channel row tiler and complete artifact-reader memory/output bounds on CPU under the **unchanged 12 GiB allocator and reserves**. A successful bounded audit could support a specifically approved final-support contract; this report does not authorise raising the cap or another forward. Preserve the full learned support—no truncation, resampling or fidelity reduction.

Stopped before final neural output, mesh extraction, materials, GLB or Generate integration.
