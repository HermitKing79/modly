# Zombie D one-shot standalone generation handoff

Base `c801fb1ac81d1565fd040aea572054c8e12fac70`. **GO for one standalone D run, subject to fresh live admission.** Preparation does not reserve or run a neural attempt. The exact published commit and command are supplied with delivery. Run once in normal, non-administrator PowerShell after closing Codex, ChatGPT Desktop and browsers.

## Authenticated input

The accepted D source is unchanged: 642,361 bytes, 1216×1294 RGBA; SHA-256 `9ac0364f409077cd3fdeb853a1baed3b851c56f4fdc539c9fde73e0e3ab79db9`. Its losslessly encoded worker PNG has the same hash. The pinned original CPU method reproduces the 980×980 crop (pixel SHA `00cbe7f07bf7a4c13df269a6da227c32b1bdbb3cc30d2168b66ba9c1d4f6d5b4`) and accepted 512×512 raster (pixel SHA `4738b7f1f3f28a3ad1747e5739ab4b6bae9b51ea78c6ff022df578415e3f4e22`). The new CPU input contract hashes to `1bd85638188a2d4b95b5926d51f95efc33686b4052890d207103a006af5c2092`.

The [reference contract](amd-zombie-d-reference-contract.json) authenticates six saved D qualification files and the original published assessment. It also pins eleven existing C view/receipt files. These hashes are rechecked before generation admission. Only D's exact private source and CPU input contract use the inherited two-file hash-authority exception; executable code/configuration still uses the original committed-file verifier.

The prepared D occupancy leaf contains **only** `source-image.png` and `input-contract.json`. It contains no conditioning, sampled occupancy, shape latent, decoder output or marker. A and C neural artifacts cannot satisfy D's source/contract/feature/shape lineage.

## Newly generated work

| Stage | Qualified bound |
| --- | --- |
| Fresh D DINO conditioning | 1 forward |
| Fresh D occupancy sampling and decoder | 12 steps / 22 sampler forwards, then 1 occupancy-decoder forward |
| Fresh D Shape-SLat sampling | 12 steps / 21 forwards from D occupancy and D conditioning |
| Fresh D geometry decode | 1 approved decoder forward; existing large-convolution tiling, no support extension |
| CPU postprocessing | One unchanged raw dual-grid extraction, exact PLY readback, eight flat-shaded CPU Blender views; no repairs, smoothing, remeshing or textures |

Canonical seed 20261013, guidance 7.5, schedules, checkpoint identities, normalization, precision and existing AMD backends are unchanged. The D handoff/stage adapter are explicit case copies of the C workflow; the stage adapter and PowerShell launcher are identical after case identity/routing substitution. Old implementations remain untouched. The D Blender entrypoint retains `sys.dont_write_bytecode=True` before third-party imports and accepts no recovery/foreign mode. Its lighting, cameras, framing and raw-geometry import remain those of the qualified renderer.

## Safeguards and execution

`Invoke-ZombieDGeneration.ps1` defaults to no child process. Explicit `-Run -ExpectedCommit <exact published SHA>` authenticates the Windows interpreter and current source commit, then requires new, exclusive D output paths. Python's default is also a no-work dry run.

Admission obtains three fresh `GlobalMemoryStatusEx` available-RAM readings, four seconds apart, each **≥22 GiB**. Another **22 GiB** check occurs immediately before pipeline/stage reservations, including the final geometry decoder marker after original CPU preparation. **24 GiB remains preferred only.** The **12 GiB** live-host floor remains active. Original GPU admission/allocator/free-VRAM limits, D: and guest/output disk floors, per-stage deadlines, process ownership, Jobs/GO/EOF and cleanup are unchanged.

The original bounds remain: 13 GiB initial free VRAM, 12 GiB allocator ceiling, Shape-SLat's 3 GiB live free-VRAM requirement; 4,430 initial / 1,000,000 intermediate / 1,842,968 final support rows and 32,768-row decoder tiles. CPU extraction/render/review retain 6 GiB process commit, 12 GiB host RAM, 2 GiB free output disk, 180 seconds per stage and four rendering threads. D's support may differ from A/C; oversized output aborts instead of extending any allowance.

Any failure stops later stages. There is no retry, fallback setting, alternate destination or automatic repair. Once a destination or attempt is consumed, preserve its evidence and stop for review. Do not rerun the command to work around a guard.

## Private outputs and visual review

All new destinations are absent at delivery, apart from the two-file prepared occupancy leaf:

- `artifacts/amd-zombie-d-generation-9ac0364f-v1` — fresh admission/gate/attempt/pipeline evidence.
- `proof/docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-d-9ac0364f-v1` — D-only conditioning, occupancy, shape and decode evidence.
- `artifacts/amd-zombie-d-comparison-9ac0364f-v1` — D raw PLY/NPY, eight new views and review outputs.

Under the comparison root, `review/paired-review.html` and `review/paired-eight-view-contact-sheet.png` compare A/D. `review/three-way-review.html` and `review/three-way-contact-sheet.png` add the already completed C views. A and C PNGs are copied byte-for-byte; **no A/C generation, extraction or rendering** occurs. Their source mesh/view hashes and exact shared camera locations/scales are checked. D is not independently recentered or refitted to improve its appearance; any clipping is reported.

Five explicit human review criteria cover fingers/hands, toes, intended wounds, black speckles/openings, and overall body fidelity/surface defects. Dark pixels can be shadows or backfaces, so they are not automatically counted as holes. Preserve visible wound rims/cavities; fewer defects caused by lost geometry is not success. Full-body views may leave some digits unassessable. Final success means generation and matched evidence completed; the quality winner remains **UNASSESSED** until the actual images are reviewed.

Preparation validation uses mocked safety tests, real read-only CPU image/profile wiring, original Git source verification, and a harmless Blender NumPy-import prefix probe. No model imports, weight loads, WSL launches, neural stages, actual D mesh extraction/rendering or attempt reservations are executed during preparation. Images, tensors, meshes, private telemetry and failure logs remain local. Only new safe source/tests/text are published on the isolated branch.
