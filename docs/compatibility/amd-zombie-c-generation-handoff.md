# Zombie C: qualified reference and one controlled generation handoff

Base `ae43bacf0b61f282382f96160fa18ed1cc957cb8`. **GO for one guarded standalone launch after final verification**, not a claim of live RAM/VRAM admission or improved generated geometry. The user runs the published command with desktop apps closed. No real generation, model-weight loading, WSL launch or neural reservation occurred during preparation.

## Reference qualification

The new 778,347-byte, 1086×1448 RGBA PNG hashes to `60862722c989772ad973870890601a6a95ed9a5cd5a9150a378e989db35ad437`; decoded pixel SHA-256 `d661bbdbbeb18e6de9235facfc83d23fb2410904fd56dc452c659cba6217a0d6`. Its 1,218,235 alpha-zero pixels confirm genuine transparency. Foreground alpha percentiles 252/253/253 are near-opaque. The raw source's saturated red/yellow background fringes have maximum alpha only 2/5; none survive as saturated fringes in the effective conditioning raster, and visual inspection found no unwanted visible edge contamination. No mask or RGB edits were made.

The exact pinned alpha-only method (source SHA `faade40274e44823e72e1659b8ced96f8815ac9e08b3abb89271077c4394982b`) produces a 970×970 crop. Its original extractor resize gives a 512×512 pre-normalization raster with pixel SHA `2ee336402eb84a3405b135515c69d255380369fcc8c9b7572948f5c2f5b0b989`. The original baseline crop/512 raster were reproduced against their known hashes. This CPU reconstruction does not execute DINO.

| Observation at 512 | Original versus C |
|---|---|
| Fingers | C shows splayed, open-palm digit silhouettes and multiple background gaps. Selected C low-alpha gap probes are 3–6 pixels; original fingers mostly overlap. These are explicit local probes, not an automatic per-finger score. |
| Toes | More drawn toe definition in C, but mostly shaded grooves and short exterior notches. Selected tip notches are one pixel; each foot remains a connected foreground region. Separate generated toes remain unproven. |
| Framing/silhouette | Foreground width 490 →397 pixels, about 81% of the original; C has no zoom advantage. Unregistered mask IoU 0.8316 is descriptive, not a shape-quality score. Both images reach the bottom frame, making tiny toe notches sensitive to cropping. |
| Face/wounds/body | Recognizable face, clothing and shoulder/chest/abdomen/thigh/shin wounds remain visible. Their rendering, shape, pose and proportions also changed. Depth and cavity walls remain ambiguous from a single image. |

This is useful new geometry information for fingers, not merely a color change. It is a comparison of revised source renderings, not isolated causality for finger placement. No guarantee is made that C will produce better hands, toes or wounds.

Local-only evidence: `amd-zombie-c-reference-20261025-v1/comparison-512.png`, `comparison-features.png`, `zombie-c-checker.png`, exact PNG snapshots and measured JSON. Keep all private references and future output local.

## One pipeline, fresh lineage

The default-dry-run [launcher](../../scripts/compatibility/Invoke-ZombieCGeneration.ps1) requires `-Run -ExpectedCommit <exact published SHA>`. It binds the qualified Windows interpreter, current HEAD and new source files. The existing generation/decoder implementations are unchanged.

The new committed [text authority](amd-zombie-c-reference-contract.json) binds the exact two private image/CPU-contract files. Only those two data files use hash authority instead of publication as Git assets. Every inherited code/configuration dependency still goes through the original bounded Git verifier; the private source/contract hashes are rechecked around it. A real CPU-only compatibility check exercised the original image reader and profile routing with model imports, model-file access, child processes and network denied.

One exclusive pipeline requests these existing stages in order, with natural owned settlement required before proceeding:

1. Fresh DINO conditioning: **1 forward**.
2. Fresh sparse sampling: **12 steps /22 forwards**, followed by **1 occupancy-decoder forward**.
3. Fresh Shape-SLat sampling from that occupancy and conditioning: **12 steps /21 forwards**.
4. Approved geometry decoder: **1 forward**, same large-convolution tiling, no support extension.
5. Bounded CPU raw mesh extraction and eight camera-matched original/C view pairs.

Original canonical seed 20261013, guidance 7.5, intervals, time rescaling, normalization, precision, checkpoints and decoder operations remain unchanged. No old zombie conditioning, occupancy, latent or neural output can stand in for C. Any rejection or failed stage stops later work; no retries, alternate settings, repair, smoothing, remeshing or textures.

## Admission and comparison

The launcher obtains three fresh `GlobalMemoryStatusEx` readings four seconds apart, each **≥22 GiB**, before importing stage modules. Another **22 GiB** check occurs immediately before pipeline/stage reservations; the geometry decoder repeats its final guard after its original CPU preflight. **24 GiB stays preferred only.** The preserved **12 GiB** live host floor and original observers, GPU admission/caps, disk checks, deadlines, owned Jobs/GO transport and named-distro cleanup remain decisive. Windows GPU-stage observation also checks the same live floor.

Unchanged envelopes include 13 GiB initial free VRAM, 12 GiB allocation cap and Shape-SLat's 3 GiB live free-VRAM requirement; 4,430 initial, 1,000,000 intermediate and 1,842,968 final rows; 32,768-row decoder tiles. Oversized C support aborts. No brick-specific allowance is borrowed. Original D:/guest/output disk floors and per-stage deadlines are retained. CPU comparison uses 6 GiB commit, 12 GiB available host RAM, 2 GiB output disk, 180 seconds per stage and four CPU rendering threads.

The [comparison plan](amd-zombie-c-comparison-plan.json) freezes the original raw PLY and saved cameras at azimuths 0,45,...315°. C is not recentered or refitted independently. Original neutral lighting, flat shading and 768×768 views are retained; clipping is reported. Future outputs include exact native view pairs, a contact sheet, a local review page and separate criteria for finger separation, toe definition, unintended holes, intended wound preservation and surface defects. Counts are secondary. Visual quality stays **UNASSESSED** until actual outputs are inspected; occluded/subpixel detail must be marked unassessable.

Current verification: 44 new mocked/native CPU tests and 42 selected inherited policy tests passed. Python/PowerShell defaults perform no generation. One initially selected inherited CPU-framework test failed in general test Python on missing `easydict`; no weights or generation were involved. Two framework-import fixtures were excluded from the final policy-only selection, which explicitly blocks model imports; no dependency/environment was changed. Full-model inference is reserved for the user-run handoff.

Exact commit, independent final review and preservation checks are recorded in the companion verification receipt. The invocation supplied with delivery is the only intended execution; use normal non-administrator PowerShell after closing Codex, ChatGPT Desktop and browsers. Preserve any rejection/failure evidence and stop for review rather than rerunning the command.
