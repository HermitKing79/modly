# Closed-solid support/logit census — Phase 9

30 October 2026. **Census complete; NO-GO for a local single-flag correction.** Original geometry remains NO-GO and unchanged. This conclusion is limited to the measured single-quad incidence screen; it does not rule out every multi-quad reconstruction.

## Scope and authentication

The original ten tensors, all payload hashes/linkages, 3,888-row input provenance, raw NPY/PLY identity and exact original quad/split/position correspondence were reauthenticated. The complete Phase 8 evidence manifest and published-source hashes at `9795e16201dda8c7e4faa308ecd85a9f448a08c6` also passed. The raw neural SHA remains `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975`; PLY SHA remains `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3`. Checkpoint contents were not loaded or read.

Exactly **1,421 boundary plus 519 triple-junction edges**, identified by their original global vertex-pair keys, produced **7,760 candidate observations and 6,481 distinct lattice quads**. Private per-edge records include endpoints/positions, all four candidate anchors/axes/corner coordinates and original indices, grid bounds, anchor/corner availability, original Boolean flag, unmodified F32 logit and IEEE-754 bits, emitted-quad identity/triangle rows, and analytical perimeter counts. Missing anchors have null signals, never invented zero logits. Actual contributing candidates sum to each edge's measured incidence.

## Support census

Counts below are **candidate observations**, not independent quads:

| Candidate state | At boundaries | At triple junctions |
| --- | ---: | ---: |
| Active: flag on, all corners present | 1,421 | 1,557 |
| Flag off, all corners present | 61 | 0 |
| Flag off, incomplete corners | 2,552 | 300 |
| Flag on, incomplete corners | 126 | 0 |
| Anchor absent | 1,524 | 219 |
| Total | 5,684 | 2,076 |

This exactly reconciles the earlier boundary totals: **1,421 valid; 2,613 off = 61 complete + 2,552 incomplete; 1,524 absent anchors; 126 flagged/missing-corner**.

Per boundary edge, **1,362 have zero fully supported inactive alternatives, 57 have one and two have two**. Only **59/1,421 (4.15%)** have any such alternative: **59/1,940 (3.04%)** of all investigated defects. The 61 observations identify **34 distinct supported inactive quads**. “One alternative” is local availability, not a unique repair solution.

Each triple junction has three verified active quads. The fourth candidate has an absent anchor at **219 edges**, and an existing off flag with incomplete corner support at **300**. There is no fully supported fourth alternative; activating a fourth would not correct triple incidence anyway.

## Would a single change transfer the defect?

Without changing any array or flag, the census looks up **all four sides against the complete original perimeter table**, including verified absent edges as incidence zero. A conservative activation candidate requires four `1→2` sides; a deactivation candidate requires four `3→2` sides. These are edge-count tests, not intersection, orientation, vertex-manifold or shape-preservation certificates.

**Zero candidates pass.** For the 34 supported inactive quads, sorted original side counts are:

- 27 quads: `[1,1,2,2]`;
- six: `[0,1,2,2]`;
- one: `[1,2,2,2]`.

Every activation would create new triple junctions on its currently two-face sides; six would also create a new boundary. The **1,288 distinct contributing quads at triple junctions** likewise yield no four-`3→2` candidate. Deactivation would create boundary edges for 1,227, erase existing source edges for 127, with overlap between those groups. Erasing an edge/sheet is not proof of closure or surface preservation. No unique unwanted branch or qualifying local region was established. Winding remains the separate Phase 8 issue.

## Original logits

Statistics are deduplicated within the named class; magnitudes are **uncalibrated model signals, not probabilities**. The source threshold remains strict F32 `>0`; no threshold was changed.

| Distinct candidate class | Count | Logit range | Median |
| --- | ---: | --- | ---: |
| Supported inactive at boundaries | 34 | −3.599307 to −0.024337 | −0.316505 |
| Unsupported off at boundaries | 2,054 | −24.818487 to −0.007463 | −8.833814 |
| Unsupported on at boundaries | 85 | +0.009969 to +4.284043 | +0.536037 |
| Active at triple junctions | 1,288 | +0.054649 to +15.751331 | +4.977235 |

No available candidate logit is exactly zero. Of the 34 supported inactive candidates, six have absolute logit ≤0.1 and none ≤0.01. Detailed sign counts, quantiles and declared magnitude-bin counts are local. These bins are descriptive, not admission/correction thresholds. Near-zero logits cannot explain away the absent corner support, and proximity to zero does not identify a safe edit.

## Supported, falsified and unresolved

- **Supported:** sparse coordinate availability plus edge-flag selection constrain closure. Most boundaries cannot gain a second quad merely by enabling an already supported incident flag. Triple incidence is already present in the three selected source quads.
- **Falsified:** all boundaries have complete inactive alternatives; all 2,613 old off observations were fully supported; a uniquely identified single-quad activation/deactivation can pass the stated incidence-preserving screen.
- **Unresolved:** pretrained-model behaviour versus earlier AMD decoding; a valid multi-quad reconstruction; full self-intersections, vertex links and global orientation. No missing vertices or geometry were invented to answer these questions.

## Execution, verification and preservation

One census worker saved all records, then **FAILED only while serializing a NumPy `int64` summary counter**. Its original partial summary, stderr and FAIL receipt remain unchanged. A JSON regression reproduced the error; an explicit integer conversion fixed it. An independently reviewed **metadata-only completion** used pinned saved records, rechecked frozen edge identities/logit bits/deduplication and original hashes, and wrote a separate exclusive `completion-v1` directory. **The census was not rerun.**

The census process settled naturally in **8.000 s**, peak private commit **417,476,608 bytes**; completion passed in **1.610 s**, peak **73,269,248 bytes**. Both retained the original 6 GiB process ceiling, 180 s deadline, 12 GiB live Windows floor, 2 GiB minimum output free-space floor and owned Job cleanup. Neural 22 GiB gates remain unchanged. Before/after verification preserved **101 source/evidence files** plus all pinned failed-run files; protected identities and Git heads were checked separately. No WSL/GPU, model load, neural attempt, geometry edit or backup/VHDX operation occurred.

**67 focused offline tests passed, zero skips**, including missing anchors/corners, supported/off/active flags, triple junctions, exact signed-zero/F32 bits, global counts, defect transfer, deduplication, guards and report serialization. Independent review checked the source and saved records separately. Large data remains local at workspace `artifacts/amd-closed-solid-support-census-20261030-v1/`; the authoritative completed summary is `completion-v1/summary.json`. The top-level partial summary is preserved failure evidence.

## One next experiment — proposed, not execution-ready

Stop local flag-correction trials. The next causally useful experiment is **one provenance-bound upstream reference-decoder parity comparison using the original 3,888-row coordinates/shape latents and identical checkpoint**, reusing this AMD output. Compare the four subdivision supports/logits and final features before extraction. This avoids repeating conditioning, occupancy and shape generation, and directly tests where the remaining upstream-versus-AMD fork occurs.

A qualified upstream runtime/hardware or an authenticated reference artifact is a prerequisite and is **not currently established**. There is no new launcher, environment, download or live authorization. Exact support/sign comparisons and dtype-appropriate numerical tolerances must be reviewed before execution. If that prerequisite cannot be met, retain NO-GO rather than beginning a speculative reconstruction campaign. No raw benchmark PASS may be manufactured by later repair.
