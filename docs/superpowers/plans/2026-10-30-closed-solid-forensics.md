# Closed-solid raw geometry forensic plan

**Goal:** Attribute the saved box's topology defects without changing geometry or running neural code.

**Architecture:** A single bounded Windows CPU child reuses the qualified Job supervisor (6 GiB process cap, 180 seconds, 12 GiB live host floor, 2 GiB output reserve). Its audit policy permits writes only to a new private evidence directory and rejects model/GPU imports, subprocesses and networking. The original extraction and generation sources remain unchanged.

**Scope:** Project-lead Phase 8 directive; original ten tensors, raw arrays/PLY and receipts; pinned Microsoft converter `de38fdd4082638cb4e6d2285f2e215280bd3f485`. No repair, mesh export, WSL, checkpoint loading, environment change or new neural attempt.

- [x] Authenticate retained input/output receipts and every tensor payload, array and PLY; snapshot source identities/hashes.
- [x] Test an independent two-column edge sort against a tuple-dictionary oracle, known winding cases and both quad splits.
- [x] Count full raw incidence/winding independently; compare complete saved edge/count sets.
- [x] Verify each original face pair against tuple-coordinate neighbour lookup, flag and weight decision; compare direct quad perimeter defects and split diagonals. Save bounded concrete defect witnesses and spatial distributions, including batch seams.
- [x] Analyze bounded two-quad orientation constraints without editing faces; distinguish local consistency from untested global orientability.
- [x] Compare deterministic surface samples under all 24 proper signed axis rotations only; retain original positions/scale/translation. Label symmetry and pose ambiguity.
- [x] Reverify source preservation, scoped regression tests, independent review; publish safe code/text only on a fresh branch. Recommend one experiment and stop.

**Review focus:** Direction signs versus incidence; pairs from the same quad; out-of-grid/missing-neighbour counters overlap; no reflection disguised as a rotation; graph parity results apply only to the tested constraint subgraph; private data never enters Git.
