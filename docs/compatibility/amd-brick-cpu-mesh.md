# First genuine brick CPU mesh — PASS

One CPU extraction converted the authenticated, complete brick neural output into a full-resolution polygon mesh. All four output files reopened successfully; source vertices, global indices, original face order, split decisions and winding were independently verified. **No decoder, GPU, WSL, model loading, texturing or geometry repair ran.** Visual fidelity remains unassessed.

Starting commit: `847357960f72f303e282e95c22666b5fad5f2ac0`, branch `codex/amd-repeatable-generation-v1`. The existing `amd_generation_mesh.py --run-contract` interface and CPU converter were used unchanged. The historical default still rejects this larger support without its authenticated `brick-1957891-v1` contract. All ten original tensor payloads and the complete 4,430 → 21,391 → 97,313 → 438,684 → 1,957,891 lineage passed the actual reader.

## Preserved output

Directory, outside the proof checkout:

```text
C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-cpu-mesh-8473579\mesh
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mesh.ply` | 77,751,514 | `ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd` |
| `vertices.npy` | 23,494,820 | `018edd85646f9b504ec52ceb311f69e0c7610e4d1ab306f41c7e73a7a77d3252` |
| `faces.npy` | 50,083,136 | `9314a921741bab5c3d87348b4ea26488ee1ab13201b33d67252d6d272997ab1d` |
| `mesh.json` | 1,346 | `12810c52c808369f360f8d826eb1f58ec66224d5b520dcf129e0b5a50ff1a37a` |

**1,957,891 vertices; 4,173,584 triangles**, with no vertex removal, simplification, smoothing or reorientation. Bounds in original D/H/W spatial order are approximately `[-0.346811205, -0.328291237, -0.498296827]` to `[0.371697068, 0.324613750, 0.499891996]`; grid 512 and original AABB scale are preserved.

There were 2,086,909 positive edge flags: 2,086,792 complete quads, split into 1,039,276 first-choice and 1,047,516 second-choice quads. The original missing-neighbour rule excluded 117 incomplete quads; none were outside the grid. Independent witnesses found **no omitted valid quad**, index error, repeated-index triangle, duplicate triangle or zero-area triangle. PLY and both NPY arrays agree exactly. Before/after hashes, timestamps and file identities match.

**Topology concern:** 387,068 edges have more than two incident faces (maximum four); 267,652 are boundary edges. This is a faithful extraction PASS, not a watertight or game-ready mesh claim. Connected components and self-intersections were not measured. No repair was attempted. [Complete independent analysis](amd-brick-cpu-mesh-evidence/independent-mesh-analysis.json).

## Resources and verification

The source-derived worst case was 11,747,346 triangles and about 340.7 MB for the three geometry files. A conservative all-phase source-array projection was 1,653,602,537 bytes, not a measured process bound. Execution used an owned, GO-gated subprocess with a **6 GiB committed-memory cap, 180-second deadline, 12 GiB available-RAM floor and 2 GiB free-disk floor**. RAM/disk were polled during extraction. The separate reader used initial resource gates and the same continuous process cap/deadline.

| Actual measurement | Result |
| --- | ---: |
| Converter elapsed / supervised extraction | 2.719 s / 4.344 s |
| Extraction peak working set / committed memory | 379,572,224 / 298,770,432 bytes |
| Lowest sampled available host RAM | 20,573,147,136 bytes |
| Independent readback elapsed / peak committed memory | 5.047 s / 170,868,736 bytes |
| Four output files total | 151,330,816 bytes |

Both workers exited **0**, Jobs were already empty, no forced cleanup occurred, and all fourteen reader mappings closed. [Extraction result](amd-brick-cpu-mesh-evidence/extraction-run/host-result.json), [reader containment](amd-brick-cpu-mesh-evidence/independent-mesh-analysis-host.json).

Focused CPU checks passed: 12 existing reader/converter tests, four per-run containment tests, two positive split witnesses, six negative geometry cases and cross-batch edge-incidence accounting. The new work is limited to the per-run host, focused tests and independent reader. [Pre-execution review](amd-brick-cpu-mesh-evidence/preexecution-review.json), [reader review](amd-brick-cpu-mesh-evidence/verifier-preexecution-review.json), [outcome review](amd-brick-cpu-mesh-evidence/outcome-review.json). The reader ran through its own reviewed host, not the unused nested `mesh_host.py --verify` route.

The original 104,148,612-byte neural artifact remains unchanged at SHA-256 `d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861`. All 100 protected files, 21 qualified source bindings, historical receipts, source edits/backups and selected Windows environment/native identities match the [before](amd-brick-cpu-mesh-evidence/preservation-before.json)/[after](amd-brick-cpu-mesh-evidence/preservation-after.json) inventories. These are scoped checks, not exhaustive environment inventories; no Linux runtime was started. Publication preserves all 76 other remote branch heads, including main. Mesh and neural binaries remain local, outside Git.

**Next action:** separately approve bounded multi-angle visual inspection of this exact unchanged `mesh.ply`. Stop before materials, GLB conversion, repair or frontend integration.
