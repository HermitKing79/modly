# Closed-solid raw geometry: Phase 8 forensic result

30 October 2026. **Forensic verification PASS; raw geometry remains NO-GO.**

The CPU extraction faithfully reproduces the saved decoded quad complex. The openings and triple junctions precede triangulation. Winding conflicts are a separate consequence of the fixed, unsigned quad-order convention shared with upstream. No CPU translation defect was demonstrated. This does **not** distinguish pretrained-model limitations from possible differences introduced earlier in AMD decoding; no upstream numerical decoder comparison was performed.

## Authentication and independent measurement

Original decode execution authority remains `d9d733e447a25a638649b24a4bcbcaea51154fa1`; original input authority remains `853a9a28da970beeb53685f967de059bb90c7364`. The existing successful receipts, complete ten tensors, all payload hashes, four subdivision linkages, original 3,888-row coordinates/latents, image/conditioning lineage and checkpoint **identity recorded in provenance** were reauthenticated. No checkpoint was loaded or rehashed in this investigation.

| Original artifact | SHA-256 |
| --- | --- |
| Ten-tensor neural output, 55,959,204 bytes | `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975` |
| Raw PLY, 38,901,900 bytes | `1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3` |
| Vertex NPY | `fa52e8076fe99cc251db578911f1e0b6eb6224a5a1118aeb18e6a28ab9926e60` |
| Face NPY | `3b530ebc0a00e73a2a8192ad855f136734e2170cb5357d28e669df3064a400fc` |

PLY/NPY readback is exact. A second method sorts two-column endpoint pairs lexicographically and reduces directed signs; the original analyzer uses flattened integer keys/bincounts. Complete edge identities and incidence agree, not just totals:

| Measurement | Independent result |
| --- | ---: |
| Vertices / triangles | 1,023,995 / 2,047,210 |
| Unique edges | 3,071,266 |
| Boundary / more-than-two-face edges | 1,421 / 519 |
| Maximum incidence | 3 |
| Two-face winding conflicts | 101,022 |
| Unused vertices | 37 |

The authenticated original receipt retains one face-bearing vertex-connected component and zero duplicate index triangles/exactly zero-area triangles. This is not proof of edge-connected sheets, vertex manifoldness or absence of self-intersection.

## Where the defects enter

An independent tuple-coordinate dictionary checked **every original quad row, original vertex index and ordering** against the saved face pairs. Every diagonal decision and vertex position also agrees. Counters exactly match `mesh.json`: **1,023,695 flagged quads; 1,023,605 valid; 90 missing-neighbour flags; zero out-of-grid; 512,820 split-1 and 510,785 split-2**. Missing-neighbour includes out-of-grid cases in the converter; the counters must not be added.

All **1,023,605 internal diagonals** have two opposite incidences. The direct quad perimeters have exactly the same 1,421 boundary, 519 triple-junction and 101,022 conflicting-winding edge IDs as the triangles. Changing a quad's diagonal cannot resolve these perimeter defects.

At the 1,421 boundaries, the four possible lattice-quad candidates give 5,684 observations: 1,421 valid, 2,613 flags off at present anchors, 1,524 absent anchors and 126 flagged-but-missing-corner candidates. Those 126 are candidate occurrences, not distinct edges. Thus **at least 1,295 boundary edges cannot be directly explained by the 90 discarded flagged quads**. The current census does not distinguish off flags with complete corner support from off flags whose other corners are missing. No missing geometry or unique correction is inferred.

## Locations and orientation

Private evidence contains 54 representative edge witnesses: original endpoint positions/grid cells, incident source quad anchors/axes, original triangle rows, directed edges and actual triangle normals. For example, edge **[12488,12500]** joins source quads **12908,12909,12910** (axes Z/X/Z), giving three incidences before triangulation.

Nearest normalized bounds-face bins are geometric labels, not semantic segmentation:

| Bin | Boundary | Triple junctions |
| --- | ---: | ---: |
| X-min / X-max | 56 / 62 | 50 / 20 |
| Y-min / Y-max | 20 / 64 | 28 / 20 |
| Z-min / Z-max | 1,019 / 200 | 35 / 366 |

Openings concentrate near the lower-Z bounds face; triple junctions near upper Z. Winding conflicts span all six bins. Of the triple junctions, 25 cross original 32,768-row processing batches; 3,943 winding conflicts cross batches. Most occur within batches. Boundary seam count is trivially zero under this incident-anchor definition; **the full exact global quad correspondence**, rather than this statistic, rules out a CPU batching omission on this output. Neural tiled-convolution seam attribution remains untested.

All winding conflicts are cross-axis: **X/Y 38,528; X/Z 38,741; Y/Z 23,753; same-axis zero**. Six bounded crops (408–590 quads each) have no contradictory two-quad orientation constraints. Their cropped boundaries are artificial; >2-incidence edges are excluded from the constraint graph. Local consistency supports a face-order explanation, not global orientability, outwardness or a closed-solid repair. No face was flipped.

## Pinned upstream audit and axis comparison

The [Microsoft inference converter at de38fdd](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/o-voxel/o_voxel/convert/flexible_dual_grid.py#L159-L264) uses the same neighbour offsets, ordered corners, two split arrays and strict weight-product comparison. Both use fixed negative-axis lattice winding without an orientation sign. CPU sorted lookup restores global row indices; its explicit bounds check is stricter, but there were no out-of-grid queries here. The upstream preliminary vertex expression is overwritten in inference; the final formula matches the CPU scale/AABB. The preserved upstream source SHA-256 is `112bd474d580f279ac196129f7989df6fdafa7ed7e08c350f2b386553e73e17d`. No native CUDA/HIP extractor was executed; GPU transcendental bit-equivalence is unclaimed.

Raw dimensions are **[0.643322408, 1.000014544, 0.805940509]**, versus reference **[1, .65, .8]**. All 24 proper signed-axis rotations were tested on 4,096 deterministic surface samples each way, without translation, scaling, continuous fitting or mesh alteration. Identity sampled RMS/original reference diagonal is **0.084020488**; best is **0.010843708**, with `(x,y,z) → (y,x,−z)`. Several symmetry-equivalent rotations score similarly. This supports an X/Y pose difference; it does not prove a single exporter permutation. Every raw position already matches the unswapped decoded-coordinate formula. Rotations preserve the topology defects. Sampled point distances are not a Hausdorff or surface-intersection certificate.

## Verification, preservation and next decision

The one CPU measurement completed in **10.360 seconds**, peak private commit **686,690,304 bytes**, under the unchanged 6 GiB process cap/180-second deadline, 12 GiB live host floor and 2 GiB minimum free output-disk floor. Owned Job exited naturally and empty, with no cleanup errors. Forty-seven source files matched their before/after hashes, lengths and mtimes. Protected-record and Git-ref checks are recorded separately. No WSL, GPU, model/checkpoint loading, neural attempt, geometry edit or backup/VHDX operation occurred.

**50 focused tests passed, zero skips**, covering independent edge/direction counts, both splits, triple junctions, tuple lookup, parity contradictions, proper rotations, CPU policy and existing extraction/recovery safeguards. Existing fixture writes were relocated into fresh private test directories. An initial relocation-wrapper argument error affected two suites; corrected wrapper reruns passed, and both logs are preserved. Independent source/evidence review and an independent 11-test rerun found no remaining material issue.

Private evidence: workspace `artifacts/amd-closed-solid-forensics-20261030-v1/`, including `defect-locations.png`, exact witnesses, counters, rotation scores, resource/cleanup and preservation receipts. Assets/images are not published.

**One proposed next experiment, separate authorization required:** read-only four-candidate support-and-logit census at exactly the **1,940 boundary/triple-junction edges**, distinguishing every off flag's complete/incomplete corner support. Test the hypothesis that the openings admit fully supported alternative quads; record ambiguity rather than selecting branches. GO only for proposing a further local experiment when required support exists and the local constraint evidence identifies a unique candidate; otherwise NO-GO for local correction. This does not authorize changing flags, deleting faces, filling holes, global winding or another neural run. Winding-only changes cannot deliver the required closed solid. Stop after this report.
