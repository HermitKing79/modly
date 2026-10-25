# Approved zombie orphan cleanup and one local structural candidate

Base: `54b049f43e676d737a0ee8f7082bba2cae7aae15`. **Phase A PASS:** a separate orphan-only variant has exactly **325,807 vertices and 673,604 triangles**, with the complete rendered surface unchanged. **Phase B GO for one future two-face winding prototype**, specified below. That structural prototype was not performed.

## Phase A: exact approved compaction

Original PLY, NPY arrays, neural output, original run contract, saved topology approval and source image were authenticated. The removed IDs exactly match all 425 saved unreferenced IDs and no others. Retained vertices keep their original relative order. Triangle rows and corner order remain unchanged after mapping back to original IDs. No referenced vertex, face, fragment, wound, hair, clothing or crown geometry was deleted or moved.

| Property | Original | Orphan-only variant |
| --- | ---: | ---: |
| Vertices | 326,232 | 325,807 |
| Triangles | 673,604 | 673,604 |
| Unreferenced vertices | 425 | 0 |
| Surface components | 10 | 10 |
| Boundary edges | 19,512 | 19,512 |
| Non-manifold edges | 31,449 | 31,449 |
| Two-face winding conflicts | 147,484 | 147,484 |
| PLY bytes | 12,671,864 | 12,666,764 |

All original edge identities, incidence multiplicities and winding sums agree through the reversible vertex mapping, rather than merely matching aggregate counts. Bounds are bit-exact and unchanged. The reduction is 5,100 PLY bytes; this is a verified data cleanup control, with no polygon reduction or claimed rendering-speed improvement.

The independent reader reopened the binary PLY with the existing separate mesh-witness parser and matched both NPY arrays. Streamed comparison checked every float32 bit of every triangle, including corner and face order. This is stronger than numeric tolerance and preserves signed zeros. The identical original/variant triangle-stream SHA-256 is `fae04e2d89bb0d8848b267343439f55222a7a5cfd687808dfc33c916fc638fc5`.

Fresh Cycles CPU silhouette renders used the same eight original orthographic camera positions/scales, with identical black emission/white background settings. All eight 768×768 RGB source/variant pairs matched exactly: **zero pixel differences and zero mask differences** at 0°, 45°, 90°, 135°, 180°, 225°, 270° and 315°. Both imported coordinate/index payloads were verified before/after rendering. The PLYs retained their bytes, hashes and timestamps.

| Asset / mapping | SHA-256 |
| --- | --- |
| Original PLY | `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c` |
| Variant PLY | `377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5` |
| Variant vertices.npy | `44b2015865e81f66a8e085bc228a6efc50345a8afa7315131f7542e98c154160` |
| Variant faces.npy | `0d256a6b6e6ad4f8b748ea6c4f9247f702f68cac52f554ada550a94645bffd90` |
| New-to-original IDs | `d696984341ae0a70e6f114e0ba19b04cda9430539516b2c453ae04c58584f2e1` |
| Original-to-new IDs, −1 only for removed orphans | `8a4de7f96f9bb7ebcfc8ae5e443208e3cb4bebdbb7f49198e4277ed71aba6759` |
| Exact removed original IDs | `1b983f2dc685165c992af78dcef3230753202847082d71257c60dabb56f3d0e4` |

`variant/manifest.json` records creation pending independent checks. It is retained unchanged as historical evidence. The final append-only `variant/acceptance.json` binds the completed independent geometry and silhouette receipts and gives the final PASS. No pending receipt was silently overwritten.

## Phase B: ordinary upper-back candidate, analysis only

The selected complete source quad is **167004**, containing face rows **334008 and 334009**. Its normalized D/H/W centroid is approximately **(−0.00153622, 0.09742922, 0.21598430)**. Its bounds are `[−0.00290579, 0.09663236, 0.21531558]` to `[−0.00034842, 0.09831667, 0.21672708]`.

All vertex, face and quad IDs in this report are zero-based row IDs. Face row IDs remain unchanged through the orphan compaction; vertex IDs have explicit original/variant mappings.

The selection region was the central posterior upper trunk: x ∈ [−.045,.045], y ∈ [.02,.12], z ∈ [.17,.23]. All four quad vertices satisfy it. The source reference and rear context/closeup show ordinary upper-back skin away from the named front wounds, face, hands, feet and clothing hem. This anatomical description is a visual assessment, not automated semantic segmentation.

The quad was selected from 59 eligible complete quads whose four perimeter edges each have two incident faces and are currently winding-conflicted. All four chosen vertices have one connected closed link cycle; their full incident fan sizes are **6, 5, 6 and 6 faces**. The union contains 15 face rows. No local vertex-link split is needed for this candidate.

Nine projected samples (four corners, four edge midpoints and centroid) are interior non-contour pixels in every original silhouette. The minimum measured contour clearance is **5.71468 pixels**, at 90°. The per-angle distances are saved locally. This is projected raster clearance, not geodesic distance or a universal semantic safety margin. Diagnostic orthographic rays to both triangle centroids first hit the respective selected triangles, supporting that this is a visible surface patch rather than hidden internal geometry.

| Binding | Value |
| --- | --- |
| Variant quad vertex IDs | 160747, 160756, 160757, 160748 |
| Corresponding original/neural row IDs | 160920, 160929, 160930, 160921 |
| Source cell coordinates | (254,305,366), (254,306,366), (255,306,366), (255,305,366) |
| Activated source axis | 2, original positive activation logit 9.22271156 |
| Original weighted diagonal | 02 |
| Candidate face rows | 334008, 334009; unchanged through Phase A |

All five touched edges, including the shared diagonal, have incidence two. Their fixed-neighbor orientation constraints are:

| Variant endpoint IDs | Current signed direction sum | Predicted sum if both selected triangles reverse |
| --- | ---: | ---: |
| 160747–160748 | −2 | 0 |
| 160747–160756 | +2 | 0 |
| 160747–160757, internal diagonal | 0 | 0 |
| 160748–160757 | −2 | 0 |
| 160756–160757 | +2 | 0 |

The prediction subtracts twice each selected face's signed edge contribution, keeping all exterior faces fixed. It creates no repaired face array and applies no structural change. Only four existing winding conflicts would disappear; every other edge remains untouched. This is a locally consistent winding correction candidate, without a sheet split, vertex motion, triangle deletion or geometric rebuild. It does not establish globally outward orientation or solve the asset's remaining non-manifold structure.

## Smallest falsifiable future prototype

**GO for one separately authorised copy-only test** with these preconditions and assertions:

1. Require the orphan-cleaned baseline PLY SHA `377a15dc…176b1a5` and the complete retained mapping/provenance receipt. Preserve this baseline as immutable.
2. On a new copy only, swap corner slots **1 and 2** (`[0,2,1]`) on exactly face rows **334008 and 334009**. Do not change face row order, vertices, mappings or any other face.
3. Require the local five signed sums to become zero and the global two-face winding-conflict count to change **147,484 → 147,480**, with no new conflicts anywhere else.
4. Require unchanged 325,807 vertices, 673,604 triangles, all per-face coordinate multisets, row IDs, bounds, ten components, 19,512 boundary edges, 31,449 non-manifold edges and eight-view silhouettes. Only the two declared corner-order rows may differ. Keep the two-face delta and source hash in an operation receipt.
5. Stop on any additional difference. Do not propagate the operation to other quads or introduce sheet splitting, normal smoothing or global repairs automatically. The experiment succeeds on the exact constraint delta and preservation evidence; no large visible aesthetic improvement is promised from two triangles.

`future-prototype-spec.json` captures that unapplied operation. Current orphan-cleaned geometry still has all **147,484** winding conflicts. No Phase B structural variant or repaired face array was generated.

## Pipeline value and limits

The new compaction primitive accepts explicit approved orphan IDs, derives reversible mappings and refuses a different removal set. Its bit-level surface comparator detects winding/order changes and signed-zero changes. The symbolic orientation and vertex-link helpers work on explicit triangle data without editing it. These tested primitives, provenance bindings, fail-closed checks and owned execution provide repeatable Modly building blocks.

The present path-bound wrapper and anatomical region are deliberately specific to this approved case. Production integration must preserve/remap any additional vertex attributes through the recorded IDs, accept explicit source/protection contracts, and retain operation receipts. This task does not add automatic retopology, rigging, LOD, UV or texturing dispatch. The asset remains a genuine high-detail reference with unresolved global topology defects.

## Execution and preservation

All stages used the preserved CPU owner: 6 GiB process commitment cap, 180-second stage deadline, 12 GiB live available host RAM and 2 GiB output disk floors; exact GO/EOF, owned Job and bounded failure cleanup. No existing neural/CPU safeguards were changed.

| Stage | Wall seconds | Peak process commitment bytes |
| --- | ---: | ---: |
| Copy-only cleanup with source authentication | 2.000 | 53,882,880 |
| Independent full geometry/mapping verification | 3.718 | 217,837,568 |
| Sixteen original/variant silhouette frames | 8.875 | 687,177,728 |
| Independent silhouette pixel verification | 0.625 | 37,732,352 |
| Local source-quad/fan/winding diagnosis | 2.063 | 180,785,152 |
| Three unrepaired patch diagnostic renders | 10.313 | 911,753,216 |
| Acceptance/specification and local sheets | 0.828 | 38,731,776 |

Every worker exited 0 with natural, verified tree settlement and no forced cleanup or cleanup errors. Minimum sampled available host RAM was 21,291,216,896 bytes, above the unchanged CPU floor. Nineteen focused tests passed: ten compaction/prediction/protocol tests and nine retained topology tests. Independent review confirmed exact mappings/surface bytes and eight RGB pairs, checked the local face fans and symbolic prediction, and found no actionable issues for this approved run.

Before/after preservation matches cover 185 prior asset/evidence files, 100 protected identities/78 bounded hashes and 88 retained historical records. All 256 original source hashes and 157 prior local refs matched. The original neural tensors, source image, original PLY/NPY arrays, prior diagnostic screenshots and environments remain in place. No neural inference, WSL, UV creation, baking, textures, global repair, hole filling, welding, fragment deletion, crown trimming or smoothing was performed.

Private output root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-orphan-cleanup-20261025-v1`. The deliverable is `variant/mesh.ply`; mappings and final acceptance are alongside it. Before/after silhouettes, independent checks, local patch analysis, exact future prototype specification and diagnostic sheets remain local. Only safe code and textual reports are published. See [source-safe verification](amd-zombie-orphan-cleanup-verification.json).
