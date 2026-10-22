# Corrected MeshLib original-patch constructor qualification

**NO-GO for lossless brick input transport under the tested settings.** The counter correction succeeded; the genuine native constructor rejected four original faces. Foliage was correctly not run. No remeshing, repair, alternate constructor, appearance operation or retry occurred.

Base: `bbdb6aca54a58c04c12d2c4e19af8f3d92dd0fb5`. Branch: `codex/amd-meshlib-patch-constructor`. Installed MeshLib remains `3.1.4.297`, with the original 78-file CPU/source/binary closure authenticated. Both original patches and their geometry, ID and external colour lineage were authenticated; only brick entered the constructor.

| Outcome | Verdict |
| --- | --- |
| Brick input transport | **NO-GO**: 777/781 faces and 456/457 valid vertices |
| Foliage input transport | **NOT RUN**: brick gate failed |
| Complete original sheet/interface correspondence | **FAILED**: original faces are absent |
| Intended pairing of surviving semantic sheets | **UNRESOLVED**: no original sheet partition is authenticated |
| MeshLib remeshing admission | **NOT AUTHORISED** |

The worker is a preserved copy of the earlier constructor probe. Its exact position/source-ID/oriented-face comparison and endpoint-topology functions, plus five historical regressions, remain unchanged. The narrow correction establishes a named one-element `int_output()` holder, sets `-23456` before binding, checks the settings getter, retains the holder without resizing and records the overwritten value. The same holder and settings getter returned **4**, agreeing with both independently enumerated failed regions and exactly the four missing original FaceIds. This is an observed native output, not an unchanged default. `allowNonManifoldEdge=True`, `duplicateNonManifoldVertices=False` and `shiftFaceId=0` remain the reviewed settings.

Exactly one native brick constructor returned. It retained all **457 allocated position rows** with bit-exact original F32-to-F64 promotion, but only **456 vertices were valid/referenced**. The independent checker verified that valid FaceIds identify exactly **777 original triangles**, without corner rotation or reversed winding; their complement is exactly the four failed faces. Every original heap input and saved source-ID copy remained unchanged. No vertex welding, coordinate movement or hidden point-row duplication was observed. Complete original geometry transport nevertheless failed. Colours remain authenticated external source rows; no native colour-attribute transport or appearance test was performed.

The worker stopped at `No skipped or failed original faces` after saving the partial readback. Its conservative operational receipt says BLOCKED. The independently reproduced missing geometry makes the **transport verdict NO-GO**. The constructor returned normally; this is not evidence of a linker/ABI fault, process crash or counter-binding failure.

| Topology measure | Original endpoint graph | Retained endpoint graph | Native edge adjacency |
| --- | ---: | ---: | ---: |
| Edges | 1,243 | 1,239 | 1,241 |
| Boundary edges | 147 | 148 | 151 |
| Edges with more than two incident faces | 4 | 1 | 0 (1,090 paired two-face native edges) |
| Edge-connected face components | 1 | 1 | 2 |

Two source endpoint pairs map to multiple native edge identities. Actual saved halfedge/left/right FaceIds independently establish reciprocal surviving adjacency, with no merge across original edge components. Three rejected faces touch three original non-manifold edges; the fourth touches none of those edges. These are measured associations, not a recorded explanation of the constructor's rejection decisions. Matching counts would not qualify semantic sheet pairing, and the observed losses already prevent full interface correspondence.

Before execution, **32 focused CPU tests** and independent source reviews passed. The success-only saved-data verifier was preserved without execution after the loss gate failed. A separate, narrowly reviewed saved-failure checker passed **10 synthetic tests**, then ran once with NumPy CPU only and MeshLib imports prohibited. It reopened all 13 partial NPY readbacks, original bounded arrays, counter snapshots and journals, and independently reproduced the omissions and native adjacency. Its diagnostic PASS means the rejection evidence is verified, not that brick transport passed. Original sources, reviews and failed attempts were not revised.

| Measured resource | Constructor worker | Saved-failure verifier |
| --- | ---: | ---: |
| Worker elapsed | 1.688 s | 1.265 s |
| Supervised elapsed | 4.250 s | 3.250 s |
| Peak process committed bytes | 170,344,448 | 125,571,072 |
| Peak Job committed bytes | 171,180,032 | 126,406,656 |

Both workers retained separate 6 GiB process/Job caps, 180-second deadlines, 12 GiB available-RAM and 2 GiB destination floors. Initial and sampled live admissions passed. Constructor minimum available RAM was **18,987,257,856 bytes**, destination free space **107,350,003,712 bytes**. Both worker exit codes were zero, both Jobs were naturally empty before cleanup, no forced termination occurred, and watched sources remained unchanged. The native supervisor returned failure as required by the loss gate. The constructor itself was below the observed timer resolution (`0.0 s` recorded); no zero-cost claim is made.

Evidence: [execution summary](amd-meshlib-patch-constructor-evidence/execution-evidence.json), [independent readback](amd-meshlib-patch-constructor-evidence/independent-readback.json), [native host](amd-meshlib-patch-constructor-evidence/constructor-host.json), [verification host](amd-meshlib-patch-constructor-evidence/verification-host.json), [initial tests](amd-meshlib-patch-constructor-evidence/focused-tests.txt), [failure-checker tests](amd-meshlib-patch-constructor-evidence/failure-tests.txt), and [independent outcome review](amd-meshlib-patch-constructor-evidence/independent-outcome.json).

Private readbacks and exact rejection/junction/source-ID witnesses remain below `artifacts/amd-meshlib-patch-constructor-bbdb6ac/import/brick` and `artifacts/amd-meshlib-patch-constructor-bbdb6ac/verify`, outside the proof checkout. Native result SHA-256: `6a47a4df80fc32610bbb2ff2072d15c38f635c621de92d8dab684d766e55ed1f`; independent diagnostic result: `433d37573237a001b0f6ca5f1f2c1bfe77dfbf533fc6396c34566669ef630aab`. No private geometry, source-ID arrays, colour arrays or images are published.

Preservation checks retain all 100 protected files, 69 historical trees/1,969 files/32 reports, original/rejected assets, selected environments/backups, caches/quarantine and installed SDK identities. Historical checkpoint protection remains stat/header-based with prior full-hash provenance; no weight payload was read. Final publication separately verifies all 112 previous remote heads and main remain unchanged.

The next justified step is a separately approved, read-only attribution of the four saved rejection witnesses against constructor constraints and original junction context. No alternative construction or remeshing is approved by this result.
