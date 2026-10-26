# Zombie D AutoRemesher 1.2.0 proof v1

**NO-GO: this one CPU run does not produce a substantially lighter Zombie D while preserving the geometry that matters.** Execution completed, but the output lost most legs/feet, distinct fingers and major face/body surfaces. It is visibly unusable. No tuning, cleanup or UV follow-up was performed.

Base evidence: `0e99c728efc53a3fdee60e6cddf6f587459aa975`. Isolated branch: `codex/amd-zombie-d-autoremesher-proof`. Aggregate measurements: [JSON](amd-zombie-d-autoremesher-proof-v1.json). Private case outside Git: `artifacts/amd-zombie-d-autoremesher-20261026-v1`. Only these text findings are published; models, images, portable binaries, raw logs and proof helpers remain local.

## Preparation and official-source audit

The pinned [1.2.0 release](https://github.com/huxingyi/autoremesher/releases/tag/1.2.0) supplies a Windows x64 portable archive. The 21,529,937-byte download matched its published SHA-256:
`f6184622cef84f0bcf032a0474df44e2076f4c10c159f1af0cb7230259af7476`.
Executable SHA-256: `103bd621bfe828d4f1fbc62ace095483eeb184e29e647e739894c3b9cc93e877`.
All inspected PE imports resolve to bundled DLLs or Windows system libraries. This is dependency inspection, not proof about arbitrary native DLL behaviour.

The tagged [CLI](https://github.com/huxingyi/autoremesher/blob/1.2.0/src/main.cpp) supports OBJ input/output, a report and target quads. A real bounded `--help` execution qualified the binary. Tagged [build configuration](https://github.com/huxingyi/autoremesher/blob/1.2.0/autoremesher.pro) uses C++14, Qt, TBB, Eigen, isotropic remeshing, meshoptimizer and its native parameterization/extraction code. No Python ABI, Torch, ROCm, WSL, model, compilation or environment update is required for this portable run.

The [root licence](https://github.com/huxingyi/autoremesher/blob/1.2.0/LICENSE) is MIT. [Third-party acknowledgements](https://github.com/huxingyi/autoremesher/blob/1.2.0/ACKNOWLEDGEMENTS.html) include GPLv3 and other terms. This local test does not clear dependency redistribution or Modly integration.

Relevant tracker evidence includes [low-target quality loss](https://github.com/huxingyi/autoremesher/issues/51), [Windows crash reports](https://github.com/huxingyi/autoremesher/issues/45), and a [large-system memory-bound proposal](https://github.com/huxingyi/autoremesher/pull/61). These reports/proposals are not proof of the cause of this result. Thin/nearby surfaces and non-manifold/winding inconsistencies remain risks for surface correspondence. The pinned [island separator](https://github.com/huxingyi/autoremesher/blob/1.2.0/src/AutoRemesher/meshseparator.cpp) follows oppositely directed halfedges, so it should not be assumed independent of source winding.

The original D PLY was authenticated:
`2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205`.
Its original NPY identities were also verified. A private PLY copy was converted to OBJ using nine-significant-digit Float32-roundtrip coordinates. Independent readback preserved all **356,876 coordinate triplets bit-exactly and all 777,400 ordered triangle rows**. No welding, repair, smoothing, UVs or materials were added.

## Exactly one native remesh

The actual command used the hash-bound portable executable, exclusive copied input/output paths and:

```text
--input <private>/input/source.obj --output <private>/raw/output.obj
--report <private>/raw/report.txt --target-quads 20000 --edge-scaling 1.0
--sharp-edge 90.0 --smooth-normal 0.0 --adaptivity 1.0 --anisotropy 1.0
```

Only the requested target differs from tagged CLI defaults. AutoRemesher's own algorithm includes simplification, resampling, smoothing/projection and extraction; none was added as a separate cleanup stage. The immutable input reference remains unchanged. CPU/TBB execution was selected with affinity to four logical CPUs (mask 0xF), Qt offscreen/software configuration and bundled software OpenGL. GPU activity was not independently sampled; no neural or GPU-compute dependency was invoked by the harness.

| Measurement | Original D | Raw native output |
| --- | ---: | ---: |
| Vertices | 356,876 | 10,849 |
| Triangles | 777,400 | 888 |
| Quads | 0 | 6,766 |
| Pentagons | 0 | 375 |
| Total polygons | 777,400 | 8,029 |
| Triangle equivalent, sum of n-2 | 777,400 | 15,545 |

The output achieved **6,766 quads**, not approximately 20,000. The 98.0004% triangle-equivalent reduction is not a quality success. Independent polygon-edge inspection found **2,931 boundary edges, 296 edges with more than two incident polygons, zero two-polygon winding conflicts, 870 connected vertex components and no unreferenced vertices**. The largest component contains 5,319 vertices. A diagnostic fan split has 185 zero-area triangles; that does not independently establish 185 degenerate native polygons. Native pentagons were retained unchanged.

Raw OBJ SHA-256: `6bb6ac2b1ecfd3e48ede1cf748015a275d32b5f94f9fba20f4190f6e3ca4207b` (534,235 bytes). No output UVs, normals, material references or textures were found. Raw output, native report and logs are retained even though quality fails.

## Visual and structural verdict

Sixteen completed CPU renders compare original/raw output from matched front, back, both hand, face, wound, shorts and feet views. Cameras, scale and lighting use the original D bounds for both meshes; there is no per-model recentering or normalization. Both use the same neutral material and flat shading. The renderer checked every candidate coordinate and original native polygon row before display. Its internal tessellation does not modify the raw OBJ. No bake or texture is involved.

- **Full silhouette/body:** most lower-body surfaces disappear; only small fragments retain almost-original global bounds. Near-matching bounds therefore do not establish preservation.
- **Hands/fingers:** separated source fingers are reduced to broken stumps/fragments. Useful hand structure is lost.
- **Face/ears/nose/mouth:** large facial regions are missing; recognizable facial structure and ear detail are not retained reliably.
- **Chest/abdominal wounds:** intended openings are overwhelmed by widespread missing torso sheets. Wound outline/cavity fidelity fails.
- **Torn shorts, feet and smaller details:** most usable surfaces are absent or fragmentary. Feet are almost entirely missing in the matched view.
- **Holes/disconnections:** extensive visible new gaps agree with the disconnected raw geometry. This is more serious than black shading speckles. No reliable improvement in body-part separation can be claimed when parts are missing.

The front comparison is `previews/front-comparison.png`; all eight matched pairs are available in the private `previews/review.html`. Existing source defects remain visible in the reference. Output loss is established, but its exact internal cause was not isolated. Logs contain **111,699 repeated-halfedge warning occurrences**, not necessarily that many unique bad edges. Tagged code can skip an island after parameterization exceptions; this run logged **zero** such exceptions, so that mechanism must not be claimed as the measured cause of the missing legs.

## Resources, stops and preservation

Original CPU safeguards remained: **12 GiB live Windows available-RAM floor, 2 GiB disk floor, 6 GiB per-process private-commit cap, 180-second stage deadline**, inherited owned-process Job containment/cleanup. Native wait was limited to 175 seconds. The 22 GiB neural admission requirement was unchanged and was not repurposed as a CPU gate.

Native algorithm self-report: **159.895 seconds**; actual native-process wait: **160.235 seconds**; supervised remesh stage: **160.781 seconds**. Peak process commit: **1,775,996,928 bytes / 1.654 GiB**; aggregate Job peak: **1,789,911,040 bytes / 1.667 GiB**. Lowest sampled host available RAM: **17.041 GiB**. All 775 remesh-stage samples admitted. These are commit and host-availability measurements, not physical working-set attribution. Remesher and owned tree exited naturally; no forced cleanup or errors. Matched rendering took **23.906 seconds**, peak process commit 680,681,472 bytes.

Preparation retained three harness failures before any native spawn: a relocated helper lookup, forbidden Windows NUL-device opening, and an unspecified audit executable. Corrected routing, an existing EOF pipe and an explicit executable path preserved the original policy/limits. Exactly one native help and **one native remesh** then executed. The initial output reader stopped on pentagons; bounded readback support was corrected without changing the OBJ or remeshing again. Six focused tests passed.

An optional image contact-sheet/mask closeout stopped on its overly strict close-up alpha-mask assumption after saving the front comparison. Its failed receipt remains; it was not retried. All sixteen original rendering PNGs and camera metadata completed successfully and are the review evidence. No silhouette IoU from that incomplete postprocessor is claimed. Once the front image showed unusable geometry, no further geometry operation, repair, tuning or UV test followed.

Closeout verified **3,346 historical files by SHA-256 and metadata**, all inherited **100 protected identities plus 88 retained records**, **173 refs before publication**, original D PLY/NPY hashes, raw-output identity and all 27 extracted portable files. Original tracked files, TRELLIS.2 installations, checkpoints, environments and historical evidence remained unchanged. Only a new findings branch is published.

Independent read-only review approved these two text findings with no blocking issue. It separately checked raw OBJ identity, all sixteen PNG hashes, native polygon counts, aggregate resource/cleanup receipts, preservation counts and the visual NO-GO. No review workload or remesh was executed.

## Recommendation

Stop this whole-character AutoRemesher route for Zombie D. **Do not run a UV-atlas experiment on this rejected output.** For salvaging this existing character, manual/segmented retopology against the immutable high-detail reference is the more credible direction: the original visibly contains important geometry that this pass discarded. This proof does not establish that another mesh-generation engine will be better, and does not justify replacing the working TRELLIS.2 pipeline. A repeatable automated pipeline still needs a separately qualified approach that preserves surfaces, not merely fewer polygons.
