# Clean target mesh and appearance-transfer feasibility — BLOCKED

Base `d10c18ee7b97c3e48c2d0f0dcfd93c480640ec05`; isolated branch `codex/amd-clean-target-feasibility`. One bounded constructor-only admission probe stopped before any remeshing. No clean target or appearance-transfer result was produced. This is a **harness/API qualification blockage**, not evidence that MeshLib damages the source or cannot remesh open surfaces.

| Requested verdict | Observed result |
|---|---|
| Clean geometry feasibility | **BLOCKED**: exact source transport/junction contract not established |
| Brick control quality | **NOT RUN / BLOCKED**: no target geometry |
| Foliage stress quality | **NOT RUN / BLOCKED**: no foliage constructor or target |
| Source-appearance transfer | **NOT RUN / BLOCKED**: geometry admission never passed |
| Runtime/memory safety | **PASS for the observed probe containment only**; target runtime remains unqualified |

Authenticated references were the existing781-triangle/457-vertex brick patch and4531-triangle/3177-vertex foliage patch, their original face/vertex IDs, accepted patch assets and learned-field lineage. Bounded files were freshly hashed; original correspondence relies on the previously qualified extraction/readback receipts, not a repeated full-model analysis. The source material remains `8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`. New target vertices would require genuine eight-neighbour spatial sampling and IEC sRGB-to-linear F32 conversion, never row-ID colour reuse. [Input authentication](amd-clean-target-feasibility-evidence/patch-authentication.json) records the exact scope.

The saved sparse support, dual-vertex offsets, three quad-intersection flags and split weights do not supply a qualified fine-resolution signed-distance or inside/outside oracle. Voxel remeshing was therefore not admitted. Metadata/source inspection identified MeshLib3.1.4.297 as the sole detailed direct-surface route. Its remesh API has boundary locks, protected edges, a pre-collapse veto and bounded splitting; its subdivision projection does not cover every later moved vertex. That limitation requires independent shape validation, rather than proving the method impossible. Other installed tools were inventoried without running alternatives.

MeshLib's convenient importer defaults to non-manifold vertex duplication and its builder can skip faces. A lossless input roundtrip was therefore mandatory before any target. Independent review admitted only one constructor per reference patch, with `allowNonManifoldEdge=True`, `duplicateNonManifoldVertices=False`, explicit failed-face region and skipped-face counter. Static import review/pins covered78 installed source/library files including72 packaged PE dependencies, with no declared CUDA/viewer/GL import in the selected closure. No native C++ remesher implementation was installed for source inspection; binding declarations alone were not treated as a quality guarantee.

The actual brick call returned. The next operation, `int(skipped.value)` at `import_probe.py:125`, raised **`RuntimeError: Expected exactly one value.`** The installed stub describes `int_output` as a scalar-list helper whose `.value` requires exactly one stored element; its unnamed integer constructor does not establish the probe's assumption that `int_output(0)` creates one readable zero. That assumption was not runtime-qualified. The saved evidence does not record the helper's actual storage length or skipped count: neither may be inferred. Valid face/vertex IDs, position/triangle readback and internal sheet pairing were not reached. This is not a demonstrated numerical or geometry failure.

The unexpected exception correctly stopped the whole worker: **one brick constructor, zero foliage constructors, zero remesh calls, zero targets and zero appearance-transfer calls**. No readback NPYs or target meshes were saved. The original brick source-only statistics were781faces,457vertices,1243edges,147boundary edges and4non-manifold edges (maximum incidence3), with one edge/vertex-connected face component. Previously authenticated foliage statistics remain83edge-connected components,219non-manifold edges and2461boundary edges; they were not remeasured by this worker. [Actual result](amd-clean-target-feasibility-evidence/constructor-result.json) and [independent outcome](amd-clean-target-feasibility-evidence/independent-outcome.json) preserve the failure.

| Measured probe resource | Value |
|---|---:|
| Worker /host elapsed | 0.937 /3.125seconds |
| Peak process committed | 168,378,368bytes |
| Peak Job committed | 169,218,048bytes |
| Process and Job ceiling | 6GiB each |
| Wall deadline | 180seconds |

All recorded12GiB available-RAM/2GiB destination floors passed. The Job settled naturally, exit0, no forced cleanup; watched sources remained unchanged. These measurements apply only to the failed input-admission probe, not a remesher. Five bounded pure-validator tests passed before execution, including source IDs, winding/cyclic corners, skipped/merged IDs and exact F32/F64 position bits. They did not exercise the genuine counter binding and are not presented as complete API acceptance.

All distance (RMSE0.005/p950.01/max0.03 of whole-model diagonal), silhouette0.98, foliage-retention0.99, RGB0.02/0.05/0.15 and scalar-RMSE0.02 gates were frozen unchanged. None was evaluated on a target because no target passed admission. No parameter sweep, constructor fallback, retry, model/GPU/Blender execution, source alteration or texture bake occurred.

[BEFORE](amd-clean-target-feasibility-evidence/preservation-before.json) and [AFTER](amd-clean-target-feasibility-evidence/preservation-after.json) document protected100, historical models/assets/reports, selected environments/backups, caches/quarantine and previous remote heads. Private probe evidence remains under `artifacts/amd-clean-target-feasibility-d10c18e/import`; no private model, image or raw geometry is published. Source and receipt bytes are preserved by evidence-local Git attributes.

**Stop this route at admission.** Its clean-target capability and20% reduction opportunity remain unproved for both cases. If continuation is desired, the cheapest separate approval is a tiny CPU qualification of the installed `int_output`/skipped-face binding before any further real-patch construction. It would not authorise remeshing or establish sheet correspondence. Do not substitute random remesh parameters, another method or a full-model run for that missing proof.
