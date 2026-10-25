# First successful standalone AMD zombie decode: CPU mesh and visual inspection

**PASS: authentic CPU mesh extraction and exact readback. The result is recognisably the source zombie.** This is a faithful geometry result with topology defects, not a watertight or production-ready mesh. No neural inference, WSL command, additional generation reservation, texture generation, repair, smoothing or simplification ran in this task.

## Authenticated source

The saved final result is `PASS`, the original supervisor is `QUALIFIED`, and its ten-tensor readback is complete. The neural harness is `2c6990b9224e9e4a0c5cd383dadcc1a3808c3e3a`. The original 17,621,624-byte `shape-decoded.safetensors` freshly matched SHA-256 `e589fec9f5ea58ec7bdb07c00c3c40911b6191d08055611fac17ddee65aa7918`.

Independent CPU authentication reopened all five copied evidence records, original host and guest run contracts, raw safetensors header and complete provenance sidecar. All ten tensor payload hashes match both the output sidecar and the saved independent host verification. The exact C2S coordinate order and positive BF16 subdivision linkage are **1,000 → 4,277 → 18,312 → 77,797 → 326,232**. Initial coordinates and latent input files were freshly authenticated against the original contract; the saved shape-job and source-image hashes also match.

The run contract remains `13a140490d933d0c9f21af701af36caf54384b0fe0e5dc9c4714d0328dea6343`, source job `913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf`, and checkpoint provenance `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. Checkpoint weights were not opened or reloaded. The saved neural minimum host availability is 20,341,559,296 bytes (18.945 GiB), above its original 12 GiB live floor. The saved owned cleanup was natural and verified.

## Genuine extracted mesh

The existing `amd_generation_mesh.py` and flexible dual-grid converter were used unchanged, with the original authenticated run contract. The inspector separately reused the source-derived brick mesh witnesses for exact transforms, quads, split weights, global indices and winding. It did not regenerate faces. All original vertices, including unused vertices, were retained.

| Result | Value |
| --- | ---: |
| Vertices | 326,232 |
| Triangles | 673,604 |
| Complete source quads | 336,802 |
| Source flags missing a corner | 22 |
| Out-of-grid quads | 0 |
| Zero-area / repeated-index / duplicate triangles | 0 / 0 / 0 |
| Exact duplicate vertex positions | 0 |
| PLY / NPY readback | Exact |
| Referenced / unused vertices | 325,807 / 425 |
| Connected surface components | 10 |
| Main component triangles | 673,338 (99.9605%) |
| Other component triangle counts | 212, 12, 12, 10, 8, 6, 2, 2, 2 |
| Boundary edges | 19,512 |
| Edges with more than two faces | 31,449 |
| Maximum edge incidence | 4 |

No valid source quad was omitted. The 22 incomplete quads follow the original missing-neighbour rule. Bounds in original D/H/W order are `[-0.5000397563, -0.1111216843, -0.4999842346]` to `[0.4997721910, 0.1112157702, 0.4998894930]`. The very small excursion below -0.5 is recorded as produced; no clamping was applied. Boundary/non-manifold counts establish that the mesh is not a closed manifold. Self-intersections, genus and printability were not assessed.

The inherited witness dictionary initially retained `components_measured: false` even though the added component calculation ran and recorded its counts. The new component helper now explicitly sets this flag. The first analysis file was preserved; an append-only clarification records the metadata correction, bound to the original analysis SHA. No extraction or analysis was repeated to change that flag.

| Local geometry file | Bytes | SHA-256 |
| --- | ---: | --- |
| mesh.ply | 12,671,864 | `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c` |
| vertices.npy | 3,914,912 | `a9d3e00a13d1faec64eb268551df11db2f42de11da6b49fa2b99d3d85e741a37` |
| faces.npy | 8,083,376 | `82dfbe05c920e74d75cc1cae90437596faf15b1d8b9c148c875baa6730816ed2` |

## Visual assessment

Eight orthographic views at 45-degree increments and one front silhouette were rendered in Blender 5.2.1 LTS, Cycles **CPU**, four threads, eight samples, 768×768. Original +Z up was retained. Blender's imported vertex and face payloads matched the original NPY arrays exactly, with identical counts and bounds, no modifiers and flat shading. The PLY hash, size and modification time were unchanged after rendering. The matte material is a temporary untextured inspection material; no learned colour or texture is shown.

The original source image and all eight actual views plus silhouette were inspected. **Recognisably a zombie:** the T-pose, skull/head proportions, recessed eye region, open mouth, hair, torso, chest/abdominal wound cavities, ragged shorts, two arms, two legs and feet remain present. The main body appears coherent across front, rear and side views; the silhouette retains the source pose and major proportions. The back and unseen sides are inferred from a single front source image, so exact rear fidelity cannot be established.

Visible limitations include simplified finger and mouth/tooth detail, rough hair and wound edges, a small crown protrusion/fleck, and faceted surfaces. The abdominal cavity appears broad relative to the reference wound. Wound openings may be intentional; the preview alone cannot identify which openings account for the measured topology defects. The eight-sample previews are grainy and bright, and flat shading emphasises facets; these rendering properties must not be counted as new geometry defects. The local head detail is only a nearest-neighbour crop of the original front render, not a new geometric or generative reconstruction.

## Bounds, containment and preservation

The existing CPU extraction precedent was retained: **6 GiB process committed-memory cap, 180-second deadline per stage, 12 GiB live available host RAM and 2 GiB output-volume disk floors**. Native host resources were polled every approximately 200 ms. The Python child was GO/EOF gated and attached to its owned Windows Job before work; the direct base interpreter avoids venv redirector memory attribution. Model/GPU imports, child creation and network are denied inside Python workers, and writes are restricted to the fresh output root. No original artifact is a permitted write destination.

| Bounded stage | Wall seconds | Peak process commitment bytes | Minimum sampled host RAM bytes |
| --- | ---: | ---: | ---: |
| Extraction, including authentication | 1.047 | 69,763,072 | 24,300,449,792 |
| Independent geometry/readback | 2.468 | 174,366,720 | 24,056,492,032 |
| Eight views and silhouette | 11.984 | 649,547,776 | 23,748,681,728 |
| Preview validation and contact sheet | 0.625 | 25,616,384 | 24,029,831,168 |

Every worker exited 0, every Job was empty at cleanup and no forced termination was required. The converter itself reported 0.422 seconds. All stages remained far below the retained limits. Blender emitted only its existing `use_nodes` API deprecation notices; rendering and payload verification completed.

Focused verification passed: seven existing mesh-reader tests, six converter tests and seven new policy/component/cleanup tests. An owned harmless fixture rejected model import, a WSL subprocess before creation, and network access. Full project/generation tests were not invoked. All 256 original source hashes, 100 protected identities/78 bounded hashes, 88 retained records, all 18 successful neural/handoff files and 155 prior local refs matched the saved baselines. Existing branch tips and environments were preserved. Newly created fixtures from existing converter tests were retained inside this task's private output root; no old fixture was moved or deleted.

Independent review freshly verified the complete source and all ten raw tensor hashes, mesh artifacts, all nine rendered images, component-clarification binding, resource receipts and visual identity. It found one Important failure-path issue in the new owner: a renderer outside the Job after attachment failure could outlive cleanup and prevent a failure receipt. The successful saved render attached correctly and was unaffected. The author fixed this in one regression-tested pass: rendering now requires the same GO/EOF pre-work gate, an unattached owned process is terminated and reaped through its original Popen handle, and cleanup exceptions are retained in a FAIL receipt. Tests first reproduced missing authorization, attachment failure and cleanup failure, then passed after the fix. No real extraction or render was repeated for the repair. The independent review covered the original code commit; this cleanup follow-up was author-verified with the final focused suite.

## Local assets

All generated assets and raw evidence remain in:

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-cpu-mesh-20261025-v1`

The actual PLY is `mesh/mesh.ply`. `previews/` contains `view-000.png` through `view-315.png`, `silhouette.png`, `eight-view-contact-sheet.png`, `front-head-detail.png` and the render metadata. Authentication, exact readback/topology, clarification and owned-stage receipts remain alongside them. Only code and this textual evidence are published.

**Next decision:** review the visible and measured topology limitations before separately authorising optimisation or texturing. This task stops with the genuine unmodified mesh and local visual evidence.
