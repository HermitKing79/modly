# AMD CPU UV and texture baking proof v1

Decision: **NO-GO for native-component or full-Zombie-D texture integration.** The separate installed Blender CPU fallback passed the tiny numerical and textured-GLB proof. Zombie D retained its exact surface geometry, but its diagnostic bake failed colour accuracy and is rejected. No AI texture generation occurred.

Base: `9cbc5709753170f108d08e036750ebc99d3762a2`. Isolated branch: `codex/amd-cpu-uv-bake-proof-v1`. Machine-readable aggregate evidence: [verification JSON](amd-cpu-uv-bake-proof-v1.json). Private case: `artifacts/amd-cpu-uv-bake-proof-20261026-v1` outside this Git repository.

## Native components and application audit

| Requested proof | Result |
| --- | --- |
| Modly native CPU UV unwrapping | **FAIL / blocked before execution**: shipped extension requires CPython 3.10 |
| Modly native CPU baking | **FAIL / blocked before execution**: same ABI mismatch; independently reproduced Python API defect |
| Blender CPU UV, bake, textured GLB export/re-import, tiny meshes | **PASS**, explicitly a fallback |
| Zombie D GLB geometry, UV/material/image references and embedded pixels | **PASS** container/readback checks |
| Zombie D diagnostic texture quality | **FAIL** independent colour reference |

Both supplied `_C.cp310-win_amd64.pyd` binaries import `python310.dll`, `torch_cpu.dll` and `c10.dll`. Their identities are recorded privately; no binary was loaded. The configured Modly application and the inspected TRELLIS/Hunyuan extension environments are Python 3.11.9; the qualified proof runtime is 3.12.10. These binaries cannot provide operators for those ABIs. No matching installed 3.10 runtime was found in the inspected runtime locations. Torch C++ ABI compatibility is also unverified. Neither installation nor compilation was attempted.

CPU implementations do exist: [UV CPU registration](../../api/uv_unwrapper/uv_unwrapper/csrc/unwrapper.cpp) and [baker CPU registration](../../api/texture_baker/texture_baker/csrc/baker.cpp). Their setup scripts require a Torch C++ extension build; the baker has optional CUDA/Metal paths, while CPU-only C++ implementations are present. Presence in source is not execution proof.

There is a confirmed [TextureBaker API defect](../../api/texture_baker/texture_baker/baker.py): `forward()` calls `interpolate(attr, rast, face_indices, uv)` although its signature accepts only `attr, rast, face_indices`. A test executes that original class with mocked rasterization and reproduces `TypeError` without importing Torch/native operators. The original component remains unchanged.

The [UI options](../../src/areas/generate/components/GenerationOptions.tsx) and [request hook](../../src/shared/hooks/useApi.ts) forward texture enablement/resolution to the [generation backend](../../api/routers/generation.py), which delegates to extensions. No shared CPU UV/bake postprocessing route was found. The locally installed TRELLIS extension's `texture_mesh` workflow invokes neural texture diffusion; that is separate from baking existing attributes. The [GLB export endpoint](../../api/routers/export.py) serves an existing GLB; it does not invent UVs or colours. Its other-format conversion does not establish texture preservation for every format.

## Executed fallback and numerical checks

Blender 5.2.1 LTS, factory-empty background process, Cycles **CPU**, four threads, emission bake, one sample, no denoising or geometry modifiers. The installed Blender fallback does not use Modly's native operators. UV cube/D settings: Smart UV Project, 66 degrees, island margin 0.002, area weight 0, aspect correction on, scale-to-bounds off. No parameters were tuned after failure.

| Tiny mesh, 128-square texture | Unit-square plane | UV-projected cube |
| --- | ---: | ---: |
| Triangles | 2 | 12 |
| Continuous UV area sum | 1.0 | 0.66489227 |
| Strict-interior covered texel centres | 16,256 / 16,384 | 10,795 / 16,384 |
| Sampled overlapping interiors | 0 | 0 |
| Maximum linear RGB error | 0.0000156164 | 0.0000471324 |

An independent CPU barycentric reference checked every uniquely covered strict-interior texel against known vertex colours, with a fixed maximum absolute error of **0.002**. Shared-edge centres were excluded. Zero-area/nonfinite/out-of-range UVs were rejected. Continuous island intersection was not exhaustively tested. Saved PNG RGB and embedded GLB RGB were byte-exact against the explicit linear-to-sRGB encoding. Both actual Blender re-import and a separate binary glTF/accessor/PNG parser verified geometry and material references.

Exported per-corner UV association was not independently compared with the pre-export bake mapping. GLB PASS therefore covers format, texture presence/pixels, valid UVs and preserved geometry; it does not establish exhaustive mapping correspondence or production suitability.

An initial evidence field mistakenly called opaque bake alpha “coverage”. Its original record is preserved; `small-coverage-clarification.json` records the actual independently rasterized coverage above. Alpha is not UV coverage. No bake was repeated to correct this interpretation.

## Zombie D result

The original PLY SHA-256 was verified before copying:
`2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205`.
It has **356,876 vertices and 777,400 triangles**, with XYZ/indices only: no usable original vertex colours, UVs, materials or decoded skin texture. The isolated copy received a clearly labelled **position-normalized diagnostic RGB pattern**. It is not AI skin or reconstructed source appearance.

UV projection preserved coordinate/index payloads exactly. UVs were finite and nondegenerate, but their area sum was only **0.0125138469**, about **1.2514%** of the atlas. At 2048-square resolution every source triangle had less than one texel of UV area; the mean was **0.06752 texel²**. This is a measured atlas-quality problem, not a suitable layout for this detail reference.

The independent colour check failed: maximum error **0.9951595068**, versus the unchanged **0.002** tolerance. No rebake, UV tuning, higher resolution or retry followed. Poor atlas utilization/subpixel coverage is observed; it has not been isolated as the sole cause of the numerical failure. The rendered diagnostic output visibly adds widespread black/blocky texture defects and must not be used as a successful texture result.

A separate readback-only closeout verified the rejected GLB's embedded RGB pixels, UVs, opaque material and exact oriented triangle-coordinate multiset. All 777,400 surface triangles, their positions and winding are unchanged, including fingers, wounds, face and silhouette. GLB has 2,327,505 vertices because export splits UV/normal seams; these duplicate coordinates do not represent added surface geometry or topology repair. No original defect was repaired.

Rejected candidate GLB SHA-256:
`a54ee89bfda48e13be462ea07e517a98efe23795196a88960342fb30e3ca9316`.

All twelve matched CPU preview images were saved: neutral original and diagnostic output for front, back, both hands, face and wounds. The preview stage then **failed** writing metadata because a NumPy Float32 camera value was not JSON serializable; its partial JSON/stderr remain unchanged. A small scalar-conversion fix is unit-tested, **not verified by rerunning Blender**. A separate image-only closeout composed those existing PNGs into `previews/comparison-sheet.png`; it does not turn either failed stage into PASS. Same camera, scale and lighting are applied to each pair in the rendering loop. The original cube's `cube/preview.png` is a successful tiny-proof preview.

## Resources and preservation

Each owned stage retained the existing **6 GiB process private-commit cap, 180-second deadline, 12 GiB Windows available-RAM floor, 2 GiB free-disk floor and owned Job cleanup**. The 22 GiB neural admission gate remains unchanged; it was not used as a replacement CPU gate.

| Stage | Seconds | Peak process private commit, bytes |
| --- | ---: | ---: |
| Synthetic bake/export/import | 6.563 | 503,091,200 |
| Independent tiny verification | 0.562 | 25,726,976 |
| D UV only | 7.797 | 1,178,005,504 |
| D bake/export/import | 15.718 | 2,505,277,440 |
| D colour verification, FAIL | 6.578 | 314,540,032 |
| Rejected GLB readback only | 1.610 | 270,397,440 |
| Preview, metadata FAIL | 27.719 | 1,462,755,328 |
| Existing-image closeout | 0.531 | 24,457,216 |

Total owned-stage elapsed time: **67.078 seconds**. Peak private commit: **2.333 GiB**. Lowest sampled Windows available RAM: **16.645 GiB**, above the live floor. Job peaks are commit measurements, not physical working set; host available RAM also includes concurrent host activity. All eight child processes exited naturally; owned trees were empty, with no forced cleanup or cleanup errors. No limit was raised and no workload was retried. Mock/unit test iterations and read-only closeouts are not generation retries.

Eight focused unit tests passed, covering the original baker argument defect, geometry/winding identity, PNG encoding/origin, UV validation, retained failure handling, JSON-safe camera scalars and model/subprocess/foreign-write denials. The no-argument launcher dry-run exited successfully without starting a worker. The broader application/model test suites were not run for this isolated CPU proof.

Independent read-only code/evidence review found no critical or important issues and approved publication as a proof-only NO-GO report. Its nonblocking request to clarify the unverified exported UV association is incorporated above. Review did not run or retry any workload.

Closeout rechecked **3,255 historical/native files by SHA-256 and metadata**, all **100 protected identities plus 88 inherited retained records**, and **172 existing refs before publication**. Original tracked files, assets and environments were unchanged. No WSL, neural/model imports, inference, dependency installation, repair, smoothing, welding, remeshing or hole filling occurred. Models, images and raw process evidence stay local.

## Remaining interface work; stopped here

1. Establish a matched Python/Torch CPU-extension installation and fix the original baker argument defect, then prove the actual native operators on the tiny references. That requires separately authorized build/install work; this fallback does not qualify them.
2. Before full-mesh use, solve and independently validate usable UV coverage and colour accuracy. The failed high-detail D layout is not evidence that a production asset can be baked reliably. No further UV experiment is authorized by this report.
3. Only after those pass, add a bounded CPU postprocess API/job and an explicit UI mode distinct from AI texture generation. Validate resolution limits, per-corner UV seam indexing, sRGB/material handling, missing-colour labels, export/reload and safe artifact registration. Do not expose a successful “textured model” state for rejected bakes.

The proof establishes a working small **Blender CPU fallback**, not working native Modly components or a production-ready Zombie D texture. Stop for review.
