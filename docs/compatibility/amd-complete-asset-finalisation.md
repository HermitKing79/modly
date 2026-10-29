# Phase 3: complete-asset geometry finalisation

**NO-GO at asset admission. No finalisation experiment was run and no output
was promoted.** The closest authenticated complete non-foliage prop is the
turret, but its intended source sheets/openings are not sufficiently classified
for the permitted local operation. This is not a claim that all finalisation
methods are impossible.

## A. Inventory and selection

Inspected the preserved Modly workspace, its source/evidence records and the
existing Windows ComfyUI input/output roots. These are the inspected locations,
not an exhaustive search of every disk or an inspection of WSL storage.

| Existing source family | Admission decision |
| --- | --- |
| Complete mechanical turret | Genuine, complete, non-foliage, with preserved reference/provenance/mesh. Closest candidate; rejected for this experiment because intended sheets and openings at the known defects are not classified. |
| Complete brick/vine | Genuine reference and mesh; accepted extensive source defects and complex foliage make it unsuitable as the fallback simple solid. Not re-audited. |
| Zombies A/C/D and derived variants | Genuine character references, excluded by this directive; not prop substitutes. |
| Brick panel, foliage exports, local patch/retopology/LOD variants | Crops or derivatives of those sources, not additional complete generated props. |
| Synthetic cubes/planes, application replay GLBs and compiler `.obj` files | Test fixtures or compilation products, not AI-generated asset evidence. |
| Two older `Pistol_*.glb` files in ComfyUI output | Byte identities established; generation/reference provenance not recovered, so not admitted. The filenames do not establish their subject or completeness. |

**No suitable complete asset was selected.** No simpler authenticated prop with
the required source-and-intent contract was found in the inspected evidence.

### Closest authenticated candidate: turret

Fresh streaming SHA-256 checks matched the original published identities:

- `artifacts/amd-turret-cpu-mesh-ad07f96/turret-full.ply`:
  `89549a36c6acc6f884174a2097ad7e346980d565c999117ea29f89cda9baad1a`.
- Preserved `selected-input.png`:
  `07557b7b9f726aad3b291f4902498e52423efadbf746fb2c5759e3e2c6adeef0`.
- Existing input/front/rear/side comparison:
  `ed3076db43a4fd106ade089e6e9e01fd06acd7f1b04429a4a2e64b34ae1a92cc`.

Original NPY arrays also matched their published full hashes; exact identities
are retained in the private manifest and linked extraction report.

The accepted baseline has **1,842,968 vertices, 4,012,360 triangles, 164,154
boundary edges and 338,449 edges with more than two incident faces**, maximum
incidence four. These are reused measurements, not a new mesh census.
Full-asset winding, intersections and source-sheet classifications are not
established by those counts. No missing measurement is treated as zero.

The reference and three existing CPU previews identify housing, pedestal,
four barrel structures, cables, inter-barrel gaps and lower muzzle openings to
preserve. They do not map every source boundary/branch to an intended part.
The previews are clipped/noisy; no new silhouette score is claimed. Neither
component size nor an open edge establishes disposable or missing geometry.

Prior face-fan/Collapse and Planar trials did not qualify a complete result;
they were not repeated. See [source extraction](amd-turret-cpu-mesh-extraction.md),
[existing views](amd-turret-visual-preview.md), [fan trial](amd-turret-face-fan-proof.md)
and [LOD qualification](amd-turret-lod-candidates.md).

### Older unqualified exports

Both GLBs have a single triangle primitive, POSITION only, a Trimesh generator
string, no source extras, materials or images:

| File | Bytes | Declared vertices / triangles | SHA-256 |
| --- | ---: | ---: | --- |
| `Pistol_00001_.glb` | 52,916,404 | 4,235,349 / 174,286 | `c45f8ffc556e581c631db5249a41e3bc9567c9d72cf304c56e17df8402030476` |
| `Pistol_00002_.glb` | 9,189,368 | 265,717 / 499,999 | `bf28497783f900c818563f2acfc9206c2a7325a3814a1555893c9b3e13af965e` |

These are header declarations, not geometry-quality measurements. The saved
`TextureOnly_Pistol_Test.json` loads the second export with the **zombie**
screenshot as its texture reference. It is not a generation receipt. No matching
source-generation contract was recovered; a targeted read-only ComfyUI database
query found no Pistol asset records. This does not prove a reference exists
nowhere. Fresh byte hashes cannot establish lineage. Both exports remain intact.

## B. Existing finalisation code: availability, not execution qualification

Read the installed exporter and CuMesh Python sources without importing their
packages. Exporter SHA-256 remains
`35b62242503944cb16c25b7a1c8730b632b2f55d7a1f3b0b6dd2051779908e4b`;
installed remesher remains
`b38a22123c381c99afff23613d188475b26af24bec04a6774100f66be76b029d`.
The accepted [upstream comparison](C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-geometry-quality-baseline-20261029-v1/REPORT.md)
was reused. Relevant entrypoints are `o_voxel/postprocess.py:to_glb`,
`cumesh/cumesh.py:CuMesh` and `cumesh/remeshing.py:remesh_narrow_band_dc`.

| Stage present locally | What it actually entails / current limit |
| --- | --- |
| Initial cleaning | Exporter fills perimeter-bounded holes **before** building the reference BVH. It cannot be enabled under this task's ban on blanket filling, nor assumed to preserve intended openings. |
| Non-manifold handling | CuMesh wrapper documents vertex splitting with duplicate coordinates. This is distinct from its face-deleting method and Modly's rejected PyMeshLab default. A lower incidence counter does not certify intended sheets or a closed exterior. Native implementation was not executed or independently revalidated. |
| Small components | Standard branch calls removal with `1e-5`. There is no semantic classification of cables, barrels or separate parts; no threshold is adopted here. |
| Simplification | Standard branch simplifies to three times the requested target, cleans twice around final simplification, then unifies orientations. Backend target is **faces**, despite the exporter docstring describing vertices. Simplification does not establish surface intent. |
| Narrow-band reconstruction | Alternative branch reconstructs connectivity from an unsigned-distance offset field using native grid/hash/dual-contouring kernels, then simplifies. It is not a CPU function or a demonstrated feature-preserving repair for these sources. |
| Projection | Installed positive-project path performs three neighbour-averaging iterations before nearest-source projection. It changes positions and may select another nearby sheet. It is not an isolated harmless snap. |
| Final orientation | Standard branch calls native unification. This changes winding, not incidence or ambiguous sheet membership; the remesh branch does not run that same standard cleanup sequence. |

For a future **unambiguously closed simple solid**, reconstruction plus measured
projection could replace branched connectivity rather than merely reindex it.
This is a direction to qualify, not a predicted PASS or an admitted method.

The exporter uses GPU tensors, native CuMesh/cuBVH, FlexGEMM and NVDiffRast;
remeshing needs native grid/hash/DC kernels. Local CPython 3.12 ROCm-labelled
wheels exist, providing packaging evidence, not AMD execution proof. `.cuda()`
names alone do not mean NVIDIA-only: ROCm PyTorch can use that API. There is
**no qualified CPU route** through this exporter. No native code was loaded.

Installed Blender, PyMeshLab, Open3D, MeshLib and CPU analysis tools supply no
qualified drop-in equivalent for this sequence and ambiguous source sheets.
Previous failures remain in force. No adapter, framework or kernel port is built.

## C. Outcome and single next technical action

**Experiment: NOT ADMITTED, not an executed FAIL.** No variant, improved
geometry, texture-readiness promotion or UV/PBR bake exists. Source defects
still need interpretation before a specific safe correction can be nominated.

The minimum next input is **one neutral three-quarter reference of a single
plain closed storage box with chamfered corners**, without a lid seam, holes,
foliage, handles, damage or attached parts. Declare one connected closed solid
as its intended geometry. Separately authorise **one** unchanged qualified
512-resolution TRELLIS.2 generation, preserving image/preparation hashes,
conditioning/occupancy/shape/decode provenance and the untouched raw mesh before
any finalisation. That removes the present opening/part ambiguity and permits
a falsifiable complete-asset test. It does not guarantee a clean raw result or
authorise rebuilding the exporter. No reference or model is generated now.

## Review and preservation

Independent review agrees with Stage A STOP and the proposed minimal input.
Only text is published on `codex/amd-complete-asset-finalisation`; no executable
or production change required new tests. Checks were streaming SHA-256,
bounded GLB headers, existing-record/visual review and static source inspection,
not numerical finalisation tests. No full-asset geometry audit was repeated.

Private inventory, fingerprints and before/after preservation receipts are in
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-complete-asset-finalisation-20261029-v1`.
Inspection kept original assets, installed sources, environments, protected
records and old evidence unchanged. Existing RAM/VRAM/disk/process/deadline and
owned-cleanup safeguards were not changed; no mesh-processing worker required
admission or ran. No WSL, GPU, neural work, installation, downloads, destructive
repair, new render, UV or bake ran. No live execution handoff is created.

Preservation checks passed for 3,346 historical file hashes, all 100 protected
identities and 88 retained records, 36 scoped inputs/old-evidence files and all
183 pre-existing local refs. This includes selected installed source identities,
not a whole-environment byte audit.
