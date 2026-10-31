# Phase 21 — Greyfield isolated static-prop engine proof

Parent: `ddc9a8162482f0347b192fb96f2421a138d7c7bc`. One controlled attempt on 31 October 2026.

**QUALIFIED PASS — conversion, S&box model compilation, isolated scene placement and neutral-preview rendering. Production use remains NO-GO pending separate qualification.** This is the optional LOWER-DETAIL STATIC PROP profile for the existing AI-derived reconstruction. User appearance approval remains PENDING. The original Phase 11 high-fidelity benchmark remains permanently NO-GO.

## Results

| Check | Result |
|---|---|
| First-party MCP connection | PASS — existing Greyfield editor, rediscovered PID 7656, engine 26.10.27c |
| Original GLB authentication | PASS — original SHA-256 unchanged before and after |
| GLB-to-OBJ geometry, UV and normal-direction transport | Qualified PASS; raw normal-magnitude test remains NO-GO and is explained below |
| Neutral source material packaging | PASS after explicit MTL metadata reconstruction; raw exporter output preserved |
| S&box import and compilation | PASS — actual compiled/up-to-date/non-error resource and loaded mesh inspected |
| Dedicated scene placement and save | PASS — separate new scene, verified rotation and unit scale |
| Geometry and dimensions | PASS for all 502 positions / 1,000 oriented triangles / 3,000 corners |
| Neutral preview appearance | PASS for visible intact rendering in four views; user appearance approval pending |
| Original PBR appearance in engine | NOT TESTED — preview material replacement explicitly used |
| Original Greyfield scene and project preservation | PASS — original scene saved/12 root objects; all 5,730 protected files unchanged |

## Authenticated source and import route

Original GLB: `artifacts/amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb`.

SHA-256: `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201`.

The exact parent commit was verified in the existing `proof` checkout. Its branch, HEAD, pre-existing untracked files and all 209 existing local refs were preserved.

The installed first-party skill and `sbox-cli` were used. Read-only discovery/status confirmed the existing Greyfield project with `greyfield lab.scene` saved, Play off and compilation idle. The initial connection investigation was not repeated. The sandbox hides the editor process, so host execution was used for the local named-pipe MCP calls.

There is no raw-mesh MCP import tool; `create_asset` was not used for a mesh. Installed Blender 5.2.1 LTS imported an authenticated isolated GLB copy and performed exactly one OBJ export. Its only process ran in the background with factory startup, auto-execution disabled, four threads and a 180-second deadline. Conversion took about 3.4 seconds; no render, neural workload, dependency installation or export retry occurred.

The documented public `Editor.EditorUtility.CreateModelFromMeshFile` utility was reviewed independently before execution, including its fixed target-path, asset registration and compile scope. It refuses an existing target and has no explicit hull, physics-mesh or LOD-generation calls. A fixed-path script additionally checked the active Greyfield Assets root, exact source OBJ SHA-256 and absence of source/compiled/indexed target resources. It was invoked once. The utility's ignored compiler return value was not treated as success; actual asset status and loaded-resource data were checked separately.

No console-command route or gameplay/source-code change was used. Other C# calls were independently reviewed read-only public model inspection and pure rotation arithmetic. Decompilation/API review evidence stays private.

## Conversion fidelity and declared normal qualification

Frozen coordinate mapping: `OBJ inches = (-GLB.z, -GLB.x, GLB.y) / 0.0254`, a proper rotation with determinant +1. This changes coordinate representation and world units while preserving physical size.

Independent standard-library parsers read the authenticated GLB and actual OBJ. No Blender arrays were substituted for the exported file.

| Measurement | Result | Declared acceptance |
|---|---:|---:|
| Unique geometric positions | 502 | Exactly 502 |
| Oriented triangles / corners | 1,000 / 3,000 | Exact complete cyclic correspondence |
| GLB / OBJ position attribute rows | 2,695 / 2,695 | Required discontinuity duplicates retained |
| Missing, reversed or extra faces | 0 | 0 |
| Maximum position error | 4.18946e-8 m | 2e-6 m |
| Maximum UV component error | 4.99943e-7 | 2e-6 |
| Maximum unit-normal direction error | 6.52231e-8 | 1e-5 |
| Minimum unit-normal dot | 0.999999999999998 | Reported direction evidence |

The initial raw-vector check failed: 870 corners on 290 faces exceeded a raw normal-vector Euclidean threshold of 1e-5, with maximum 6.87954e-5. OBJ stores four-decimal normals whose lengths range from 0.999931208 to 1.000067728; source GLB normals are approximately unit length. Independent diagnostics separated vector length from direction. Comparing unit directions at the unchanged 1e-5 directional threshold passes every corner. This is a documented correction to comparison semantics, not a relaxed directional limit, altered mesh, normalized output or second export. The initial `conversion-raw-audit.json` NO-GO and all raw outputs remain retained. Raw vector magnitudes are not claimed identical.

The raw MTL also contained a dangling texture path and omitted explicit diffuse modulation. A separate `verified_source` package preserves OBJ and PNG bytes exactly, changes only `map_Kd` to the adjacent identical PNG basename, and adds `Kd 1 1 1` from the authenticated GLB's white base-color factor. This is declared neutral-material metadata reconstruction. The original MTL remains unchanged. The independent gate verifies decoded PNG pixel identity, roughness 0.35, metallic 0, opacity 1 and the explicit modulation. It does not infer engine shader equivalence.

## Actual engine resource and fidelity

New resources, relative to Greyfield's Assets root:

- Model: `modly_tests/phase21_static_prop/experimental_static_prop.vmdl`.
- Verified source: `modly_tests/phase21_static_prop/experimental_static_prop_inches.obj` and adjacent MTL/PNG.
- Saved scene: `modly_tests/phase21_static_prop/modly_phase21_staticprop_test.scene`.

S&box canonicalizes the requested `Modly_Phase21_StaticProp_Test.scene` resource path to lowercase. No pre-existing file was overwritten.

`asset_info` reports `IsCompiled=true`, `IsCompiledAndUpToDate=true`, `IsCompileFailed=false`; `asset_files` reports zero unresolved references. The model loads with `IsError=false`. The inspected mesh contains 2,695 attribute vertices, 3,000 indices and a LOD0 index count of 3,000. Its referenced default material is also non-error.

The public `GetVertices` / `GetIndices` APIs are marked experimental. Their complete returned single-model data was independently compared; no claim is made about additional unexposed draw-call or LOD streams.

ModelDoc applies a further measured proper rotation: `Engine XYZ = (OBJ.z, OBJ.x, OBJ.y)`. The unique rotation was selected using geometry alone, with full oriented-face correspondence, before checking attributes. It was not fitted separately to UVs or normals. ModelDoc's authored import scale is exactly 1, with zero authored import translation/rotation.

| Engine measurement | Result | Declared acceptance |
|---|---:|---:|
| Positions / oriented triangles / corners | 502 / 1,000 / 3,000 | Exact correspondence |
| Maximum recovered position error | 4.18946e-8 m | 2e-6 m |
| Missing/reversed/extra faces or unused readback vertices | 0 | 0 |
| Maximum UV error | 0.000487644349 | 0.0005, one consistent global convention |
| UV convention | Uniform V flip relative to OBJ | No per-face convention fallback |
| Maximum unit-normal direction error | 0.00369818105 | 0.015 |
| Minimum unit-normal dot | 0.999993161728 | 0.9999 |
| Corners failing engine normal gates | 0 | 0 |
| Physics body parts | 0 | 0 |

Compiled normal directions differ slightly within the declared tolerance; they are not claimed bit-identical, and the exact packing cause was not independently verified. Complete corner tests retain the original sharp/normal discontinuities within the stated engine tolerance. UVs have measurable serialization error; production atlas padding/mip behavior remains untested.

The `.vmdl` contains one render-mesh node and material nodes, with no physics/collision/LOD generation nodes. No collision or LOD generation was requested. Extra compiled LOD coverage is not qualified by the selected public readback route.

## Saved scene, rendering and dimensions

The separate scene was created using the supported GameResource `create_asset` tool, which refuses overwrite, then opened as another tab. Its active path was checked before placement. Dedicated scene tools placed the imported ModelRenderer, neutral key/fill/ambient lights and four proof cameras. Only this scene was saved.

The inverse importer rotation was checked using public rotation basis vectors before use. The prop has world rotation `-90,-90,0`, world scale `1,1,1`, and position `0,0,15.8921919` inches. This restores the verified OBJ axes and puts its bottom at approximately zero. There is no second metre-to-inch scaling.

Scene XYZ dimensions are approximately **1.001431855 × 0.644692310 × 0.807271741 metres**, equivalent to the original 0.6446923 × 1.0014318 × 0.8072717 metre extents after the documented axis rotation. Intended production size and gameplay viewing distances still require approval.

The model resource's default material is `materials/default.vmat`; the scene deliberately uses `materials/default/white.vmat` with neutral gray tint 0.5 as a **preview replacement**. Original roughness-0.35 PBR equivalence, source texture sampling and normal baking are not established by these images.

Four 1280×960 native MCP camera images show front/rear three-quarter, side and top views. A fifth native MCP editor-camera screenshot provides viewport proof. Images are copied byte-for-byte to local evidence with SHA-256 manifests; no pixels were edited. The model is fully framed, with coherent silhouette and bevels, finite faceted shading, and no obvious holes, inverted faces, tearing or missing-resource coloration. Uniform preview color does not visually test UV chart behavior; numeric UV correspondence is the evidence for transport.

Local screenshots under `artifacts/amd-static-prop-sbox-engine-proof-20261031-v1/screenshots/`:

- `three-quarter.png`
- `rear-three-quarter.png`
- `side.png`
- `top.png`
- `editor-viewport.png`

The new scene remains active, saved and framed on the selected prop. The original `greyfield lab.scene` remains open, saved and at its original 12 root objects. Play was never entered.

## Verification and preservation

Two independent verification modules and **17 local adversarial tests passed**, including wrong units, reversed winding, missing triangles, changed/scrambled UVs, normal inversion, unused vertices, collision and LOD configuration rejection. These tests derive fixtures at runtime from the private authenticated GLB; they remain local and are not a portable public CI suite. Both actual source-package and compiled-engine data pass their qualified gates. The public read-only verifier modules require authorized private input files supplied through their CLI options.

Final live status: compilation idle, Greyfield game/editor compilers successful with zero errors and zero warnings; another installed retargeter library reports 62 warnings in its compiler state. No new Warn/Error console output appeared after the import-time cursor. No source edit in this task modified or triggered a rebuild of that library.

Author and independent reviewer preservation checks both found all **5,730** baseline authored files unchanged, including original Greyfield scene/assets/code/settings, the authenticated GLB, its prior UV/export evidence and tracked Modly source/history. No new authored project files appeared outside `Assets/modly_tests/phase21_static_prop/`. All 209 pre-existing local refs and the checkout's pre-existing untracked state remain unchanged. GitHub publication uses a new branch; existing branch preservation is checked separately.

Private evidence includes raw/qualified conversion audits, material packaging records, reviewed operation scripts, native MCP requests/results, full mesh snapshot, independent engine/scene/visual audits, five screenshots, project artifact hashes and preservation manifests. Private game assets, mesh arrays, screenshots, asset-dependent test files and first-party decompilation payloads are not published. Only the two separately reviewed read-only verifier modules and this textual report are selected for GitHub publication.

## Remaining requirements and stop decision

Explicit user appearance approval for the optional LOWER-DETAIL STATIC PROP profile remains pending. Before Greyfield production use: qualify intended size/viewing distance, original/final PBR shader and texture transport, finished materials, UV padding/mips, normal baking if desired, separately authorized collision/LOD behavior, gameplay performance and readiness, and distribution/licensing requirements.

Normal baking, production textures, original engine PBR equivalence, collision behavior, LOD behavior, Play-mode behavior and actual gameplay readiness remain NOT TESTED. No existing gameplay, startup scene, prototype object or project setting was changed. No source geometry was regenerated or promoted. No further experiment is initiated.
