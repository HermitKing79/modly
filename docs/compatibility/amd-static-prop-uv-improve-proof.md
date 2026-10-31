# Phase 20C — controlled UV improvement and conditional GLB export

Parent: `89bbbe7b6f050c8ef6efa9088a598f0a5d7557d9`. One authorised UV candidate and one conditional export, 31 October 2026.

**PASS — UV qualification and experimental neutral-PBR GLB export/readback. USER_APPEARANCE_APPROVAL_PENDING.** This is the separate LOWER-DETAIL STATIC PROP transport proof for the existing AI-DERIVED RECONSTRUCTION. Phase 11 remains permanently NO-GO. No geometry was regenerated, repaired, replaced, simplified or smoothed, and no game-ready or organic-asset qualification follows.

## Diagnosis before the attempt

Phase20B's density ratio **2.133809957682501** was predominantly projection compression within broad charts, not numerical interpretation or broken 3D geometry. On the five multiface charts, a fitted affine 3D-to-UV map had rank4, condition6.95–76.68, UV residual at most **4.85e-8**, and projection singular-value ratio within **2.22e-7** of one. Densities follow the orthogonal projection relation `scale² * abs(normal dot axis)` to maximum relative discrepancy **9.16e-5**. Face-axis angles reach **62.05 degrees**, explaining strong compression. Singleton charts are rank3 and do not identify a unique projection axis; no axis conclusion is drawn from them.

The pinned [Blender Smart UV implementation](https://github.com/blender/blender/blob/9e2066aef7ef/source/blender/editors/uvedit/uvedit_unwrap_ops.cc#L2924) averages projection directions, assigns faces to axes and projects coordinates. Our measured maps agree with that mechanism. Simply rescaling the old islands cannot fix within-chart variation.

The three old boundary-degree4 coordinates correspond to original candidate vertices **59, 206 and 356**. Each joins the coordinates of **two distinct seam-separated corner fans**, each with boundary degree2. All ten old charts are topological disks: Euler characteristic1, one boundary cycle and no bad fan degree. Thus the apparent nonmanifold boundary count came from coordinate welding; **there are real zero-area UV point contacts, but no demonstrated abstract chart-topology or 3D manifold defect**. This distinction is retained rather than calling the contacts absent or harmless for padding.

## Frozen alternative and suitability gates

The candidate's dihedral distribution has ordinary adjacent faces mostly below1 degree and major crease changes around44–51 degrees. The single **greater-than20-degree dihedral seam rule** was measured offline before any UV creation. It produced **26 disks, 280 seams**, one genuine singleton triangle, and maximum area-weighted-axis normal spread **17.623 degrees**. No assumed box dimensions, prescribed face count or per-triangle fragmentation was used.

The approved pipeline was first-party Blender **ANGLE_BASED** unwrap, `fill_holes=False`, aspect correction true, subsurf/original-bounds false; uniform `average_islands_scale(scale_uv=False, shear=False)`; then rigid rotation/global-uniform packing with `FRACTION` margin0.01, overlap merging false. ABF is not claimed to guarantee area uniformity. The exact settings and seam-plan hash were frozen in `amd-static-prop-uv-improve-strategy.json` before execution. Native unwrap can report solve failure despite returning FINISHED; the independent audit therefore checked its logs as well as its arrays. No solve-failure warnings occurred.

The original finite/in-bounds/nonzero/continuous-overlap and **area-density ratio<=1.1** gates were retained. Added predeclared suitability gates required the **exact planned face partition and seam set**, disk/fan topology, consistent chart orientation, no coincident distinct UV vertex fans, and maximum per-triangle directional singular-value ratio **<=1.1**. This last experiment-specific limit rejects area-preserving but badly anisotropic layouts; it is not a new universal Phase11 criterion. Intermediate ABF, normalized and final UVs were retained. No sweep, adjustment or retry occurred.

## Independent UV results

| Measurement | Phase20B | Phase20C |
|---|---:|---:|
| Area-density max/min | 2.13380996 — FAIL | **1.00513029 — PASS** |
| Independent Float64-cross ratio | 2.13380984 | **1.00513030** |
| Directional stretch max/min | Not a prior gate | **1.00223803 — PASS** |
| Charts / seams | 10 / 149 | **26 / 280**, exact planned partition |
| Topological disks | 10 | **26** |
| Coordinate/fan boundary coincidences | 3 prior boundary warnings | **0**, including across charts |
| Bad boundary fan degrees | 0 after proper fan analysis | **0** |
| Positive-area UV overlaps | 0 | **0**, complete exact SAT over2,331 broadphase pairs |
| Nonfinite / out-of-bounds / zero UV triangles | 0 / 0 / 0 | **0 / 0 / 0** |
| Atlas triangle-area coverage | 59.7613% | **67.2002%** |

Area density is UV-unit² per original asset-unit². Minimum density is **0.1822245871** at candidate face **388**; maximum **0.1831594515** at face **537**. Every triangle has its source/UV area, density, topology and directional-stretch evidence. Coverage is a Float64 nonoverlapping area sum, not a mip/padding certificate. Packing margin semantics differ from the old Smart UV setup, so the coverage increase is a measurement, not an attribution to seams alone.

Acceptance retains inherited Float32-cross/Float64-norm area arithmetic, with an independent Float64-cross check. Independent NumPy builds compare computed metrics at rel1e-9/abs1e-14; geometry/UV hashes, identities, verdicts, failures and structures remain exact. This comparison tolerance does not waive a1.1 gate. Export-serialized UVs were checked again: density ratio **1.00513594**, directional stretch **1.00223681**, with the same partition and all topology/overlap gates passing.

## Actual GLB and reimport verification

The experimental GLB is **94,652 bytes**, SHA-256:

`afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201`

- Working geometry remained exactly **502 Float32 positions and 1,000 oriented triangles**, with bit-identical coordinate/index arrays before and after every UV operation and export. The reused Float32 copy has the previously declared serialization bound; the original high-precision OFF and all neural geometry remain unchanged.
- GLB uses **2,695 referenced, unique attribute vertices**. Its **2,193 required duplicates** encode UV/flat-normal discontinuities; a bijection maps all1,000 oriented surface triangles and corners back to the502 original positions. No cosmetic vertex splitting was used to change the geometric topology verdict.
- Independent container parsing verified bounded accessors, scene/identity transforms, geometric topology, exact source-corner UV correspondence, material/texture references and packed texture pixels. Geometric boundary/nonmanifold/winding and vertex-link defects remain zero, with one component. The frozen geometry qualification is retained by unchanged correspondence; no CGAL binary was invoked.
- Actual isolated Blender reimport arrays were saved and independently checked. UVs match the explicitly rounded Float32 V-flip roundtrip exactly; flat-normal minimum dot against geometric face normals is **0.9999998797**. There are no unused or redundant exported attribute rows. Imported dimensions are **0.6446923 ×1.0014318 ×0.8072717**, unchanged from the prepared candidate. Identity transforms and one-metre-per-unit convention were verified; intended physical size remains unqualified.
- The embedded image and Blender-reimported packed PNG have identical decoded neutral pixels. The material is explicitly **NEUTRAL_TEST_PBR_NOT_AI_TEXTURE**, metallic0, roughness0.35, opaque and single-sided. This is transport validation, not finished AI texturing or a source-normal bake.

## Evidence, resources and preservation

Private root: `C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-static-prop-uv-improve-20261031-v1/`.

- **GLB:** `asset/experimental-static-prop.glb`.
- **UVs:** `asset/uv.npy`, `asset/uv-abf.npy`, `asset/uv-normalized.npy`; full `asset/uv-diagnostics.json`, independent `uv-audit.json`.
- **Readback:** `verification.json`, `reimport-verification.json`, `asset/export-blender.json`, actual reimport arrays/packed PNG.
- **Manifest:** `final-manifest.json`; `diagnosis.json` and `seam-plan.json` preserve the original-cause analysis and frozen patch plan.

No new appearance renders were made. The24 Phase20B images and their **INCONCLUSIVE_LIMITED_CLOSEUP_INFORMATION** assessment are unchanged. User appearance approval remains PENDING; normal baking remains SKIPPED; S&box remains NOT TESTED.

Nine bounded stages totalled **83.236 seconds**, excluding engineering/review. UV took **12.344s**, audit **3.031s**, export/reimport **7.640s**, independent verification **4.016s**. Peak owned Job commit was **364,503,040 bytes**; minimum observed Windows available RAM was **16,432,902,144 bytes**. All stages retained6GiB process/Job, four-core affinity,180s per stage,12GiB host floor and2GiB disk floor. Every tree exited naturally with an empty Job; no forced cleanup or resource escalation occurred.

Authentication/preservation checked the full **128-file Phase20B history**,44 Phase20 files,166 Phase19 files,314 Phase18 files,552 Phase14 files,47 original source bindings and188 protected identities. Original assets, criteria, environments, checkpoints, backup and historical holds are unchanged. **Seven focused tests passed**, including corner-fan topology, source row/cyclic correspondence, isometric layouts, equal-area anisotropic rejection, invalid/zero-layout diagnostics and audit, exact verdict/hash comparison, export refusal and no-render routing. Actual live reimport verification adds the full asset test.

Independent reviews approved diagnosis, the single strategy, code, settled UV qualification, conditional export and final output. Only reviewed code/configuration/text are published. No WSL, GPU/ROCm, neural operation, model replacement, installation, download, new reconstruction or CGAL rebuild/execution occurred. This does not approve CGAL binary distribution or Modly UI integration.

**Decision: the controlled UV/export proof passes.** Remaining requirements are informative material/distance views and explicit user appearance approval, finished PBR/texture and padding/mip validation, S&box units/import/collision/LOD tests, and separate distribution licensing decisions. No further experiment or asset promotion is initiated.
