# MODLY Phase 24 — masonry PBR transport admission review

**NO-GO at Stage 1 for the existing OBJ route. Stop before conversion.** The authenticated panel passes the independent source-identity and layout audit, but the reviewed OBJ normal serialization has not qualified at the frozen direction tolerance. Source-specific arithmetic models exceed that tolerance. No Blender process, capability probe, converter, export, alternative conversion, render or S&box editor operation ran in this phase. No converted asset exists; actual converted-file readback is **NOT RUN**.

This is the conditional instruction's pre-execution stop, not an observed failed export and not a complete-prop assessment. One justified alternative, installed first-party binary FBX, is identified below for separate consideration. It was not executed. Phase 11's permanent NO-GO, Phase 22's strict sampling NO-GO and USER_APPEARANCE_APPROVAL_PENDING remain unchanged.

## Parent and authenticated inputs

GitHub parent: `1655252c30d9697f57fd6a50a449caf47bb5edfa`, tree `b839abc2d9ec77f4f73bd5b1c1c4a7d3aa0f9f6f`, sole parent `8cc55a93d34911854e59d9e6fbf49e1a3321dfd4`. The immutable remote Phase 23 report was fetched and independently matched byte-for-byte to the local reviewed report: 20,529 bytes, SHA-256 `c8e5282a2fbaba2666f345b110c4f1d153603d2d03d2f68cd9d3e3e9bbc04ce5`, Git blob `032f9feed6a6eb4ceea2427af72f892f697a0f64`. The selected revision is the reviewed mip-safe panel, not its earlier failed atlas variant.

| Input | Exact private location | Bytes | SHA-256 |
|---|---|---:|---|
| GLB | `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-mip-safe-atlas-e8244cb\revision\run\candidate\patch.glb` | 638,700 | `dc772cd55fc523c5427ab4771da27e62cd6905d5109f00b2fca47403704da857` |
| Base color | `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-mip-safe-atlas-e8244cb\revision\run\candidate\bake\base-color.png` | 540,248 | `0048a6342df959735cb4687dc22c9a1c4f013459bb5ffb93d94d4ed0080652ad` |
| Metallic/roughness | `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-mip-safe-atlas-e8244cb\revision\run\candidate\bake\MR.png` | 17,162 | `307f5a0c17c974c4fadeac0a2ae16c00bcd0368a666c92313a943282cf05314f` |

Adjacent bake-array identities were also reauthenticated: `vertices.npy` → `144dad7717dcaf45bb5cccc1129152a5dc8265d25046c603cfaaf54425bf807c`; `faces.npy` → `996560f4a812d4751d0d6ec494abfdd9aea122c5788ff839d4140d6d1831ee46`; `uv.npy` → `61ae6a3c231aa2cc74f7995fda91da9ae5fe32ffe48227fd0633280e00839656`; `bake.json` → `370ffe817cb7aff3882048f0cf5512f464e8e8cb7df7551cecf323633feeb33c`; `face-island-ids.npy` → `16162bfbbb17d30a7ec4fcbe0e07e0b13065296078e0a7012d7ffaa60c875f09`. No input or checkpoint was rewritten.

## Independent source readback — not conversion readback

A separate standard-library GLB/NPY/PNG parser inspected every source triangle and corner, without Blender or its in-memory arrays. Another reviewer independently decoded the normal accessor and reproduced the implementation-informed arithmetic models.

| Source finding | Independently observed result |
|---|---|
| Geometry | 781 ordered oriented triangles; 457 geometric positions; 2,343 attribute corners, each referenced once |
| Face preservation against adjacent bake arrays | All 781 match in order, winding and multiplicity; no duplicate oriented faces or zero-area faces |
| Position correspondence | Maximum Euclidean difference **0 m**, after the proper bake-to-glTF rotation `(x,y,z) → (x,z,-y)` |
| UV0 correspondence | All 2,343 corners checked; maximum component difference **2.9802322387695312e-8**, with the recorded `(u,v) → (u,1-v)` convention |
| Scene structure | One identity-transform mesh node; no animation, skin or morph targets |
| Normals | All finite/nonzero; source lengths **0.9999999435060836–1.0000001207596383**; maximum absolute unit-length deviation **1.2075963828905856e-7** |
| Connectivity | Four edges with incidence three, 147 boundary edges; no repair or new intersection qualification |

The glTF XYZ bounds are minimum `(0.19012212753295898, -0.09282952547073364, -0.10188096761703491)` and maximum `(0.19571399688720703, -0.04551100730895996, -0.06921809911727905)`. Declared dimensions in metres are **0.005591869354248047 × 0.04731851816177368 × 0.03266286849975586**. These are source dimensions under the glTF unit convention, not a new real-world calibration or a converted-file scale result.

Source audit PASS does not establish OBJ, FBX or engine preservation. **Actual converted-file maximum position, UV and normal-direction errors: N/A — conversion not run.** Actual output winding, dimensions, normal magnitudes, texture packaging and PBR transport are likewise unmeasured.

## Frozen gates and historical normal failure

[ACCEPTANCE.md](ACCEPTANCE.md) was frozen before any converter execution. It requires exact 781-face winding/multiplicity and complete corner correspondence; position error ≤2e-6 m; UV component error ≤2e-6; and unit-normal direction chord/L2 error **≤1e-5 at every corner**, approximately 0.000572958°. Output normals may not be rewritten to make the comparison pass.

Keep three different metrics explicit. Phase 23 proposed a raw-component 1e-5 check. Phase 21's actual diagnostic used raw-vector Euclidean 1e-5. Phase 24 also requires the separately unit-normalized direction comparison at 1e-5. Neither historical raw threshold is relaxed, and normalization of temporary comparison operands cannot relabel a failed raw comparison as PASS.

The Phase 21 source was a different GLB, SHA-256 `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201`. Its actual OBJ failed the raw Euclidean gate at **870 corners / 290 faces**, maximum **6.879543022580293e-5**. Its separately measured unit-direction maximum was **6.522307083676927e-8**, which passed. This explains why vector magnitude and direction must be separated; that old asset's directional result does not qualify this panel. The preserved diagnostic SHA-256 is `bccfe4e451f907377075fa306aa57d56aec739a3a9e13e61aa109942a6613e0a`.

## Why the OBJ route is not admitted

The installed Blender executable is `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`, 113,014,232 bytes, SHA-256 `284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb`. Windows file metadata says 5.2. The separately retained Phase 21 RNA snapshot says 5.2.1 LTS and exposes `bpy.ops.wm.obj_export` without a normal-precision/digits control. Its SHA-256 is `4133f9e3eaf334b7de86541fcfc72b1b253bc59ca6bb8d5540f03a8729f2efd7`. No fresh version or RNA probe was launched.

The [official v5.2.1 normal preparation](https://raw.githubusercontent.com/blender/blender/v5.2.1/source/blender/io/wavefront_obj/exporter/obj_export_mesh.cc) normalizes transformed normals and rounds components to four decimals before deduplication. The [official OBJ text writer](https://raw.githubusercontent.com/blender/blender/v5.2.1/source/blender/io/wavefront_obj/exporter/obj_export_io.hh) emits four-decimal normal components. The native C++ source is not shipped in the installed tree; this version-tagged implementation and retained actual-output evidence support the precision challenge, without proving every instruction in the installed binary.

The route review modeled source normalization followed by the implementation's float32 component-rounding expression and four-decimal serialization. Double-precision normalization and separately emulated scalar-float32 normalization gave the same aggregate result. An independent direct nearest-decimal source model also gave that result, but is a distinct model. A separate ideal-rounding sensitivity calculation differs at decimal boundaries.

| Source-only model | Maximum unit-direction chord | Corners / faces over frozen 1e-5 |
|---|---:|---:|
| Implementation-informed float32 rounding; both normalization variants | **3.1473214219013556e-5** | **27 / 9** |
| Independent direct nearest four-decimal source model | **3.1473214219013556e-5** | **27 / 9** |
| Double normalization, then ideal nearest-decimal sensitivity model | **3.3137410518395154e-5** | **30 / 10** |

Raw diagnostics remain separate. For the direct/implementation-informed result, measured against the **original raw source vectors**, maximum raw Euclidean difference is **5.2955871506970916e-5**, with **1,857 corners / 619 faces** over 1e-5. The independent direct model's maximum component difference is **4.996633529663619e-5**, with 1,848 corners over 1e-5. Its rounded lengths are **0.9999484686722611–1.0000530085950445**, and maximum absolute magnitude difference is **5.295586852582801e-5**. These failed raw comparisons are retained, not concealed by the direction calculation. Sensitivity-model raw values use their own reference conventions and are not substituted for these original-source comparisons.

These are arithmetic predictions, **not actual Blender output measurements**. Importer custom-normal encoding/evaluation, compiler contraction and deduplication remain unmeasured. The models therefore do not establish exact actual-file failure counts or prove that every imaginable four-decimal vector is incapable of approximating these directions. They establish a concrete, source-specific precision hazard without a demonstrated supported control that meets admission. The conservative decision is **OBJ NOT QUALIFIED / NO-GO before execution**. A trial export is not used to resolve this already unqualified admission, and the historical direction exception is not transferred.

## Authenticated PBR channel manifest

Both adjacent PNGs match their embedded GLB byte streams exactly, including valid PNG chunk CRCs. No images were regenerated, extracted into new image files, recompressed or copied into a converted package.

| Role | Source image contract |
|---|---|
| Base color | `base-color.png`, 1024×1024 RGBA8; RGB interpreted as **sRGB**, alpha retained as linear data; original bytes/hash above |
| Metallic/roughness | `MR.png`, 1024×1024 RGB8; **linear/non-color** data, **G → roughness**, **B → metallic**; R is not an occlusion map in this material |
| UV set | Both maps use UV0 / TEXCOORD_0 |
| Factors | glTF defaults: baseColorFactor `[1,1,1,1]`, roughnessFactor `1`, metallicFactor `1` |
| Opacity and sidedness | Default alphaMode **OPAQUE**; explicit doubleSided **true**. Stored base-color alpha does not imply blending |
| Addressing/filtering | S/T **CLAMP_TO_EDGE (33071)**; magnification LINEAR (9729); minification LINEAR_MIPMAP_LINEAR (9987) |
| Other maps | No normal, occlusion or emissive texture; no source-normal bake; default emissive factor zero |

Omitted scalar properties are interpreted under glTF defaults, not observed MTL or Blender material values. The two roles are authenticated independently in `source-pbr-manifest.json` in the private evidence directory. OBJ/MTL cannot be assumed to carry this spatial metallic/roughness, colour-space, addressing, opacity and sidedness contract faithfully. A future supported-format package must retain the exact original PNGs plus a separate authenticated manifest; this phase does not claim that transport has occurred.

## One justified alternative, not executed

**Installed first-party binary FBX** is the single proposed alternative. The supported-format basis was established in the parent review through Facepunch's [Model Editor documentation](https://sbox.game/dev/doc/editor/model-editor). Installed `io_scene_fbx/export_fbx_bin.py` uses float64 normal arrays; `fbx_utils.py` dispatches them to the binary writer, and `encode_bin.py` writes float64 array bytes. This avoids the identified four-decimal OBJ storage bottleneck.

The installed exporter source SHA-256 is `8fd324e4ee0d25cc89cd53507b0f93c057b17b699601ac90f81085d5be09e44a`; helper `66a15d79d5eaf490176b439f22fb01963efcd46e06702eff224a84135e636fb3`; writer `a9211f2d40c31587cf96b6f3bb000e600c9f729585c9af3b00132f65c2a3e82c`. These installed files and the glTF importer were inspected as text. Float64 storage alone does not prove import fidelity, axis/unit handling, corner correspondence or PBR equivalence.

The smallest justified next step, requiring a separate decision, is to qualify a frozen binary-FBX route and an independently reviewed actual-file parser **before** a one-conversion attempt. It must preserve every original oriented face/corner, dimensions, UV0 and normal directions at the unchanged gates, both exact PNGs and the explicit PBR metadata. Stop before execution if settings, independent readback, process ownership or containment cannot be qualified; stop after any discrepancy without a retry or sweep. No FBX implementation, capability probe or conversion was run here. No S&box import is authorized by this result.

## Preservation, independent review and publication boundary

The exclusive private evidence directory is `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-masonry-pbr-transport-20261031-v1`. The installed official S&box skill and current CLI documentation were read; no editor connection or project action was needed for this format-admission stop.

Fresh preservation verification passed for **6,964 protected files / 895,079,653 bytes**, including all 6,924 inherited entries, all 20 selected input/tool identities, historical evidence and authored Greyfield files. No protected file changed or disappeared; no authored Greyfield or historical evidence files were added. The existing local `proof` checkout's Git HEAD, refs and status remain unchanged. Mutable editor caches and Git internals were excluded from file hashing; this is not a claim to freeze those live caches. Independent reviewers rechecked the inherited and expanded baselines and immutable GitHub parent.

Before/after selected-process inventories are identical: the pre-existing Greyfield `sbox-dev.exe` PID 7656 remains, with **zero Blender processes** in either snapshot. No converter worker was launched. Retained potential-execution limits remain 6 GiB process/Job, 180 seconds, at least 12 GiB available RAM and 2 GiB destination free space. These are future admission ceilings, not measured conversion usage or an exercised containment claim. Only bounded source parsing, hashing, source-text inspection and evidence writing were performed.

Three independent reviews covered route precision, source/normal/PBR audit, and final admission/preservation/publication. Publication is restricted to the two original text files `REPORT.md` and `ACCEPTANCE.md` under `docs/compatibility/amd-masonry-pbr-transport-evidence/`, based directly on the authenticated parent. No model, image, exact normal array/witness, vendor source, checkpoint, private capture or raw preservation/process inventory belongs in that allowlist. The separate local publication receipt must verify exact bytes, sole parent, added-only allowlist, one new absent branch and unchanged prior remote heads before completion. No local checkout/ref mutation is required.

The four non-manifold edges, crop boundaries, incomplete intersection evidence, cropped/holed art quality, production scale/placement, collision/LOD/performance and input/component rights remain unresolved. The parent assessment's private-asset and rights restrictions remain in force. Conversion cannot repair them. Engine PBR material mapping, actual filtering/mips, native rendering and appearance approval would require separately authorized future work even after a qualified format conversion.

**Final scope: source reauthentication PASS; current OBJ route admission NO-GO; Stage 2 conversion and Stage 3 converted-asset readback NOT RUN. STOP.**
