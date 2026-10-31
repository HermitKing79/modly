# MODLY Phase 22 — S&box UV texture and PBR transport proof

## Decision

**UV TEXTURE SAMPLING: NO-GO.** The single authorized attempt produced five actual S&box renders. The unchanged, predeclared image test failed on original GLB zero-based triangle **659** in the top view: **34/38 eligible samples (89.473684%)**, below the **95% per-face gate**. High aggregate agreement does not override this failure. Independent replay reproduced the result byte-for-byte. No shader, image, camera, mask, classifier or numeric threshold was changed after observing the result, and no replacement render was attempted.

**Neutral PBR: source-property representation and compiled mip-zero image identity supported; rendered transport fidelity unqualified.** A separate new neutral material compiled, but it was never applied or rendered. The UV failure stopped further engine changes. No claim of equivalent BRDF, tone response, original-source normals or bit-identical PBR rendering is made.

The diagnostic material and isolated scene remain local investigation assets, without production qualification or promotion. Phase 11's original high-fidelity benchmark remains permanently **NO-GO**. **USER_APPEARANCE_APPROVAL_PENDING** is unchanged. This attempt is finished; another experiment requires a new explicit decision.

## Authenticated starting point and isolation

The exact GitHub parent is [b88200257f672bb8713c8f0936bf6595bf9164c5](https://github.com/HermitKing79/modly/commit/b88200257f672bb8713c8f0936bf6595bf9164c5), tree `ba0e0bb6ac0081135d36778a534fe1445f0500ff`. Its Phase 21 report and both conversion/engine verification scripts were fetched and matched exactly. All twelve Phase 21 artifact hashes, conversion reports, source geometry/material analysis and compiled-model audit were reauthenticated locally. A tuple/list serialization comparison was corrected to JSON-normalized exact equality before the preservation baseline; no numeric tolerance was relaxed.

| Preserved input | SHA-256 |
|---|---|
| Phase 20C `experimental-static-prop.glb` | `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201` |
| Phase 21 `experimental_static_prop.vmdl` | `113d688135246d71cbd85c9c74dbd65af9f75160cce8efddce921c5964c8547c` |
| Phase 21 compiled `experimental_static_prop.vmdl_c` | `9a83938757967a81cb7f37d9a505ae14dc491b109ba19df42c4d2b830cfd7a11` |
| Phase 21 `Modly_Phase21_StaticProp_Test.scene` | `2833c1f861a8ab6093269cde0b8833752f89788562cec7af502112173ebdc686` |

Installed instructions were read from `C:\Users\wattz\.agents\skills\sbox\SKILL.md` and the first-party CLI's `--skill sbox-cli` output. Discovery found the existing **Greyfield** editor, PID **7656**, engine **26.10.27c**, project `C:\Users\wattz\Documents\S&box Projects\greyfield\greyfield.sbproj`. Every live call attached to that existing editor through `sbox-cli --connect ... --call ...`. No editor instance, installation or external asset was added.

Supported scene, component, asset, status, dependency and console tools were used. The documented `run_csharp` fallback covered asset registration, native shader/schema inspection, texture readback and camera bitmap capture where dedicated tools were insufficient. Installed first-party shader templates and public API evidence established the shader route. No undocumented reflection operation or guessed shader property was used for live application.

All new project files are under `Assets/modly_tests/phase22_uv_pbr/`. The new scene is `modly_tests/phase22_uv_pbr/modly_phase22_uv_pbr_test.scene`. Its isolated `MODLY_PHASE22_UV_DIAGNOSTIC_ONLY` placement references the original Phase 21 model with a new material override; it does not duplicate, modify or regenerate model geometry. Position `(0,0,15.892192)`, angles `(-90,-90,0)`, scale one, white tint. Only a ModelRenderer, proof cameras and neutral lights were added. No collision, physics, LOD or gameplay integration was performed.

The baseline and final verification cover **6,460 protected original files**. Final hashes show **zero changes, zero missing originals, zero changed original local refs and zero new authored files outside the permitted test directory**. Both original scenes remain saved. The local `proof` checkout remains at `ddc9a8162482f0347b192fb96f2421a138d7c7bc`, including its five pre-existing untracked paths. Remote publication uses a new branch without moving that checkout.

## Diagnostic image, UVs and compiled dependencies

`make_diagnostic.py` authored one deterministic **1024×1024 RGB8** PNG using Python's standard library, a fixed seed and a local bitmap font. It contains 64-pixel SHA-coded tiles, 32-pixel complementary color checks, hexadecimal column/row labels, U+/V+ arrows and asymmetric corner regions. All RGB channels are binary 0/255; alpha is opaque; the PNG has an explicit sRGB chunk. No external image, font, neural service or download was used.

An independent implementation reconstructed and compared **all 1,048,576 diagnostic pixels**, validated PNG chunks/CRCs and confirmed the copied original neutral PNG's identity. The generator's frozen SHA-256 is `723eda9607de90770bafacb3389cddc91e0d1ccf94d35b40c428856652afb299`.

| Image identity | Diagnostic | Original neutral copy |
|---|---|---|
| PNG SHA-256 | `35161147faf014e2069660a8681af66762f3d3ad4cc859904087becbb328a6d3` | `2364e7ddb8b271a95db206b39bfbdf938d9e427d46dda2745357264c8451c89e` |
| Full native mip-zero RGBA SHA-256 | `9f4ae0a3e72bc313dca571c554d8075f726fa6411988c9852ea9ac845d474c12` | `09e5b3f39bf89b7c0a9726ed3925c75e3298e8bc5708cbc4a86646c83673d24a` |
| Dimensions / native format / mip count | 1024² / RGBA8888 / 1 | 256² / RGBA8888 / 9 |

Both complete native mip-zero buffers matched independently decoded source pixels, with 25 additional exact XY/RGBA witnesses each. Other neutral mip contents were not independently read back. Both native textures were loaded, valid and non-error. Their public wrapper `ResourcePath` was actually **null**, and `Texture.Flags` was **None**. These values were retained; no resource name or color-space measurement was invented from them.

The independent named dependency graph supplies the resource identities separately. Each current compiled material has exactly two named references, its custom shader and generated texture, with **zero unresolved references**:

- `diagnostic_pointwrap_unlit.vmat` → `diagnostic_pointwrap_unlit.shader` and `diagnostic_uv_color_png_f3ad0e25.generated.vtex`.
- `neutral_pbr.vmat` → `neutral_pbr.shader` and `neutral_source_basecolor_png_ad7af70b.generated.vtex`.

Every name above has the prefix `modly_tests/phase22_uv_pbr/`. The diagnostic shader derives from the installed first-party unlit template, preserves its vertex path, reads UV0 with explicit **PointWrap / SampleLevel 0**, declares sRGB input and returns opaque unlit color. Texture compression is avoided with RGBA8888 output. This is a diagnostic sampler contract, separate from the original neutral material's filtered sampler.

Fresh read-only `Model.GetVertices/GetIndices` inspection authenticated the existing compiled model: **2,695 attribute vertices, 3,000 indices, 502 unique geometric positions, 1,000 oriented triangles, zero physics shapes**. Full original GLB → OBJ → engine correspondence passed again; the maximum UV component difference is **0.00048764434890746777**. Geometry and UV correspondence are prerequisites, not visual sampling proof.

The original asset operation stopped when generic `.shader` compile flags did not satisfy its expected gate. Independent installed-source review established that shaders use a separate native compiler; generic `CanRecompile/IsCompiled` flags are unsuitable shader-completion evidence. The unchanged six authored files were retained. The native watcher completed both shaders, ten combinations each; exact compiled hashes, native validity and custom texture schema were then checked before only the two materials were compiled. No shader rewrite or forced shader recompile was used as a retry. Shader/Material `IsError=false` is inherited structural metadata and was not treated as native error detection.

Initial on-demand missing-compiled-shader messages preceded native shader publication; two dependency-info warnings preceded material completion. These historical messages remain in the private logs. Final materials were compiled/current with no compile-failed flag; native shader interfaces and texture contents passed. The final editor had zero C# compile errors and no new Warn-or-higher entries after cursor 78 through cursor 83. This does not turn the later sampling NO-GO into a pass.

## Actual engine sampling and independent reference

Five unmodified native engine PNGs were captured at **1280×960** using dedicated cameras in the new saved scene. The capture operation temporarily set only each new camera's public `CustomSize`, rendered its bitmap, read public camera/projection/sentinel metadata, restored the previous nullable size, and asserted saved state before writing private evidence. No existing scene was saved. FOV was 42 degrees horizontal; post-processing was disabled. The final output resolve/filter footprint was not independently instrumented.

| View | Eligible samples | Matching samples | Agreement | Source faces with ≥3 samples |
|---|---:|---:|---:|---:|
| Three-quarter | 144,262 | 144,262 | 100% | 362 |
| Rear three-quarter | 157,223 | 157,223 | 100% | 337 |
| Side | 200,408 | 200,408 | 100% | 245 |
| Top | 132,250 | 132,230 | 99.984877% | 321 |
| Underneath | 195,525 | 195,525 | 100% | 361 |
| All recorded views | 829,668 | 829,648 | 99.997589% | 826 distinct |

The independent CPU oracle uses the authenticated **original GLB UVs** for predictions. Engine UVs do not generate expected screenshot colors. It applies the authenticated coordinate transform, perspective-correct interpolation, depth visibility and backface handling. Fifteen public projection sentinels per view verify the analytic Source coordinate projection; maximum error over all views is **0.000638769 pixels**, below the frozen **0.05-pixel** limit.

All reference masks were constructed before actual screenshot pixels were opened. Eligibility uses reference-only geometric/visibility margins, a UV component uncertainty of 0.0005, the allowed projection error and a minimum three-texel source-color plateau. Unknown actual categories count as failures and cannot remove a sample. Fixed binary-category thresholds do not claim exact screenshot RGB or a measured sRGB transfer curve.

Frozen gates require at least 5,000 eligible pixels and 100 faces with ≥3 samples per view, 300 tested faces in the union, six nonblack categories with ≥100 samples, ≥99% per-view agreement, ≥98% per sufficiently sampled category, and **≥95% per face with ≥20 samples**. Continuity requires at least 30 paired nonseamed-edge witnesses and ≥99% pair agreement. Negative controls must differ from the positive reference by ≥20%, agree with actual pixels by ≤80% and trail the positive result by ≥15 percentage points. All gates except the single face gate passed; they were not adjusted after capture.

| Deliberately wrong reference | Aggregate agreement with actual pixels |
|---|---:|
| U reversed | 24.044678% |
| V reversed | 15.500055% |
| UV stretched and offset | 16.979563% |
| Texture pixels rotated 90° clockwise | 16.786474% |

All controls were rejected on the same positive masks, per view and in aggregate. Recorded sampled regions provide strong evidence against those particular orientation/stretch/wrong-image errors. They do not qualify every texel or all possible distortions.

Source topology has 1,500 edges, 280 intentional UV seams and 26 charts. Across the views, **935/1,000 triangles were visible**, **826 had at least three eligible samples**, **174 remained untested**, and **65 were never visible**. Nonseamed continuity matched **941/941 paired witnesses across 395/1,220 nonseamed edges**. Deliberate seams were classified separately; cross-seam color equality was not required. Unsampled seams, small regions, boundary pixels and excluded regions remain unqualified. No whole-model completeness or absence of every missing/incorrect region is asserted.

### Retained failure and bounded diagnosis

All twenty mismatches occur in the top view: nineteen unknown mixed-color classifications and one black-reference sample classified as gray/white. The sole gate failure is triangle 659's four unknown pixels among 38 eligible samples. Two have RGB `(223,101,223)` with expected magenta; two have `(101,223,101)` with expected green. No exact pixel coordinates, private mesh arrays or screenshot payloads are published.

Independent ray/plane calculations reproduce the reference interpolation to **1.22e-15**. Actual compiled UV rounding preserves the expected category at all 38 tested points on triangle 659. At its four failures, the entire frozen ±0.05-pixel projection rectangle plus ±0.0005/component UV envelope preserves the expected source category. These verified point-center uncertainties do not explain the mixed output colors.

A separate read-only half-pixel sensitivity calculation reaches neighboring texture categories while remaining within the same geometric triangle. This demonstrates image-footprint sensitivity at that foreshortening. The recorded public metadata does not identify the engine sample locations or resolve/filter stage that produced the mixed values. The finding is therefore an **unresolved image-sampling/reference-footprint discrepancy**, without a correctness waiver or a demonstrated specific engine/UV defect. There was no threshold retuning, post-result mask change, image replacement, new render or PBR appearance attempt.

The frozen oracle SHA-256 is `b0aadae4d6c3ebb859aeada7f135621d219a8cbe26d80124d1296a30e3aab1ae`. Its seventeen self-contained synthetic tests passed, including projection, perspective interpolation, depth, clipping, wrapping, classification, texture identity/dependency tampering, negative controls, seams, coverage and unknown-pixel handling. Independent actual replay produced the same result file SHA-256, `e988366234ea888028083fffd030ab2f57afdfb08a9efdf2db8a857b5cb4e718`. A fresh final reviewer separately replayed the evidence and checked preservation. Schema v2's null runtime path/named dependency separation was reviewed before capture; numerical gates were unchanged.

## Separate neutral PBR property assessment

| Property | Authenticated original | New S&box representation and limit |
|---|---|---|
| Base color | 256² RGB PNG, every pixel `(160,160,160)`; default factor `(1,1,1,1)` | Same PNG bytes and full native mip-zero RGBA; factor one represented by no extra multiplication. Rendered mid-gray transfer was not measured. |
| Roughness | `0.3499999940395355` | Explicit `m.Roughness` float assignment. Engine roughness/BRDF response is not proven equivalent. |
| Metallic | `0` | Explicit `m.Metalness=0`; scalar representation supported. |
| Opacity | Default OPAQUE, RGB image without alpha | Opaque template path and `m.Opacity=1`; no mask/blend transport claim. |
| Texture coordinates/sampler | UV0, repeat U/V; LINEAR magnification and LINEAR_MIPMAP_LINEAR minification | Explicit UV0 and installed `g_sTrilinearWrap`; nine native mips present, only mip-zero contents independently checked. Uniform source color cannot demonstrate filtered nonuniform sampling correctness. |
| Color space | glTF base-color RGB is sRGB; factor values are linear | Explicit source-evidenced `Srgb` input and `SrgbRead(true)` declaration. Raw byte identity does not measure the mid-level sampling transfer function. |
| Normals | Existing converted model normals only | First-party `Material::Init(i)` path retained. No normal map or source-normal bake; signed neural source normals/ray correspondence remain unqualified. |

The glTF base-color, scalar-factor and alpha conventions are defined by the [Khronos glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html). Numeric sampler modes and wrapping are authenticated against the [official sampler schema](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/sampler.schema.json); defaults are also specified in the [official material schema](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/material.schema.json). Installed S&box shader templates and sampler declarations provide the engine-side evidence. This mapping supports faithful source parameter representation at the declared interface, while actual shader math, lighting, mip construction, filtering, tone/output transforms and rendered PBR equivalence remain unqualified.

No default engine material key was guessed to approximate the source. The separate neutral shader derives from the installed material template and calls `ShadingModelStandard::Shade(m)` with the authenticated scalar values. Neither this fact nor successful compilation establishes bit-identical PBR rendering. The diagnostic binary unlit renders cannot substitute for a neutral PBR render.

## Local evidence and remaining production requirements

Private evidence root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-static-prop-sbox-uv-pbr-20261031-v1`.

Retained actual images are `screenshots/three-quarter.png`, `rear-three-quarter.png`, `side.png`, `top.png` and `underneath.png`. The capture manifest authenticates each native image and per-view metadata. Supporting private files include `preservation-before.json`, `preservation-final-engine.json`, `fresh-model-uv-authentication.json`, `compiled-material-texture-probe.json`, the original MCP logs, `sampling-root-result.json`, independent replay/failure/envelope reviews and independent PBR assessment. Textures, shaders/material assets, scenes, screenshots, mesh/reference arrays and private per-pixel data remain local. Public code contains no embedded private images or mesh data; its evidence replay still requires the withheld local inputs.

Final live state: Greyfield active on the new Phase 22 scene, all three open scenes saved, Play off, compiler idle, last C# compile successful with zero errors. Camera temporary state was restored. The original Phase 21 model and scene retain their exact identities.

Production use still requires a separately authorized resolution of the failed image-sampling gate with an independently specified output-footprint model; suitable complete surface/texel/seam coverage; measured mid-level color-space, filtering/mip and rendered neutral PBR behavior; and user appearance approval. Source-normal fidelity and the original high-fidelity target remain unqualified. Any collision, physics, LOD, gameplay or production promotion would require its own scope. No such work is started here.

Publication is limited to independently reviewed code/text under a new `codex/amd-static-prop-sbox-uv-pbr` branch descending directly from the authenticated parent. The publication receipt separately records the resulting commit, exact allowlist, remote text readback and unchanged prior branch heads. It contains no production asset promotion.
