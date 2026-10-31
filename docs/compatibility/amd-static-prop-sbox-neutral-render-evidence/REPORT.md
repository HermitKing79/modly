# MODLY Phase 22C — isolated neutral PBR render proof

**PASS — limited one-view neutral material operability and native-rendering smoke proof.** Exactly one ordinary three-quarter render completed with the authenticated neutral material, followed by verified restoration of the original diagnostic assignment and camera size. Independent image reviews found a solid gray shaded model with no obvious error material pattern or missing visible face. Faceted/triangular shade variation is visible on some bevels and corners; normal fidelity and production appearance remain unqualified.

## Scope and historical decisions

This is one bounded material-operability and native-rendering smoke proof. It does not qualify exact glTF/S&box BRDF or display-pixel equivalence, filtered nonuniform UV sampling, mip fidelity, normal-source fidelity or production appearance.

The historical Phase 22 sampling **NO-GO** and Phase 11 permanent high-fidelity **NO-GO** remain unchanged. **USER_APPEARANCE_APPROVAL_PENDING** is unchanged. No CMAA2 investigation, graphics-preference change, shader/texture/model/UV/geometry edit, normal bake, collision/LOD/gameplay integration or production promotion is within this proof.

## Authenticated starting point

Parent: [9a7efb13495fc9b44d02bbbfe4fb94c5e1534839](https://github.com/HermitKing79/modly/commit/9a7efb13495fc9b44d02bbbfe4fb94c5e1534839), tree `769654bd5c2244d9e6c932dd7790966a523e9df3`. Its five published code/text files were fetched at that exact commit and matched local UTF-8 bytes. The sole parent and tree were checked using public GitHub commit metadata.

Installed S&box instructions were read from `C:\Users\wattz\.agents\skills\sbox\SKILL.md` and the current first-party `sbox-cli --skill sbox-cli` output. Only the existing Greyfield editor was discovered/attached through `--connect/--call`. PID **7656**, engine **26.10.27c**, started **2026-10-31 15:17:12.818088 UTC**, owner **HERMIT\wattz**, exact existing Greyfield project command line. Independent process inspection confirmed ownership and that no second editor was present.

Project: `C:\Users\wattz\Documents\S&box Projects\greyfield\greyfield.sbproj`. Active existing scene: `modly_tests/phase22_uv_pbr/modly_phase22_uv_pbr_test.scene`. All three open scenes were saved, Play was off and the compiler was idle with its last compile successful and zero errors. No unsaved scene was discarded.

| Preserved identity | SHA-256 |
|---|---|
| Original GLB | `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201` |
| Existing Phase 21 model source | `113d688135246d71cbd85c9c74dbd65af9f75160cce8efddce921c5964c8547c` |
| Existing compiled model | `9a83938757967a81cb7f37d9a505ae14dc491b109ba19df42c4d2b830cfd7a11` |
| Neutral material source | `9f084377486c7cab460b2927127de0f2b80c4e763b0ad942a22f2a43d3b38c5b` |
| Neutral shader source | `5a79ec8b48b1f4d936da1463226c1ba2655efd5e33dfc043e718bd6b31c5fd12` |
| Compiled neutral shader | `6e958990e31caf8cd877c78ca9018262d072c2dd8774defbdca4c4f8d1aeed9a` |
| Original neutral PNG | `2364e7ddb8b271a95db206b39bfbdf938d9e427d46dda2745357264c8451c89e` |
| Existing Phase 22 scene file | `280f17fcc78c4db7c6085bb10e6f7cc2cc8b43459a303eb85168428e68890e20` |
| Original Phase 21 scene file | `2833c1f861a8ab6093269cde0b8833752f89788562cec7af502112173ebdc686` |

The baseline authenticates all earlier protected files and all 22 Phase 22 project artifacts. Its **6,865 protected files** include the **335-file Phase 22** and **44-file Phase 22B** historical evidence sets, authored Greyfield files and inspected engine assemblies/docs. Existing local Git HEAD/refs/status are preserved. Mutable `.sbox` runtime caches, `.git` internals and existing Python bytecode are excluded; no authored scene/resource is excluded. Historical directory additions are checked separately.

## Source properties and fresh native resources

The original GLB material and embedded image were parsed again independently. The source image is **256×256 RGB8**, every pixel **(160,160,160)**. Roughness is **0.3499999940395355**, metallic **0**, alpha mode defaults to OPAQUE, base-color factor defaults to one, and no normal texture is present. Source sampler enums are LINEAR magnification and LINEAR_MIPMAP_LINEAR minification, with repeat wrapping defaults.

The unchanged reviewed custom neutral shader samples **UV0** through installed **g_sTrilinearWrap**, declares sRGB input/read behavior, assigns sampled RGB to Material.Albedo, assigns the original roughness, zero metalness and opacity one, retains `Material::Init(i)` model normals, and calls `ShadingModelStandard::Shade(m)`. There is no normal map or bake. These declarations establish the intended representation, not measured mid-level GPU color transfer or identical BRDF math.

Fresh read-only checks find `modly_tests/phase22_uv_pbr/neutral_pbr.vmat` already compiled/current with no compile-failed flag and **zero unresolved references**. Its exact two named dependencies are `neutral_pbr.shader` and `neutral_source_basecolor_png_ad7af70b.generated.vtex`, both under the same Phase 22 resource directory. No asset was repaired, regenerated or recompiled.

The actual bound `g_tNeutralBaseColor` texture is loaded, native-valid, non-error, **256×256×1 RGBA8888**, with **9 mips**. Its complete mip-zero RGBA digest is `09e5b3f39bf89b7c0a9726ed3925c75e3298e8bc5708cbc4a86646c83673d24a`; all 65,536 pixels match the independently decoded original with opaque alpha, including 25 explicit coordinate witnesses. Higher mip contents were not read. The runtime wrapper path remains **null**; the named dependency graph and actual texture binding/readback are separate evidence channels.

The exact previously reviewed read-only material/texture probe was reused unchanged and returned the same complete resource snapshot as Phase 22, now through a fresh call. Material/Shader `IsError=false` alone is not a native error test; those types inherit that getter. This proof uses current compiled flags, exact identities, native validity, actual custom interface/texture content, console evidence and the actual render together.

## Bounded render and restoration

The independently reviewed script `capture_neutral.candidate.cs`, SHA-256 `c30593b2687a0a2107f53ded67a316fe30b35ae044ffa7e4e92b7689897f1f73`, was executed **once** against the same owned editor process. It called the existing three-quarter CameraComponent's ordinary `RenderToBitmap(bitmap,false)` exactly once at **1280×960**. No top view, retry or second experiment was performed.

Before application, it checked the exact saved scene, all three sessions' saved flags, Play off, exact prop/renderer/camera IDs, thirteen source/compiled/scene file hashes, fresh native texture/dependency evidence, and current neutral resource validity. The existing renderer had attachments disabled, no prop children and no material groups. The neutral material and texture were fully preloaded/validated before changing the assignment.

The only temporary setters were the existing renderer's MaterialOverride and the existing proof camera's CustomSize. At capture time the recorded override was `modly_tests/phase22_uv_pbr/neutral_pbr.vmat`, reference-equal to the preloaded neutral object. The assigned material exposed the expected shader and actual `g_tNeutralBaseColor` native texture handle **680**, 256² RGBA8888 with nine mips. The model and SceneObject references were unchanged; tint was white and scale one.

The inspected installed public material setter synchronously updates its backing reference and dispatches through UpdateObject/SceneObject.SetMaterialOverride/native SetMaterialOverrideForMeshInstances before returning. No independent public native global-material getter is available in this proof. The assignment evidence is therefore the exact managed reference/readback, the source-evidenced synchronous native update and the actual native render together; no separate native-material readback is invented.

The existing camera retained position `(65,-75,55)`, its original rotation, FOV **42 horizontal**, near/far **10/10000**, full viewport, perspective projection and disabled post-processing. Public View, projection matrix and 1280×960 ScreenRect were captured immediately after rendering. Existing neutral lights were unchanged: white directional key intensity 1.2, white fill .35, and ambient .18. Their full public states were recorded and compared before/after.

Current graphics preferences were observed only: Cmaa2, Multisample8x, render scale 1, upscaler Off and upscaler scale 1. These are user preferences, **not measured effective native per-view AA/resolve state**. They were not changed or investigated further. Disabled post-processing is not treated as an AA-off assertion.

| Private output | Identity |
|---|---|
| `neutral-three-quarter.png` | SHA-256 `ed8456d5eb0360d28b98d53e9dce1f5548012aad9dc3e601fc3983fdc7e3a684`; valid opaque RGBA PNG; 1280×960; 19,094 bytes |
| `neutral-render-metadata.json` | SHA-256 `8c434a23ef05a543a046b282d3807f1a14ad8e546c66e68d653f06f1a62de949` |
| Native render operation | One RenderToBitmap call; CLI exit 0, no stderr |

The native image shows the existing beveled box in neutral gray, with distinct top and two side faces and brighter bevels. No diagnostic color-check pattern, obvious error-material pattern, invisible/black prop or obvious missing visible face was observed. Subtle facet/triangle tonal variation is visible along bevels/corners; its cause was not diagnosed. This visible variation is disclosed rather than qualified as correct source normals. A single view cannot establish all-face appearance or detect every defect. Gray output values are lighting/pipeline results; they are not expected to equal the source PNG's RGB160 and do not measure the sampling transfer function.

Nested `finally` blocks attempt restoration on exceptions. This successful call restored and verified the **exact original Material object** and original **nullable CustomSize (null)** before output; no failure path was exercised, and restoration itself is not guaranteed if a setter throws. Before writing either fresh output file, the script required reference-equal original material/model/SceneObject, exact equality of its finite authored prop/renderer/camera/hierarchy/tint/light snapshots, unchanged global preference values, unchanged scene-session saved flags, and all thirteen pinned source/compiled/scene hashes. Transient render caches were observed, not asserted bit-identical. No scene or asset was saved.

A separate post-render read-only editor query confirmed the original `diagnostic_pointwrap_unlit.vmat` assignment on the same renderer, the original model, white tint, attachments disabled and no children. All three scenes remained saved; Play remained off; the compiler remained idle with zero errors; no new Warn-or-higher console entries appeared after cursor 83. Full final preservation verification found **all 6,865 protected files unchanged**, no new project files, no additions to either historical evidence set and unchanged local Git HEAD/refs/status. The existing Phase 22 scene hash remains `280f17fcc78c4db7c6085bb10e6f7cc2cc8b43459a303eb85168428e68890e20`.

Independent source/resource/process, exact-operation and actual-image/result reviews support this **limited PASS**. The proof establishes that the authenticated material operates and produces a visible native render in this one recorded view, with the specified restoration. It does not establish radiometric or source-normal correctness.

## Remaining qualification

A successful neutral render can establish operability only in its recorded configuration and visible view. It cannot repair or override Phase 22's nonuniform UV-sampling NO-GO. Exact glTF/S&box BRDF, lighting, exposure/tone/display conversion, intermediate sRGB transfer, filtering/mip/LOD behavior, complete surface/normal/seam coverage and default gameplay rendering remain separate production requirements. The source is uniform neutral gray, so this proof cannot diagnose nonuniform UV texture placement.

The next production work requires its own scope: measured color-space/BRDF/filtering/mip behavior, broader surface and normal-quality review, default gameplay configuration qualification and user appearance approval. This smoke result does not authorize those experiments or reopen either historical NO-GO.

Private evidence root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-static-prop-sbox-neutral-render-20261031-v1`. The actual image, metadata, probes, raw editor logs, preservation manifests and independent review receipts remain local. Public code is an exact proof record requiring the withheld local inputs; it embeds no private image or mesh data. The generic attachment helper does not authorize arbitrary editor calls. Only the reviewed read-only batches and one reviewed render were executed.

Only independently reviewed code/text is published directly from the authenticated parent on a new `codex/amd-static-prop-sbox-neutral-render` branch. The separate publication receipt records exact additions, sole parent/tree, remote UTF-8 content readback and unchanged prior heads. **STOP after the reviewed result; no automatic follow-up experiment.**
