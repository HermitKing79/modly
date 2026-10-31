# Independent Phase 22 sampling oracle

`independent_sampling_oracle.py` reads only authenticated local inputs and creates a **new** JSON report. It refuses report overwrite. It makes no engine calls, project writes, installations or network requests. Original mesh UVs provide the expected values; engine UV readback is not used to predict screenshot pixels. Private texture/capture/reference inputs must not be published.

The five names, dimensions, camera positions, target, prop transform, classifier, negative controls and thresholds are constants in the script. They are fixed before any actual Phase 22 screenshot inspection. No values are estimated from screenshot colors or feature positions.

## Run

From this evidence directory:

```text
python -B test_independent_sampling_oracle.py
python -B independent_sampling_oracle.py --glb ORIGINAL.glb --texture diagnostic_uv_color.png --metadata capture-metadata.json --report NEW-independent-sampling-report.json
```

Exit 0 means all gates passed; exit 2 means NO-GO. A prerequisite failure creates a NO-GO report rather than substituting inputs or lowering a threshold. The default original GLB path is the authenticated Phase 20C artifact. The source SHA is fixed at `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201`.

`independent-sentinel-points.json` has already been generated from the authenticated GLB and the fixed approved prop transform. It contains exactly 15 named world points. Regenerating it requires a new output name:

```text
python -B independent_sampling_oracle.py --sentinel-template NEW-sentinel-points.json
```

## Exact input metadata schema

JSON keys are case-sensitive. Use numeric arrays and JSON booleans; do not serialize vectors, quaternions, matrices or rects as comma strings. Paths may be absolute or relative to the metadata file. Every referenced input is authenticated by SHA-256 before use.

```json
{
  "schema": "modly.phase22.sampling-captures.v2",
  "source_glb_sha256": "afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201",
  "texture_sha256": "SHA256_OF_GENERATED_DIAGNOSTIC_PNG",
  "sampling_contract": {
    "sampler": "PointWrap",
    "sample_level": 0,
    "srgb_read": true,
    "texture_format": "RGBA8888",
    "opaque": true,
    "shader_resource": "ACTUAL_PUBLIC_SHADER_RESOURCE_PATH",
    "material_resource": "modly_tests/phase22_uv_pbr/diagnostic_pointwrap_unlit.vmat",
    "runtime_texture_resource_path": null,
    "dependency_texture_resource": "modly_tests/phase22_uv_pbr/diagnostic_uv_color_png_f3ad0e25.generated.vtex"
  },
  "texture_probe": {
    "path": "compiled-material-texture-probe.json",
    "sha256": "SHA256_OF_PROBE_JSON"
  },
  "texture_dependencies": {
    "path": "mcp/08-material-dependencies/stdout.txt",
    "sha256": "b14cb99840cb22df5cb7ea84b5e650a17d8f09f56a64a8ea5f0ff1029ee70b4c"
  },
  "prop": {
    "guid": "ACTUAL_PROP_GUID",
    "renderer_guid": "ACTUAL_RENDERER_GUID",
    "model_resource": "modly_tests/phase21_static_prop/experimental_static_prop.vmdl",
    "world_position": [0, 0, 15.892192],
    "world_rotation": [0, 0, 0, 1],
    "world_scale": [1, 1, 1],
    "tint_rgba": [1, 1, 1, 1]
  },
  "captures": [
    {
      "name": "three-quarter",
      "png_path": "screenshots/three-quarter.png",
      "png_sha256": "SHA256_OF_UNMODIFIED_ENGINE_PNG",
      "width": 1280,
      "height": 960,
      "camera": {
        "guid": "ACTUAL_CAMERA_COMPONENT_GUID",
        "enabled": true,
        "world_position": [65, -75, 55],
        "world_rotation": [0, 0, 0, 1],
        "world_scale": [1, 1, 1],
        "basis": {
          "forward": [1, 0, 0],
          "left": [0, 1, 0],
          "up": [0, 0, 1]
        },
        "fov_axis": "Horizontal",
        "field_of_view": 42,
        "z_near": 10,
        "z_far": 10000,
        "orthographic": false,
        "custom_projection_matrix": null,
        "custom_size": [1280, 960],
        "viewport": [0, 0, 1, 1],
        "screen_rect": [0, 0, 1280, 960],
        "enable_post_processing": false,
        "projection_matrix": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "view": {
          "position": [65, -75, 55],
          "rotation": [0, 0, 0, 1],
          "field_of_view": 42,
          "z_near": 10,
          "z_far": 10000
        },
        "sentinels": [
          {
            "name": "bbox-000",
            "world_position": [0, 0, 0],
            "screen_normal": [0, 0],
            "behind": false
          }
        ]
      }
    }
  ]
}
```

The zero/identity transform, basis, matrix and sentinel values above are **schema placeholders**, not accepted camera/prop inputs. Fill them from public engine readback. Prop rotation must equal the Phase 21 proper rotation with matrix rows `[[0,1,0],[0,0,1],[1,0,0]]`; use the actual engine quaternion in XYZW order. The source-to-engine local mapping is `(GLB.y,-GLB.z,-GLB.x)/.0254`; the approved prop rotation and translation are then applied.

Supply exactly these five capture entries, all aimed at `(0,0,15.891)`:

| Name | World camera position |
|---|---|
| `three-quarter` | `(65,-75,55)` |
| `rear-three-quarter` | `(-65,75,52)` |
| `side` | `(0,-95,32)` |
| `top` | `(32,-38,110)` |
| `underneath` | `(-60,-55,-35)` |

Each camera entry needs all 15 sentinel records from `independent-sentinel-points.json`, with engine outputs added. Use public `CameraComponent.PointToScreenNormal(worldPoint, out bool behind)` for `screen_normal` and `behind`. Snapshot public `View`, `ProjectionMatrix` and `ScreenRect` immediately after rendering and before sentinel calls; sentinel calls refresh the underlying camera cache. The dedicated `CustomSize=(1280,960)` must already match the bitmap dimensions. Record `ProjectionMatrix` as M11,M12,...,M44. The oracle preserves this matrix as input evidence but derives projection analytically, so it does not assume an undocumented matrix transpose.

Engine projection is checked against the 15 fixed world points with a maximum 0.05-pixel error. The camera quaternion, captured Forward/Left/Up basis, fixed look target, raw world transform and public View must agree. Unexpected composed view effects, viewport changes, orthographic projection, custom projection or post-processing fail prerequisites.

The `texture_probe.path` file is the direct JSON object returned by the reviewed `readonly_shader_material_texture_probe.cs`, containing `Classification`, `NoSceneOrResourceMutationRequested` and `Records`. It may include both diagnostic and neutral records; exactly one must match the diagnostic material path. The oracle verifies:

- Actual MaterialPath, ShaderResourcePath, runtime TextureResourcePath (including the actually observed null) and RuntimeTextureName (`g_tDiagnosticColor`).
- Material/texture valid, non-error and loaded state; 1024×1024×1 dimensions, at least one mip and `RGBA8888` format.
- ImageSourceSha256 equals the independently authenticated PNG file hash.
- MipZeroRgbaByteCount equals 4,194,304 and MipZeroRgbaSha256 equals SHA-256 of the independently decoded source PNG's complete row-major RGBA buffer.
- All 25 exact `(X,Y,R,G,B,A)` row/column witnesses match that decoded buffer.

No second compiled PNG export is required. A full-buffer hash comparison establishes every decoded mip-zero byte, including orientation. Source shader/compilation evidence establishes the declared sampler and sRGB read contract separately; binary RGB screenshot categories do not independently measure a tone curve or BRDF.

### Pre-capture schema v2: native wrapper path is unavailable

The actual public `Material.GetTexture("g_tDiagnosticColor")` wrapper reports a null `ResourcePath`, while the native texture is valid, loaded, non-error and returns every expected mip-zero RGBA byte. The actual material dependency response independently names `modly_tests/phase22_uv_pbr/diagnostic_uv_color_png_f3ad0e25.generated.vtex` and the exact custom diagnostic shader. Schema v2 records these as separate facts. It does **not** replace the null runtime path with the dependency path. The old ambiguous `texture_resource` key is rejected.

`runtime_texture_resource_path` must reproduce the public wrapper value exactly; it is null for this capture preparation. `dependency_texture_resource` is the independently observed named generated asset. The `texture_dependencies` file is SHA-authenticated before parsing its first six JSON documents (three diagnostic material records, then three neutral material records). The remaining diagnostic/log text is retained in the report. The oracle requires exactly one diagnostic asset-status record, file record and References graph; successful/current compiled flags; no unresolved references; matching material source/compiled file identities; and exactly two named dependencies, comprising this generated Texture and this custom Shader. Missing/duplicate records, false compile flags, unresolved references, changed texture/shader paths or a changed dependency-file hash fail even when the runtime wrapper path is null.

This is an adaptation of input provenance before the first actual screenshot capture, based on supported public API behavior. The source PNG, shader, sampler, classifier, UV uncertainty, reference rasterizer, masks, numeric gates and negative controls are unchanged. The complete native readback hash proves the returned handle's image contents; the material lookup argument and custom shader path identify the runtime binding; the independently authenticated graph names the compiled resource. No claim is made that the public wrapper exposed that resource name.

## Frozen sampling method and coverage

All five reference masks are constructed before any screenshot pixels are opened. Opaque single-sided triangles are projected in Source coordinates (+X forward, +Y left, +Z up). A pixel-center Z buffer picks the visible face; UVs and their derivatives use perspective-correct barycentric interpolation. PNG V increases downward. The shader contract is point/wrap sampling at explicit mip level zero.

The mask excludes a two-pixel geometry/visibility neighborhood, source texture feature boundaries, known UV quantization of 0.0005 and the entire allowed projection error. Required source texture plateau radius is `max(3, ceil(.0005*T + .05*max(|dUV/dx|+|dUV/dy|)*T + 1e-7)+1)` texels, where `T=1024`. Margins use wrap addressing and are conservatively bounded at 32 texels; larger requirements are unavailable coverage. Near/far clip crossing is an explicit NO-GO, never a silently dropped triangle.

Source channels must be exactly 0 or 255, alpha 255, using all eight categories. Actual screenshot categories use fixed thresholds: max<0.08 is black; otherwise each channel is high at ≥0.65×max or low at ≤0.35×max. An intermediate channel is unknown and counts as a mismatch. Actual colors never determine sample eligibility.

Negative controls use the same positive mask: `u'=1-u`, `v'=1-v`, `(u',v')=(1.2u+.037,.8v+.071)`, and a 90-degree clockwise rotation of the texture pixels. Each control must differ from the positive reference in at least 20% of samples, agree with actual pixels at most 80%, and trail positive agreement by at least 15 percentage points. These controls are checked per view and over the union.

Other frozen gates are 5,000 eligible pixels and 100 faces with at least three samples per view; 300 faces in the union; six nonblack categories with at least 100 samples in the union; 99% agreement per view, 98% per sufficiently sampled category and 95% per face with at least 20 samples. Nonseamed continuity requires 30 eligible paired edge witnesses and 99% pair agreement. Paired samples lie 2–6 pixels inside both incident faces and must have matching reference plateau categories. Deliberate UV seams are classified separately and are not required to match across the seam.

The report lists per-view visible/tested/untested faces, backfaces, front faces without a visible pixel, visible faces without enough eligible pixels, per-face/per-category scores, and union missing coverage. It separately lists tested and untested nonseamed edges. A PASS applies to these recorded sampled regions; it does not assert all-face/all-texel appearance or user approval.

## Verification already performed

Seventeen self-contained synthetic tests cover numeric projection, quaternion handedness, independent sentinel rejection, manual perspective interpolation, depth visibility, backface culling, clip failure, fixed classification, wrap and top-left orientation, texture margins, full compiled-buffer identity with the actual null-wrapper-path behavior, strict named dependency validation and hash rejection, negative-control rejection, seam classification, continuity pair faults, minimum coverage, and unknown actual pixels. They require no private images or live engine.

Independent source preflight confirms 502 geometric positions, 1000 triangles, 1500 edges, no boundary edges, 280 seams and 26 UV charts. Analytical fixed-view preflight finds 429/415/289/498/403 visible source faces. This is reference preparation, **not actual engine sampling evidence**.
