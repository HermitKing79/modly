# Native colour qualification and first full-model view: PASS

Base `4c38a7a00e2fffa692eb40c4942f786ca5b07bc6`.
Branch `codex/amd-brick-native-colour-render`.

**IMPORT QUALIFIED / VISUAL PASS.** The existing imported scene was opened
directly in Blender 5.2.1. **Zero new GLB imports; one CPU render.**
All 1,952,142 referenced vertices and 4,173,584 original oriented triangles
remain exact, including position bits, face/corner order and multiplicity.
All 12,520,752 stored native corner RGBA bytes and the original shader/material
were verified unchanged before and after rendering. The previous ideal-rounding
verifier, its NO-GO receipt and historical BLOCKED report remain unchanged.

## Material qualification

The CPU worker replayed all 56,910 original source-spatial queries and 2,048
frozen interior holdouts with native-byte-decoded colours. The original
evaluator and a separate implementation both passed all **112 metric rows**.
There was no new sampling, colour fitting, threshold or rounding ceiling.

After opening the saved scene, the renderer read Blender's actual scene-linear
BYTE_COLOR getter, mapped it back to original source IDs and proved bit-identical
RGB at repeated corners. The independent evaluator again passed all 112 rows
on these actual values before the single render. This accounts for Blender's
native LUT and working-space semantics without assuming float32 bit equality
with an ideal IEC conversion.

| Worst actual-native RGB metric across all 112 rows | Measured | Unchanged gate |
|---|---:|---:|
| RMSE | 0.006029394 | 0.02 |
| p95 absolute error | 0.013316770 | 0.05 |
| Maximum absolute error | 0.047903276 | 0.15 |

Worst scalar RMSE: metallic 0.0001737232; roughness 0.0001471594; OPAQUE alpha
0.0000999755, each below the unchanged 0.02 gate. Standard decoded-byte linear
conversion loss against all original F32 colour rows was RMSE 0.0007269937,
maximum 0.0036164522. These last values are measurements, not new gates.

- [All original 112 numerical rows](amd-brick-native-colour-render-evidence/qualification-metrics.json)
- [All actual native-getter rows](amd-brick-native-colour-render-evidence/actual-native-metrics.json)
- [Candidate review](amd-brick-native-colour-render-evidence/independent-candidate-review.json)
- [Independent visual review](amd-brick-native-colour-render-evidence/visual-review.json)
- [Independent final review](amd-brick-native-colour-render-evidence/independent-outcome.json)
- [Measured outcome](amd-brick-native-colour-render-evidence/outcome.json)

## Actual image and execution

Private directory, outside the proof Git worktree:
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-native-colour-render-4c38a7a/`

| Private file | Bytes | SHA-256 |
|---|---:|---|
| `render/view-pos-x-ev2.png` | 416,986 | `cb9b6df90fe9ec1d9dbf7b5b38878776c6b975cb8f6042a1a8d2f6e86e8ad52b` |
| `inspect/comparison-pos-x.png` | 701,848 | `8e8485d6dfbcafb19bfcf61b68a786ac6afdfb5e349c156a0df37cb395221f82` |

The +X orthographic view is 640x640, Cycles CPU, 32 samples, CPU OpenImageDenoise,
four threads, original neutral studio/camera and labelled +2 EV exposure.
Flat shading and vertex RGB with approved constant metallic/roughness/OPAQUE
factors were retained. No spatial PBR bake or geometry repair was performed.

| Owned stage | Seconds to natural empty Job | Peak process / Job committed bytes |
|---|---:|---:|
| CPU qualification | 5.797 | 327,467,008 / 328,318,976 |
| Saved-scene checks, single render and postchecks | 69.938 | 2,056,413,184 / 2,056,413,184 |
| PNG reopening and comparison | 2.016 | 26,398,720 / 27,246,592 |

The render script's render-plus-postcheck interval was 15.185 seconds; pure
engine time was not separately instrumented. Minimum observed available RAM
was 18,375,901,184 bytes; destination free space remained at least
112,181,583,872 bytes. Unchanged 6 GiB process AND Job caps, 180-second deadlines,
12 GiB RAM floor and 2 GiB disk floor all held. No forced cleanup or retry.
Blender logged an optional CUEW library-probe warning; the configured CPU render
and CPU denoising completed successfully. The original stdout is preserved.

Brick courses, mottling, curling vines and thin leaves are recognisable. The
warm brown/orange vegetation and varied brick colours remain close to the
previous PBR view, without an obvious global darkening or missing surface.
Dark brick gaps, holes, wispy/ragged foliage and stray strands remain visible.
Exact geometry checks attribute no new face loss to this presentation; a single
view cannot establish watertightness, hidden-surface quality or game readiness.
The comparison is contextual: the old direct-row gamma2.2 PBR image and the new
source-position-sampled vertex RGB use different colour/material semantics.

Ten focused checks passed. Scoped preservation confirms protected100, original
assets, saved scene, selected environment/source/backups, 50 historical trees,
1,312 files, 23 reports and cache/quarantine evidence unchanged. No checkpoint
weights were loaded; no WSL or neural inference ran. Models and images remain
private; only source-safe code, hashes and aggregate numerical evidence publish.

Stop for visual review. No export, geometry optimisation, baking or integration
is implied by this result.
