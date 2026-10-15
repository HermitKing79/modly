# Compact learned-colour turret GLB — PASS

Base `d3eec480522442a79f90c87d28fd4985ec7b9c86`; branch `codex/amd-turret-compact-colour-glb`.

**Worthwhile smaller alternative:** changing only `export_normals=True` to `False` removes normal-driven duplication while preserving every original triangle and genuine learned RGB. No geometry or material repair was performed.

| Measurement | Existing GLB | Compact GLB |
|---|---:|---:|
| Bytes | 479,788,924 | **92,305,152** |
| glTF vertices | 11,989,981 | **1,839,821** |
| Triangles | 4,012,360 | **4,012,360** |
| Attributes | POSITION, NORMAL, COLOR_0 | **POSITION, COLOR_0** |

Savings: **387,483,772 bytes / 80.7613%**, and **84.6553% fewer exported vertices**. All 1,839,821 source vertex rows referenced by triangles are represented. The other 3,147 original rows are unreferenced by every triangle; unchanged loose-point export settings omit them. No triangle surface or component is missing.

## Portable output

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-compact-glb-d3eec48\turret-learned-colour-compact.glb`

SHA-256: `6f3a71eb1dc683a83a83ba0b1c6e0add87391ef1d7ee4a9e0faf4092eb3f6c2d`.

Verification and export/import metadata are beside the file. The binary stays outside Git. It contains standard FLOAT32 linear RGB `COLOR_0`, triangle indices and embedded geometry, with no external resources, images or required compression extensions. The same opaque material uses constant metallic0, roughness0.8 and no emission; spatial learned PBR data remains preserved separately.

## Actual checks and shading

One byte-identical copy of the prior four-vertex/three-triangle fixture was exported once and imported once: **4 vertices instead of9**, exact positions, triangle coverage, winding and RGB, no NORMAL. Only then was the turret exported once and reopened once in a separate factory-startup Blender5.2.1 process. The installed executable/exporter identities are unchanged. [Settings comparison](amd-turret-compact-colour-glb-evidence/export-setting-comparison.json) confirms normals are the only export-setting change apart from output path.

The reused verifier checks every triangle corner and the complete triangle multiset, including winding and multiplicity. Exact world positions/orientation/scale are recovered after the standard `(x,y,z) → (x,z,-y)` conversion. All display-RGB bytes match; the independent raw accessor check measures **zero linear F32 RGB error**. Triangle hash remains `727a1d8ab867fbac6896df334eaedd31be66ffab5d5e46d6eeffcbf75de4d1f3`.

**Both source and fresh import have zero smooth faces and no custom normals.** No authored smooth/custom-normal information was discarded. The [glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#geometry) requires clients to compute flat normals when NORMAL is absent. This matches the source's flat-shading intent. Exact cross-viewer highlight/pixel identity is unproved: computed normals can differ slightly from the old exporter's rounded stored normals. No comparison render or automatic shading adjustment was needed or performed. Keep the original explicit-normal GLB for consumers requiring that representation.

Full export took **9.453s**, peak committed memory **1,885,712,384 bytes**; fresh import plus exhaustive verification took **24.282s**, peak **1,699,950,592 bytes**. Both exited0, naturally emptied their owned Jobs and required no forced cleanup. Unchanged full-process limits were **8GiB**, export **300s**, import **180s**; tiny fixture limits were2GiB/60s. No retry or unrelated test suite occurred.

[Complete verification](amd-turret-compact-colour-glb-evidence/verification.json) · [Fresh import](amd-turret-compact-colour-glb-evidence/full-import.json) · [Independent review](amd-turret-compact-colour-glb-evidence/independent-review.json).

The original479.8MB GLB, both source Blender scenes, original PLY, neural output and complete material field retain their published hashes. Main and all72 previous remote heads are preserved. No Torch, ROCm, WSL, neural execution, installs, geometry processing, UV/PBR bake or environment changes occurred. Stop after this compact alternative.
