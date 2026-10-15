# Portable genuine learned-colour turret GLB — PASS

Base `357e541ac13ecb5e1d30c1a866820cb6d81646cf`; branch `codex/amd-turret-learned-colour-glb`.

**PASS:** One full-resolution GLB was exported and independently reopened in a fresh Blender process. Every original triangle, its winding, positions and learned RGB survived. The file embeds all resources and needs neither the original Blender project nor external images.

## Actual asset

`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-coloured-glb-357e541\turret-learned-colour.glb`

- **479,788,924 bytes** (457.56 MiB).
- SHA-256: `ea9ea02c87311e91ddd455a2d0664ec5a2965adc966b108f72c3e3ce1f98df15`.
- All **4,012,360 triangles** from the original **1,842,968-vertex** surface are represented. Normal/attribute splitting produces **11,989,981 glTF vertices**; this is representation duplication, not added geometry.
- Opaque material: genuine learned RGB through **linear FLOAT32 `COLOR_0`**, constant metallic **0**, roughness **0.8**, no emission. No images, textures or external URIs.

`verification.json`, `full-export-metadata.json` and `full-import-metadata.json` are beside the GLB. The large asset remains outside Git. Import the `.glb` directly in Blender to view it.

This is an intermediate direct-row learned-colour asset. The genuine spatial metallic, roughness and alpha remain in the unchanged source `texture-field.safetensors` at `C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-owner-067f668\field`; they have **not** become portable PBR textures. Known original topology defects remain unchanged. No repair, smoothing, simplification, UV work or baking occurred.

## Verification and execution

The installed Blender **5.2.1 LTS** exporter recognizes the original `Vertex Color(Col).Color → Principled.Base Color` connection. A four-vertex, three-triangle noncoplanar fixture used actual learned RGB from four source rows. Its single export created `COLOR_0`; fresh import confirmed geometry, winding, shader linkage and colours. An initial verifier assertion incorrectly required an explicit imported layer name: the installed importer intentionally selects its active render colour layer with an empty name. That failed record is preserved. One corrected fresh import of the same fixture passed; the fixture was not re-exported.

Only then was the turret exported **once**, followed by **one** separate factory-startup Blender import. The source scene was never saved. The output uses the standard determinant-positive axis conversion `(x,y,z) → (x,z,-y)`; inverse conversion and fresh import recover exact original positions, orientation and scale.

Independent checks compare the complete triangle multiset, including cyclic corner order, winding, multiplicity and RGB. They permit split/reordered vertices, not lost/reversed faces. All recovered 8-bit display RGB matches exactly. A separate bounded GLB accessor reader verifies the original **linear F32 RGB with maximum error 0**; it does not rely on Blender's imported BYTE_COLOR storage. Canonical triangle hash: `727a1d8ab867fbac6896df334eaedd31be66ffab5d5e46d6eeffcbf75de4d1f3`.

| CPU process | Complete host time | Peak committed bytes | Deadline / cap |
|---|---:|---:|---|
| Full export | 15.500 s | 3,210,952,704 | 300 s / 8 GiB |
| Fresh import + exhaustive verification | 50.172 s | 6,008,889,344 | 180 s / 8 GiB |

Both exited **0**, owned Windows Jobs were empty, and **no forced cleanup** occurred. Admission required 12 GiB available host RAM and 3 GiB disk; actual full-export admission recorded 20,674,129,920 RAM bytes and 111,548,616,704 disk bytes. Four focused CPU tests covered split/reordered vertices, cyclic rotations, reversed/missing triangles, altered colours and linear-RGB corruption. No broad suite was repeated.

[Final verification](amd-turret-learned-colour-glb-evidence/verification.json) · [Fresh import](amd-turret-learned-colour-glb-evidence/full-import.json) · [Export containment](amd-turret-learned-colour-glb-evidence/full-export-host.json) · [Import containment](amd-turret-learned-colour-glb-evidence/full-import-host.json) · [Independent review](amd-turret-learned-colour-glb-evidence/independent-review.json).

## Preservation and limits

The source PBR scene (`555ef223…`), original RGB scene (`b5c91db6…`), original PLY (`89549a36…`), neural output (`dc466749…`) and genuine material field (`7b24c946…`) all retain their complete published SHA-256 identities. Before/after hashes and the 71 original remote heads are retained in evidence; main and historical branches are preserved. Existing unrelated untracked caches remain untouched; new bytecode caches are ignored.

No Torch, ROCm, WSL, model execution, environment changes, installations or native builds were used. Validation covers the actual GLB payload and Blender reopening; other viewers and game engines were not tested. The full-resolution file remains large. Stop after this verified export; no texture bake or further processing was started.
