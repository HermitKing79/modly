# High-resolution AMD master in the actual Modly app

**Overall: BLOCKED for complete workflow qualification.** The installed Modly **0.4.3** genuinely imported, displayed and exported the accepted full-resolution brick-and-vines GLB. Five acceptance areas pass; camera interaction and in-session unloading remain unverified. No production code was changed, and the master was imported only once.

Source base: `64d07f5c5a10a48a1cbff028cb6143d2a444ae89`. Branch: `codex/amd-high-resolution-master-app`. The installed binary was authenticated separately; this is not an exact-checkout-build claim. See the [runtime audit](amd-high-resolution-master-app-evidence/installed-runtime-audit.json), [saved execution summary](amd-high-resolution-master-app-evidence/execution-summary.json) and independent outcome review in the evidence directory.

| Acceptance area | Result | Actual evidence |
| --- | --- | --- |
| GLB import | PASS | Normal Import → Mesh → native file picker; tiny 4-vertex/3-triangle control first, then one master import. Both original API requests returned 200. |
| Full-resolution viewport | PASS | Whole brick-and-vines object framed and visible in the captured desktop view; one mesh with 12,520,752 indices and no mesh groups. No simplification operation invoked. |
| Original vertex colours | PASS | Live Float32Array POSITION and linear COLOR_0 buffer hashes exactly match the original. MeshStandardMaterial has vertexColors enabled, white base colour, original metallic/roughness factors, double-sided surfaces and flat shading. |
| Camera interaction/responsiveness | BLOCKED | UI import/export responded, but orbit, pan, zoom and a frame-rate/latency assessment were not established before the fixed session budget ended. An attempted drag encountered the export dialog and does not count. |
| Triangle statistics | PASS | Actual viewer and geometry both report **4,173,584 triangles / 1,957,891 vertices**. The helper/postpass render counter is not used as mesh statistics. |
| Original-byte GLB export | PASS | Normal Export → .glb → native Save As, followed by independent post-exit full-file hash verification: **97,073,940 bytes**, SHA-256 `1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae`. |
| Model unloading/resource cleanup | BLOCKED | No in-session replacement/unload test or GPU disposal measurement. App close returned 0 and the owned Windows Job emptied without supervisor force, which does not prove model-level cleanup. |

The master remains unchanged. Three.js builds a BVH that may reorder index triples internally; no exact in-memory index-order hash was captured. The complete original exported GLB remains byte-identical, so this result does not claim the live BVH index buffer retained source triangle order. No asset-generation or mesh-edit operation was invoked; original asset buffers and material factors were retained.

The preview shows the original brown brickwork and foliage under Modly's existing lighting. It was initially small in the viewport, and detailed visual inspection from multiple angles remains outstanding. Exact colour buffers and material routing establish transport fidelity, not pixel equivalence to the earlier Blender studio render.

## Measured resources and startup limitations

The graphics application session enforced **6 GiB per process AND aggregate Job**, a **900-second total interactive deadline**, **180 seconds per import**, **12 GiB available Windows RAM** and **2 GiB destination-free disk** floors. All **896** live resource samples admitted. Elapsed session: **896.891 s**, including debugger setup, diagnostic startup work and desktop interaction, not asset load time. Individual import timings and FPS were not captured precisely.

- Peak process committed memory: **725,319,680 bytes (691.7 MiB)**.
- Peak aggregate Job committed memory: **1,929,433,088 bytes (1.797 GiB)**.
- Minimum available Windows RAM: **16,604,430,336 bytes (15.465 GiB)**.
- Minimum destination-free disk: **136,734,617,600 bytes**.
- Owned app return code 0; Job active processes 0 before/after cleanup; no forced Job cleanup.

Electron also reported the following CPU-memory snapshot while the master export dialog was open. These are **not dedicated VRAM** or simultaneous per-process peaks.

| Process | Working set (KiB) | Private bytes (KiB) |
| --- | ---: | ---: |
| Browser | 174,604 | 108,196 |
| Renderer | 395,788 | 305,384 |
| GPU process | 288,708 | 444,364 |
| Network utility | 61,436 | 12,976 |

Dedicated GPU memory was not measured. The stock PythonBridge shuts down its backend using `taskkill /T /F` and logs backend exit 1; therefore natural app settlement is not described as graceful exit of every child.

Two startup issues consumed most of the session budget. The first restricted-context process exited `0x80000003` before a page or asset loaded; its exact internal cause is unproved. In the second, trusted desktop-context launch, the temporary observer awaited renderer initialization before original navigation. One independently reviewed blank-page initialization unblocked registration and the unchanged original app page. This is a **diagnostic-harness limitation**, not evidence of a Modly asset failure or unassisted startup success. All original failure receipts are preserved. The observer's four mocked tests did not cover this cold-renderer dependency.

The tiny fixture's import, buffers, material and statistics were verified before the master. The source-review checklist additionally requested a tiny export/download hash first; that additional check was omitted. The user's small import/view requirement was met, but the complete reviewer checklist was not. The independently hashed full-master export remains an observed fact; no claim is made that every reviewed admission step was completed.

## Private outputs and preservation

Actual viewport screenshot:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-high-resolution-master-app-64d07f5\desktop-session\screenshots\master.jpg`

Actual UI-exported master:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-high-resolution-master-app-64d07f5\desktop-session\exports\modly-original-master.glb`

Screenshots, complete private receipts and the GLB stay outside Git. Source-safe hashes are in the execution summary. No neural, WSL, Blender, MeshLib, AI-weight loading, optimization, remeshing, installation or dependency upgrade was performed. All 100 protected files, authenticated original assets, selected environment/source/backup identities and 114 previous remote branch heads match the preservation baseline.

The initial AFTER preservation receipt is retained as **FAIL**: its profile file-set comparison treated 2,096 backslash names as additions and the same 2,096 forward-slash names as removals. Read-only normalization of the saved sets establishes zero genuine additions/removals; all **2,103** original profile file hashes, sizes and timestamps, and all **12** installed-app/input identities match. A separate attribution supplement records this audit-formatting defect without deleting or changing the original receipt. Checkpoint preservation uses unchanged file metadata and bounded header hashes plus historical full-hash provenance; weight payloads were not reread.

**Next step:** one separately approved, short existing-app session focused only on orbit/pan/zoom and master-to-tiny replacement with resource-disposal measurements. Do not add another importer or infer cleanup from application exit. This partial result does not yet qualify the entire High-Resolution Original Mesh workflow or authorize generation integration.
