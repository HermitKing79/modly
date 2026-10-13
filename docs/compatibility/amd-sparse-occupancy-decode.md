# AMD saved sparse-structure occupancy decode

**PASS for this bounded decoder/artifact milestone.** One genuine `SparseStructureDecoder` forward decoded the committed sampled latent into finite native F32 `[1,1,64,64,64]` logits. One connected component contains 1,606 occupied voxels. All artifacts are preserved and independently readable; the worker released every owned GPU tensor and exited normally. The preview is a thin slab, not a validated detailed object shape.

Starting branch/commit: `codex/amd-sparse-structure-sampling` / `26fc3aa988c57f082542a80636a63f436218f3fb`. Continuation: `codex/amd-sparse-occupancy-decode`. Reviewed execution harness `fe02093614605a0621c738b0e7827b287660fd14` was committed, pushed and verified through GitHub before the exclusive attempt marker. See [preparation](amd-sparse-occupancy-evidence/preparation.md), [CPU tests](amd-sparse-occupancy-evidence/cpu-tests.json), [review](amd-sparse-occupancy-evidence/independent-review.json) and [execution](amd-sparse-occupancy-evidence/run/results.json).

## Exact inputs and execution

The original sampled latent remained unchanged: `amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors`, 134,688 bytes, SHA256 `8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2`. Its bounded header, embedded complete sampling/DINO/source provenance, `sparse_latent` key, F32 `[1,8,16,16,16]` shape, payload hash and every finite value were verified before GPU initialization. No synthetic replacement or additional conditioning/sampling execution occurred.

Only `D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16` loaded. Checkpoint SHA256 `1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a`; all 74 entries matched strictly, with no missing/unexpected keys and 147,585,412 parameter/buffer bytes. Source/config hashes are preserved in [provenance](amd-sparse-occupancy-evidence/occupancy-provenance.json). The inspected two 2× pixel-shuffle stages require native 64³ output; input/output remain FP32 and the genuine decoder body uses FP16.

Runtime: existing Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, CUDA version metadata `None`; RX 9070 XT/gfx1201. The actual PowerShell launcher set `WORKSPACE_DIR=C:\Users\wattz\Documents\ChatGPT\modly\proof`, and the actual workspace verifier passed CPU-only before launching its inheriting worker.

| Measurement | Observed |
|---|---:|
| CPU strict load |1.578s|
| Synchronized complete forward |25.157s|
| GPU stage including release |26.000s|
| Overall owned worker |33.984s|
| Peak allocated / reserved VRAM |1,127,093,760 /1,648,361,472bytes|
| Allocated / reserved after release |0 /0bytes|
| Available VRAM before placement / after release |16,937,582,592 /16,822,960,128bytes|
| Process exit / strict loads / root forwards |0 /1 /1|
| Exact input/output copies / workspace clears |1 /1 /1|

The 12 GiB allocator cap and 120/180/360-second CPU/GPU/overall deadlines were unchanged throughout the attempt. Fresh RAM of 15,873,228,800 bytes exceeded 4 GiB admission. Nine previously validated cache entries were copied into fresh writable owned scratch; historical cache files were never live runtime paths. Job attachment preceded preparation; owned-tree exit was verified without forced cleanup. Six healthy periodic Windows observations had a maximum 5.031 s gap. Durable module/convolution entry/return records distinguish host return from the confirmed after-forward GPU synchronization. No compiler child was observed in 14 process activity samples; in-process compilation cannot be excluded. No timeout, driver reset, native error or retry occurred.

## Preserved occupancy and preview

Exact threshold: **`decoded > 0`**. No pooling, morphological processing or threshold changes. Binary mask 64³, coordinates I32 `[1606,4]` in `[batch,D,H,W]` order. Occupancy fraction 0.00612640380859375 (0.61264038%); inclusive bounding box `[0,20,31]` through `[63,43,32]`, extents 64×24×2. Six-neighbor connectivity: one component of 1,606; 50 occupied cells touch a volume boundary. Output is nonempty and passes the predeclared nondegeneracy criterion, but it visibly forms a thin slab and touches both D boundaries. This proves execution/data preservation, not shape fidelity or a correctly shaped zombie.

All files below are at **`C:\Users\wattz\Documents\ChatGPT\modly\proof\docs\compatibility\amd-sparse-occupancy-evidence\`**:

| Artifact | Bytes | SHA256 |
|---|---:|---|
|[occupancy-logits.npy](amd-sparse-occupancy-evidence/occupancy-logits.npy)|1,048,704|`b30e463e1fb9530a630dd8696f991ed663c4da105d611009993feacf8d187c7c`|
|[occupancy-mask.npy](amd-sparse-occupancy-evidence/occupancy-mask.npy)|262,272|`d48345608add4a0c864e060b367966e15d1278f3eb52cf55255996a38aa87fdb`|
|[occupancy-coordinates.npy](amd-sparse-occupancy-evidence/occupancy-coordinates.npy)|25,824|`674202ea9471148271d7f3b58ffb65a4f460dd1dbfaa64c3001a0df2e8cf8b81`|
|[voxel-preview.obj](amd-sparse-occupancy-evidence/voxel-preview.obj)|107,027|`cd1ab5d12addb3cd545d80d6bdd0d18ae7b2348fe05e176a2bcd7fd37199568a`|
|[voxel-preview.png](amd-sparse-occupancy-evidence/voxel-preview.png)|21,422|`3f6f27fef3786ce91bc57fe0ec888097983e5d0a41a05e2c6755374c498e5e9c`|

The worker verified all three NPY readbacks before dropping GPU outputs. [Independent stdlib verification](amd-sparse-occupancy-evidence/independent-result-analysis.json) checked complete raw payloads, hashes, provenance, exact threshold, ordered coordinates and connectivity, then preserved byte-identical copies. Coordinates use valid Fortran-order NPY storage; the independent reader's initial row-major assumption was diagnosed and corrected, recorded [here](amd-sparse-occupancy-evidence/independent-reader-layout-correction.json), with no artifact/harness changes or GPU rerun. Logits/mask are C-order. Recorded payload hashes use logical C-order bytes; complete-file hashes cover actual NPY storage.

CPU-only exposed-cell preview: 3,392 vertices and 3,390 outward unit quads, independently matched against every exact exposed mask face. OBJ x/y/z maps to D/H/W. It stayed below 50,000 faces/12 MiB OBJ and 2 MiB PNG limits. No CuMesh or native geometry code was invoked. This diagnostic voxel surface is not TRELLIS.2's detailed shape mesh.

## Verification and preservation

22 focused CPU tests passed: real input reader/tamper rejection, native shape/dtype and production transfer guards, threshold/writers, cleanup order/failures, supervisor lifecycle and bounded preview. Independent source review found no blocker; its own sandbox-limited rerun passed three tests and hit WinError 5 path restrictions in six, without claiming a full rerun PASS. Final CPU artifact and OBJ readbacks passed. Installed Modly/AMD inventories, source hashes and raw model/cache metadata match before/after; [Git preservation](amd-sparse-occupancy-evidence/git-after.json) confirms all prior branches/evidence and existing `nodes.py`/backup changes unchanged. Main remains `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`.

No execution blocker remains for this saved-latent decode. The concrete quality limitation is the thin occupancy slab. Keep that finding intact and review the saved conditioning/input/sampling contract before claiming useful object geometry. No DINO, sparse-flow, shape-SLat, texturing, native mesh repair or GLB work followed this attempt. The earlier CuMesh 719 risk remains unresolved and was outside this execution. Stop here; further model work requires separate approval.
