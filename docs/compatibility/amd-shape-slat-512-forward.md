# AMD shape-SLat 512 single-forward proof

**PASS.** Exactly one genuine `SLatFlowModel` forward completed on RX9070XT, with finite F32 output, unchanged sparse coordinates, synchronization, healthy observation, zero allocated/reserved PyTorch bytes and clean owned-process exit. This proves execution compatibility for this forward; sampled shape or mesh quality remains unproved.

Base `27232cf6b2ba7f03e59327bc69065364a6960eb3`; branch `codex/amd-shape-slat-512-forward`. Independently reviewed harness published and verified **before** execution: `b4055367758c7ef3ba1dd79903bfe01a316ba8dd`. [Raw result](amd-shape-slat-512-forward-evidence/run/results.json), [independent artifact readback](amd-shape-slat-512-forward-evidence/independent-artifact-analysis.json), [execution analysis](amd-shape-slat-512-forward-evidence/independent-execution-analysis.json).

The preserved turret conditioning is finite F32 `[1,1029,1024]`, SHA256 `78a0b54210568624639c8cacb7a803d83d4b838b782ea54e7aceab749d47e17a`. Its native logits and mask are unchanged. Microsoft's pinned 512 path thresholds native decoder logits at zero, applies `max_pool3d(mask.float(),2,2,0)>0.5`, then selects `argwhere[:,[0,2,3,4]].int()`. Exact CPU comparison with real PyTorch produces **3,743** unique lexicographic int32 `[batch,D,H,W]` tokens; bounds `[5,0,6]`..`[26,31,25]`. The saved derived occupancy and coordinates remain separately labelled 32³ artifacts.

The genuine local `slat_flow_img2shape_dit_1_3B_512_bf16` has 1,292,253,728 parameters, 640 BF16 checkpoint entries, file SHA256 `ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f`. The weights-free real architecture matches every key/shape: 420 runtime BF16 and 220 runtime F32 entries, with reversible widening scoped to source-required F32 parameters. No persistent buffers are absent. Thirty ordinary unlearned RoPE frequency attributes are initialized using the genuine CPU constructor; first-block phases are shared in the sparse cache during the forward.

The pinned source's explicit PyTorch SDPA branch was selected through its setter; sparse convolution was disabled because this model uses none. FlashAttention exception fallback, external sparse kernels, Triton compilation/launch, remote factories, other checkpoints and extra model calls remained prohibited. Runtime: existing Python3.12.10 / PyTorch2.13.0+rocm10.0.0 / HIP7.15.26333, CUDA build metadata null, gfx1201. Actual inputs: seed20261013 F32 noise `[3743,32]`, canonical first-model timestep1000, saved conditioning and genuine SparseTensor layout. Output is a finite F32 **velocity prediction** `[3743,32]`, sparse batch/channel shape `[1,32]`, with exact original int32 coordinates. It is not a sampled or denormalized shape latent.

Six CPU contract tests and seven production transfer/diagnostic tests pass. The corrected owned CPU architecture probe passed without real model loading, forward calls, or GPU initialization. Original CPU test-harness failures remain preserved and explained in `cpu-preparation-notes.json`; they are not native computation failures. Read-only RAM admission was 18,576,961,536 bytes against 11,411,771,648 required bytes. Preflight verified 5,308 historical files, all existing branches, both installed environments and AMD source edits/backup.

The reused supervisor retained owned Job containment, healthy periodic Windows observation, CPU/GPU/overall deadlines180/240/480 seconds, minimum13GiB available-VRAM admission and 12GiB allocator cap. Real-load RAM:18,111,471,616 bytes available /11,411,771,648 required. Exactly one strict load and root forward; each30 blocks/60 attentions once; timestep and first-RoPE frequency transfers once. No retries/warmup.

| Measurement | Actual result |
|---|---:|
| Strict CPU load + input preparation |10.891s|
| CPU peak working set / private commit |5,675,151,360 /6,806,732,800 bytes|
| Synchronized AMD placement |0.625s|
| Model host return / forward including synchronization |3.907 /**3.922s**|
| Supervised worker elapsed |24.703s|
| Peak PyTorch allocated / reserved |**2,896,867,328 /3,072,327,680 bytes**|
| AvailableVRAM before / after forward / after release |16,937,582,592 /13,751,353,344 /16,815,292,416 bytes|
| PyTorch allocated / reserved after release |**0 /0 bytes**|
| Periodic Windows maximum gap |5.047s; all healthy|
| Worker exit / active owned Job processes |0 /0; no forced cleanup|

Teardown dropped shared caches and all model/tensor references, synchronized, cleared proven BLAS/BLASLt workspaces once, and verified collection. Four process-activity samples observed only the venv launcher/interpreter; no compiler child was observed. Samples do not exclude in-process compilation. No GPU-attempt native error, guard failure, timeout or reset occurred.

Preserved publication artifacts, also retained byte-identically in owned scratch:

| File | Contents | Bytes | SHA256 |
|---|---|---:|---|
|[shape-flow-output.npy](amd-shape-slat-512-forward-evidence/shape-flow-output.npy)|F32 `[3743,32]` velocity|479,232|`8929865823ae95c60897296b6e10b29eb89aed458505a3067d1a1f8a010d79da`|
|[shape-flow-coordinates.npy](amd-shape-slat-512-forward-evidence/shape-flow-coordinates.npy)|I32 `[3743,4]` coordinates|60,016|`949a9e6a2548abcad66354384502c06834d4d56ea3f2fa86b619c25926379ed4`|

Independent stdlib NPY container/payload and NumPy readback agree with worker hashes. All119,776 output values are finite: min−5.2884293, max4.9230051, mean−0.03223495, std1.2078959. Coordinates exactly match canonical32³ inputs.

Post-run passive inventories are byte-identical; all5,308 historical files, prior local/remote branches including main, and AMD source edits/backup remain unchanged ([preservation](amd-shape-slat-512-forward-evidence/git-after.json)). Installed environments, drivers, checkpoints and production mock routes were preserved. Historical caches are intact; new private scratch is retained with hashes. No DINO, sparse-structure stages, shape decoder, CuMesh, textures or GLB ran.

Shortest next milestone: **one separately approved canonical512 shape-SLat sampling sequence**, using these same coordinates and saved conditioning, saving its latent without decoding. Audited shape parameters differ from sparse-structure settings:12 steps, rescale_t3.0, guidance7.5, guidance_rescale0.5, interval[0.6,1.0], sigma_min1e−5 ([read-only contract](amd-shape-slat-512-forward-evidence/next-source-contract.json)). It requires a reviewed multi-forward budget and containment contract; do not relax the single-forward counters. No next-stage implementation or execution has started.
