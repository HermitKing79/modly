# Canonical AMD shape-SLat512 sampling

**PASS.** One genuine12-step shape-SLat512 sequence completed on RX9070XT: **21 model forwards (12 positive/9 negative)**, finite F32 sample, unchanged3,743 coordinates, healthy observation, zero allocated/reserved PyTorch bytes and clean owned-process exit. This establishes numerical sampling compatibility; decoded geometry quality remains unproved.

Base `5eb759d186b8f661c954b0a0b680370ff04057cf`; branch `codex/amd-shape-slat-512-sampling`. Exact independently reviewed harness published and remote-verified before execution: **`133dcb41f866488ff017595547f8d8c77bc446f3`**. [Raw result](amd-shape-slat-512-sampling-evidence/run/results.json), [independent artifact checks](amd-shape-slat-512-sampling-evidence/independent-artifact-analysis.json), [execution checks](amd-shape-slat-512-sampling-evidence/independent-execution-analysis.json), [independent results review](amd-shape-slat-512-sampling-evidence/independent-results-review.json).

The human approved correcting22 to the source-faithful21 forwards, with every other parameter unchanged ([approval](amd-shape-slat-512-sampling-evidence/authorization-correction.json)). Rescaled step9 has t=0.6 and step10 t=0.5; only steps1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“9 lie in inclusive[0.6,1.0]. Genuine `FlowEulerGuidanceIntervalSampler`:12 steps, sigma_min0.00001, guidance7.5, guidance_rescale0.5, rescale_t3.0, seed20261013, explicit dino_lock0. Native sparse population-standard-deviation CFG and Euler arithmetic were retained without epsilon/clamp/repair. Output is final `sample(...).samples`, rather than predicted x0. Denormalization uses audited CPU F32 `raw*std+mean`.

Saved turret conditioning F32 `[1,1029,1024]` SHA256 `78a0b54210568624639c8cacb7a803d83d4b838b782ea54e7aceab749d47e17a` and canonical int32 `[3743,4]` batch,D,H,W coordinates were reused. No DINO, sparse-structure sampling or occupancy decode repeated. Genuine `slat_flow_img2shape_dit_1_3B_512_bf16` SHA256 `ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f` loaded strictly once:640 entries, existing220 reversible source-required F32 widenings, no missing/unexpected entries or global loader relaxation. Python3.12.10 / PyTorch2.13.0+rocm10.0.0 / HIP7.15.26333; CUDA build metadata null; RX9070XT gfx1201. Explicit SDPA, no sparse convolution or CPU computation fallback.

The proven worker, Job containment, observer, diagnostics and BLAS/BLASLt teardown were reused. Original single-forward and production mock routes remain intact.25 focused CPU tests passed with GPU initialization and real weights blocked. Actual production hooks and genuine sparse sampler were replayed using CPU fixtures. Review corrections tightened NaN rejection, compact tracing and final evidence classification. An isolated test-suite audit-hook collision is preserved in [preparation notes](amd-shape-slat-512-sampling-evidence/cpu-preparation-notes.json).

WORKSPACE_DIR was explicitly set and CPU-verified in the actual invoking PowerShell process, inherited by the launcher/worker ([preflight](amd-shape-slat-512-sampling-evidence/final-launch-preflight.json)). CPU/GPU/overall deadlines180/240/480 seconds, host requirement11,411,771,648 bytes, minimum13GiB availableVRAM and12GiB allocator cap remained enforced. No warmup/retry. Each root/30blocks/60attentions ran21 times; staged timestep and frequency transfers21 each; RoPE frequency copied once and phases reused. Exact final counters are independently cross-checked against21 durable root entry/return pairs and12 synchronized step completions.

| Measurement | Result |
|---|---:|
| Available host RAM before strict load / required |17,830,862,848 /11,411,771,648 bytes|
| Strict CPU load + input preparation |10.656s|
| Synchronized GPU placement |0.594s|
| Complete sampling + final synchronization |**8.890s**|
| Supervised worker |29.672s|
| Peak PyTorch allocated / reserved |**2,911,625,728 /3,078,619,136 bytes**|
| AvailableVRAM before / after sampling / after release |16,937,582,592 /13,725,663,232 /16,785,408,000 bytes|
| PyTorch allocated / reserved after release |**0 /0 bytes**|
| Peak host working set / private commit |5,674,065,920 /6,805,651,456 bytes|
| Periodic Windows checks |3 healthy; maximum gap5.031s|
| Exit / active owned Job processes |0 /0; no forced cleanup|

Raw and denormalized F32 `[3743,32]` arrays each contain119,776 finite values; sparse logical shape `[1,32]`. Coordinates are byte-identical to source-derived32Ãƒâ€šÃ‚Â³ coordinates. Raw min/maxÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢6.2533054/5.8535156, mean0.01862432, std1.1917493. Denormalized min/maxÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢28.125721/29.861464, std6.4120173. Standard NPY headers/payloads, NumPy readback, source normalization and provenance crosslinks were independently validated. Publication copies are byte-identical to preserved owned scratch.

| Saved artifact | Bytes | File SHA256 |
|---|---:|---|
|[Raw latent](amd-shape-slat-512-sampling-evidence/shape-latent-raw.npy)|479,232|`03fd8208614fcf20c37e40e59c9eb48972c3c7a3ee18a31bab1b0537e86ab2a1`|
|[Denormalized latent](amd-shape-slat-512-sampling-evidence/shape-latent-denormalized.npy)|479,232|`90eaf72030f303b209a4d02ae9adc7d4ecd39662344f35f0adc8120bc1e871b9`|
|[Coordinates](amd-shape-slat-512-sampling-evidence/shape-latent-coordinates.npy)|60,016|`949a9e6a2548abcad66354384502c06834d4d56ea3f2fa86b619c25926379ed4`|
|[Initial noise](amd-shape-slat-512-sampling-evidence/shape-initial-noise.npy)|479,232|`6ba908a9c41e74fc6201e46ac428b4603dddf24bd4ea38d7ab9774ae422235d0`|
|[Full provenance](amd-shape-slat-512-sampling-evidence/shape-latent-provenance.json)|14,918|`4834412e700434f5645cd919dd0d75e9209b6b5d67cc7cb765e98b76890f9a81`|

Initial CPU-generated noise payload SHA256 `b5fa87e715f728aabcd218a109f492f59c51999039581a20b640c860e77a9c7b` matches the prior deterministic forward fixture. Raw/denormalized payload SHA256 and normalization values are retained in provenance/analysis. Full per-model module counters remain in raw results, hash-linked from bounded artifact provenance.

Teardown dropped sampler histories, shared caches, model/output/input references, synchronized and cleared proven BLAS/BLASLt workspaces once. All owned weak references collected; worker and Job exited. The native operation histogram independently confirms36 segment reductions (4 per guided step) and1,260 SDPA calls (60 per forward). Six process samples observed only Python interpreters, without compiler children; discrete samples do not exclude short-lived or in-process compilation. No native error, guard failure, timeout or reset occurred.

Both installed environment inventories and checkpoint-before/after records are byte-identical. All5,372 historical files, prior local/remote branches including main, AMD edits and backup remain unchanged ([preservation](amd-shape-slat-512-sampling-evidence/git-after.json)). Source/config/checkpoints, installed caches, drivers and production worker behavior were preserved; fresh native caches remain confined to retained owned scratch. No shape decoder,1024cascade, CuMesh, mesh extraction, textures or GLB ran.

**Next approval boundary:** audit the genuine shape decoder's exact local contract and native-operation requirements, then consider one bounded decode using this saved denormalized latent and unchanged coordinates. No such audit, model loading or execution has begun. Stop here; sampling PASS does not establish usable geometry.
