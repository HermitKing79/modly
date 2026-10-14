# Read-only Linux RX 9070 XT route comparison

Assessed on 2026-10-14. Reference: [kr4ckhe4d/trellis2-9070xt](https://github.com/kr4ckhe4d/trellis2-9070xt/tree/0220d73971fd0be242b2b311ac00d590e1b26301), pinned HEAD `0220d73971fd0be242b2b311ac00d590e1b26301`. Documentation and three execution/export files were read; no repository was cloned, platform switched, environment accessed or modified, or code executed. [Source identities](amd-cached-neighbour-mixing-evidence/linux-source-assessment.json) preserve the exact Git blob hashes.

## What the Linux implementation demonstrates

The author reports a complete 512³ crown-image result on RX 9070 XT/gfx1201: about 190,000 triangles, an 8.5 MB GLB with base-colour and metallic-roughness textures, 21.4 seconds warm and about 9.7 GB peak memory. Its Python 3.12/PyTorch `2.13.0+rocm7.1`/ROCm 7.2.4 stack differs from our Windows ROCm 10 stack. Cold model loading and first-use compilation are much slower. These are author-reported measurements, not results reproduced here. [HANDOVER](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/HANDOVER.md)

The important qualification is raw geometry: nonfinite decoder vertices are reported, then removed during export processing, potentially losing detail. A clean exported GLB therefore does not prove finite raw decode or correct shape. The documented 1024³ path fails through memory fragmentation; 512³ is the supported target. Its preview silhouette metric is also documented as unreliable. [HANDOVER](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/HANDOVER.md)

## Techniques and Windows applicability

| Area | Linux implementation | What could be reused on Windows |
|---|---|---|
| Native convolution | `explicit_gemm`, TF32 disabled; different from our cached masked-implicit SplitK leaf | Assess the installed explicit-GEMM path as an alternative to dynamic Triton coverage. It still needs genuine native neighbour/wrapper and resource proof. The Linux rationale cites another gfx target; it does not establish a defect in our tested gfx1201 kernel. |
| Decoder arithmetic | Mixed BF16 body with FP32 norms/input/output | A separately approved precision experiment, requiring strict loading/conversion analysis. It changes the original F16 arithmetic and cannot be silently transplanted. |
| Residency | `low_vram=True`, active-model residency, texture latent parked on CPU during shape decode | Portable Python lifecycle concepts. Reuse our strict per-component loader and measured teardown rather than its whole-pipeline factory. |
| Allocator/native memory | Removes HIP expandable segments before torch import; releases cached PyTorch memory before native baking | Useful process-local configuration and stage-boundary checks. Its raw HIP allocations explain why PyTorch statistics alone are insufficient. Do not alter our environment or acceptance thresholds on that basis. |
| Worker ownership | Persistent GPU child; API remains GPU-free; Linux parent-death signal | Reuse Modly's extension subprocess architecture and existing Windows Job Objects. Linux `prctl` itself is not portable. Persistent model residency also does not satisfy our disposable-proof zero-release criterion. |
| Texturing | Genuine texture sampling; vectorized PyTorch rasterization replacing a software adapter | Potential portable later fallback, subject to numerical/output-size/memory checks. No texture inference or baking has been proved in our experimental Windows worker. |
| GLB export | Signature-aware O-Voxel `to_glb` adapter, material/sparse attributes, optional WebP textures | Export argument mapping and independent material/GLB validation are reusable ideas. Native extraction, UV work and our installed rasterizer remain separate proof boundaries. |
| Native repairs | Vendored CuMesh HIP memcpy correction; guarded hole filling | Source-level fixes require binary provenance or separately approved rebuilding. Its warn-and-continue policy must not replace our fail-closed handling. Our earlier CuMesh 719 is not shown to share that cause. |

These mechanisms are supported by the actual [fix code](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/inference/amd_fixes.py), [GPU worker](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/inference/gpu_process.py) and [GLB interface](https://github.com/kr4ckhe4d/trellis2-9070xt/blob/0220d73971fd0be242b2b311ac00d590e1b26301/inference/server_glb.py). Linux `.so` binaries, `/opt/rocm` build paths and ABI/runtime-specific caches cannot be copied into the Windows `.pyd` environment. The Python techniques are not inherently Linux-only, but their local native dependencies still require validation.

The Linux code also uses ordinary `from_pretrained(model_id)`, an alternative background-removal model, and warnings after some patch/hole-fill failures. Its offline-cache defaults are not equivalent to our explicit local paths, complete strict state dictionaries and validated alpha-only input. Those safety and model-selection policies must remain explicit.

## Cheapest evidence-based route

**Recommendation: use the Linux implementation as the stronger reference for reaching a first fully textured 512³ asset, and stop synthetic testing after the approved Windows case.** This is a direction recommendation, not local Linux readiness or permission to install it.

Continued Windows integration still requires the genuine neighbour/postprocessor/full-wrapper path, 40 convolutions across nine channel pairs, four learned subdivisions with pre-allocation admission, actual neural decoding, extraction, texture models and export checks. The existing cache-only inventory has possible LOGN12 gaps; actual turret expansion is unknown. Its resource planners are not attached and do not prove all temporary/native workspace bounds. A two-axis numerical kernel PASS closes none of these integration gaps.

The next controlled experiment should be **one offline, neural-only decode of the preserved turret latent using the Linux reference's explicit-GEMM/BF16 path**, separately approved with that precision change stated. First establish whether a compatible Linux runtime already exists and verify exact source/checkpoint identities, offline strict loading, bounded support growth, allocator/native-memory limits, owned containment and deadlines. Require finite genuine features, valid dynamic coordinates, synchronization and clean release; stop before extraction. If no suitable runtime exists, report prerequisite UNKNOWN and seek a separate setup decision; do not rebuild it automatically.

This reuses the preserved conditioning/shape result and avoids repeating DINO or sampling. A later textured-GLB proof must separately assess finite raw geometry, any sanitization/remeshing loss, texture/material integrity and rendered shape quality. No Linux execution, implementation, Windows backend change, texture work or further synthetic test is authorized by this assessment.
