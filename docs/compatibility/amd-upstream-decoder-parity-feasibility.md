# Phase 10 — upstream decoder parity feasibility

30 October 2026. Base `7ef1f1703b499bc556d897229397505d2b746ee7`.

**NO-GO FOR UPSTREAM PARITY EXECUTION.** Windows' read-only adapter inventory identifies only the RX 9070 XT. No qualified external reference machine has been supplied. Inventory of six preserved, non-fixture decoded-output paths finds only the existing AMD result bound to this box's shape job; there is no independently authenticated matching upstream result. This is a scoped project/local-hardware finding, not a claim about every machine the user might own. No parity harness, reference environment or execution handoff was built.

## Authenticated inputs

The existing bounded CPU reader freshly verified the successful AMD receipts, complete ten tensors/payload hashes, four subdivision linkages, source image/conditioning/shape provenance and raw PLY/NPY identity. Phase 9 evidence and published-source hashes also match. Original input/output identities remain:

| Item | SHA-256 |
| --- | --- |
| Raw shape latent | `d0a5c8b5f490484227ed6a822400b56dfeb9432b7637542903c7ffaba97c8505` |
| Decoder input F32[3888,32] | `caabc77639b7ccb355836fe2dee6f36bdbcdbd0bb7d32b207854ac74f1455f3e` |
| Coordinates I32[3888,4] | `dc958a216097954dbc1844669a24b645503c2e87f69fd2c08fb1aa372d3dbdec` |
| Checkpoint, 948,490,494 bytes | `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581` |
| Checkpoint JSON config | `5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209` |
| Original ten-tensor decode | `ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975` |

The checkpoint was freshly streamed for SHA-256 with unchanged file identity, size and mtime; it was **not deserialized or loaded as a model**. Original source files/arrays were not changed.

## What a decoder-only reference actually needs

This assessment explicitly pins and captures **16 official source files at `de38fdd4082638cb4e6d2285f2e215280bd3f485`**, independently verifying their Git blob identities and SHA-256. Earlier evidence pinned an AMD runtime policy and the Microsoft converter; it did not qualify this official decoder runtime. This new source pin is an audit reference only.

The configured decoder has latent channels32, output channels7, channels `[1024,512,256,128,64]`, blocks `[4,16,8,4,0]`, ConvNeXt blocks and four learned C2S upsamplers. A strict parameter-key/shape/dtype load of the same checkpoint plus the original sparse inputs suffices for the **parent** [SparseUnetVaeDecoder forward](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/trellis2/models/sc_vaes/sparse_unet_vae.py#L398-L505), with `eval`, `guide_subs=None`, `return_subs=True` and fresh spatial caches. It returns seven raw features and subdivision outputs before the [FDG wrapper's mesh conversion](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/trellis2/models/sc_vaes/fdg_vae.py#L52-L110). No image encoder, occupancy/shape sampler, texture decoder, rendering or exporter computation is needed. Importing the FDG wrapper normally pulls in Mesh/O-Voxel; avoiding those imports would still require reviewed parent-loader equivalence.

Minimum numerical runtime: compatible Python/PyTorch, the pinned sparse tensor/linear/norm/C2S modules, and a working FlexGEMM sparse-convolution backend with its native/Triton dependencies. This configuration contains no attention block; full-pipeline FlashAttention is not a decoder mathematical requirement. Safetensors/NumPy input and strict state-loading support are also needed. The [official setup](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/setup.sh#L70-L81) lists Torch2.6 CUDA12.4 and a ROCm branch; that branch is not proof of qualified AMD reference execution. Its FlexGEMM clone is unpinned, so extension source/binary/ABI identities would need separate freezing. No such installation or qualification occurred.

The saved JSON's `resolution:256` is not evidence of a wrong 512 decode: it is the wrapper's mesh-grid setting. The [official pipeline explicitly sets the requested resolution](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/trellis2/pipelines/trellis2_image_to_3d.py#L357-L378); the admitted 32-grid latent passes four learned expansions to a 512 grid. A comparison must bind the effective512 interpretation and unchanged architecture separately.

## Memory and qualification gate

Microsoft documents Linux and **at least 24 GB NVIDIA VRAM** for its general supported workflow, verified on A100/H100. That is **not a measured decoder-only minimum**. The checkpoint's approximately0.883GiB file size is not an activation/workspace budget. Our AMD12GiB allocator ceiling is neither a measurement of NVIDIA demand nor proof that a12GB card fits. Decoder-only NVIDIA peak, workspace and safe admission remain **unmeasured**. [Official prerequisites](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/README.md#L56-L78)

A previously working, authenticated upstream host would make loader/output capture/comparison qualification a bounded engineering task. Starting from an unqualified host entails dependency/native-binary qualification, precision verification and memory profiling; it is not a cheap launch-script change. No supported hour estimate or decoder-only GB estimate exists. A12GB reference remains unqualified; no paid/cloud route is proposed.

## Comparison protocol — fixed before any future execution

1. Bind identical original input hashes, checkpoint/config, official source commit, effective512 grid, model modes and complete dependency/binary/hardware identities. Record every parameter/activation dtype, convolution algorithm, TF32 setting, tiling and cache policy. Do not rerun or renormalize the latent.
2. Preserve autonomous subdivision prediction with the same strict `logit>0` rule and C2S bit-to-coordinate convention. **Do not force AMD masks into the reference** to manufacture agreement. Save all four parent-coordinate/logit pairs and final coordinates/features; reject duplicates, nonfinite values and incomplete provenance.
3. Retain original execution input order and record emitted row-order agreement separately. Canonicalize comparison copies lexicographically by integer `(batch,D,H,W)` coordinates; carry the same permutation into features/logits. Compare complete coordinate sets and Boolean masks exactly. Do not rotate, swap axes, rescale or fit coordinates. The Phase 8 pose rotation is irrelevant here.
4. Compare logits only on identical parent support/context. At the first support divergence, report missing/extra coordinates and sign disagreements; subsequent common-coordinate comparisons are diagnostic, not same-context parity. Require exact support/flag signs for **quad-connectivity parity**, even when numerical differences are small. Triangle-connectivity parity additionally requires identical weighted diagonal decisions, including strict-`>` ties. Record signed zeros, both logit margins and near-zero sign crossings; never ignore them by an epsilon mask.
5. Decode each stored FP16/BF16 value correctly to F32 for comparison. Report per-stage/channel maximum and RMS absolute error, quantiles, exact-equality fraction and symmetric relative error `2|a−b|/(|a|+|b|)` (both-zero→0). Record representation spacing separately: BF16 spacing near1 is2^-7, FP16 is2^-10. Those are **not network error tolerances**; final F32 storage does not imply an all-F32 computation. Numerical pass tolerances require pre-reviewed, dtype/backend-specific calibration before a run, not post-hoc tuning. They remain an explicit unqualified prerequisite here.
6. Official FP16 body/[default masked GEMM](https://github.com/microsoft/TRELLIS.2/blob/de38fdd4082638cb4e6d2285f2e215280bd3f485/trellis2/modules/sparse/conv/config.py) versus AMD BF16 body/explicit GEMM is a combined precision/algorithm/backend comparison. Same checkpoint bytes do not mean identical effective parameter values: the [AMD runtime policy](https://github.com/HermitKing79/modly/blob/7ef1f1703b499bc556d897229397505d2b746ee7/experimental/extensions/trellis2-amd/inference_decoder/runtime.py#L9-L34) rounds216 body tensors to BF16 while retaining76 F32 tensors. Do not alter the working AMD route to hide this confound.

**Interpretation:** matching support/flags/features would argue against an AMD-decoder-specific explanation **for these fixed latents**. It would not isolate pretrained-model behaviour from earlier AMD conditioning, occupancy or Shape-SLat effects. A mismatch identifies a divergence requiring localization; it does not automatically prove an AMD bug. One asset cannot establish general quality parity.

A tiny CPU fixture could check a specific C2S/index or rounding invariant. C2S coordinate linkage and extractor indexing are already verified; a synthetic FP16/BF16 example would only restate the known precision distinction. Such a fixture cannot exercise the real native GPU kernels, learned subdivisions or full-network error propagation, so it cannot lift this gate. No additional fixture campaign was started.

## Stop and smallest alternative direction

Do not build a speculative CUDA/CPU decoder replacement, buy hardware, create another launcher or run another neural attempt. Reopen parity only when an already available supported host or a provenance-complete matching reference artifact is supplied and its remaining qualification gates pass. Existing22GiB neural admission,12GiB live host floor and all other safeguards remain unchanged.

There is **no demonstrated neural-geometry fix to promote** from this investigation. The smallest distinct product direction worth separate review is an **explicitly labelled, dimension-driven CPU constructive/parametric mode for constrained hard-surface props**, using the already available procedural/Blender capability and refusing unsupported shapes. TRELLIS.2 stays the working neural engine. This would be a separate representation/input contract, not repair of this box, a general image-to-3D claim, or permission to substitute benchmark ground truth for neural output. If that restricted product mode is unwanted, retain NO-GO rather than extend this forensic/repair campaign. No alternative work began.

Offline authentication completed naturally in4.750s, peak committed memory100,655,104 bytes, under existing owned CPU limits. All104 selected source/evidence hashes and protected-record checks are retained privately at workspace `artifacts/amd-upstream-decoder-parity-feasibility-20261030-v1/`. Sixteen official source blobs received independent verification; the protocol/decision received independent review. No WSL, GPU workload, package/model download, installation, environment modification, geometry variant, VHDX or E: backup operation occurred. Only this text assessment is published.
