# Repeatable AMD Generation v1 — BLOCKED

**The Generate-button acceptance goal is not met.** A previously unused brick image completed genuine image conditioning, sparse-structure sampling/decoding and shape sampling. Its genuine neural geometry decode stopped safely after 33/40 convolutions. No new-image mesh, learned material field, viewport result or GLB exists. Frontend integration stopped at this inference gate.

## Preservation and application boundary

Baseline: GitHub-verified `6ea0abb20b3391ff3dd4daceeb2e1a9db903afa2`; development branch `codex/amd-repeatable-generation-v1`. The actual checkout is `proof`, separate from installed Modly. Tracked files were initially clean; four existing bytecode-cache directories were preserved. All 76 prior remote heads were recorded, including main at `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`. Twenty local-only files were backed up and independently read back. [Final preservation evidence](amd-repeatable-generation-v1-evidence/final-preservation-summary.json) records protected source/artifact hashes and environment metadata.

The application path is `WorkflowPanel.handleGenerate` → workflow store → HTTP `/generate/from-image` → registry/`ExtensionProcess` → `api/runner.py` → generator → workspace GLB → `Viewer3D/useGLTF` → download. Electron starts the API; generation uses HTTP. The viewport/export consumers exist. The experimental AMD extension requires `mock=true`; geometry preparation leaves models unloaded and generation disabled. Common frontend parameters and Refine path names differ from its mock schema. Neither those mismatches nor its 30-second mock timeout were changed around an unproved engine. Local Node dependencies/build output were absent; no installation, frontend build or Generate/viewport test was attempted.

The successful turret chain consists of separate Windows sampling, WSL neural decoding, CPU dual-grid extraction, learned-material decoding and Blender GLB proofs. Those assets were never substituted for this image. No installed application/environment, model, driver or native source was changed.

## Genuine new-image results

Input: the previously selection-only, vine-covered brick fragment, **1018×1018 RGBA**, 184,424 bytes, SHA-256 **`a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293`**. Alpha bounds `[198,46,820,973]`; established alpha-only crop/512-conditioning, without RMBG. It is materially different from the turret.

| Executed stage | Actual result | Synchronized computation | Peak Torch reserved bytes |
|---|---|---:|---:|
| DINOv3 | PASS, one conditioning forward |27.922s|1,371,537,408|
| Sparse structure | PASS,12steps/22forwards |8.266s|3,038,773,248|
| Sparse occupancy decoder | PASS, one forward;20,829 occupied64³ cells |22.468s|1,648,361,472|
| Shape-SLat | PASS,12steps/21forwards; F32 `[4430,32]` |9.391s|3,152,019,456|
| Neural shape decoder | PARTIAL, one parent-forward invocation;33/40 convolutions returned |Not completed; last admission at GPU-stage16.599s|4,389,339,136|

Every completed Windows stage had healthy observation, natural owned-process exit and measured zero allocated/reserved Torch bytes after release. Exact canonical 2× max-pool occupancy produced **4,430** unique `[batch,D,H,W]` rows, bounds `[4,5,0]`–`[27,26,31]`. Historical 3,743 was a fixture bound; this adapter was CPU-qualified up to 4,430 while preserving historical defaults and hard memory limits. This did not promise decoder completion.

Independently reopened durable artifacts:

| Artifact | SHA-256 |
|---|---|
| `conditioning.safetensors` |`471c04ea88df611a84cb311880e600e703c034254946b01807763e11c569f2b2`|
| `shape-latent-denormalized.npy` |`e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808`|
| `shape-latent-coordinates.npy` |`17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c`|

Large outputs remain local and uncommitted. Relative to the checkout, conditioning is in `docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/`; shape arrays/provenance are in its `shape/run/scratch/` directory. Original noise, raw latent, exact denormalization, payload hashes and coordinate identity were independently checked. [Prefix receipt](amd-repeatable-generation-v1-evidence/new-image-prefix-result.json), [shape receipt](amd-repeatable-generation-v1-evidence/new-image-shape-result.json), [neural receipt](amd-repeatable-generation-v1-evidence/new-image-neural-result.json).

## Exact execution

RX 9070 XT / gfx1201; Windows driver 32.0.31041.1004. Windows stages used existing Python 3.12.10 / Torch 2.13.0+rocm10.0.0. Neural decoding used existing named Ubuntu WSL2 Python 3.12.3 / Torch 2.9.1+rocm7.2.1.gitff65f5bc, ROCm 7.2.1 and qualified FlexGEMM. Native `.so` SHA remained `91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01`. All three TunableOp controls/actual flags were off.

Reused audited local DINOv3, `ss_flow_img_dit_1_3B_64_bf16`, `ss_dec_conv3d_16l8_fp16`, `slat_flow_img2shape_dit_1_3B_512_bf16` and `shape_dec_next_dc_f16c32_fp16`. The decoder's 292-tensor checkpoint was strictly loaded and freshly SHA-256 verified as `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`, with reviewed **216 BF16 / 76 F32** runtime precision. No install, new model, source/native rebuild or precision fallback occurred.

`WORKSPACE_DIR` was set to this checkout. Windows commands used the existing AMD interpreter with `-B`. `OCC` denotes the new occupancy evidence directory:

```text
amd_generation_occupancy.py --image OCC/source-image.webp --evidence OCC --harness-commit 189c670cba769ad3274b0a535b761d5873826f51
amd_generation_shape.py --occupancy-evidence OCC --evidence OCC/shape --harness-commit 93ce31df21346ce2a5327c2978d4c89f702ec780
amd_generation_decode.py --shape-evidence OCC/shape --evidence OCC/decode --harness-commit d73bc6906e2be1c57c2b5d51d48b921601963adf --run-id run-brick-6ea0abb-v1
```

A 5-second read-only Git check timed out before neural marker/WSL launch. A bounded 15-second read succeeded; only the Git-read timeout was corrected before publication and the single real decode. Exact HEAD/content checks and GPU deadlines were retained. [Failure](amd-repeatable-generation-v1-evidence/prelaunch-git-timeout.json), [review](amd-repeatable-generation-v1-evidence/git-timeout-review.json).

## Decoder stop and bounded CPU repair

The genuine parent path completed supports **4,430→21,391→97,313→438,684**, three learned subdivisions/six C2S operations and three native global-neighbor maps. Fourth subdivision/final convolution were not reached. Exact stop:

`AdmissionStop: PARTIAL: blocks.2.8.conv2: visible allocation plus reserve/headroom exceeds memory limits`

The 128→128 plan required 7,555,454,532 visible bytes, padded to **9,444,318,165**. At rejection: allocated **1,261,897,216**, reserved **3,546,284,032**, device free **11,385,540,608** bytes. The unchanged predicate requires `max(allocated,reserved)+ceil(1.25×visible)+1GiB≤12GiB`, and `free≥ceil(1.25×visible)+2GiB`. It rejected before the next convolution/map allocation. **No actual OOM, numerical failure or native incompatibility was observed.**

An opt-in CPU-tested repair permits one unused-cache release and remeasurement at an otherwise rejected admission. Live allocation must remain equal, reservation cannot increase, and the original predicate must pass. Historical default behavior is unchanged. No operation retry/math/budget change. This patch was **not GPU-executed**.

Independent source/arithmetic review proved eviction alone insufficient: convolution 39 (`blocks.3.4.conv1`,128→512) has padded plan **10,639,732,065** bytes, leaving **1,171,427,999** live bytes under the predicate. GPU parameters plus original residual and distinct normalized input require at least **1,173,204,700**, before coordinates/maps/other buffers. This is a conservative-policy blocker, not proof that 16 GB hardware cannot execute a tiled implementation. No second decoder run was spent on it. [CPU proof](amd-repeatable-generation-v1-evidence/new-image/occupancy/decode-cpu-admission-review.json), [independent review](amd-repeatable-generation-v1-evidence/cache-reclaim-independent-review.json).

Cleanup measured **zero owned allocated/reserved Torch bytes**, collected GPU tensors/CPU model, healthy observation, naturally empty Job and successful named-Ubuntu termination. Linux exit1 reflects admission stop, not a successful forward. Worker/supervisor elapsed 42.934 / 60.578 s; CPU load 11.917 s; CPU high-water 2,786,291,712 bytes. Minimum Windows available RAM 14,604,640,256 and D: free 74,985,840,640 bytes stayed above floors. Device free after Torch cleanup 14,935,961,600 bytes; device-global memory is not claimed zero.

## Changes, checks and limits

Reused strict loaders, sampler arithmetic, AMD SDPA, native kernels, CPU dual-grid converter, observer, Job containment and artifact readers. New stage adapters authenticate source/predecessor lineage, exclusive paths and bounded run-derived counts. WSL host supervises decoder plus allowlisted result transfer within one owned timeout; incomplete output cannot become PASS. CPU mesh-reader/converter and texture-sampling candidate are preserved but **were not executed for this image**.

Focused CPU checks: occupancy adapter 9; shape 18; neural reader/tiler 15; mesh 7; final host/Git-timeout 16; cache correction 10 plus 2 original admission tests independently rerun. [Stage reviews](amd-repeatable-generation-v1-evidence/stage-adapter-review.json), [host review](amd-repeatable-generation-v1-evidence/host-review.json). Texture CPU qualification and historical-test limitations are in [its receipt](amd-repeatable-generation-v1-evidence/texture-candidate-cpu-review.json), including an excluded hardlink permission case. These are contract tests, not an application end-to-end PASS. No broad suite/build, texture inference, GLB verification, sequential Generate test or UI cancellation test passed or is claimed.

## One recommended next action

Qualify source-faithful **row tiling of remaining large-support 128-channel explicit-GEMM convolutions**, reusing the final-convolution tiler: CPU equivalence/admission for 128→128 and 128→512, preserving global neighbors, coordinates, precision and limits. Only after separately reviewed qualification should a fresh neural attempt resume from the **saved new shape latent**, without repeating DINO/sampling. No cap increase or frontend integration around this blocker.

**Stopped:** new-image geometry completion, learned materials, GLB and Generate integration remain unproved. Completed intermediate data and every prior successful turret asset are preserved.
