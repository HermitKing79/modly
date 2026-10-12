# AMD offline model audit and headless import gate

**PARTIAL — 12 October 2026.** The standard TRELLIS.2 geometry/texturing
checkpoint metadata and DINOv3 assets are present and passed the bounded audit.
The essential **`briaai/RMBG-2.0` background-removal assets are missing from the
searched local model/cache roots**. Gate 2 was **not attempted**, following the
instruction to stop when essential weights are missing. This is an asset
blocker, not an observed pipeline-import failure.

Work is confined to `codex/amd-offline-model-audit`, based on scaffold commit
`7c2c9634db52c7e78bfa90be16ed6bf969dc2dbe`. The [scaffold report](amd-worker-scaffold.md)
and previous proof branches/reports remain preserved. No worker or production
application code changed. The AMD source remains at
`3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, with modified `nodes.py` and
`nodes.py.before-profiler-fix` preserved.

## Actual local requirements and results

The audited profile is **`microsoft/TRELLIS.2-4B`, standard BF16/FP16, including
512 and 1024 geometry and texture routes**. Requirements were traced through
the actual local `pipeline.json`, `texturing_pipeline.json`, AMD pipeline/base
factories, model-class map, DINO feature extractor and BiRefNet constructor.
The [full audit](amd-offline-model-audit-evidence/final-assets/audit.json) records
configurations, resolved paths, source hashes, installed dependency versions,
optional profiles and exact blockers. Each checkpoint has a separate complete
tensor-header capture.

All prefixes below require paired `.json` and `.safetensors` files. Except the
sparse decoder, they reside under
`D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS.2-4B\ckpts`.

| Required checkpoint prefix | Role | Metadata result |
|---|---|---|
| `ss_flow_img_dit_1_3B_64_bf16` | Sparse structure flow | VERIFIED, 640 BF16 tensors |
| `ss_dec_conv3d_16l8_fp16` | Sparse decoder; source override to `models\microsoft\TRELLIS-image-large\ckpts` | VERIFIED, 74 F16/F32 tensors |
| `slat_flow_img2shape_dit_1_3B_512_bf16` | 512 geometry flow | VERIFIED, 640 BF16 tensors |
| `slat_flow_img2shape_dit_1_3B_1024_bf16` | 1024 geometry flow | VERIFIED, 640 BF16 tensors |
| `shape_dec_next_dc_f16c32_fp16` | Geometry decoder | VERIFIED, 292 F16 tensors |
| `shape_enc_next_dc_f16c32_fp16` | Mesh encoding / separate texturing input | VERIFIED, 284 F16 tensors |
| `slat_flow_imgshape2tex_dit_1_3B_512_bf16` | 512 texture flow | VERIFIED, 640 BF16 tensors |
| `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` | 1024 texture flow | VERIFIED, 640 BF16 tensors |
| `tex_dec_next_dc_f16c32_fp16` | Texture decoder | VERIFIED, 284 F16 tensors |

DINOv3 is at `models\facebook\dinov3-vitl16-pretrain-lvd1689m`. Its
`config.json`, `preprocessor_config.json` and `model.safetensors` are present.
The header has **415 F32 tensors, 303,129,600 parameters** and a patch-embedding
shape of `[1024, 3, 16, 16]`. Its 1024 conditioning channels match the flow
configurations. Both pipeline configs still declare the remote DINO identifier;
any later offline factory must explicitly bind the verified local directory.

The validator reads only JSON and declared safetensors headers, bounded to
1 MiB/16 MiB. It checks tensor dtypes/shapes, byte counts, offsets, complete
nonoverlapping payload coverage, source class declarations, both projection
shapes for five flow checkpoints, and latent-normalization lengths. The actual files contain nonempty
tensor payloads and differ from the scaffold's empty metadata-only fixtures.
**Tensor payloads were not read or loaded.** Header/stat verification does not
authenticate tensor values or establish complete state-dict/architecture
compatibility; no model constructor ran. GGUF/pickle fallback is unavailable.

## Essential auxiliary blocker and offline boundary

The actual `Trellis2TexturingPipeline.from_pretrained()` constructs the configured
image extractor and `BiRefNet` unconditionally. `BiRefNet.__init__()` calls
`AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)`.
The texturing config selects **`briaai/RMBG-2.0`**. Preprocessed/alpha images do
not remove this constructor requirement in the existing loader.

No RMBG/BiRefNet assets were found in the enumerated ComfyUI model tree, installed
Modly model tree, default Hugging Face hub cache or torch hub cache. No alternate
HF cache path was configured. The intended local
`models\briaai\RMBG-2.0`, `models\background_removal\RMBG-2.0`, and cache
`C:\Users\wattz\.cache\huggingface\hub\models--briaai--RMBG-2.0` are absent.
The background-removal directory contains only its placeholder file.

Missing requirements are **a local RMBG `config.json`, a complete supported
checkpoint set** (`model.safetensors`, or the corresponding shard index and
all shards; the upstream loader also recognizes PyTorch filename alternatives),
**and every custom-code dependency named by `config.auto_map`**. Neither the
configuration nor any weight/index candidate exists in the inspected locations.
Exact custom Python filenames cannot be derived without that configuration;
none were guessed or fetched remotely. [Search evidence](amd-offline-model-audit-evidence/search-summary.json)
lists the bounded roots and candidates.

No Hugging Face factory or download function was called. Configs and source
were inspected as data; local references reject remote/escaping paths using the
scaffold's local-volume/reparse-point rules. The final audit ran under offline
environment flags in a 30-second supervised process, with no torch/native
imports. The worker remains mock-only and cannot consume this incomplete plan
for real loading. Remote IDs were recorded, not silently rewritten or accepted
as offline paths.

ReconViaGen is unselected and additionally lacks paired `ss_vggt_cond`,
`slat_vggt_cond`, and `ss_flow_img_dit_L_16l8_fp16` checkpoints. FP8 and
Pixal3D/MoGe are separate, unselected profiles; their assets were not substituted.
The existing `tex_enc_next_dc_f16c32_fp16` pair also passed metadata checks, but
is not required by the selected two pipeline configurations.

## Headless imports, tests and preservation

**Gate 2: NOT ATTEMPTED.** No `folder_paths` or `ProgressBar` shim was added.
The source still needs `folder_paths.models_dir` and `comfy.utils.ProgressBar`
when imported outside ComfyUI; their installed source paths are recorded.
The selected runtime's dependency metadata is present, including Python 3.12,
PyTorch `2.13.0+rocm10.0.0`, torchvision `0.28.0+rocm10.0.0`, ROCm `10.0.0`,
transformers `4.56.0`, safetensors `0.8.0` and the existing AMD native packages.
Presence/version checks do not prove those real pipeline imports succeed.
GPU initialization was not queried; no GPU package was imported by this audit.

Seven [narrow tests](amd-offline-model-audit-evidence/tests.stderr.txt) passed,
covering a stream that rejects tensor-payload reads, malformed/truncated and
overlapping headers, duplicate keys/header bombs, metadata-only fixtures, flow
shape mismatches, nonlocal paths and isolated CLI import. The first CLI check
exposed isolated Python's missing script-directory path; the explicit project
helper path fixed it. The final asset audit intentionally exits **1 / PARTIAL**
with no timeout. [Captured command/result](amd-offline-model-audit-evidence/audit.json)
and stdout/stderr are preserved. Application suites were not rerun because
production code is unchanged; their prior missing-dependency failures remain
documented in the scaffold report.

The [before](amd-offline-model-audit-evidence/inventory-before.json) and
[after](amd-offline-model-audit-evidence/inventory-after.json) inventories are
identical. [Preservation summary](amd-offline-model-audit-evidence/preservation.json)
records their hashes and the unchanged source status. Evidence includes installed distribution/native/source hashes and
complete venv/model/cache file metadata, with directory allocation size
normalized as in the earlier proof. No packages, drivers, models, source edits
or installed application files were modified. There were no installations,
downloads, native rebuilds, model loads, pipeline imports or GPU runs.

## Smallest subsequent proof

First resolve and audit the existing/local RMBG asset directory and its custom
code, then complete the approved headless import gate. After those prerequisites,
the smallest separately approved model-load proof is **only the sparse decoder**
`ss_dec_conv3d_16l8_fp16` (147,591,972 file bytes): verify its loaded parameter
keys/dtypes, measure CPU and AMD GPU memory before/after placement, and verify
release on process exit. Use a healthy Windows observer and strict process
timeouts; execute no forward pass, geometry or texture stage.

The previously observed CuMesh error 719/hole-filling crash remains unresolved.
Neither this metadata audit nor the earlier tiny fixture validates full geometry,
textures or a correctly shaped, textured GLB. **Stopped at the Gate 1 asset
blocker; no automatic loading or inference follows.**
