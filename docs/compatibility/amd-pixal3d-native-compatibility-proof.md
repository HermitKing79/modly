# Native Pixal3D INT8: isolated single-image proof

25 October 2026. **GO for one future isolated compatibility proof, not for generation execution today.** TRELLIS.2 remains Modly's working primary engine. Native Pixal3D has a credible maintained AMD path; no CUDA-porting project is justified. A successful 16 GB / 32 GB run under Modly's limits is **not yet measured or guaranteed**.

The [non-executable proof contract](amd-pixal3d-native-single-image-proof-contract.json) fixes the source revisions, model allowlist, one-image stage plan, isolated paths, existing limits and stopping conditions. `execution_ready` and `execution_authorized` are both false. This preparation did not create an environment, install/download packages or weights, import Torch/models, launch WSL, initialize the GPU, reserve an attempt or run inference. It inspected bounded public source/metadata and existing local evidence only.

## What was checked

- ComfyUI **0.42.0**, commit `fe298e4e20f1ff26d30e8639267bbd0b57811390`; exact core requirement **comfy-kitchen 0.2.40**, commit `f8b7e97a1418d2e529bd5405fbf86312be198ae6`.
- Official single-image workflow at `67984637eefcd4825dcf5c134e39034785be1682`.
- Comfy-Org Pixal3D file metadata at `f37641be376725d324c56626007cc477f1249a69`, plus pinned MoGe metadata; no model bodies were fetched.
- Installed Windows ComfyUI is **0.33.4** and lacks the native Trellis/Pixal node file; its kernel package is **0.2.31**. It must not be updated in place. Saved successful WSL evidence records Python 3.12.3, Torch 2.9.1/ROCm 7.2.1, Triton 3.5.1; this is historical identification, not a fresh WSL query or native Pixal qualification. [Existing runtime](amd-wsl-rocdxg-gpu-resume.md).

## AMD dependencies

Native core uses PyTorch sparse gathers/GEMMs and Comfy's attention dispatch, with a pure-PyTorch NAF neighborhood-attention implementation. It does not require the older wrappers' NATTEN, spconv, FlexGEMM binary, CuMesh or nvdiffrast for this geometry path. Sparse-convolution workspace and NAF attention are chunked. This is a different implementation from CUDA-only custom-node wrappers. [Native sparse convolution](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/ldm/trellis2/flexgemm.py), [NAF](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/image_encoders/naf.py).

The required kernel release explicitly supports `gfx1201` through HIP/WMMA, including INT8 linear/ConvRot. Current core disables the CUDA backend on ROCm and keeps global Triton opt-in. The older same-GPU ConvRot report was closed after successful retesting with the official HIP backend; it is not evidence that today's native path is unusable. That report concerned Krea2, **not Pixal3D**. [Kernel support](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.40/README.md#hip-backend-amd-rdna2--rdna3--rdna35--rdna4), [resolved report](https://github.com/Comfy-Org/comfy-kitchen/issues/78#issuecomment-5164841371), [current dispatch](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/quant_ops.py).

Python 3.12 ABI3 Linux x86-64 and Windows wheels are published. Their availability does not prove that the installed wheel exposes a usable HIP extension on our runtime. The first proof must check that, single-device visibility and actual finite/reference-correct operations; no local kernel compilation, global Triton workaround or fallback porting. [Release metadata](https://pypi.org/pypi/comfy-kitchen/0.2.40/json).

## Supported resource estimates

| Item | Evidence-supported amount |
| --- | ---: |
| Single-image INT8 checkpoint | 5,584,555,824 bytes, 5.20 GiB on disk |
| DINOv3 + bundled NAF | 1,215,214,176 bytes |
| Shape VAE | 1,095,844,024 bytes |
| MoGe-2 | 661,859,924 bytes |
| Four-model total | **8,557,473,948 bytes / 7.97 GiB**, before environment/caches/output |
| Retained native NAF feature maps | **3 GiB at default FP16**, 6 GiB if FP32 is selected |
| NAF chunk target | 0.5 GiB, not a total-memory cap |
| Sparse-convolution gather target | 0.25–0.5 GiB, plus other tensors |
| Native BF16 shape-decode planner at 512 | About **2.58 GiB**, an internal heuristic, not measured peak VRAM |

Model sizes come from official file metadata. File size is not runtime residency. [Pixal model files](https://huggingface.co/Comfy-Org/Pixal3D/tree/main), [MoGe files](https://huggingface.co/Comfy-Org/MoGe/tree/main/geometry_estimation). The retained-map calculation is `1024 channels × (512² + 512² + 1024²) × element size`; native conditioning still computes these maps for a 512 geometry target. Current text-encoder default is FP16 despite the checkpoint filename. Offload, allocator workspace, caches and process overhead prevent an honest total-RAM/VRAM or runtime prediction without the isolated test. [Conditioning/planner](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy_extras/nodes_trellis2.py), [dtype/offload policy](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/model_management.py).

## Smallest useful single-image test

Use the unchanged Zombie D RGBA, SHA-256 `9ac0364f409077cd3fdeb853a1baed3b851c56f4fdc539c9fde73e0e3ab79db9`; compare against raw D PLY `2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205`. Do fresh native preprocessing, MoGe camera estimation and DINO/NAF conditioning. Explicitly derive foreground alpha; do not depend on mask-polarity heuristics. Use native pad 1.1, save the effective 512 raster, and record that framing differs from TRELLIS's preparation.

Keep the template's initial passes: structure seed 56, 12 Euler/normal steps; first shape seed 42, 20 Euler/normal steps; CFG 7.5 with their native wrapper settings, recorded in the contract. Decode the **first shape pass** directly at coord resolution 32 / geometry resolution 512. Omit the 1536 upsample and second shape pass, texture stages and background-removal model. Do not run the full supplied template unchanged. [Official template](https://github.com/Comfy-Org/workflow_templates/blob/67984637eefcd4825dcf5c134e39034785be1682/templates/3d_pixal3d_trellis2_image_to_model.json).

Capture raw vertex/index buffers, GLB/NPY/PLY and hashes immediately after native shape decode. The native GLB writer directly packs mesh buffers; validate exact readback before previews. No remeshing, hole filling, welding, decimation, smoothing, UVs, baking or textures enter this proof. Some newer postprocessing kernels have CUDA-only coverage; they are unnecessary here. [Raw decoder/converter](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/ldm/trellis2/vae.py), [raw serialization](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy_extras/nodes_save_3d.py).

Preserve the 22 GiB three-reading admission, four-second spacing, final 22 GiB pre-reservation guard, preferred-only 24 GiB, 12 GiB live host floor, 13 GiB initial free VRAM, 12 GiB allocator ceiling, 3 GiB live shape reserve, disk/support caps, stage deadlines and owned cleanup. A Torch cap alone does not cover native allocations; retain external host/GPU observation. New native runtime supervision must be implemented and verified before a generation command is issued. Any guard, NaN, timeout or backend failure stops the proof without retry or increased limits.

After success, use the existing CPU render method for eight matched views against D, with explicit Y-up/Z-up convention registration on presentation copies. Assess fingers, toes, wounds, dark artefacts and overall fidelity on the **raw** geometry. This compares engines, quantization, preprocessing and target resolution together; it cannot attribute a quality difference to the pretrained model alone. Shared TRELLIS shape VAE/dual-grid design also means Pixal need not eliminate our topology defects. No multi-view test follows automatically.

## Exact next safe action

Separately authorise provisioning **only** `/home/modly/pixal3d-native-proof-v1/.venv` and its new core/model/cache directories, with Windows metadata under `D:\AI\Modly-Pixal3D-Proof-v1`. Start from the existing qualified WSL Torch build in that separate environment, lock dependencies, and run **one bounded model-free HIP/ConvRot, attention/NAF and tiny sparse-convolution/converter proof before downloading model weights**. Missing compatible binaries or a failed probe means stop; no environment upgrade in place and no days of porting.

None of those paths was created by this preparation. The Windows destinations are confirmed absent; guest-path exclusivity must be checked during isolated setup. A passing probe permits preparation/review of the one-shot guarded generation launcher and separately approved four-file provisioning; it does not itself authorise inference. No ready-to-run generation handoff is claimed at this source-only stage.

Offline verification checked 57 public source/metadata records, exact sampler settings, model sizes, memory arithmetic, guard values and fresh Windows destinations without importing the engine. Preservation rechecked 3,197 historical private files, 100 protected identities, 88 retained records and 169 previous local refs. Independent read-only review approved the three new text files with no findings; runtime compatibility, enforced native supervision and actual fit remain unproven. See [preparation verification](amd-pixal3d-native-proof-preparation-verification.json).
