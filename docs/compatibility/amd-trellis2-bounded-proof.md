# Modly AMD-only TRELLIS.2 bounded compatibility proof

**Result: PARTIAL — 12 October 2026.** Python 3.11-to-3.12 subprocess transport is proved. The existing GGUF utility imports and decodes four synthetic CPU quantization blocks under PyTorch 2.13. The installed Modly native binaries are demonstrably CUDA builds. A complete AMD native-import suite, model loading, GPU geometry, texturing, and a correctly shaped textured GLB are **not proved**.

An AMD-only Modly adapter is realistically achievable as an engineering direction, because the application already isolates generators and AMD-native component binaries exist locally. This is an inference from architecture and prior user observations, **not validation of an operational generator on the RX 9070 XT**. The previously observed CuMesh error 719 and native crash remain release blockers.

## Scope and Git safety

- Initial workspace: `C:\Users\wattz\Documents\ChatGPT\modly`, empty repository, unborn `master`, no remote and no existing working-tree changes. It was not a checkout of the fork.
- Investigation checkout: `C:\Users\wattz\Documents\ChatGPT\modly\proof`, shallow source-only clone of `https://github.com/HermitKing79/modly.git`. Initial `main`/HEAD was `f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea`, clean, with that fork as `origin`. GitHub independently returned the same main HEAD. Upstream main also matched this SHA.
- All report/test additions are on **`codex/amd-trellis2-bounded-proof`**, created before edits. No existing checkout was reset, cleaned, overwritten, or force-pushed. Main is not modified by this proof.
- Official extension source: `caa85c7a1aa6c6ba9cb3f07f13878ba8e3864912`, separate small source checkout at `..\gguf-source`. Installed `generator.py` and `setup.py` match that checkout after newline normalization; [source comparison](amd-trellis2-evidence/source-comparison.json) records their byte hashes.
- Both AMD repositories' main HEADs: `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`. The existing local AMD checkout has that HEAD and remote `dmonkman/ComfyUI-Trellis2-AMD`, with **pre-existing modified `nodes.py` and untracked `nodes.py.before-profiler-fix`**. Both were preserved. It must not be described as a clean checkout of the user's fork.
- The installed `trellis2_gguf` package is downloaded from an unpinned Aero-Ex main archive by setup, rather than tracked in the official extension repository. Its inspected file SHA-256 hashes are in [inventory.json](amd-trellis2-evidence/inventory.json). Aero-Ex main was `b3b9e951874c5fb5c936566c93b908299be08b2c` at inspection; equality with the installed package is **not established**.

Neither installed application nor its Python environment was changed. The selected package/PE/source inventory was identical before and after testing. No packages, PyTorch, ROCm, drivers, GPU components, or model weights were installed/downloaded. No real generator `load()`, inference, texture operation, GPU tensor allocation, rasterizer creation, hole filling, or recompilation was attempted. Source-only clones were the only network downloads. Temporary pip-free mock venv/source fixtures lived in the investigation workspace and were removed.

## Compatibility boundary

1. Electron launches extension setup using the application's Python, passing `python_exe`, GPU information and ROCm specifications. The [setup launcher](https://github.com/HermitKing79/modly/blob/f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea/electron/main/setup-launcher.ts) rewrites **torch requirements/indexes** to ROCm; it does not convert custom CUDA wheels. [GPU detection](https://github.com/HermitKing79/modly/blob/f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea/electron/main/gpu-detect.ts) defaults Windows to torch `2.11.0+rocm7.14.0` and torchvision `0.26.0+rocm7.14.0`.
2. Official [extension setup](https://github.com/lightningpixel/modly-trellis2-gguf-extension/blob/caa85c7a1aa6c6ba9cb3f07f13878ba8e3864912/setup.py) creates the extension venv from the supplied interpreter, installs NVIDIA-selected torch versions unless intercepted, then installs CuMesh/FlexGEMM/nvdiffrast/o-voxel from a CUDA wheel index. The selector permits older torch-minor wheel fallback, which is not evidence of C++ ABI compatibility. It also downloads Aero-Ex source and City96 GGUF files and applies Python patches.
3. [Registry discovery](https://github.com/HermitKing79/modly/blob/f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea/api/services/generator_registry.py) chooses subprocess mode when an extension venv/setup exists. [ExtensionProcess](https://github.com/HermitKing79/modly/blob/f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea/api/services/extension_process.py) runs `extensions/<id>/venv/Scripts/python.exe api/runner.py`. It does not require the child's Python minor version to match the API interpreter and has no general external-interpreter setting.
4. [Runner](https://github.com/HermitKing79/modly/blob/f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea/api/runner.py) imports `generator.py`, emits `ready`, then instantiates it. JSON lines carry image bytes, parameters, progress and output paths. Generate and refine are separate manifest node identities. A `ready` message says nothing about model/native execution.
5. The [GGUF generator](https://github.com/lightningpixel/modly-trellis2-gguf-extension/blob/caa85c7a1aa6c6ba9cb3f07f13878ba8e3864912/generator.py) calls `Trellis2ImageTo3DPipeline.from_pretrained(enable_gguf=True, gguf_quant=..., precision="bf16", keep_models_loaded=False)`. Shape generation and texture refinement are distinct paths; refinement consumes an existing mesh path. The load path can download weights, install missing packages, fetch GGUF code, patch modules, create DINOv3 files and execute a tiny GPU CuMesh remesh probe. These methods were not called.

**Python 3.12 feasibility:** a separate extension venv can run under the unchanged Python 3.11 API. The production launcher and runner were exercised successfully with a pip-free 3.12.10 mock venv. Stock setup still chooses the supplied application's interpreter; an AMD runtime provisioner/interpreter selection must be explicit in a later milestone. Adding a 3.12 site-packages directory to Python 3.11 cannot substitute for that boundary.

## Exact observed dependency/ABI differences

All native wheels below have Windows x64 machine type `0x8664`. `win_amd64` identifies the **CPU platform**, not AMD GPU support. Full wheel metadata and direct PE DLL dependencies are in [inventory.json](amd-trellis2-evidence/inventory.json).

| Component | Installed Modly extension (Python 3.11.9) | Existing AMD environment (Python 3.12.10) |
|---|---|---|
| PyTorch | `2.11.0+rocm7.14.0`, `cp311-cp311-win_amd64` | `2.13.0+rocm10.0.0`, `cp312-cp312-win_amd64` |
| torchvision | `0.26.0+rocm7.14.0` | `0.28.0+rocm10.0.0` |
| ROCm package | `7.14.0` | `10.0.0` |
| CuMesh | `0.0.1+cu126torch2.11`, `cp311` | `1.0+rocm10.0`, `cp312` |
| FlexGEMM | `1.0.0+cu126torch2.11`, `cp311` | `1.0.0+rocm10.0`, `cp312` |
| nvdiffrast | `0.4.0+cu126torch2.11`, `cp311` | `0.4.0+rocm10.0`, `cp312` |
| o-voxel | `0.0.1+cu126torch2.11`, `cp311` | `0.0.1+rocm.10.0`, `cp312` |
| Triton Windows | `3.6.0.post26`, `cp311` | `3.8.0.post29`, `cp312` |
| transformers | `5.2.0` (official setup pins this) | `4.56.0` |
| GGUF / SDNQ / accelerate | `0.19.0` / `0.2.8` / `1.15.0` | No corresponding distribution found in selected metadata inventory |

- Modly CuMesh `_C`, FlexGEMM native kernel, o-voxel `_C` and nvdiffrast plugin directly import **`cudart64_12.dll` and `python311.dll`**. CuMesh BVH and nvdiffrast also import **`c10_cuda.dll`**. These are NVIDIA CUDA runtime dependencies alongside ROCm torch, not native AMD GPU components. This establishes a backend mismatch consistent with the user's CuMesh DLL-import failure; this task did not isolate the original Windows loader's first missing symbol/DLL.
- AMD equivalents directly import **`amdhip64_7.dll` and `python312.dll`**; BVH/nvdiffrast import **`c10_hip.dll`**. `amdhip64_7.dll` is the observed DLL name within the ROCm 10 package stack, not evidence that the package was silently downgraded to ROCm 7. Python and torch C++ ABI versions also differ; these binaries cannot be transplanted into the 3.11/torch 2.11 extension.
- GPU-facing components need the matching ROCm torch build, DLL discovery and a working Triton AMD backend. Package labels and matching Python tags alone cannot prove exported C++ symbols, architecture code objects, or kernel execution.
- Calls such as `.cuda()` and `torch.cuda` are also used by HIP PyTorch. Their spelling alone does not require NVIDIA binaries. [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.14/notes/hip.html) explicitly documents this API reuse. Actual binary dependencies and `torch.version.hip` distinguish the backends.

## GGUF findings

- The AMD implementation uses `trellis2`, JSON plus **safetensors**, with a different `from_pretrained` signature (`keep_models_loaded`, `use_fp8`, `use_reconviagen`, `isPixal3D`). The installed Aero-Ex path uses **`trellis2_gguf`**, GGUF tensor wrappers, key remapping, quantization selection and lazy model management. Simply replacing the import or passing Modly GGUF weights to the AMD safetensors loader is unsupported. See the pinned [AMD model loader](https://github.com/dmonkman/ComfyUI-Trellis2-AMD/blob/3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5/trellis2/models/__init__.py).
- Inspected GGUF dequantization implements Q8_0/Q4_K/Q5_K/Q6_K using torch reshape/view, integer bit operations, scales and ordinary linear algebra. City96's so-called native support here is Python `ops/dequant/loader`, not a mandatory CUDA GGUF shared library. Unsupported quantization types may fall back to NumPy on CPU. That finding is limited to the inspected utility paths, not an audit of every model operation.
- The official standalone adapter stubs `folder_paths` and part of `comfy`. City96 ops additionally require `comfy.ops`, `comfy.lora` and `comfy.model_management`; the inspected package can therefore take its internal `GGMLOpsFallback` path. The installed fallback Linear has Modly's patch to inherit `GGMLLayer` and `torch.nn.Linear`. The CPU smoke test exercised this **fallback**, not City96 native ComfyUI integration.
- Dense attention defaults to SDPA in the official generator. Sparse configuration does not accept `sdpa` as an environment value, but the inspected sparse full-attention implementation catches missing FlashAttention/xformers and falls back to per-sequence torch SDPA. Thus a remaining `flash_attn` configuration label does not prove NVIDIA execution or failure. The AMD source additionally supports Aule attention. Performance/VRAM on large sparse sequences is unmeasured.
- Transformers 4.56.0 versus pinned 5.2.0 is a dependency difference, **not a proved incompatibility**: both inspected image extractors import `DINOv3ViTModel`, and the GGUF source contains handling for pre-5 and 5+ layouts. No transformers upgrade/import or DINO weights load was performed.
- Native o-voxel locally already exposes tiled FDG conversion. Interface equivalence with the GGUF patch, remeshing and texture export is unproved. GGUF and SDNQ are separate quantization strategies; optional SDNQ/Triton integer-matmul support was not exercised.
- The official generator documents corrupt 512 texture K-quant weights and prefers Q8_0 for that model. This is an **upstream code claim**, not a reproduced observation here. Future acceptance must check image-faithful textures, not just GLB export completion.

## Tests and actual results

Tests were bounded, read-only against existing environments, with `-B` to suppress bytecode writes. Native/CPU tests hid GPUs and forbade network/process actions after a read-only Windows platform lookup. No full pipeline import or model dependency reconstruction was attempted.

| Proof | Actual result | Limit |
|---|---|---|
| Wheel metadata / PE inspection | Exit **0**; exact versions, Python tags and direct DLL imports recorded | Static dependency evidence; no original failure reproduction |
| Actual Modly launcher + runner, 3.11.9 host to 3.12.10 child | Exit **0**; mock generate/refine routes, UTF-8 progress/parameters, image bytes, node metadata, output paths and unload passed | Mock returned `.json` fixtures, not models or GLBs |
| Official GGUF generator class under pip-free Python 3.12 | `ready` passed in the same test | No `load()`, dependency resolution, generation or texturing |
| Native import attempt 1 | Exit **1** before torch initialized: test guard blocked CPython's Windows-version subprocess | Harness limitation; raw trace retained |
| Native import controlled retry after caching platform lookup | Exit **1** at FlexGEMM/Triton's `rocm-sdk path --root` discovery subprocess, blocked by the guard | CuMesh `_C`, `_cubvh`, `_xatlas` imports completed before this point, inferred from the ordered script reaching FlexGEMM; no per-module success log. o-voxel/nvdiffrast checks were not reached. Complete suite is **inconclusive**, not proof that AMD binaries fail |
| Existing GGUF utility + staged pure-Python `gguf 0.19.0`, AMD torch 2.13 | Exit **0**; Q8_0 (32 values), Q4_K/Q5_K/Q6_K (256 each) matched NumPy reference exactly on CPU; `gpu_initialized=false` | Tiny synthetic blocks; no weights, GGUF matmul, GPU dequantization or model validation |

Evidence: [mock output](amd-trellis2-evidence/mock.txt), [initial guard failure](amd-trellis2-evidence/imports-initial-guard-failure.txt), [native retry trace](amd-trellis2-evidence/imports-stderr.txt), [GGUF CPU output](amd-trellis2-evidence/gguf-cpu.json). Empty output/stderr files in the evidence directory are intentional captures. Source hashes identify what was inspected.

Reproduction commands from the `proof` checkout, **for these bounded tests only**:

```powershell
$audit = 'scripts/compatibility/amd_trellis2_audit.py'
$modlyVenv = 'C:/Users/wattz/Documents/Modly/extensions/trellis2/venv'
$amdVenv = 'D:/AI/Trellis2AMD/venv'
$py312 = 'C:/Users/wattz/AppData/Local/Programs/Python/Python312/python.exe'
& $py312 -B -I $audit inventory --modly-venv $modlyVenv --amd-venv $amdVenv --amd-source 'D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD' --extension-source '../gguf-source'
& "$modlyVenv/Scripts/python.exe" -B -I scripts/compatibility/amd_trellis2_mock.py --python312 $py312 --gguf-extension '../gguf-source'
& "$amdVenv/Scripts/python.exe" -B -I $audit imports
& "$amdVenv/Scripts/python.exe" -B -I $audit gguf-cpu --modly-venv $modlyVenv
```

The native command's final guard failure is expected in the saved harness. It intentionally does not retry or relax its guard to compile anything. The initial platform-guard trace predates the one small harness correction; it remains evidence rather than being discarded.

## Alternative comparison and smallest next milestone

| Route | Reusable work | Additional boundary and risk | Assessment |
|---|---|---|---|
| Adapt Modly's existing GGUF extension | Current generate/refine contract, GGUF loader, quantization, model path/DINO handling, staged low-VRAM loading | Replace CUDA-oriented setup with an AMD-only runtime contract; align Python/torch native ABI; prohibit auto-install/update paths; verify GGUF model/attention and ROCm native API integration; audit NVIDIA capability/ptxas patches | Plausible and retains quantization, but **not proved smaller or safer** than the adapter route |
| Dedicated AMD extension/adapter using pinned AMD TRELLIS source | Existing ROCm component binaries, safetensors loader and prior user-observed model-stage execution; Modly IPC remains reusable | Headless `folder_paths`/progress shim, explicit local model layout, staged VRAM policy, generate/refine mapping and textured export; no blind import of full ComfyUI `nodes.py` | **Recommended first implementation direction**; reduces the initial GGUF uncertainty, but does not solve CuMesh crashes or prove 16 GB sufficiency |

The [AMD README](https://github.com/dmonkman/ComfyUI-Trellis2-AMD/blob/3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5/README.md) reports end-to-end validation on RX 6800 XT (`gfx1030`), with `gfx1201` included in fat binaries. It explicitly distinguishes architecture coverage from pipeline validation. It is not RX 9070 XT end-to-end evidence. The local 6 October startup log identifies RX 9070 XT/torch 2.13 but also contains a then-missing profiler module error; it is historical evidence, not a current generation test.

**Recommended next milestone, requiring approval: an offline AMD worker scaffold with a fail-closed runtime preflight.** Use a separate fork development checkout and explicit read-only interpreter/source paths to the existing AMD environment; do not provision or repair either installed environment. Implement only a headless adapter scaffold, not a production installer or image generation feature.

Acceptance criteria:

1. Real Modly IPC reaches `ready` under 3.12 for generate/refine with exact interpreter, source commit and package provenance reported. AMD detection uses `torch.version.hip`; no NVIDIA SM/ptxas path or automatic package/model download runs. Failure never falls back to CPU or CUDA binaries silently.
2. Correct SDK path discovery is explicit; a narrowly allowed read-only discovery command may be assessed separately. All seven native imports in the saved test complete with paths from the intended ROCm environment, no compilation/install and no GPU initialization. A successful import still only satisfies preflight.
3. A headless AMD pipeline class imports without importing full ComfyUI nodes, loading weights or invoking native operations. Model paths resolve to already-present safetensors/configs offline; missing artifacts fail clearly. GGUF weights are explicitly unsupported in this first adapter milestone.
4. IPC/mock routing, actionable missing-dependency errors and subprocess termination pass; both installed environment inventories and the pre-existing AMD source changes remain unchanged. Any need to rebuild a native component or reconstruct dependencies stops the milestone.

**Separate later runtime gate:** with further approval, isolate small CuMesh hole-filling/remesh and nvdiffrast texture-bake fixtures in disposable processes on `gfx1201`, with strict timeouts and no blind retries, before any complete image-to-GLB run. A resolved or avoidable CuMesh 719 failure must have reproducible evidence. Disabling hole filling alone is not proof of correct geometry; `o_voxel.postprocess.to_glb()` also calls it. A Python exception handler cannot reliably recover a fatal native crash or a poisoned GPU context.

The eventual end-to-end acceptance remains: one fixed zombie image, one nonempty GLB with coherent limbs/body shape, UVs and embedded image-faithful PBR textures, verified by reopening it; measured peak VRAM within the RX 9070 XT's available budget; no native crash, missing geometry, corrupt textures, downloads or silent backend substitution. That acceptance was **not attempted** here.

**Stop condition reached:** further runtime assurance would require extending the native preflight and resolving the complete model dependency boundary. No implementation is authorized or performed beyond this report and bounded proof scripts.
