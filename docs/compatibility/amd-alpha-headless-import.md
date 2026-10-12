# AMD alpha-only boundary and real headless imports

**PASS for this bounded milestone — 12 October 2026.** Both real TRELLIS.2
pipeline classes imported from the preserved AMD source in isolated Python 3.12
processes. Explicit alpha-only preprocessing passed without an RMBG model and
produced pixel-identical conditioning images to the upstream RGBA path on the
tested fixtures. **No model weights loaded, conditioning model executed,
inference ran, textures baked or assets generated.**

Branch `codex/amd-alpha-headless-import` starts at offline audit commit
`d0f3426f85509e9705cec7852beef13b31fb2e70`. The [earlier audit](amd-offline-model-audit.md)
remains correct for the standard factory: local RMBG-2.0 assets are still missing.
Main and all previous proof branches are preserved. Changes affect only the
experimental AMD adapter, bounded proof tools, tests and documentation.

## Why alpha can remove the segmentation requirement

Both real `preprocess_image()` methods check for RGBA with any alpha below 255.
They use that image directly, cap its longest dimension at 1024, find foreground
at **alpha > 204**, compute the existing square crop, multiply RGB by alpha and
return RGB. RMBG runs only in the other branch. The ordinary constructors accept
`rembg_model=None`; with `models={}` they initialize Python attributes without
constructing AI models.

The geometry factory already defers its image extractor and background model.
The texturing `from_pretrained()` factory still eagerly constructs both DINO and
BiRefNet, even for an eventual transparent input. Therefore **transparent input
does not bypass that factory**. The validated route uses an explicit adapter
boundary with no background-model object; no global segmentation patch, fake rembg callable,
replacement segmentation model or source modification is used.

`alpha_input.py` rejects non-RGBA, opaque, all-transparent, faint-only and
degenerate masks. It requires fully transparent background and a foreground
spanning at least three pixels in both dimensions, rechecking after the upstream
resize. Animated images, oversized images and invalid bytes fail clearly.
Limits are 32 MiB encoded, 4096 pixels per dimension and 16 Mi pixels. Validation
does not establish semantic cutout quality or whether the zombie is fully visible.

The mock adapter accepts explicit `image_mode="alpha-only"` and records this
validation in its JSON receipt. Default `image_mode="rembg"` retains existing
mock behavior. Neither mock route creates models. The unchanged upstream RMBG
path remains available for a future standard loader with verified local assets.
Alpha-only model loading is **not implemented**: a subsequent loader must omit
the eager BiRefNet factory explicitly, bind local DINO/checkpoint paths, reject
network fallback and populate real model slots under separate approval.

## Actual headless import and preprocessing evidence

The minimal `headless.py` bridge provides only `folder_paths.models_dir` and
`comfy.utils.ProgressBar` with real progress state/callbacks. It rejects existing
ComfyUI modules rather than replacing them. Other interfaces and previews fail
explicitly. No installed ComfyUI entrypoint or modified `nodes.py` is imported.

[Final complete results](amd-alpha-headless-import-evidence/imports-stateless/results.json)
record source/module hashes, dependency versions, native provenance, schemas,
side effects, full command lines and process outcomes:

| Route | Real class / source module | Exit / elapsed | Result |
|---|---|---|---|
| Generate | `Trellis2ImageTo3DPipeline`, `trellis2.pipelines.trellis2_image_to_3d` | 0 / 12.266 s | PASS |
| Texture / Refine | `Trellis2TexturingPipeline`, `trellis2.pipelines.trellis2_texturing` | 0 / 12.546 s | PASS |

Both resolve beneath
`D:\AI\Trellis2AMD\ComfyUI\custom_nodes\ComfyUI-Trellis2-AMD`, commit
`3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`. The interpreter is
`D:\AI\Trellis2AMD\venv\Scripts\python.exe`, Python **3.12.10**, PyTorch
**2.13.0+rocm10.0.0**, HIP **7.15.26333**, CUDA **null**. Dependency metadata
includes torchvision **0.28.0+rocm10.0.0**, transformers **4.56.0**, ROCm
**10.0.0** and the pinned AMD native components. PE/HIP provenance and the
inventoried CPU xatlas utility hash passed; no CUDA binaries were substituted.

Each real constructor shell has `models={}` and `rembg_model=None`. On **16×12**
and **1200×1080** RGBA fixtures (including a soft-alpha pixel), both methods
returned identical RGB bytes to their unchanged upstream preprocessing:
**6×6** and **510×510**, respectively. This proves the conditioning-image
boundary for those inputs; DINO features and model conditioning were not tested.

## Guards, side effects and informed corrections

Workers have **30-second** deadlines; the only subprocess they may launch is
the previously verified read-only `rocm-sdk path --root`, capped at **5 seconds**
and rechecked by source/executable hashes and missing-devel state. Network and
writes outside disposable workspace caches are prohibited. Model-file reads,
trained-module construction, parameter/buffer registration, all tensor
operations, module invocation, Triton compilation, autotuning and launches are
guarded. No factory or model loader was called.

The [verified Windows observer](amd-alpha-headless-import-evidence/imports-stateless/windows-before.json)
passed before launch and after each worker; controller status was OK and no
matching reset/WHEA/power events appeared. Import-time metadata initialization
was unavoidable in the inspected FlexGEMM/native import chain. Each worker
changed GPU context state **false → true**, identifying **RX 9070 XT / gfx1201**,
17,095,983,104 reported bytes. PyTorch tracked **0 allocated / 0 reserved tensor
bytes**. This does not measure native driver/context allocations or total VRAM.

Three prior guard-boundary stops are preserved, all with context **false → false**
and a healthy observer; texturing was not attempted in those runs:

1. [Initial attempt](amd-alpha-headless-import-evidence/imports/results.json):
   the all-dispatch guard rejected TorchVision's `_cuda_version` operation.
2. [Query allowlist attempt](amd-alpha-headless-import-evidence/imports-verified-query/results.json):
   the assumed schema omitted the registered return name `_0`. A separate
   [15-second schema probe](amd-alpha-headless-import-evidence/torchvision-schema-probe.json)
   measured the schema without executing the operation or initializing GPU.
3. [Activation guard attempt](amd-alpha-headless-import-evidence/imports-final/results.json):
   the exact query passed twice with `-1`; the module-construction guard then
   rejected import-time stateless activation objects in Transformers.

The query exception now requires the exact zero-argument integer schema,
at most two calls, pinned installed files/version and result `-1`. The
[exact official C++ revision](https://github.com/pytorch/vision/blob/8fb87713a24951e639c494b0f2a8a81b5f8e33a6/torchvision/csrc/vision.cpp)
returns a compile-time constant without tensor/GPU work; the
[inspection evidence](amd-alpha-headless-import-evidence/torchvision-version-query.json)
is retained. Seven inspected, pinned stateless activation classes may create
at most eight import-time Python objects (exactly eight observed); their execution and parameter/
buffer registration stay forbidden. No trained model class is allowlisted.
These were diagnosed harness corrections, not environment repairs or blind
native retries. Read-only review checked each exception before the successful run.
The guards constrain trusted inspected code, not arbitrary hostile native code.

## Tests and preservation

**34 focused tests passed:** 9 alpha-input, 5 headless-interface/supervisor,
7 existing model-metadata and 13 scaffold-regression tests. [Captured commands
and logs](amd-alpha-headless-import-evidence/tests/) are retained. Initial
test failures demonstrated the absent helper/unsupported parameter; a Windows
sandbox temporary-directory restriction was avoided with workspace-only scratch.

[Both mock IPC routes passed](amd-alpha-headless-import-evidence/ipc/results.json):
valid RGBA receipt, opaque-input rejection, same-process recovery and reaped
shutdown. Their ordinary worker identities still report **GPU uninitialized**
and **weights not loaded**. The tiny GLB container and metadata-only safetensors
in IPC scratch are labeled fixtures, not generated geometry or real model weights.

The broader application Python test command (`node scripts/run-pytests.mjs`)
did **not complete** within 120 seconds. Early error markers and partial output
are captured; no completed test summary or reliable failure-name list was
produced. The owned host/child tree was terminated and cleanup verified.
[Exact limitation](amd-alpha-headless-import-evidence/tests/application-suite.json)
is preserved. The `Volume Decoding` text in its log is a synthetic string in
`api/tests/test_extension_process.py`, not an inference run. No unrelated failure
was repaired. Node application code was unchanged; its suite was not rerun.

The [before](amd-alpha-headless-import-evidence/inventory-before.json) and
[after](amd-alpha-headless-import-evidence/inventory-after.json) inventories
are identical, including both installed venvs, distributions/native/source
hashes and normalized model/cache file metadata. Their baseline also matches
the previous offline audit. Full compressed metadata manifests are preserved.
The modified AMD `nodes.py` and untracked `nodes.py.before-profiler-fix` remain
unchanged. These inventories are not full content hashes of every file/model.
No installations, downloads, rebuilds, driver edits or installed application/
source modifications occurred.

## Next separate proof

The alpha-only conditioning-image boundary and real headless classes are proved.
The next smallest candidate for separate approval is an **explicit local-only alpha-mode
loader** followed by loading **only the sparse decoder**, checking parameter
keys/dtypes and CPU/AMD memory before/after placement and release on process exit.
Use the healthy observer and strict deadlines, with no forward pass. Do not call
the eager texturing factory or add DINO/flow/texture loads automatically.

RMBG remains missing for the standard route. Actual model loading, DINO feature
execution, generation/texturing quality, 16 GB capacity and GLB validity remain
unproved. The previous CuMesh error 719/hole-filling native crash remains an open
risk. **Stopped after the import/alpha milestone. No automatic model loading or
GPU generation follows.**
