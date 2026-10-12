# Experimental AMD headless scaffold

The default mode emits **JSON mock receipts**. The separately selected
alpha-only preparation mode returns genuine pipeline configuration diagnostics.
Neither mode generates geometry, textures or GLB files.
Generate and Texture / Refine have separate worker identities. Their manifest
mesh ports reserve the eventual integration contract; mock receipts must not be
fed into a mesh viewer or another workflow mesh node. This milestone verifies
registry discovery and subprocess IPC, not the graphical generation workflow.

Use this checkout's API with `EXTENSIONS_DIR` pointing to the absolute repository
`experimental/extensions` directory. Default production discovery is unchanged.
Explicitly set `MODLY_ENABLE_EXPERIMENTAL_AMD=1` and
`MODLY_AMD_RUNTIME_CONFIG` to an absolute local JSON configuration file:

```json
{
  "python": "D:/AI/Trellis2AMD/venv/Scripts/python.exe",
  "source_root": "D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD",
  "models": {
    "generate": ["D:/offline-models/shape/checkpoint-prefix"],
    "refine": ["D:/offline-models/texture/checkpoint-prefix"]
  }
}
```

These are placeholders; no model files are shipped or downloaded. Each absolute
prefix must have existing `.json` and `.safetensors` siblings. Provide explicit
lists for both nodes. Validation reads bounded JSON/header metadata only. It does
not verify tensor shapes, completeness of a TRELLIS pipeline, architecture
compatibility, or successful loading. GGUF and remote identifiers are rejected.
Fixture files in the proof are empty metadata-only safetensors, not AI models.

The runtime requires Windows Python 3.12, the pinned PyTorch 2.13 / ROCm 10
package versions, AMD64/cp312 native files and HIP imports. The inventoried CPU
xatlas UV utility is pinned by hash. Source commit and pipeline file hashes are
reported; the user's modified `nodes.py` is neither imported nor changed.
The adapter itself imports only stdlib and Modly's base contract. Upstream
pipeline imports, ComfyUI shims, model factories and GPU discovery are deferred.

The separate bounded [headless import proof](../../../docs/compatibility/amd-alpha-headless-import.md)
imports the real AMD geometry and texturing classes through `headless.py`.
It implements only local `folder_paths.models_dir` and callback-based
`comfy.utils.ProgressBar`; existing ComfyUI modules cannot be replaced and
unimplemented interfaces/previews fail explicitly. These imports can initialize
GPU metadata. Preparation uses the established passive Windows observer and
bounded, owned worker lifecycle. Ordinary mock-worker startup retains its
no-GPU-initialization guard.

For explicitly pre-segmented input, mock parameters can include
`"image_mode": "alpha-only"`. This requires a single-frame RGBA image with
fully transparent background pixels and a nondegenerate foreground above alpha
204, including after the upstream 1024-pixel resize. Opaque, empty, faint-only
and degenerate masks fail; there is no conversion, segmentation or fallback.
Inputs are limited to 32 MiB encoded, 4096 pixels per dimension and 16 Mi pixels.
The receipt records validation only. The default `"image_mode": "rembg"`
retains the prior mock route behavior; it does not load a background model.

`alpha_input.preprocess_alpha_image()` validates the mask and delegates to the
real pipeline's unchanged CPU crop/composite method with `rembg_model=None`.
The proof uses empty constructor-created pipeline shells, never model factories.
The upstream texturing `from_pretrained()` still constructs BiRefNet regardless
of image alpha. A future alpha-only texturing loader must explicitly omit that constructor
and bind the verified local DINO model. Geometry preparation is implemented below;
alpha-only texturing preparation remains unimplemented.
Neither alpha validation nor passing headless imports establish model-loading
or generation readiness. Mask quality still depends on the supplied cutout.

Mock parameters must contain `"mock": true`. Refine additionally requires an
absolute existing GLB path under the current worker output directory via
`existing_mesh_path`; validation checks its container header, not its geometry.
The same image-byte IPC and progress/error/done messages as other extensions
are used. Unknown parameters and inference requests fail. Network/subprocess
actions, writes outside the workspace, tensor operations and GPU initialization
are prohibited in the default mock mode. The explicit preparation mode permits
observed GPU metadata initialization and only the pinned, read-only SDK discovery
command, capped at five seconds. These guards constrain trusted scaffold
code and are not a security sandbox for arbitrary native extension code.

Paths on UNC/device/mapped network volumes and Windows reparse points are
rejected before resolution. Use ordinary local files on local drives.
Startup, mock load and mock response limits are 30, 10 and 30 seconds.
IPC writes have a separate five-second limit before the response wait.
`shutdown()` requests cooperative exit with a five-second bound and uses the
existing verified hard stop if needed. Startup errors never trigger pip repair,
interpreter substitution or automatic retries. A later explicit call may start
a fresh worker after it was stopped. No installed application is modified.

See [the scaffold report](../../../docs/compatibility/amd-worker-scaffold.md)
and `scripts/compatibility/amd_worker_scaffold_proof.py` for bounded verification.

## Offline alpha-only geometry preparation

Use a separate runtime configuration; omit the mock `models` lists:

```json
{
  "python": "D:/AI/Trellis2AMD/venv/Scripts/python.exe",
  "source_root": "D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD",
  "mode": "geometry-preparation-alpha-only",
  "geometry": {
    "models_root": "D:/AI/Trellis2AMD/ComfyUI/models",
    "pipeline_dir": "D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B",
    "dino_dir": "D:/AI/Trellis2AMD/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
  }
}
```

Keep the opt-in and experimental extension discovery settings described above.
For the registry's `trellis2-amd/generate` ExtensionProcess, call:

```python
diagnostics = worker.prepare_geometry(
    rgba_png_bytes, {"image_mode": "alpha-only"}, cancel_event=cancel_event
)
worker.unload()    # drops pipeline structure and lazy component handles
worker.shutdown()  # cooperative exit, with bounded hard-stop recovery
```

This API-level route uses the existing runner IPC. No graphical preparation
button or asset response is provided. It rejects Generate/inference and Refine
operations in this mode. The default mock configuration and production extension
discovery remain unchanged.

Validation reads bounded local JSON and safetensors headers, requires the audited
eight-role checkpoint map and DINOv3 files, and checks model classes, flow tensor
shapes and conditioning width. The genuine geometry factory creates nine `None`
model slots; DINO and RMBG remain absent. The adapter binds absolute local model
paths, removes RMBG from the in-memory preparation configuration and delegates
the alpha crop/composite to upstream CPU preprocessing. It does not edit local
model JSON, upstream sources or the standard RMBG route.

Diagnostics contain `pipeline_prepared=true`, six unloaded geometry/DINO component
requirements, `weights_loaded=false`, `glb_generated=false` and
`ready_for_inference=false`. `StrictLocalComponent.load_cpu()` provides reusable,
lazy loading from explicit files: parameters and persistent buffers must match
keys, shapes, dtypes and payload bytes exactly. Metadata-device construction,
strict assignment and cleanup were tested only with tiny synthetic checkpoints.
Nonpersistent buffers left on the metadata device fail explicitly. Real loading
is disabled by the preparation worker's process-wide guards.

Opaque/invalid images and pre-import asset errors can be corrected in the same
worker. Import, factory, safety-guard, transport, timeout or observer failures
stop and reap the worker. Preparation cancellation uses the same controlled
hard stop, rather than waiting for upstream code to poll an event. Each request
has a 30-second response limit; observer queries have a 10-second limit.

See [the loader report](../../../docs/compatibility/amd-offline-alpha-geometry-loader.md)
and `scripts/compatibility/amd_geometry_loader_ipc.py` for saved verification.
