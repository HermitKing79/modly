# Offline alpha-only AMD geometry loader

**PASS for configuration preparation through real application IPC — 12 October
2026.** The experimental Generate worker prepares the genuine local TRELLIS.2
geometry structure, validates alpha-only input and returns unloaded component
requirements. **No real checkpoint was loaded, model placed, forward executed,
textures baked or GLB generated.** Generation readiness remains false.

Branch `codex/amd-offline-alpha-geometry-loader` starts at
`f518deae75bb4626d17d29812b995d4488d90edf`. The
[sparse-decoder loading report](amd-sparse-decoder-loading-proof.md) and
[alpha/headless report](amd-alpha-headless-import.md) were reviewed first.
The decoder transfer proof was not repeated. Changes are reusable application
code, focused tests and evidence, confined to the experimental AMD integration.

## Factory reuse and offline contract

The inspected AMD `Pipeline.from_pretrained()` checks local `pipeline.json`, then
creates `None` slots; its model-loading calls are commented out. The geometry
subclass constructs Python samplers and attributes while leaving DINO/RMBG
unloaded. Its fallback download branch is excluded by existing absolute local
directory validation, blocked network calls and pinned factory source hashes.
No factory replacement or upstream source patch was needed. Texturing's eager
factory is not used.

The explicit `geometry-preparation-alpha-only` runtime mode uses the existing
`ExtensionProcess.prepare_geometry()` → `runner.py` → AMD adapter lifecycle.
The default mock configuration, no-GPU mock startup guard and production
extension discovery are preserved. [Usage](../../experimental/extensions/trellis2-amd/README.md)
documents the separate configuration and API call; no graphical preparation
workflow is added.

The adapter validates local pipeline/configuration files, per-role model classes,
safetensors headers, flow shapes, DINO conditioning width and source provenance.
It binds the audited absolute checkpoint map under
`D:\AI\Trellis2AMD\ComfyUI\models`, including the external
`microsoft\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16` prefix.
DINO is `facebook\dinov3-vitl16-pretrain-lvd1689m`. All eight declared roles
are validated; only five geometry roles plus DINO have lazy handles. Texture
slots and the injected shape encoder remain `None`. RMBG is removed only from
the adapter's in-memory configuration; standard upstream sources remain intact.

`StrictLocalComponent` defers construction and weight reads until explicit CPU
loading. It constructs from local JSON on the metadata device, checks complete
parameter/persistent-buffer keys, shapes and dtypes, assigns with `strict=True`,
and compares every resulting tensor's bytes to the local safetensors payload.
Unexpected/missing entries, metadata changes and unresolved nonpersistent
buffers fail closed; failed handles do not retry automatically. The preparation
worker disables real loading entirely. Only tiny synthetic CPU checkpoints
tested this reusable interface; other real component architectures are unproved.

## Actual results and side effects

[Saved IPC evidence](amd-offline-alpha-geometry-loader-evidence/ipc/results.json)
records the installed Modly Python 3.11 host launching the existing AMD Python
**3.12.10 / PyTorch 2.13.0+rocm10.0.0 / HIP 7.15.26333 / CUDA null** worker.
Native binary hashes/ABI and source paths are retained. The source is the
preserved AMD checkout at `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`.

| Boundary | Recorded result |
|---|---|
| Genuine factory | `Trellis2ImageTo3DPipeline`, nine `None` slots, CPU structure, DINO/RMBG absent |
| Lazy requirements | Six unloaded handles; absolute local configuration/weight paths and header hashes |
| Alpha route | Valid 12×10 RGBA cutout → upstream 6×6 RGB conditioning image; DINO never executed |
| Input recovery | Opaque image rejected before real imports; valid preparation succeeds in the same PID |
| Lifecycle | Unload/reload and preparation in the same worker; cooperative shutdown exit 0 and reaped |
| Cancellation | Fresh worker cancelled before preparation, hard-stop exit 1 and reaped |
| Asset/inference claims | Generation request rejected; real weights loaded 0, GLB files generated 0 |

Import-time GPU metadata changed from uninitialized to initialized on **RX 9070
XT / gfx1201**, with **0 allocated / 0 reserved tensor bytes**. Both Windows
observations returned normally, validated the System log and reported controller
OK with no monitored reset/error events. The preparation guards prohibit tensor
operations except the two pinned TorchVision integer version queries (`-1`),
AI module construction/loading, forwards and Triton launches/compilation. Eight
verified stateless activation objects are allowed during Transformers import.
Model-file reads are blocked during import/factory/preprocessing; asset validation
uses bounded header reads. Payload-read zero is the enforced metadata-only code
contract, not a trace of arbitrary native-library I/O.

One verified `rocm-sdk path --root` discovery ran with a five-second cap, returned
1 in 1.531 seconds for absent `rocm[devel]`, and was handled by Triton. No package
was installed. This missing-devel branch is pinned because other SDK states can
mutate package records. Scratch caches stay in the disposable workspace; saved
output lists contain no generated asset. Zero tensor allocation does not prove
zero internal driver activity or complete future GPU stability.

## Verification and interruption recovery

[Focused test records](amd-offline-alpha-geometry-loader-evidence/targeted-tests.json)
plus [mock regressions](amd-offline-alpha-geometry-loader-evidence/mock-regression.json)
contain **67 passing tests**: assets 8, IPC/lifecycle 9, alpha input 9, headless
interfaces 5, metadata 7, supervisor 2, decoder contract 9, synthetic strict loader
5 and existing scaffold 13. In-flight cancellation, timeout, observer failure,
fatal worker errors and malformed metadata were tested with mocks; actual process
termination was demonstrated by the IPC shutdown/pre-cancellation scenario.
The [initial regression record](amd-offline-alpha-geometry-loader-evidence/regression-initial.json)
preserves the mock-config regression found and fixed before successful verification.
Per-role class/syntax validation was tightened after the IPC run and verified by
targeted tests plus [passive real-asset checks](amd-offline-alpha-geometry-loader-evidence/asset-validation.json).

After the reported power interruption, saved assertions, test counts, matching
IPC records and syntax of 19 Python files were independently rechecked in
[recovery evidence](amd-offline-alpha-geometry-loader-evidence/recovery/saved-evidence-verification.json).
No successful GPU/IPC test or checkpoint proof was rerun. The subsequent power-off
cause is unknown and lies outside the saved observer window.

[Preservation](amd-offline-alpha-geometry-loader-evidence/preservation.json) records
byte-identical before/after inventories: AMD venv 94,687 entries, Modly venv
61,313 entries, model/cache trees, dependency records and selected native/source
hashes. The [fresh passive recovery comparison](amd-offline-alpha-geometry-loader-evidence/recovery/preservation-verification.json)
found byte-identical inventory JSON. Raw tree metadata differs only in directory
`st_size` reporting: 8,453 venv rows and 50 model/cache rows. All paths, regular-file
sizes, mtimes and modes match; normalized trees and selected content hashes match.
Every raw delta is preserved in the compressed recovery evidence. The source retains
only `M nodes.py` and `?? nodes.py.before-profiler-fix`. No installed environment,
driver, package, weight or existing source edit was changed. Main and all seven
previous proof branches are preserved; only this new development branch is
published normally. Remote ref/content verification follows publication.

A read-only recovery review independently confirmed saved test/IPC evidence,
current source hashes and directory-only metadata reconciliation, with no code
blocker to publication. IPC remains historical evidence: recovery did not rerun
it or establish present GPU stability after the power interruption.

## Smallest next milestone

With separate approval, load only the already-proved sparse decoder through the
new strict interface and execute **one deterministic, minimal decoder forward**
on an architecture-validated synthetic latent, under the healthy observer and a
strict timeout. Record output shape/finite values, VRAM and release, then stop.
This exercises a genuine model operation without loading flow/DINO/texture
models or exporting geometry. Full generation, texturing and CuMesh error 719
remain unresolved. No inference work is authorized by this loader milestone.
