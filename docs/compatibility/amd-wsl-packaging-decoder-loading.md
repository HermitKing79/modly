# Isolated packaging admission and genuine decoder CPU loading — PASS

Base: `e24808e873785637ad754049b54d74e719c94e7e`. Branch: `codex/amd-wsl-packaging-decoder-loading`. Exact reviewed execution harness published and GitHub-verified at `67a1f1eeb4e73cc5edbceea92a540e268ca27150`; `loading.py` SHA-256 `3751cb3666b42e1285717b6e73f523919c659c28485e7f771e9f00ded63bb2a5`.

**PASS:** one dependency installation, one actual CPU import acceptance and one genuine strict CPU checkpoint load completed. No retries, neural forward, GPU tensor computation or native binding invocation occurred. This proves CPU decoder readiness; it does not prove neural decoding or geometry quality.

## Isolated dependency and acceptance

The only addition was [PyPA packaging 25.0](https://pypi.org/pypi/packaging/25.0/json): universal pure-Python `packaging-25.0-py3-none-any.whl`, 66,469 bytes, SHA-256 `29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484`. Official metadata and wheel contents confirm stable/non-yanked identity, Python >=3.8 and no dependencies. [Staging evidence](amd-wsl-packaging-decoder-loading-evidence/packaging-staging.json) records 23 files / 238,344 bytes added exclusively under `/home/modly/native-decoder-preparation-6dd94a85/site-packages`. No resolver, upgrade, build or other package download was used.

Actual safetensors 0.6.2 source inspection established its immediate import closure: standard library, packaging Version, Torch and its existing Rust extension. The previous successful parent-module import evidence was reused. The [single dependency worker](amd-wsl-packaging-decoder-loading-evidence/dependency-run/guest/results.json) imported genuine `Version` and callable `safetensors.torch.load_file` from the intended staging directory. Twelve genuine GPU-entry functions were instrumented: **zero calls**, initialization false before/after, zero checkpoint loads/tensor operations/native calls/rejections. Guest time 6.301 seconds; outer time 15.764 seconds. Its supervisor exited 0 with healthy monitoring, naturally empty Job and named-distro termination 0.

The read-only model preflight passed before reserving the separate loading marker. [Independent readiness review](amd-wsl-packaging-decoder-loading-evidence/independent-loading-preexecution-review.json) gave GO after examining the actual dependency result and unchanged execution boundaries.

## Genuine checkpoint result

Runtime: Ubuntu `Modly-AMD-Ubuntu-24.04`, Python 3.12.3, Torch `2.9.1+rocm7.2.1.gitff65f5bc` (distribution `2.9.1+rocm7.2.1.lw.gitff65f5bc`), HIP `7.2.53211-e1a6bc5663`, Triton `3.5.1+rocm7.2.1.gita272dfa8`. RX 9070 XT / gfx1201; Windows driver unchanged at `32.0.31041.1004`.

Original local checkpoint: `D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS.2-4B\ckpts\shape_dec_next_dc_f16c32_fp16.safetensors`, 948,490,494 bytes, SHA-256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. Configuration SHA-256 `5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209`.

The [actual worker result](amd-wsl-packaging-decoder-loading-evidence/run/guest/results.json) records genuine `SparseUnetVaeDecoder` construction on meta and the unchanged `ShapeDecoderComponent.load_cpu()` / `StrictLocalComponent` contract:

- All **292** original F16 entries matched exact keys, shapes and payload bytes; no missing/unexpected keys.
- **216 F16 / 76 F32** CPU parameters; all 76 approved widenings passed byte-exact F16 round trips. No meta parameters, zero buffers, evaluation mode and gradients disabled.
- Genuine staged FlexGEMM loaded and all 18 bindings resolved; native `.so` SHA-256 matched `91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01`. **Zero binding calls, zero model forwards, zero guard rejections.**
- Model, component and every parameter weak reference were collected after release.

The inherited checkpoint metadata row still carries its earlier `deserialized=false`/header-only label. The current `decoder_diagnostics.weights_loaded=true`, complete final tensor records and fresh stream hash establish the actual load; historical metadata was not rewritten.

| Measurement | Actual result |
| --- | ---: |
| CPU decoder loading/verification | 11.002 s |
| CPU RSS before / loaded / after release | 762,458,112 / 1,760,768,000 / 807,452,672 bytes |
| CPU high-water RSS | 2,699,878,400 bytes (2.51 GiB) |
| Host RAM admission / minimum observed | 16,779,304,960 / 10,998,259,712 bytes |
| GPU free VRAM at admission | 14,369,837,056 bytes (13.38 GiB) |
| Torch GPU allocated / reserved: baseline, peak, release | **0 / 0 bytes throughout** |
| Guest / supervisor / outer duration | 22.560 / 40.610 / 41.501 s |

HIP metadata initialization was permitted and actually occurred during admission; this model remained CPU-only. Native import and synchronization took 0.258 seconds. The preserved limits were 6,972,719,480-byte CPU RAM admission, 13 GiB free VRAM, 12 GiB allocator ceiling, and 60/120/240/220/270-second native/CPU/guest/authorized-worker/host deadlines.

The [supervisor](amd-wsl-packaging-decoder-loading-evidence/run/results.json) and worker qualified, exit 0. All nine periodic observations were healthy. Windows Job accounting showed zero active processes before/after cleanup, naturally empty without force; only Modly Ubuntu was terminated, exit 0. Stderr is empty. [Independent outcome review](amd-wsl-packaging-decoder-loading-evidence/independent-loading-outcome-review.json) verifies acceptance from raw records and unchanged loaders.

## Preservation and next action

[Final verification](amd-wsl-packaging-decoder-loading-evidence/preservation-verification.json) confirms 306 historical files, five controls, two loaders, four pre-existing bytecode files and all **52 prior local/remote heads** unchanged, including main. Windows before/after audits match for both installed environments, 12 binaries, AMD source edits/backups, four saved inputs and checkpoint metadata/header; the decoder config was verified separately. Linux before/after audits preserve the baseline venv, packages/dpkg, runtime source hashes, library identities and all 922 original staged payloads; the only staged package addition is the verified 23-file packaging package. `docker-desktop` remains the default and stopped. Previous failed markers/results were retained.

**Shortest next action, separately approved:** prepare one genuine parent neural-only decode of the unchanged full 3,743-row turret latent and coordinates in this WSL environment. Before execution, bound the four learned subdivision stages and all 40 sparse convolutions: predicted token/coordinate/feature growth, neighbor maps, SplitK/native workspace, retained spatial caches and peak live memory must be admitted before allocation. This CPU load proves neither those native computations nor compatibility of Windows kernel caches with WSL. Any required Triton specialization/build/tuning miss needs an explicit reviewed policy; import success alone grants no compilation permission. Preserve 13 GiB admission, the 12 GiB ceiling, original F16/F32 arithmetic, zero-allocation teardown and process monitoring. Stop before O-Voxel extraction, CuMesh, textures or GLB export. No forward is authorized or executed here.
