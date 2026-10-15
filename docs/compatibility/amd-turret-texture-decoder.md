# Genuine guided turret texture decoder

Base `c9e4d3b3555fc3f35a998053592e13c476b60463`; branch `codex/amd-turret-texture-decoder`.

**UNKNOWN — one contained worker launch stopped before Torch, native import, checkpoint loading or any decoder forward. No texture field was produced.** The exact reviewed harness `d0c3405a573b769d03b50371f4cdc38466cf6144` was pushed and GitHub-verified before execution. No retry or earlier model stage ran.

## Exact blocker and observed release

At `decode.py:215`, the initial `artifact_io.check_directory()` imports `amd_texture_field_artifact`, whose module-level import requires NumPy. The worker adds the previously staged dependency directory to `sys.path` only later, at line 227. The baseline WSL venv does not provide NumPy; the preserved staging manifest places NumPy 2.2.6 under `/home/modly/native-decoder-preparation-6dd94a85/site-packages`.

The actual exception is **`ModuleNotFoundError: No module named 'numpy'`**. This is a launcher dependency-order defect. It is not a demonstrated AMD, F16, FlexGEMM or checkpoint incompatibility. Windows CPU qualification and static review missed this WSL bootstrap requirement. The failed worker never freshly revalidated the staged dependency closure.

| Actual observation | Result |
| --- | ---: |
| Authorized contained worker launches | 1 |
| Decoder forwards / real checkpoint loads / native imports | **0 / 0 / 0** |
| Guest time after authorization | 0.061 s |
| Complete supervised attempt | 10.485 s |
| Worker exit | 1 |
| Windows observer | Healthy; no recorded GPU fault events |
| Windows Job active before / after cleanup | **0 / 0** |
| Forced Job cleanup | **False** |
| Named Ubuntu termination | Exit 0 |
| Initial available host RAM | 18,721,562,624 bytes |
| Lowest observed available host RAM | 16,132,980,736 bytes |
| GPU identity/admission, peaks and zero-allocation measurements | **Not reached; no measurements claimed** |

The permanent output leaf remains empty:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-decoder-c9e4d3b\field`

There is no real output file or output hash. The preserved small `probe` is synthetic CPU writer evidence and must not be confused with learned material data. [Exact worker error](amd-turret-texture-decoder-evidence/run/guest/results.json) · [supervision and monitoring](amd-turret-texture-decoder-evidence/run/results.json) · [verified result summary](amd-turret-texture-decoder-evidence/verified-result-summary.json).

## Completed CPU qualification; execution remains unproved

**72 focused tests: 71 passed, one Windows symlink-privilege skip.** They cover the installed meta architecture and strict schema/precision contract, adjacent strict-loader regressions, actual saved-input reader and original C2S topology, live guided hooks, F16/F32 tiled-convolution equivalence, six-channel writer/reader and launcher/receipt preflight. Actual outside-Git CPU persistence and independent readback passed. Independent static review found no open blockers, but this did not qualify the real WSL bootstrap order.

- Original 284 F16 checkpoint entries; 208 body parameters retained as F16, exactly 76 source-required norm/projection parameters restored to F32 with byte-exact F16 round trips. No BF16 model substitution or checkpoint change. Strict real loading was not reached.
- Original F32 denormalized `[3743,32]` texture latent and I32 coordinates. Four original BF16 guides retain every bit and use `> 0`. CPU/source tests reconstruct exact ordered supports: 3743 -> 17015 -> 79169 -> 378891 -> 1842968. Live GPU order checks were not reached.
- Reviewed original 40 convolutions and eight C2S expansions. The final convolution alone uses full global neighbor indices with source-equivalent F16 explicit-GEMM row tiling: 57 tiles, 32768 rows, tail 7960. Its real F16 execution remains unproved.
- Intended field is raw F32 `[1842968,6]`, original unclamped F32 `raw * .5 + .5`, and exact original I32 `[1842968,4]` geometry support. Channels are red, green, blue, metallic, roughness, alpha. No numerical field values exist from this run.
- Unchanged gates: 13 GiB initial free VRAM / 12 GiB host RAM; 12 GiB allocator ceiling; 2 GiB device reserve plus 1 GiB opaque-workspace allowance; reviewed sparse-allocation plans and exact final-support exception. Native/CPU/GPU limits 60/120/240 s, guest/host 450/480 s; binary GO, Windows Job, health monitoring, five-second natural-exit grace and named-distro termination.

A pre-execution committed-byte check also caught Git LF versus existing working-tree CRLF in an unchanged metadata helper. Git now preserves its reviewed existing bytes; executable text and all candidate hashes stayed unchanged. No runtime guard was relaxed.

A helper briefly started the unrelated application suite during preparation; it was interrupted, is not claimed as passing, and led to no unrelated fixes. The visible Volume Decoding progress was a literal existing test fixture. No genuine model stage ran there.

[CPU qualification](amd-turret-texture-decoder-evidence/qualification.json) · [precision/schema](amd-turret-texture-decoder-evidence/texture-cpu-contract.json) · [input identities](amd-turret-texture-decoder-evidence/input-qualification.json) · [persistence](amd-turret-texture-decoder-evidence/persistence-preflight.json) · [preflight](amd-turret-texture-decoder-evidence/preflight.json) · [pre-execution review](amd-turret-texture-decoder-evidence/independent-preexecution-review.json).

## Preservation and next action

Fresh post-run checks matched 51 protected artifacts/evidence files, 17 AMD source/configuration files, five saved inputs, five previews, the existing AMD-source backup, both Windows environments' distribution/venv metadata and selected native binaries. The two actual decoder inputs were independently rehashed unchanged: texture `5506c4d0199bef64423b41b8bed560e2b7def9d29c5bb1c3072c0ea31ee1ddc4`; geometry/guides `dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22`. All 64 pre-existing remote heads, including main, are unchanged. Historical evidence and attempt markers remain intact. Driver remains `32.0.31041.1004`, controller OK; both WSL distributions are stopped and docker-desktop remains default.

No installation, rebuild or environment-write command ran. This worker failed before its private scratch directory creation and installed-runtime checks; a fresh comprehensive Linux package inventory is **not** claimed. [Windows preservation](amd-turret-texture-decoder-evidence/windows-preservation-after.json) · [independent outcome review](amd-turret-texture-decoder-evidence/independent-outcome-review.json).

**Smallest corrective step for separate approval:** move the existing staged `sys.path` setup before the first writer import, qualify that exact WSL bootstrap with CPU-only dependency resolution, then authorize one fresh guided decoder attempt. Reuse staged NumPy; no installation, dtype change, safety relaxation or earlier model rerun is indicated. The failed candidate and marker should remain unchanged. No correction or further attempt was made after this failure.

No mesh changes, RGB preview, UV/baking, material export or GLB work occurred. A learned-RGB preview remains conditional on first producing and independently validating the genuine six-channel field.
