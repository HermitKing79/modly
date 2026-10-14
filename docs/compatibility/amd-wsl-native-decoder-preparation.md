# WSL native qualification and turret decoder preparation — UNKNOWN

The milestone stopped at a demonstrated **CPU test-harness admission defect**. No genuine FlexGEMM native import, real decoder checkpoint loading, neural forward or geometry extraction occurred. Native compatibility is **UNKNOWN**, not a demonstrated native incompatibility. The current execution candidate is **NO-GO** until its device-context guard is corrected and independently accepted.

Base: `ab97328d12163e94a8e854a1c1efd5dcf92b81db` (`codex/amd-wsl-flexgemm-build`). Continuation: `codex/amd-wsl-native-decoder-preparation`. All earlier evidence is preserved.

## What was completed

| Boundary | Actual result |
|---|---|
| Isolated package staging | Verified; baseline venv and dpkg inventory unchanged |
| Real installed Torch/Triton CPU policy acceptance | Earlier explicit-device test passed; later complete default-device test failed safely |
| Genuine FlexGEMM import and 18 binding resolution | **NOT RUN**; no native attempt marker or run directory |
| Genuine parent decoder import/construction/strict CPU loading | **NOT RUN**, conditional on native qualification |
| Component loader regressions | 8 tiny synthetic-checkpoint tests and 5 unchanged shared strict-loader tests passed |
| Whole-turret forward admission | **UNKNOWN / blocked**; visible checks exist but are not installed execution hooks |

Ubuntu remains `Modly-AMD-Ubuntu-24.04`, Python 3.12.3, Torch distribution `2.9.1+rocm7.2.1.lw.gitff65f5bc` (runtime omits `.lw`), Triton `3.5.1+rocm7.2.1.gita272dfa8`, ROCm 7.2.1 / ROCDXG 1.2.0. No baseline runtime replacement occurred. Windows driver remains `32.0.31041.1004`.

Only the previously built FlexGEMM 1.0.0 wheel and newly admitted NumPy **2.2.6** / safetensors **0.6.2** were staged under `/home/modly/native-decoder-preparation-6dd94a85/site-packages`. Their exact wheel hashes, provenance and dependency closure are in [dependency admission](amd-wsl-native-decoder-preparation-evidence/python-dependency-admission.json). New downloads total **17,013,453 bytes**. Native wheel SHA-256 remains `cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285`.

One pip target installation completed, followed by a package-name check failure (`flex_gemm` versus `flex-gemm`). A metadata-only reconciliation using canonical package naming verified the existing target; no reinstall or download retry occurred. Both captures remain intact. Final isolated scratch regular-file size is **87,915,175 bytes**; Ubuntu free space **9,620,484,096 bytes**, D: free space **76,014,514,176 bytes**, above the 5/50 GiB floors. [Final Linux preservation](amd-wsl-native-decoder-preparation-evidence/linux-preservation-after.json) verifies 922 staged payloads and unchanged baseline metadata, package versions and dpkg inventory.

## Decisive blocker and tests

The installed Torch 2.9.1 `DeviceContext` can inject a CUDA device through `__torch_function__` **after** the factory wrapper has checked its arguments. On the observed `with torch.device('cuda'): torch.empty((1,))` route, the module's `CURRENT_DEVICE` did not expose that active context. The request reached `_lazy_init` before the production frontend policy rejected it.

The CPU probe had replaced `_lazy_init` with an independent rejection, so the original initialization function did not execute. The process exited **1** after **9.922 seconds**. Its stdout was empty and no final success/GPU-state JSON was produced; these limitations are preserved. Implicit CPU/meta routing passed before the negative CUDA test failed. This does not demonstrate a FlexGEMM or HIP defect.

Read the [exact trace](amd-wsl-native-decoder-preparation-evidence/policy-cpu-meta-confirmation.stderr.txt), [actual probe](amd-wsl-native-decoder-preparation-evidence/policy-cpu-meta-confirmation.py) and [final independent policy decision](amd-wsl-native-decoder-preparation-evidence/independent-policy-review.json). The earlier narrower CPU PASS and static approvals remain preserved and are explicitly superseded for execution admission by [final disposition](amd-wsl-native-decoder-preparation-evidence/final-disposition.json).

During preparation, scoped positive/negative CPU regressions passed: 14 policy tests, 6 supervisor/authorization/result tests, 8 allocation tests, 8 component-loader tests and 5 unchanged strict-loader tests. The 14 policy regressions preceded the final meta/default-device confirmation; they are not a final GREEN assertion for the frozen policy. These scoped regressions do **not** override the failed complete installed-runtime acceptance. Loader fixtures used host CPU Torch 2.12.0+rocm7.14.0 with GPU initialization blocked; genuine WSL policy probes used the unchanged Torch 2.9.1 stack. No synthetic convolution tests were run.

## Prepared loader and resource boundary

`ShapeDecoderComponent` delays genuine parent construction until native import completes. It checks the pinned configuration and entire historical 292-parameter manifest, temporarily normalizes exactly 76 meta parameters to F16, uses the **unchanged** `StrictLocalComponent` to verify every original F16 payload byte, then restores only those 76 parameters to F32 with bit-exact F16 round trips. The other 216 remain F16; zero buffers are required. It rejects key, shape, dtype and source mismatches and never calls `from_pretrained`, a model forward or GPU placement. Only synthetic loading has been demonstrated in this milestone.

The source-pinned parent boundary omits only child wrapper `resolution` and constructs `SparseUnetVaeDecoder(out_channels=7, **args)`; child O-Voxel extraction remains excluded. No BF16 substitution or production mock-route change was made.

The original denormalized F32 `[3743,32]` latent SHA-256 is `90eaf72030f303b209a4d02ae9adc7d4ecd39662344f35f0adc8120bc1e871b9`; I32 `[3743,4]` coordinates remain `949a9e6a2548abcad66354384502c06834d4d56ea3f2fa86b619c25926379ed4`. Full files and row order remain unchanged. Decoder checkpoint SHA-256 is `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`, independently streamed once this milestone; no tensor deserialization occurred.

The [source allocation audit](amd-wsl-native-decoder-preparation-evidence/allocation-source-audit.json) identifies 40 convolution sites and four learned subdivision sites (each C2S has separate feature/skip expansion calls). Worst-case support is `[3743,29944,239552,1916416,15331328]`. The visible uint32 explicit-GEMM bound is `16N + 21NK + 4NKCi + 2NCo`, K=27. This is not a total workspace bound. ATen indexing/BLAS workspace, whole-call overlap, retained activations/caches and hash probing remain unresolved. The current source defaults to masked SplitK; the explicit branch is documented separately, without changing the algorithm. Both proposed CPU guards reject before callbacks while coverage is unknown. No full forward admission or installed subdivision interception is claimed. The 12 GiB allocator ceiling and 13 GiB free-VRAM requirement are unchanged.

## Preservation and next action

Both Windows AI environments, native files, AMD edits/backups, saved turret files and prior Git evidence are independently checked in [Windows preservation](amd-wsl-native-decoder-preparation-evidence/windows-preservation-after.json). Ubuntu was terminated by name; `docker-desktop` remains stopped and the default registration is preserved. Final Windows observations were healthy. No native-risk worker was launched, so Job/native-import cleanup acceptance was not exercised.

**One recommended next action:** separately approve a surgical CPU-only correction that inspects the genuine active TorchFunctionMode `DeviceContext.device` before invoking the original factory, preserving Torch's original constructor-cache identities. One bounded installed-runtime acceptance must prove implicit CPU/meta and explicit/implicit GPU rejection before any GPU-entry probe. If it fails, stop. If it passes, the deferred single native import and strict CPU loading still require approval; full turret workspace admission remains a separate unresolved execution gate. No forward is authorised by this report.
