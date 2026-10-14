# Genuine WSL decoder CPU loading — UNKNOWN

Base: `c8614c093d11e7d88c16e9e475a1eb5b5e9cd72c`. Branch: `codex/amd-wsl-decoder-path-correction`. Execution candidate was published and GitHub-verified at `83e5d7f6ff810cee274b9d2f106c3809c3c21d9b`, SHA-256 `26b2679293ce7ab0d2eab190e45bc4d8ab3bcbb34734aa0e2f395b515a2b62fa`, before the single contained attempt.

**Both path contracts passed and genuine FlexGEMM qualified. The real CPU loader was entered but failed because `safetensors.torch` could not import `packaging`. Meta construction and checkpoint tensor loading were not reached. No installation, runtime repair or retry followed.**

## Exact runtime boundary

The guest validated both explicit `/mnt/d/...` string arguments through unchanged `absolute_file()` before Torch/FlexGEMM imports, verified the original checkpoint/configuration identity and preserved the full 3,743-row turret. Genuine FlexGEMM loaded from the existing isolated staging directory, its expected `.so` identity matched, and all 18 bindings resolved without invocation.

The traceback then shows `ShapeDecoderComponent.load_cpu()` entering `StrictLocalComponent.load_cpu()`. At strict_loader.py line 37, `from safetensors.torch import load_file` reached staged safetensors/torch.py line 5 and raised:

```text
ModuleNotFoundError: No module named 'packaging'
```

This occurs before the strict loader's meta context/factory and before any `load_file()` call. No checkpoint tensors were deserialized, genuine parent model constructed, parameters widened, GPU model placement performed or forward/native computation executed. The raw `decoder_cpu=NOT_ATTEMPTED` status was never updated to success; it does not mean the loading method was never called. Full [guest results and traceback](amd-wsl-decoder-path-correction-evidence/run/guest/results.json) and [supervisor results](amd-wsl-decoder-path-correction-evidence/run/results.json) are preserved without relabelling.

| Measurement | Actual observation |
|---|---|
| Python / Torch runtime | 3.12.3 / 2.9.1+rocm7.2.1.gitff65f5bc |
| Torch distribution / HIP | 2.9.1+rocm7.2.1.lw.gitff65f5bc / 7.2.53211-e1a6bc5663 |
| GPU / admission free VRAM | RX 9070 XT, gfx1201 / 14,528,552,960 bytes |
| Native synchronized import | 0.262 seconds; 18 callable bindings; zero invocations |
| Fresh guest RAM before loading | 15,675,420,672 bytes |
| Pre-loading CPU RSS / high-water snapshot | 762,695,680 / 762,695,680 bytes; resource.ru_maxrss 761,933,824 bytes |
| Torch GPU allocated/reserved: before / after native import / release | 0/0 bytes at all three observations |
| Final policy | cpu_meta; zero tensor operations, native calls and rejected calls |
| Authorized guest / supervisor / outer duration | 12.315 / 25.484 / 26.385 seconds |
| Guest / outer exit | 1 / 1 |
| Windows observations / cleanup | Six healthy observations; Job naturally empty, active before/after cleanup 0/0, no forced cleanup; named Modly distro termination exit 0 |

HIP metadata initialization was permitted and occurred. GPU allocation peak, model construction/loading time, complete-attempt CPU peak and loaded-model/parameter collection are unmeasured because the dependency import failed first. The CPU figures above describe only the observed pre-loading snapshot. No claim of decoder readiness or native computation failure is supported.

## Correction and preflight evidence

The guest constructor now passes exactly two strings. Host preflight uses Windows `PREFIX_HOST` strings through the genuine `absolute_file()` API and checks their exact POSIX `/mnt/d/` equivalents. The existing shared checkpoint-record reader is unchanged. The original 948,490,494-byte checkpoint SHA-256 matches on host and guest: `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. Config SHA-256 is `5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209`. No weight copy is published.

The two explicitly allowed preflight correction cycles fixed the workspace-only mapper used for D: paths and Windows formatting of the Linux path literal. All failed captures are retained. [Final host preflight](amd-wsl-decoder-path-correction-evidence/preflight-3.stdout.txt) exited 0 in 0.965 seconds. It used the genuine component's metadata-only constructor and source verification, checked all 292 original F16 header entries, and confirmed no loaded weights or AI/native imports. The genuine strict F16/F32 loader and its intended 76/216 precision contract remain unchanged and unexercised on tensor values in this attempt.

RAM/VRAM admission, allocator cap and deadlines stayed unchanged: 6,972,719,480-byte CPU RAM floor; 13 GiB available VRAM; 12 GiB Torch cap; 60/120/240/220/270-second native/CPU/guest/authorized-worker/host limits. Exclusive marker, binary GO, owned Job, periodic monitoring, five-second natural-exit grace and named-distro termination were preserved.

## Preservation and next action

Windows before/after audits match both environments, 12 selected native binaries, AMD source edits/backups, four turret artifacts and decoder metadata/header. Linux before/after audits match baseline venv/packages/dpkg/source/library metadata and all 922 staged payloads. Passive audits import no AI/native modules. Modly Ubuntu finishes stopped, with Docker/default registration preserved. Historical evidence, previous markers, loaders and existing bytecode remain unchanged; main and prior branch tips are preserved. No dependency installation, native build or production worker modification occurred.

**UNKNOWN:** an importable `packaging` dependency is missing from the actual isolated loader environment. This is a concrete dependency boundary, not a demonstrated model/checkpoint or native binary incompatibility.

**Smallest next action:** separately approve qualifying the missing pure-Python `packaging` dependency for isolated staging, including an explicit CPU-only `safetensors.torch` import check before any newly approved strict model load. Preserve the baseline venv and every F16/F32 rule. Do not proceed to neural decoding, subdivision or native workspace experiments until the genuine CPU load succeeds. Stop here.
