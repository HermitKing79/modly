# Genuine WSL decoder CPU loading — prepared candidate

Base: `c8614c093d11e7d88c16e9e475a1eb5b5e9cd72c`. Branch: `codex/amd-wsl-decoder-path-correction`.

The preserved-copy candidate changes both guest `ShapeDecoderComponent` arguments to explicit `/mnt/d/...` strings and exercises the genuine Windows `absolute_file()` API with `PREFIX_HOST` strings. The unchanged shared checkpoint reader validates exact size, timestamp, provenance and full SHA-256 on host and guest. No loader, model precision rule, installed source, native binary or runtime policy changes are included.

The complete host preflight passes at candidate SHA-256 `26b2679293ce7ab0d2eab190e45bc4d8ab3bcbb34734aa0e2f395b515a2b62fa`. It constructed the genuine `ShapeDecoderComponent` metadata interface, verified its source manifest/configuration and all 292 original F16 header entries, and confirmed `loaded=False` and `weights_loaded=False`. An import audit denies Torch/Triton/FlexGEMM/NumPy/safetensors modules in host preflight. No real model factory or weight loading occurred.

Two explicitly permitted caller/preflight correction cycles were consumed: replacing a workspace-only path mapper for D: model paths, then using POSIX formatting for the Linux literal during Windows comparison. All three preflight captures remain intact; the final preflight exited 0 in 0.965 seconds. No runtime marker was reserved by those checks.

Passive Windows preservation matches both environments, AMD edits/backups, native identities, saved turret inputs and decoder metadata/header. Following the successful host gate, the standard-library-only Linux before audit verified baseline venv/packages/dpkg/source/library metadata and all 922 staged payloads. Guest available RAM was 15,812,939,776 bytes. It imported no AI/native modules; only the named Modly Ubuntu distro was stopped afterward.

The one genuine loading worker will validate Linux strings/configuration before Torch or FlexGEMM imports. It will then qualify the existing native binary without invoking any bindings and reuse the unchanged `ShapeDecoderComponent.load_cpu()` / `StrictLocalComponent`: genuine parent on meta; all 292 original F16 entries with strict keys/shapes/dtypes/bytes; exactly 76 reversible F32 widenings and 216 remaining F16; CPU/eval/grad-disabled; zero buffers or unresolved metadata parameters. Timings, CPU RSS/high-water marks, primitive parameter metadata and weak-reference collection are recorded. No forward or GPU model placement is authorized.

Limits are unchanged: 6,972,719,480-byte host/guest RAM admission; 13 GiB available VRAM and 12 GiB Torch allocator cap for incidental metadata initialization; native/CPU/guest/authorized-worker/host deadlines 60/120/240/220/270 seconds. Binary GO, owned Windows Job, periodic health monitoring, monitored five-second natural-exit grace and named-distro cleanup remain intact. The existing full 3,743-row turret is unchanged.

This preparation commit contains no genuine model-loading result. Independent review and GitHub verification precede exactly one loading attempt. Any runtime failure stops without correction or retry. No model weights or full tensors will be published.
