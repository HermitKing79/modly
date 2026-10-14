# WSL decoder checkpoint-schema correction — UNKNOWN

Base: `c4552110f80bd86e3e19d892f5b397b495f7fa94`. Branch: `codex/amd-wsl-checkpoint-schema-correction`.

**The exact checkpoint-schema correction passed the shared read-only reader. The complete preflight failed on another deterministic caller-path contract. No execution marker was reserved, WSL launched, native package imported or model loaded. The conditional genuine loading attempt was withheld; no repair or retry followed.**

## Corrected checkpoint contract

The fresh candidate [loading.py](amd-wsl-checkpoint-schema-correction-evidence/loading.py) uses one `checkpoint_record()` function in both host preflight and the actual worker. It reads `decoder_checkpoint_metadata_and_header_only`, validates the required fields with exact value types (excluding booleans as integers), confirms explicit file identity, regular-file status, size, timestamp and prior full-hash provenance, then streams the file hash without deserializing tensors. It rejects a changed file/hash or size/timestamp change during the read.

One actual host preflight used the real preservation JSON and original checkpoint. It confirmed 948,490,494 bytes, `mtime_ns=1791274722716299500` and SHA-256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. The original JSON audit fields returned in the checkpoint record describe the passive preservation audit; `freshly_streamed_sha256` records this separate preflight hash pass. No weights were deserialized or copied.

The candidate SHA-256 is `aa9efe254e7bd235698c00c44ab0a5c98d80a3ad5797e3e30715159301ab953b`. Its new scratch/output paths preserve the previous failed worker, marker and evidence. Loaders, precision rules, native modules, policy and supervisor implementations are unchanged. Deadlines remain 60/120/240/220/270 seconds for native/CPU/guest/authorized worker/host. RAM admission remains 6,972,719,480 bytes; 13 GiB VRAM admission and 12 GiB allocator ceiling remain in the unexecuted worker.

## Exact preflight blocker

The unchanged worker calls:

```python
ShapeDecoderComponent(PREFIX.with_suffix('.json'), PREFIX.with_suffix('.safetensors'))
```

Both arguments are `pathlib.Path` objects. `ShapeDecoderComponent` passes them to `StrictLocalComponent`, whose `absolute_file()` delegates to `local_path()`. That existing validator explicitly requires a string before resolving any file. This is independent of Windows/WSL or GPU compatibility.

Preflight invoked the genuine existing `absolute_file()` on the actual worker's argument type. It failed with `ValueError: component configuration must be an explicit absolute local path`, through `api/services/amd_runtime.py` lines 64 and 40. The complete [stdout](amd-wsl-checkpoint-schema-correction-evidence/schema-preflight.stdout.txt), [traceback](amd-wsl-checkpoint-schema-correction-evidence/schema-preflight.stderr.txt) and [capture](amd-wsl-checkpoint-schema-correction-evidence/schema-preflight-capture.json) are preserved. This was the single read-only host preflight: exit 1, 0.983 seconds. No synthetic checkpoint, model construction or GPU-entry substitution was used.

Before rejecting the path arguments, preflight checked the original helper directory, selected guest path, exclusive-marker absence, checkpoint configuration/control/loader identities and host resource floors. Host available RAM was 17,304,338,432 bytes; D free space 75,839,397,888 bytes. The other directly consumed evidence fields and paths were independently audited, including staging/native identities, 18 binding names, 97 source-check rows and the 292-parameter manifest. The previous Linux after-inventory is explicitly labelled historical; no fresh Linux package/RAM/VRAM observation was attempted after the failed gate.

## Preservation and decision

Fresh Windows [before](amd-wsl-checkpoint-schema-correction-evidence/windows-preservation-before.json)/[after](amd-wsl-checkpoint-schema-correction-evidence/windows-preservation-after.json) audits match both environment inventories, 12 selected binaries, AMD source edits/backups, four saved turret artifacts and decoder metadata/header. The full saved 3,743-row latent and coordinates remain unchanged. Original checkpoint/config, strict loaders, native artifacts, guards, historical branches, evidence and preexisting bytecode are preserved. No WSL command, package installation, native build, GPU initialization, model load, forward or geometry operation occurred.

**UNKNOWN:** the genuine strict loading contract was not exercised. This is a demonstrated caller-input mismatch, not an incompatibility in the 292 original F16 checkpoint entries or audited 76 F32 widenings. Their load time, CPU memory high-water mark and model collection remain unmeasured.

**Smallest next action:** separately approve serializing both explicit local checkpoint paths to strings at the existing loader call boundary, with a shared platform-aware path preflight against the genuine `absolute_file()` validator. Retain every strict F16/F32 rule and safety limit. Only after that gate passes should one newly authorized genuine CPU load occur. No further native compatibility investigation or GPU neural forward is warranted by this failure. Stop here.
