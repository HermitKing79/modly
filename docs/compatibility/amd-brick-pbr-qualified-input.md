# Brick guided PBR with the qualified WSL input adapter

Base: `a064378cd9530addc38e8be1688088777881b2fd`.
Branch: `codex/amd-brick-pbr-qualified-input`.

**PASS — one genuine guided F16 PBR forward completed.** The full brick material
field is saved and independently verified before GPU release and after natural
worker exit. No retry, geometry regeneration, baking or rendering occurred.

The exact execution harness was published and GitHub-verified at
`9924445f0183788dbac4ee928e5207cec3fe7dfd` before the attempt. The result publication
adds evidence and documentation; it does not change the executed implementation.

| Actual result | Measurement |
| --- | --- |
| Genuine parent forwards / original convolutions | 1 / 40 |
| C2S operations / authenticated native maps | 8 / 5 |
| Six large-convolution sites / final convolution | 84 / 60 completed tiles |
| Final support | 1,957,891 rows; original geometry coordinate order |
| Strict checkpoint / runtime parameters | 284 entries; 208 F16 / 76 F32 |
| Synchronized forward duration | 14.059039387 seconds |
| Entire GPU stage / worker / supervisor | 27.587835607 / 64.010420516 / 84.015 seconds |
| Complete host invocation, including preflight | 155.3313668 seconds |
| Peak Torch allocated / reserved | 3,367,268,864 / 4,410,310,656 bytes |
| Initially free VRAM / lowest sampled admission free VRAM | 15,713,161,216 / 11,826,372,608 bytes |
| Lowest sampled Windows available RAM | 14,986,592,256 bytes (13.957 GiB) |
| Guest process memory high-water mark | 2,912,833,536 bytes |
| Post-release Torch allocated / reserved | 0 / 0 bytes |

The three fresh prelaunch RAM readings were 21,639,503,872, 21,642,563,584 and
21,640,773,632 bytes; the pre-marker reading was 21,593,247,744 bytes. D: free
space never fell below 74,230,857,728 bytes in the saved samples. All 200 saved
GPU admission checks and 84 host-resource samples passed. All 16 saved Windows
health observations were healthy. These are actual observations, not projections.
The 14.621497981-second journal value is elapsed GPU-stage time at the synchronized
forward event, distinct from the 14.059039387-second forward duration.

Saved private artifact:
`C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-pbr-qualified-a064378\field\texture-field.safetensors`

- Bytes: **125,318,120**.
- SHA-256: `8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394`.
- Raw and transformed material fields: finite F32 `[1957891,6]`, channels RGB,
  metallic, roughness and alpha. Original I32 `[1957891,4]` coordinates match the
  geometry byte for byte. The original transformation is preserved without clamp.
- Sidecar: `texture-field-metadata.json`, 13,493 bytes; SHA-256
  `4f65c6730f4a8669d9e62ba4f151f15b19f6cfd6b7e7dcd81289ad923bbdd8c2`.

The independent Windows receipt was acknowledged before tensor release. A fresh
CPU process and the independent reviewer both subsequently reopened the complete
artifact. Model/input owners were collected, hooks and native/convolution patches
restored, the original workspace cleanup ran once, and the worker exited with code
0. The owned Windows Job emptied naturally; no forced cleanup was used. Only the
named Modly distro was terminated. No visual material-quality claim follows.

Actual evidence: [outcome](amd-brick-pbr-qualified-input-evidence/actual-outcome.json),
[worker result](amd-brick-pbr-qualified-input-evidence/run/guest/results.json),
[operation journal](amd-brick-pbr-qualified-input-evidence/run/guest/phases.jsonl),
[supervisor result](amd-brick-pbr-qualified-input-evidence/run/results.json),
[fresh post-exit readback](amd-brick-pbr-qualified-input-evidence/independent-post-exit-readback.json),
[independent outcome review](amd-brick-pbr-qualified-input-evidence/independent-outcome-review.json).

Fresh host/worker copies select the exact previously qualified read-only input
adapter, manifest and five-file Windows SID/ACL/file-ID proof. CPU and GPU paths
share `prepare_inputs(authenticated_profile, verified_plan_context, record)`.
Historical readers, model arithmetic, source/checkpoint identities, precision,
tiling, native dispatch, receipt acknowledgement and cleanup remain unchanged.

The actual production worker's CPU route passed under genuine WSL UID/GID 1000:
all 123 candidate source identities and the saved geometry contract were verified;
16 input-security checks passed; all four original guides and full coordinate order
were authenticated. All 15 prepared arrays were collected. Worker time was
**7.674588246 seconds**. No Torch, model, native GPU import or GPU entry occurred.
The Windows observer was healthy, the guest exited normally and the Job emptied
without forced cleanup. Only a CPU marker was used.

**29 focused host/worker tests passed.** Independent review found no source blocker
and granted GO conditional on fresh admission. Evidence:
[CPU verdict](amd-brick-pbr-qualified-input-evidence/cpu-input-acceptance/CPU-verdict.json),
[actual worker](amd-brick-pbr-qualified-input-evidence/cpu-input-acceptance/guest/results.json),
[independent review](amd-brick-pbr-qualified-input-evidence/pre-execution-review.json),
[source closure](amd-brick-pbr-qualified-input-evidence/candidate-identities.json).

The fresh permanent output prepared before execution was:
`C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-pbr-qualified-a064378\field`.
Its ownership, file IDs, alias checks, free space and exclusive 42-byte fsync/readback
probe passed; the field was empty at candidate publication. A fresh production
marker and execution plan were used. Original limits remained: 18.5/12 GiB prelaunch/live
Windows RAM, 50 GiB D: free, 2 GiB destination free/write access, 13 GiB initial free
VRAM, 12 GiB allocator cap, 2 GiB device reserve, 1 GiB opaque allowance, 32,768 tile
rows, native/CPU/GPU/host deadlines 60/120/240/480 seconds and original I/O deadlines.

Completed required computation: 284 strict checkpoint entries, 208 F16/76 F32 runtime,
one forward, 40 convolutions, eight C2S calls, five global native maps, 84 large
tiles and 60 final tiles; exactly 1,957,891 original coordinate rows. A PASS also
requires finite persisted raw/transformed material fields, independent Windows
acknowledgement before GPU release, post-exit readback, zero owned Torch bytes,
healthy monitoring and natural process settlement. No visual-quality claim follows.

The [BEFORE audit](amd-brick-pbr-qualified-input-evidence/preservation-before.json)
and [AFTER audit](amd-brick-pbr-qualified-input-evidence/preservation-after.json)
confirm all 100 protected files and 400 scoped identities unchanged, including
original inputs, historical evidence/markers, selected installed-environment
metadata/native binaries and AMD edits/backups. The audit records its exact hash
versus metadata-only scope. No private model output or image is committed. The
original 84 GitHub heads are snapshotted privately and checked at publication.

The next separately approvable step is a CPU-only learned-PBR preview on the
accepted brick mesh, first confirming vertex/material-row correspondence. Stop
here: no baking, rendering, geometry changes, resampling or follow-on execution.
