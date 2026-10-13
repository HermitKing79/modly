# First genuine sparse-structure sampling — PARTIAL

2026-10-13. Branch `codex/amd-sparse-structure-sampling`, based on
`38e73a5e02476a597ab0b91bd4f2fc58754baf4d`.

**No GPU sampling occurred.** The parent launcher exited with status 1 at
`amd_blaslt_teardown.verify_workspace()`: the invocation omitted
`WORKSPACE_DIR=C:\Users\wattz\Documents\ChatGPT\modly\proof`.
The complete observed stderr and invocation are preserved in
[launch-result.json](amd-sparse-structure-sampling-evidence/launch-result.json) and
[launcher.stderr.txt](amd-sparse-structure-sampling-evidence/launcher.stderr.txt).
This was a launcher configuration safety failure, not a native computation,
model-state, GPU transfer or sampling failure. No retry was performed.

The failure precedes parent preflight, fresh execution RAM admission, attempt
marker creation, owned-worker launch and all model/GPU operations. Neither
`attempt-started.json` nor `run/` exists. Real checkpoint loads, GPU
initializations, completed sampling steps and forwards are all **zero**.
No final latent, synchronized GPU runtime, VRAM measurements or GPU release
evidence exists for this milestone. Previous successful computations and
historical PARTIAL classifications remain unchanged.

## Completed preparation

The reviewed harness was committed and
[published as 003d98165039a2b0aac02fb8c0d376ba6d429d9f](https://github.com/HermitKing79/modly/commit/003d98165039a2b0aac02fb8c0d376ba6d429d9f)
before the launch. [Publication evidence](amd-sparse-structure-sampling-evidence/harness-publication.json)
records matching local, `git ls-remote` and GitHub API identities.

[Passive preflight](amd-sparse-structure-sampling-evidence/passive-preflight.json)
verified the genuine 640-entry sparse-flow checkpoint SHA256
`ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`,
configuration, source hashes and saved DINO feature SHA256
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`.
Windows observation was healthy. Available RAM was **18,200,150,016 bytes**
against the unchanged **11,411,181,632-byte** requirement. This passive sample
does not replace the unexecuted fresh pre-marker/pre-load admissions.

The pinned real `FlowEulerGuidanceIntervalSampler` uses sigma_min=1e-5,
steps=12, rescale_t=5, guidance_strength=7.5, guidance_rescale=0.7,
interval=[0.6,1], dino_lock=0. Gaussian noise is CPU-generator seed20261013,
F32 `[1,8,16,16,16]`; positive conditioning is the saved F32 `[1,1029,1024]`
tensor and negative conditioning is `zeros_like(cond)`.

| Step range | Required complete forwards | Actually executed |
|---|---|---|
| 1–10 | Positive then negative: 20 total | 0 |
| 11–12 | Positive: 2 total | 0 |

All **47 focused CPU tests passed**: [17 installed-runtime regressions](amd-sparse-structure-sampling-evidence/cpu-final-reviewed.json)
and [30 host regressions](amd-sparse-structure-sampling-evidence/focused-host-reviewed.json).
They exercise the real sampler/CFG arithmetic with synthetic CPU fixtures,
actual production module/factory/dispatch functions, exact schedule, root
identity, 91 counters per invocation, one frequency transfer per admitted root,
23rd-call rejection, history collection, supervisor protocol/deadlines and
bounded exclusive artifact integrity. GPU initialization, backend execution and
real-weight reads were blocked in the runtime replay. Its 68 compact events
pass production journal validation under the worker cap192.

[Independent read-only review](amd-sparse-structure-sampling-evidence/independent-review.md)
found no remaining blockers after adding validated sigma_min artifact metadata.
CPU tests also caught and corrected the new profile's deadline/CPU-gate selection.
Neither correction weakens safety checks. Earlier red evidence remains saved.

The candidate retains CPU/GPU/overall limits **180/240/480 seconds**,
13GiB available VRAM admission, 12GiB allocator cap, periodic healthy Windows
checks, owned Job containment and zero allocated/reserved cleanup. Timestep
construction stages only the inspected sampler factory through the proven
explicit CPU-to-GPU copy. The final F32 latent writer caps the file at256KiB,
stores all sampling/source/feature provenance and validates a readback hash.
These GPU contracts remain untested for an actual sampling sequence.

## Preservation and one next experiment

[Structural preservation evidence](amd-sparse-structure-sampling-evidence/preservation-summary.json)
confirms identical installed dependency inventories and byte-identical raw
venv/model/cache metadata, **3,455 historical evidence files** unchanged, and
unchanged AMD source edits/backup, main and all historical local/remote branches.
The production experimental worker, strict loader and installed applications
were not changed. No downloads, installations, native rebuilds, DINO rerun,
decoder/shape execution, mesh operations, textures or GLB output occurred.

The cheapest falsifiable next step is **only the same bounded12-step experiment**:
set the explicit checkout `WORKSPACE_DIR` in the parent launch environment,
verify the existing workspace check CPU-only, then obtain approval to launch
the exact reviewed/published harness with new exclusive evidence and all
unchanged gates. A further infrastructure or native compatibility investigation
is not justified by this failure. Do not proceed to occupancy decoding until
genuine sampling, artifact integrity and zero-allocation release pass.

Stopped after this pre-execution safety failure; no automatic retry or additional
model execution.
