# Brick guided PBR CPU integration

Base: `3942181cec1aaac2e38698f08c61a68a69af4851`. Branch: `codex/amd-brick-guided-pbr-cpu`.

This implements the user's approved bounded CPU integration. No GPU, WSL worker, native binding, real checkpoint loading, installation, download or model forward is authorised. Torch may be imported only in isolated CPU tests with GPU-entry functions denied. Preserve all historical sources, artifacts, environments and branch heads. New reusable code belongs to `experimental/extensions/trellis2-amd/inference_texture`; tests/evidence remain focused on this path.

## Shared contract

`profile.authenticate(profile_name, geometry_contract_document)` returns an immutable, revalidated brick-only Profile. It exposes `initial_rows=4430`, `parent_rows=(4430,21391,97313,438684)`, `child_rows=(21391,97313,438684,1957891)`, `final_rows=1957891`, `run_contract_sha256` and `document()/sha256`. Bind the exact original geometry contract, texture latent, guide file, coordinate identity, texture checkpoint/config and source identities. A profile string or larger row count alone never grants admission.

`inputs.prepare(profile, texture_path, geometry_path)` uses genuine existing readers; returns original denormalized F32 texture latent/coordinates, four untouched BF16 guide records, final coordinates and authenticated metadata. `inputs.GuideOrderValidator(prepared)` supplies the historical `before_c2s/after_c2s/finalize` interface with exact eight-call lineage and coordinate/mask checks. No historical global constants are changed.

## Tasks

1. Preservation: snapshot current protected/environment/artifact identities, then compare after. Root owns Git branch/publication. Preserve four preexisting untracked bytecode directories.
2. Profile/input/artifacts: implement the shared contract and bounded writer/independent reader opt-in. Historical no-profile behavior remains unchanged. Preserve existing exclusive-write, ownership, symlink/hardlink, confinement, finite-value, transform, integrity and provenance checks. Use source-pinned isolated copies/wrappers where protected modules cannot change. Test authentic inputs, malformed/foreign profiles/artifacts and bounded synthetic serialization/readback; label synthetic data explicitly.
3. F16 core: reuse the genuine frozen F16 row helper. Exactly six 128-channel large sites, 14 tiles each (84 total), followed by 60 final64 tiles, last24579. Keep global native map indices, cache identity, full source row order, original weights/bias and source F16/F32 semantics. Source-pinned isolated hooks/runtime avoid modifying historical generic modules. Test mathematical equivalence, cross-tile/sentinel/bias/nondivisible coverage, site/order/native provenance, admission refusal and cancellation/restoration.
4. Integration: compose one usable neural path, memory plans and output contract; no GPU CLI/run. Test CPU control paths with representative fixtures. Account for full model/guides/maps/caches/retained tensors in baseline and additional allocations under original formula. Admission must precede full output and each tile allocation.
5. Independent review: review source equivalence, precision, global-map/cache ownership, memory assumptions, consumer authentication and cleanup. Fix justified CPU defects. Publish only source-safe textual/code evidence and a truthful PASS/BLOCKED/NO-GO readiness report.

## Fixed limits

12GiB allocator, 13GiB initial free VRAM, 2GiB device-free reserve, 1GiB opaque headroom, original1.25 padding, <=32768 tile rows; unchanged native/CPU/GPU/host deadlines (GPU240s). No truncation, altered masks/support, BF16 decoder substitution, cap increase, autotuning or compiler/native build. Full output is raw/transformed F32[1957891,6] + I32[1957891,4],125305024 payload bytes. Numerical CPU fixtures are not genuine model outputs or GPU evidence.

## Ownership

Artifact implementer owns `__init__.py`, `_source.py`, `profile.py`, `inputs.py`, `artifact.py` and their focused tests. Core implementer owns `tiling.py`, `admission.py`, `runtime.py`, `neural.py` and core tests. Root coordinates interfaces, qualification CLI/evidence, memory review, publication and final result. Do not spawn additional agents or commit independently.

## Accepted independent-review corrections

- I1: A callback interface alone does not qualify output persistence. Add a small concrete ready/independent-host-ack bridge using the original 30-second protocol, fresh exact-leaf Windows SID/ACL/file-ID attestation, marker/path/profile/hash matching and real CPU child-writer/parent-reader acceptance. Artifact implementer owns `receipts.py` and focused tests. No WSL/GPU execution or general transport manager.
- I2: File hashes and `co_filename` alone do not authenticate executed code. Core implementer owns a narrow source-code qualifier comparing actual callable code/dependencies to compiled pinned source; spoofed-code-with-genuine-filename CPU regressions must fail before placement/native entry. No model imports or native builds.

The initial BLOCKED review remains immutable. Re-review only the corrective diff and covering CPU evidence, then make the final readiness decision. Actual GPU/resource/native measurements remain future live gates, never CPU success claims.
