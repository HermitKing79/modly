# Repeatable AMD Generation v1 implementation plan

**Goal:** One new image produces genuine geometry and learned material data, then a verified GLB through Modly's Generate path. Existing turret assets are never substituted.

**Approved intent:** User requested independent execution, preservation, cheapest new-image proof before frontend integration, and a stop on unsupported inference. Architecture discovery found fixture-specific wrappers rather than a connected engine; independent review confirmed their arithmetic is reusable.

**Preserved baseline:** `6ea0abb20b3391ff3dd4daceeb2e1a9db903afa2`; branch `codex/amd-repeatable-generation-v1`. One hundred protected files verified; twenty local-only files backed up.

- [x] Discover application/runtime topology and reproduce non-mock admission rejection without GPU work. This is a diagnostic checkpoint, not an inference verdict.
- [x] Parameterize the existing Windows DINO → sparse sampling → occupancy coordinator with one authenticated new input and exclusive evidence. Published at `189c670cba769ad3274b0a535b761d5873826f51`; genuine new brick-image sequence passed with healthy observation and zero release. This is occupancy only, not full generation.
- [ ] Derive canonical32³ coordinates from those outputs. Measured N=4,430; CPU qualification replaces the historical fixture-derived 3,743 initial bound with an explicit adapter-only 4,430 bound. Historical defaults stay unchanged. Keep12steps/21forwards, canonical seed/settings and all original memory limits, plus adapter-only 2GiB free +1GiB opaque live headroom. Reject4,431 before loading shape weights. Publish reviewed candidate, then run once.
- [ ] Parameterize the proven WSL shape decoder with authenticated fresh shape arrays, actual learned counts and unchanged math. Early support remains≤1M, final support≤1,842,968, tile32768,40convolutions/eight C2S, same precision/native identities. Retain live memory admission,13GiB initial VRAM,12GiB allocator,2GiB reserve/1GiB opaque headroom and existing deadlines.
- [ ] Convert genuine saved neural output with the existing CPU flexible dual-grid converter. Validate and preserve mesh before spending on texture stages. An untextured GLB is only intermediate evidence.
- [ ] Parameterize existing texture sampling and guided decoder using this same job's shape/data/subdivision lineage. Save and independently verify learned material output. Reuse Blender compact learned-colour export; retain raw PBR data separately.
- [ ] Only after the CLI proof succeeds, add a real experimental AMD runtime mode to the existing Generate HTTP/subprocess lifecycle; reconcile common parameters, stage-aware deadlines, one-GPU-job lock, progress, cancellation and result loading. Use existing viewport/GLB download, not a new application.
- [ ] Focused CPU tests and independent review for every changed contract. At most the necessary genuine new-image proof and subsequent application-backed validation; stop on any unsupported native/resource failure, without automatic inference retries. Report PASS only for the full button/viewport/export acceptance.

## Worker interfaces

Windows occupancy adapter: `amd_generation_occupancy.py --image PATH --evidence DIR --harness-commit SHA`; `--prepare-only` validates/stages source contract without model imports. Fresh outputs keep existing stage receipts and authenticated paths.

Windows shape adapter: `amd_generation_shape.py --occupancy-evidence DIR --evidence DIR --harness-commit SHA`; reads preceding PASS receipts and hashes, derives pooled coordinates and fresh source-faithful noise. Existing proof defaults stay unchanged.

WSL neural adapter: `inference_decoder/decode.py --run-contract PATH`. Schema `modly.amd.neural-run.v1`: initial_count≤4430, input_paths/input_sha256 keyed `shape-latent-denormalized.npy`/`coords.npy`, source_job_sha256, exclusive WSL-owned output_directory and scratch_directory, guest repo_root, native_stage, fixed intermediate/final caps. Host exports only independently verified owned results to the job's Windows directory.

No historical report/receipt is manufactured or rewritten. New run-specific references replace fixture references; source/model audits remain immutable. Reject forged provenance, mismatched coordinates, unsupported counts, ambiguous paths, unsafe allocation plans and incomplete cleanup. CPU tests must establish those boundaries before new GPU work.
