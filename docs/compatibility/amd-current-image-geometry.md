# Current-image AMD geometry admission v1

Baseline: `c6345043b7aa15425161fa53d99ac0f922faf12d`. Branch: `codex/amd-current-image-geometry-admission`.

| Gate | Result |
|---|---|
| Current-input geometry admission implementation | **PASS — CPU qualified** |
| CPU mathematical equivalence | **PASS** |
| Resource-bound enforcement | **PASS — CPU checks; live GPU admission remains mandatory** |
| Historical brick preservation | **PASS** |
| Upstream product-context producer | **NOT QUALIFIED** |
| Guided material inference for a new image | **NOT QUALIFIED** |
| Full AMD orchestrator | **NOT QUALIFIED** |
| Fresh image-to-3D GPU run | **NOT RUN** |

## Product change

The separate `modly.amd.current-image-neural-run.v1` contract binds the submitted image/parameters and complete product job/request/run/nonce context to an independently supplied completed shape-job hash, both NPY file hashes, denormalized F32[N,32] features, I32[N,4] coordinate payload/order, image input-contract digest and inherited source/checkpoint/native/runtime/precision identities. Nested bindings are immutable. These pins are expected policy identities, not freshly verified installed source/runtime/checkpoint bytes.

`amd_generation_decode.prepare_current_input()` calls the existing full shape/occupancy lineage verifiers, checks fresh single-link canonical files and the actual request, and requires the predecessor receipt's exact product context. A same-image receipt from another job does not qualify. Legacy receipts lacking that context cannot enter this path. The completed-job hash must come from trusted host stage authority, never worker claims or user parameters. Temporary inherited verifier routing is restored on success and failure.

The contract connects to existing support admission, provenance and ten-tensor output persistence. Independent readers verify finite output, exact metadata, starting-coordinate identity and every original C2S coordinate chain. The existing product stage reader accepts the resulting metadata without changes.

This is **not GPU execution authority**: the CLI execution route, worker and neural entry reject current-image contracts before path claims or AI imports. No Generate option was enabled. Upstream stages still need genuine job-owned context propagation before a real current-input execution can be qualified.

## Historical versus resource limits

Generic limits remain initial **4,430**, intermediate **1,000,000**, final **1,842,968** rows. The exact **1,957,891** extension, input hashes and parent tuple remain exclusive to `brick-1957891-v1`. Row caps are a qualification envelope, not a hardware-capacity guarantee.

Convolution/C2S/tiling arithmetic and selective 216 BF16 / 76 F32 policy are unchanged. Actual learned children are checked before either expansion. Full global maps, outputs and tile scratch retain preallocation admission; input indices remain global. The six large sites, 32,768-row tile maximum, 12 GiB allocator, 13 GiB initial free requirement, 2 GiB device reserve, 1 GiB opaque allowance, 1.25 padding and deadlines are unchanged. Earlier untiled operations can still fail live admission within these row caps.

At the generic final cap the source-derived final64 plan has **57 tiles**, last **7,960 rows**, **3,069,154,208 visible bytes** and **3,836,442,760 padded bytes** before map allocation. Linked maximum parent support bounds ten-tensor payload at **123,439,072 bytes**, below 256 MiB. These are CPU arithmetic projections, **not measured GPU peaks or promised completion**. [Independent review](amd-current-image-geometry-evidence/independent-review.json) records full assumptions.

## Evidence and limits

[95 distinct focused CPU tests passed](amd-current-image-geometry-evidence/cpu-final/result.json), in **31.02 seconds** including interpreter startup across four test processes; an independent reviewer repeated **78 distinct checks**. Coverage includes wrong image/job/nonce, stale receipts, foreign coordinates, policy/source changes, oversized intermediate/final counts before allocation, tiled versus untiled references, cache/lifecycle rules, actual tiny safetensors writing/readback, stage-reader interoperability and execution denial. F32 uses established tolerance; exactly representable BF16 fixtures agree exactly.

Fixtures are labelled synthetic. No current genuine shape job or full-sized new output was produced. Bridge wiring tests substitute the already-qualified predecessor verifier; they do not claim an end-to-end inference proof. No WSL, GPU initialization, AI weights, FlexGEMM binding, installation, download or production attempt marker ran. Tests used existing API and Windows AMD interpreters; no fresh WSL-runtime qualification or process-memory peak is claimed.

[Source comparison](amd-current-image-geometry-evidence/historical-source-preservation.json) confirms nine mathematical/product sources and eleven historical host/acceptance functions/classes unchanged. [Preservation](amd-current-image-geometry-evidence/preservation-review.json) verifies protected100 metadata/identities, 78 small hashes, accepted-master saved hash provenance, prior evidence, caches, main and all 126 prior remote heads; it does not claim a new bulk model/artifact rehash. Private generated assets remain local.

Next: separately qualify genuine upstream product-context propagation and the complete current-input worker integration before requesting one GPU attempt. Material inference and orchestrator execution remain separate gates.
