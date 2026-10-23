# Upstream image-to-shape job integration

Baseline: `5501dfd43263a9a155bed50413264b692c171905`. Branch: `codex/amd-upstream-image-shape-integration`.

| Gate | Verdict |
|---|---|
| Upstream job-context integration | **PASS — CPU host interface** |
| Image-to-shape lineage authentication | **BLOCKED for genuine execution; CPU identity/byte-chain checks pass** |
| Current geometry admission compatibility | **PASS — CPU only** |
| Historical geometry preservation | **PASS** |
| Genuine upstream neural execution | **NOT RUN** |
| WSL orchestration | **NOT QUALIFIED** |
| Fresh image-to-3D generation | **NOT RUN** |

## Implemented boundary

`api/services/amd_upstream_job.py` supplies a non-executing host owner. It freezes the actual image hash, canonical parameter object, exact `trellis2-amd` model identity and original product job/request/run/nonce. An exclusive private request and nonce reservation prevent reuse. Host-selected source pins, runtime image/input-contract pins and output expectations are fixed before any stage output exists.

Conditioning, occupancy and shape receive ordered, opaque, one-use tickets. Before a ticket is issued, every source, runtime input, predecessor receipt and artifact is reopened and authenticated. Acceptance requires a typed **trusted-host** observation matching the exact ticket, stage, full context and nonce, with successful natural settlement. Worker JSON or a claimed PASS cannot substitute for that observation. Outputs must be newly created after their ticket and satisfy stable canonical/single-link/file-identity/hash/size checks. Any failed acceptance, issuance, completion or cancellation permanently blocks the owner.

Completion writes a **new sidecar**, retaining the original shape receipt and NPY bytes. Only the exact sealed object registered with its issuing owner can supply the trusted completed-job digest; copied seals, foreign jobs and uninitialized objects fail. Sidecars explicitly state `CPU_CONTRACT_ONLY` and `genuine_execution_qualified=false`. They authenticate bytes and host association, not neural computation.

`prepare_current_input(..., upstream_completion=handoff)` validates that authority before and after the unchanged original shape/occupancy readers. It permits a raw shape receipt without a product-context field only through this sidecar; a conflicting context is still rejected. Actual F32 latent and I32 coordinate reads/order remain independently verified.

The new route also avoids requiring publication of the private submitted image: only the **exact two** authenticated occupancy image/input-contract paths are removed from the inherited Git source closure. Their real bytes, hashes, freshness and paths must match both the host pins and original occupancy reader. Code, policy and all other source checks remain required. The historical route is unchanged.

## What remains blocked

Existing proof workers do not yet supply product-owned observations through a qualified production execution adapter. The trusted host must produce these observations from the actual owned supervisor, never deserialize them from worker messages. The original shape verifier also needs a complete pinned checkpoint/config and five-artifact semantic receipt closure before the new owner can be used as genuine execution evidence. Byte authentication alone cannot prove that a network ran or used requested parameters.

No executor, launcher, registry entry, successful `amd-worker` simulation or Generate capability was added. The generated orchestrator block and all current-image GPU entry denials remain intact. A serialized sidecar cannot recreate the in-memory authority after restart. Genuine execution readiness is therefore **BLOCKED** despite passing CPU interfaces.

## Verification and preservation

[38 distinct focused checks pass](amd-upstream-image-shape-evidence/qualification.json); selected final runs total **22.47 seconds**, including interpreter startup. An [independent reviewer](amd-upstream-image-shape-evidence/independent-review.json) repeated **28 distinct checks** with no unresolved CPU-contract finding. Tests cover request/parameter/model substitutions, reused nonces/tickets, stale/pre-existing/changed files, incomplete and conflicting receipts, copied authority, cancellation and actual byte/NPY handoff. Synthetic predecessors are clearly labelled; neural verifiers are substituted only in bridge wiring tests. No genuine inference proof is claimed.

The affected ten bridge regressions were checked; the old 95-test geometry suite was **not repeated**. One exploratory existing router-test selection stopped progressing and was interrupted with **no verdict**; it was not retried or repaired. Three bounded checks independently confirm the generated gate, disabled unopted AMD discovery and ordinary legacy generator discovery. This is not a claim that the full router suite passes.

[Source preservation](amd-upstream-image-shape-evidence/source-preservation.json) confirms **19 complete sources and six historical host functions/classes unchanged**, including original occupancy/shape implementations, geometry arithmetic/contracts/resource admission, registry, router, runner and generated-dispatch block. [Preservation receipts](amd-upstream-image-shape-evidence/preservation-review.json) record the protected100 scope, accepted-asset provenance, historical evidence, caches and previous remote heads. No bulk checkpoint/GLB rehash, new GPU/RAM peak measurement or environment modification is claimed. No GPU, WSL, neural model, native fixture or OS Job ran.

Next: separately implement and CPU-qualify the real upstream observation/semantic-receipt adapter against the existing supervisors and pinned model/config/artifact closure. Do not launch neural inference until that gate is independently accepted.
