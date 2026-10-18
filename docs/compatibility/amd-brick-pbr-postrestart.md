# Brick guided PBR: post-restart execution

Base: `4d1af7f79aee1a40732ccc2d805fc5320416a376`. Branch: `codex/amd-brick-pbr-postrestart-decode`.

## Pre-execution qualification

Status at harness publication: **PREPARED**. Actual outcome: **BLOCKED before WSL launch**. This authorizes no retry. The experiment composes the unchanged CPU-qualified `inference_texture` implementation with the original owned Windows/WSL supervisor, exact binary GO, healthy periodic monitoring, natural Job settlement and named-distro termination.

The original 18.5 GiB prelaunch requirement applies. Three initial post-restart Windows readings were 22,647,754,752 / 22,644,805,632 / 22,645,833,728 bytes. Independent host review subtracts the historical 7,581,499,392-byte whole-host decline and retains about 2.03 GiB above the original 12 GiB live floor. That comparison is an empirical screen, not a brick RAM bound. Fresh measurements, recent downward RAM changes and host-controlled release barriers after native preparation and strict CPU loading remain mandatory.

Limits remain 13 GiB initially free VRAM, 12 GiB allocator ceiling, 2 GiB device reserve, 1 GiB opaque allowance, 32,768 rows per tile, 50 GiB D: free, 2 GiB destination free, and native/CPU/GPU/host deadlines of 60/120/240/480 seconds. Original precision remains 208 F16 / 76 F32 parameters from all 284 checkpoint tensors.

The authenticated `brick-1957891-v1` inputs retain support 4,430 → 21,391 → 97,313 → 438,684 → 1,957,891. Acceptance requires one parent forward, 40 convolutions, eight C2S operations, five native maps, 84 large-site tiles and 60 final tiles. Production persistence requires genuine ownership checks, independent Windows acknowledgement before GPU release and fresh readback after natural exit, plus measured zero owned Torch allocated/reserved bytes.

Twenty-three focused CPU adapter tests passed (12 host and 11 worker). The complete supervisor lifecycle test uses explicitly simulated CPU receipts; it is not native or neural execution evidence. Independent review resolved stale-bytecode and early output-routing issues and found no remaining blocking code defect, conditional on the committed source closure and live gates. The 103-file candidate manifest is checked against both current source and committed Git blobs. Existing CRLF checkout files may match LF Git blobs only with exact CRLF-to-LF equivalence, verified Git EOL metadata, core.autocrlf=true, and no filter/encoding/ident transformation. Runtime source hashes remain byte-exact; protected files are untouched. No qualified decoder/core source was changed.

Evidence: [CPU tests](amd-brick-pbr-postrestart-evidence/cpu-test-summary.json), [host-memory review](amd-brick-pbr-postrestart-evidence/host-memory-review.json), [independent review](amd-brick-pbr-postrestart-evidence/independent-preexecution-review.json), [source manifest](amd-brick-pbr-postrestart-evidence/candidate-identities.json), [preservation before](amd-brick-pbr-postrestart-evidence/preservation-before.json).

Fresh private destination: `C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-pbr-postrestart-4d1af7f\field`. The empty destination has real Windows SID/ACL/file-ID attestation and a successful exclusive, durable, exact-readback write probe. Private model outputs and images remain outside Git.

## Actual result

Execution harness `710e995f4ec44f59d6e46525e7fff3bdb6fb3efb` was published and verified before the single host launch. All 80 pre-existing remote heads were unchanged.

Three fresh Windows RAM readings were 22,494,556,160 / 22,484,410,368 / 22,468,280,320 bytes (20.93–20.95 GiB). Final pre-marker availability was 22,414,417,920 bytes (20.88 GiB), D: free was 74,269,659,136 bytes and destination free was 112,657,211,392 bytes. These resource gates passed. The unchanged 12 GiB live floor was not tested under a running WSL decoder.

The real plan-building path raised `ValueError: Canonical approved C: path required`. `reserve_attempt()` first published the exclusive production marker, then attempted `to_guest(GEOMETRY/'run-contract.json')`. That authenticated saved contract lives in the original neural artifact directory, outside the adapter's approved root list. Plan publication, host marker creation and `run/` creation were never reached. The saved external marker remains intact; the output field directory remains empty.

This is a confirmed **adapter integration defect**, not a RAM failure or AMD/F16/native compatibility result. No WSL worker, Torch runtime, model load or GPU forward started. There is no material field, GPU runtime, peak VRAM or measured Torch release to report. Host exit was 1; no worker process required forced cleanup. No retry occurred.

The CPU tests and earlier independent review missed the complete real plan serialization: the lifecycle simulation replaces `reserve_attempt()`, and command-routing tests do not exercise that function with the actual saved geometry path. Passing lower-level tests therefore did not establish the required end-to-end launch readiness. The [independent outcome review](amd-brick-pbr-postrestart-evidence/independent-outcome-review.json) confirms this coverage gap and the BLOCKED result. The failed candidate is preserved unchanged for diagnosis.

Exact error and measurements: [actual outcome](amd-brick-pbr-postrestart-evidence/actual-outcome.json), [traceback](amd-brick-pbr-postrestart-evidence/unhandled-host-failure.json), [fresh readings](amd-brick-pbr-postrestart-evidence/fresh-prelaunch-readings.json), [durable resource journal](amd-brick-pbr-postrestart-evidence/phases.jsonl). A sandbox-only CIM query restriction was preserved separately; the identical read-only ownership preflight passed outside that sandbox before the real host launch. The [additive commit-representation review](amd-brick-pbr-postrestart-evidence/independent-commit-normalization-review.json) supersedes the initial adapter hashes for the published candidate.

The [independent AFTER preservation audit](amd-brick-pbr-postrestart-evidence/preservation-after.json) found zero issues: all 400 scoped identities match BEFORE, including fresh full hashes for all 100 protected files, 16 qualified CPU sources, existing caches, environments and AMD edits/backups. Models, latent, geometry and historical artifacts remain intact. No private images or model outputs were published.

**Next step:** narrowly qualify the exact original geometry-contract guest path and CPU-test the complete production plan serialization before any new marker. Use a fresh reviewed execution candidate only after separate approval. Do not broaden path access or rerun the decoder automatically. No baking, rendering or geometry change was performed.
