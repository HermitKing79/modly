# AMD completed-worker product boundary

Base: `279f8c0925a951ca85255fb7774b9bc72487818b`. Branch: `codex/amd-completed-worker-product-slice`.

This slice connects an authenticated completed GLB to the existing generation job, workspace library and viewer URL. It does **not** implement fresh image-to-3D orchestration. No GPU inference, model loading, desktop viewer session, installation or environment changes were performed.

| Requested verdict | Result |
|---|---|
| AMD worker/product integration boundary | **PASS**, for the opt-in authenticated completion contract; no genuine producer is enabled. |
| Genuine completed-artifact promotion | **PASS — historical replay/integration test**, not a new generation. Tiny control and accepted full master copied exactly. |
| Existing UI/workflow compatibility | **PASS** for API, workflow status, library discovery/open and viewer-URL contracts. No new desktop rendering/cleanup qualification. |
| Fresh AMD image-to-3D | **NOT RUN**. Product-owned stage orchestration, input-derived profiles and GPU cancellation qualification remain required. |

## What is real, and what remains disconnected

| Capability | Established state |
|---|---|
| Image conditioning, occupancy/shape sampling, geometry and guided material inference | Genuine AMD executions demonstrated separately in historical supervised proofs. |
| CPU dual-grid extraction, source-spatial learned RGB sampling and portable GLB export | Genuine accepted full brick artifact: 1,957,891 positions, 4,173,584 triangles, 97,073,940 bytes. |
| Existing product | GeneratorRegistry, ExtensionProcess/runner IPC, image/workflow jobs, workspace assets and GLB viewer already exist. |
| Current installed AMD extension | `amd-offline-scaffold-v1`: explicit mock or alpha-only preparation; no complete image-to-GLB execution. No manifest/capability was changed here. |
| Missing connection | A product-owned orchestrator must execute the independently proven stages for the **current** submitted image, preserve their contracts, supervise cancellation/resources and issue current job-bound completion evidence. Historical brick profiles cannot authenticate an arbitrary new image. |

`amd_generation_texture_decode.py` is preparation, not a genuine PBR forward. The historical qualified PBR worker is a separate supervised composition. The successful Windows sampler and WSL native decoder environments must not be treated as one interchangeable runtime. Neither a saved GLB nor a preparation receipt proves fresh generation.

## Implemented production boundary

`modly.amd-completed-glb.v1` is an opt-in extension result contract. A trusted **host callback**, never an IPC field or user parameter, supplies the permitted root and independently pinned worker/export/host/verification evidence. The completion context binds job, request, run, nonce, model, input bytes, canonical parameters and mode.

The existing runner accepts only a typed completion descriptor for a negotiated contract. The host rejects stale/foreign requests, incomplete evidence, bad hashes, external GLB resources, malformed/bounded buffers, foreign paths, traversal, aliases, reparse/symlink paths and multiply linked input files. In `generated` mode all four proof receipts must additionally bind the current context and pass freshness checks. Historical receipts are admissible only in explicitly experimental `replay_fixture` mode. The contract assumes the future host verifier independently qualifies stage evidence; a worker-authored `PASS` string is insufficient.

Publication copies bytes in bounded chunks, hashes/fsyncs/readbacks the copy, commits provenance first and the GLB last without overwrite, and rechecks source/output identities. Cancellation cleans only this attempt's owned files. The original GLB, geometry, RGB, materials and coordinate transform undergo no conversion. The bounded implementation admits self-contained GLBs up to 128 MiB; broader formats/extensions are not silently admitted.

Results use existing `JobStatus.output_url`, with additive `result_origin` (`amd-worker` / `amd-replay`) and provenance metadata. Assets land under `Workflows/<collection>/amd-<job>/model.glb` with `model.json`, so existing workspace discovery and viewer opening apply. Errors/cancellation do not publish a successful job. The existing Import Mesh route is unchanged. No public replay model, fixed-fixture Generate option or new inference endpoint was installed.

## Verification and measured replay

See [focused CPU checks](amd-worker-product-integration-evidence/focused-tests.json), [independent boundary review](amd-worker-product-integration-evidence/independent-boundary-review.json), [IPC review](amd-worker-product-integration-evidence/independent-ipc-review.json) and [bounded replay harness](amd-worker-product-integration-evidence/replay_check.py).

The [measured replay](amd-worker-product-integration-evidence/replay-result.json) completed in **3.625 seconds**, with **46,317,568 bytes / 44.2 MiB peak Job commit**, 45,494,272 bytes peak process commit, and minimum observed Windows available RAM 17,152,692,224 bytes. Both process and Job ceilings were 6 GiB; deadline 180 seconds; live RAM/destination floors 12/2 GiB. The Job was empty before cleanup, with no forced termination. Full-file host readback after worker exit matched both original hashes. The launcher-handle PSAPI working-set reading is retained separately and is not substituted for the actual Job peak.

| Replay case | Bytes | SHA-256 |
|---|---:|---|
| Tiny control | 1,108 | `a4fe7dd08d3fd6c9f14d53c1be09f1b2a3b4ab134c4100aee381602f7a6826f1` |
| Accepted historical master | 97,073,940 | `1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae` |

Private promoted master: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-worker-product-integration-279f8c0\replay\workspace\Workflows\Replay\amd-22d0fb7b-9479-4cd7-8dff-84abec955448\model.glb`. The original accepted asset remains in its original location unchanged.

The replay harness uses the actual image-job API and ExtensionProcess completion receiver/authenticator; only its worker reply transport is replaced by a clearly labelled saved reply. It does not claim a live runner/pipe roundtrip, GPU generation or desktop rendering. The tiny control's receipts are synthetic test records; the master uses its four original independently pinned successful export/host/verification receipts. Both actual jobs report `Completed-artifact replay; no inference` and `result_origin=amd-replay`.

The combined Python run passed 152 of 154 tests, with no failures and two explicit skips: POSIX mode enforcement on Windows and unavailable symlink-creation privilege. After the final staging-identity fix, all 18 affected core tests passed again; all six final route tests passed, including the actual workflow result projection. Existing asset-library/viewer projection tests passed 21/21. These overlapping focused reruns are not additional unique test counts. Negative cases cover foreign/stale/partial/corrupt receipts, malformed GLBs, unsafe paths/links, replacement races, overwrite, cancellation and cleanup.

The [actual replay-workspace check](amd-worker-product-integration-evidence/workspace-result.json) verified both GLBs are listed/openable and resolve to the exact job viewer URL. Its sidecar check verifies artifact ID and source link through the existing read API; the replay worker separately verifies full sidecar equality with API metadata. Ordinary `model.json` is intentionally omitted from the library list. The initial extra listing assumption and corrected test are retained in [the precheck finding](amd-worker-product-integration-evidence/workspace-precheck-finding.json); no production library change or replay rerun was needed.

[Preservation after execution](amd-worker-product-integration-evidence/preservation-after.json) found zero issues: 2,280 scoped file identities match before, including all 100 protected files, original master/app inputs, the original profile, selected environment/native identities and source backups. All 117 pre-existing remote heads, including main, matched before publication. Checkpoint/model payloads were not loaded or rehashed; broader historical scopes retain their previously authenticated evidence. No private assets are included in this branch.

The previous high-resolution viewer qualification remains **BLOCKED** after forced containment. Its later cleanup qualification also remains **BLOCKED**, at an empty-viewer observer gate before any asset load, despite natural app shutdown. This slice neither repeats those investigations nor upgrades their verdicts. API/library compatibility is not a new full-resolution desktop cleanup qualification.

## Next genuine UI-generation admission

The next implementation gate is one source-pinned product orchestrator and host verifier, with no fixed-artifact fallback. It must:

- Bind the submitted original image, preprocessing/DINO, occupancy and shape latent lineage to the product context; authenticate all installed checkpoints/configuration/native sources.
- Compose existing Windows sampling and named WSL decoding workers, preserving their original precision and native-map/tiling contracts. The geometry decoder's selective BF16 contract and guided PBR decoder's selective F16 contract are distinct.
- Replace exact-brick support/hash assumptions with a separately reviewed input-derived admission contract before claiming arbitrary-image support. Preserve all original sparse support; no silent truncation or fallback.
- Propagate real stage progress, cancellation and deadlines to every owned worker/descendant, including I/O acknowledgement before GPU release. Existing synchronous extension IPC alone is not a validated GPU process supervisor.
- Run source-faithful CPU extraction, position-sampled learned colour and portable GLB export; independently verify finite tensors, coordinate/guide lineage, complete operations, exact saved bytes, resource release and current job-bound evidence before publication.

Existing relevant gates remain 18.5 GiB prelaunch / 12 GiB live Windows RAM, 13 GiB initial free VRAM, 12 GiB allocator cap, 2 GiB device reserve, 1 GiB opaque allowance, 32,768 tile rows, 50 GiB D: free and 2 GiB output free. Historical stage deadlines differ: native/CPU/GPU/host decoder 60/120/240/480 seconds plus I/O; sampling and occupancy retain their own published deadlines. CPU extraction/export/readback retains 6 GiB process **and** Job caps and 180-second workers. These are admission requirements, not measurements of a new generation.

No additional model download is justified by this inspection. Fresh GPU generation requires separate approval after that orchestration/cancellation/profile gate. Existing open/non-manifold geometry and constant portable metallic/roughness are known output limitations; this boundary does not convert vertex colours into spatial PBR textures.
