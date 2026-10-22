# Final high-resolution resource cleanup qualification

**Overall: BLOCKED before asset loading.** The single authorized Modly 0.4.3 session reached the original empty viewer, but the mandatory ownership gate failed with **`Error: Missing installed closure attributes`**. Source existence of Three's constructor-owned WebGLAttributes did not establish its accessibility through this V8 function scope. No tiny fixture or master was imported, and no alternate capture method or retry was attempted.

**Orderly shutdown passed for this empty preflight session.** Normal Electron `app.quit` was requested at **70.7118 seconds**, after the failed gate. Observer restoration completed at **70.7138 seconds**; before-quit and will-quit receipts followed. The app returned0 and the owned Job was already empty, with **no supervisor-forced cleanup**. Total session **76.281 seconds**, within the360-second cap and well before the240-second latest shutdown initiation.

Base `bc9515fbbdc8e28cdc58b47bc1b32f4ed3369ad4`; isolated branch `codex/amd-high-resolution-cleanup-qualification`. This report preserves the previous five camera/replacement PASS results and the previous forced480.75-second result unchanged. No production source changed.

## Independent verdicts

| Requirement | Result | Scope |
| --- | --- | --- |
| High-resolution model-specific cleanup | BLOCKED | No master load; new scene/disposal/BVH/GPU-buffer observations are UNVERIFIED. |
| Tiny-fixture cleanup | BLOCKED | No tiny load; no new resource lifecycle exercised. |
| Cache/reference release | BLOCKED | Exact module-cache and loader scope capture progressed, but renderer-attribute capture failed before observer installation. No loaded-cache invalidation or active-reference census was performed. |
| Orderly app shutdown | PASS | Empty preflight session only; original app quit returned0, owned Job0 before/after cleanup, no supervisor force. |
| High-Resolution Original Mesh workflow | Partial acceptance retained | Prior import/display/colour/statistics/export/camera/replacement evidence stands. Complete model-owned cleanup remains unverified; full workflow acceptance is not claimed. |

## Source review and falsifiable criteria

Installed versions were pinned: Three0.171.0, React Three Fiber8.18.0, Drei9.122.0, suspend-react0.1.3, React18.3.1 and the installed postprocessing/BVH sources. The actual renderer bundle matches SHA-256 `4d87e2a24b88510a55689f40755adb21e73cffd0499f5365d9510d12d80536fa`.

Viewer cleanup calls `useGLTF.clear(url)`, geometry/material disposal and BVH disposal. Suspend-react's cache is module-local; importing another copy would not qualify it. The proposed observer therefore required actual source-pinned cache/GLTFLoader handles and WebGLAttributes records before any asset load. It would calibrate buffers with `context.isBuffer=true` while loaded, then check record removal and buffer invalidation after unload, alongside material-property removal, exact-URL cache absence and a bounded known-active-reference census. Neither cold resource counts nor WeakRef disappearance was an acceptance condition.

The prior observer's root registry and restoration were reviewed. The new candidate uses weak roots, weak model/resource/buffer histories and scalar outputs; it has no camera/render/input wrappers. Complete current/alternate fiber reachability or garbage collection is not claimed. Its13 focused mocked tests pass, but they do not prove native V8 scope accessibility. The runtime gate exposed precisely that unqualified boundary and correctly prevented asset loading.

The installed postprocessing component also has a **source-level composer-disposal concern**: source creates a composer without an explicit composer disposal cleanup. No new loaded session, resource attribution or reproducible high-resolution model-specific leak establishes its runtime impact here. Residual shared/lazy renderer resources, CPU allocator retention and dedicated GPU bytes remain **UNVERIFIED**. This concern should be tracked separately rather than used to infer a model leak or trigger another acceptance loop.

## Execution and measurements

Known isolated-profile launch and its single reviewed blank-page initialization were reused. The first diagnostic attach was blocked by sandbox localhost EACCES before attachment; authorized access then attached to the same already-paused owned process. There was no second app launch. Original-page readiness and updater/profile isolation were observed before the failing empty-viewer gate.

The normal-quit reserve was armed before original main resume for220 seconds; the host retained a240-second shutdown-request guard and360-second overall ceiling. On the gate failure, collection stopped early. Restoration removed the owned React hook/root globals, new-document registration and renderer debugger; the cleanup observer itself had never installed. Remote scope handles were released through the helper's finally block.

- Process AND aggregate Job limits: **6 GiB** each; available RAM floor **12 GiB**; destination-free floor **2 GiB**; import ceiling180seconds retained and unused.
- **76** live resource samples, all admitted.
- Peak process committed memory: **560,934,912 bytes** (**534.949 MiB**).
- Peak aggregate Job committed memory: **971,251,712 bytes** (**926.258 MiB**).
- Minimum available Windows RAM: **17,236,537,344 bytes** (**16.053 GiB**).
- Minimum destination-free disk: **135,822,696,448 bytes**.
- Dedicated GPU byte usage: **not measured**. CPU Job memory is not VRAM.

The stock PythonBridge logs `Stopped` and backend exit1 through its established backend shutdown path, whose pinned source uses `taskkill /T /F`. This is distinct from supervisor Job termination. The app's return0 and empty Job establish normal application settlement here; they do not establish graceful exit of every backend child or cleanup after a loaded master.

## Evidence, preservation and next step

[Source review](amd-high-resolution-cleanup-evidence/source-review.json), [frozen criteria](amd-high-resolution-cleanup-evidence/acceptance-plan.md), [candidate review](amd-high-resolution-cleanup-evidence/session-review.json), [execution summary](amd-high-resolution-cleanup-evidence/execution-summary.json), [independent outcome review](amd-high-resolution-cleanup-evidence/outcome-review.json), and [preservation receipts](amd-high-resolution-cleanup-evidence/preservation-after.json) contain source-safe findings and hashes.

Full private receipts, logs and the existing empty-viewer capture remain under:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-high-resolution-cleanup-bc9515f`.

The original master, protected100, original profile, selected installed-environment/source/backup identities and116 pre-existing remote heads are checked within the same bounded preservation scope as the previous mission. Neural checkpoint/model payloads are not reread. Original assets, failures, quarantined/untracked caches and historical branches remain unchanged. Publication verification follows commit/push and is reported with the exact verified commit; private screenshots, full traces, models and raw asset data are excluded from Git.

**Recommendation:** stop cleanup research here. Record the unverified, currently noncritical renderer ownership concern separately. The next separately approved product milestone should prioritize integration of the already verified AMD generation/colour/export pipeline into the existing desktop workflow, carrying this accurately scoped acceptance rather than claiming complete resource qualification. No integration or further app run occurs in this task.
