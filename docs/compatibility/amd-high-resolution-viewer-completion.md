# High-resolution viewer completion: BLOCKED

The installed Modly **0.4.3** displayed the original 1,957,891-vertex / 4,173,584-triangle master, supported the tested camera controls, replaced it with the tiny fixture and removed that fixture in-session. Complete resource release is unproved. The operator did not close the app before the fixed deadline; the supervisor forcibly emptied the owned Job at **480.75 seconds**. This is a failed orderly-session requirement, not a demonstrated product crash or leak. The High-Resolution Original Mesh acceptance milestone remains **BLOCKED**.

Base: `3cdcaa97d528ea1fe334b91bd695dd13a2e33712`. Isolated branch: `codex/amd-high-resolution-viewer-completion`. No production code changed. There was one normal UI master import, one tiny-fixture replacement and one selected-model Delete; no export or asset modification. The previously qualified isolated-profile startup, including its known single blank-page initialization, was reused. This is an installed-binary result, not an exact-checkout-build or unassisted-startup claim.

## Separate acceptance results

| Check | Verdict | Saved observation |
| --- | --- | --- |
| Orbit | PASS | Horizontal and vertical camera quaternion changes of 1.40027 and 0.56012 radians; corresponding screenshots show orientation changes. |
| Pan | PASS | User-performed native right-drag: 121 trusted right-button move records and 1.11549-world-unit target displacement; visible position change. |
| Zoom and practical framing | PASS | Five inward wheel events reduce camera-target radius 22.62%; five outward events restore it within 1.8e-15 world units. Useful projected conservative bounds occupy 53.56% of viewport height and fit inside the frustum. |
| Interaction responsiveness | PASS, within observed CPU scope | Main-frame interval p95 at most 16.9 ms; maximum 17.7 ms. Matched trusted-input to successful main-render CPU return is at most 12.5 ms for orbit/zoom and 66.7 ms for pan. Six pan events remain unmatched. |
| Master-to-small replacement | PASS | Tiny fixture displays 4 vertices / 3 triangles; master detaches and tiny becomes the active model. |
| High-resolution resource disposal | BLOCKED for complete release | Master geometry and material each emit one disposal event and its BVH is absent. Model/resources remain observable through WeakRefs; loader-cache release was not independently established. |
| In-session empty-scene cleanup | BLOCKED for complete release | After Delete, no active SceneMeshModel; both models detach, tiny geometry/material each dispose once and its BVH is absent. Residual renderer counts remain above the initial baseline. Session teardown was forced. |

The mesh's position, quaternion and scale remain identical through all six gesture windows. Initial conservative projected height was 33.45%; zoom makes it inspectable at a useful size. The human pan deliberately moves part of the conservative bounds outside the frustum; zoom-out brings them back inside. This does not establish automatic framing for every gesture.

The responsiveness observations distinguish real main R3F ticks from additional render passes. They measure renderer CPU event receipt through pose change and successful exact-main scene/camera render return, supported by before/after screenshots. They do **not** measure GPU presentation or OS-to-display latency. All orbit/zoom events matched; pan has 115 matched and six unmatched records. Observer overhead and unseen input completeness are unqualified.

## Disposal and resource evidence

| Settled checkpoint | WebGL geometries | Textures | Programs | Active model |
| --- | ---: | ---: | ---: | --- |
| Initial empty | 8 | 4 | 4 | None |
| Master loaded | 21 | 10 | 11 | Master |
| Tiny, approximately 5 seconds | 21 | 10 | 11 | Tiny |
| Empty, approximately 5 seconds and final | 20 | 7 | 10 | None |

Installed source calls `useGLTF.clear(url)` on unmount; direct cache enumeration was unavailable. The observer retained weak references and scalar disposal records, without forcing garbage collection or clearing caches. Surviving WeakRefs and residual counters do not alone establish a leak, its cause or which shared/helper resources remain. Disposal, scene detachment and BVH removal are directly observed; complete CPU/cache unreachability and dedicated GPU-memory release are not.

The temporary observer was stopped and its restoration checkpoint saved before containment. Its six focused mocked regressions cover exact-main-store discovery, fail-closed foreign stores, render-wrapper behavior/restoration, trusted event correlation, disposal/BVH observations and separate frame/render-pass accounting. Independent pre-execution source review and saved-outcome review are retained; mock tests do not substitute for runtime evidence.

## Measured bounds and preservation

The session retained **6 GiB process AND aggregate Job caps**, **480 seconds overall**, **180 seconds per import**, **12 GiB available Windows RAM** and **2 GiB destination-free disk** floors. All 480 live resource samples admitted.

- Peak process committed memory: **715,317,248 bytes** (682.18 MiB).
- Peak aggregate Job committed memory: **1,896,833,024 bytes** (1.767 GiB).
- Minimum available Windows RAM: **16,595,570,688 bytes** (15.455 GiB).
- Minimum destination-free space: **136,204,550,144 bytes**.
- Deadline receipt: `Bounded8minute UI session`, elapsed **480.75 seconds**, forced Job cleanup **7 active processes to 0**. Natural settlement was not verified. No retry occurred.

Electron process-memory snapshots, including the GPU process, are CPU-memory measurements. WebGL object counts are separate graphics-resource observations; dedicated VRAM was not measured.

Fresh before/after preservation checked all **100 protected files**, all 12 authenticated installed-app/input files, the original profile's 2,103 files, 42 selected environment identities and 13 source/backup identities. All match. The accepted master remains **97,073,940 bytes**, SHA-256 `1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae`. All 115 pre-existing remote heads, including main, matched before publication. Publication verification is pending at report preparation; the final response must supply the verified new-branch commit and post-push comparison of those heads. Historical broader/checkpoint provenance is reused within the scope recorded in the preservation receipts; neural/checkpoint payloads were not reread. Original failures and unrelated untracked caches remain intact.

## Evidence and private views

- [Frozen plan](amd-high-resolution-viewer-completion-evidence/acceptance-plan.md), [source review](amd-high-resolution-viewer-completion-evidence/session-review.json), [independent outcome review](amd-high-resolution-viewer-completion-evidence/outcome-review.json).
- [Before preservation](amd-high-resolution-viewer-completion-evidence/preservation-before.json), [after preservation](amd-high-resolution-viewer-completion-evidence/preservation-after.json), [saved-evidence verification](amd-high-resolution-viewer-completion-evidence/saved-evidence-verification.json).
- Private evidence root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-high-resolution-viewer-completion-3cdcaa9`.
- Useful zoom: `screenshots\zoom-useful.jpg`; horizontal/vertical orbit: `screenshots\orbit-horizontal.jpg` and `screenshots\orbit-vertical.jpg`; human pan: `screenshots\pan.jpg`; tiny replacement: `screenshots\tiny-replacement.jpg`; empty scene: `screenshots\empty.jpg` within that private root.

Images, full memory/input traces and assets remain private. Published evidence contains source, tests, numerical observations and hashes only. No neural execution, WSL, Blender, MeshLib, neural model loading, geometry editing, installation or generation work occurred.

**Next step:** separately approve a short cleanup-only verification with an explicit early app close, directed at the remaining cache/resource ownership evidence. Reuse the camera findings. There is insufficient evidence for a production leak fix, and this result does not authorize generation-workflow integration. No further session is started here.
