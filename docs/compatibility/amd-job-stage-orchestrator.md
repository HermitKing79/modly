# Job-owned AMD stage orchestrator v1 — BLOCKED

Base `c5f7ccd3cdb3de371405c09856a356ebc41a6ba3`; branch `codex/amd-job-stage-orchestrator-v1`.

Production sequencing, input/artifact contracts and Windows/WSL supervisors were implemented and independently reviewed. **The single actual CPU qualification failed in the Windows controls. No WSL stage, orchestrated chain or GPU inference ran.** No correction or retry followed that execution.

| Required verdict | Result |
|---|---|
| Windows stage ownership | **BLOCKED overall.** Target tree was contained and cancelled naturally; independent sentinel supervision failed. |
| WSL stage ownership | **BLOCKED / NOT RUN.** Source and mock contracts are reviewed; no actual Linux settlement proof. |
| Input-derived admission | **BLOCKED for a genuine chain.** CPU byte/tensor/lineage checks pass; preprocessing closure and new-image neural profiles remain unqualified. |
| Process-tree cancellation | **BLOCKED overall.** Actual target cancellation passed; healthy independent-job isolation was not qualified. |
| Job status and progress | **PASS in CPU contract tests.** No native end-to-end stage-chain claim. |
| Completed-result boundary compatibility | **PASS in focused controls/negative tests.** Existing authenticator/publisher unchanged; no 97 MB replay. |
| Fresh AMD GPU inference | **NOT RUN.** No public real AMD Generate capability enabled. |

## Implemented boundary

- `amd_stage_orchestrator.py` binds the actual image SHA-256, canonical params/model, product job/request/run/nonce and exclusive private run root. It pins host-selected stage code, orders dispatch, durably journals real timestamps, authenticates exact predecessor receipts/artifact bytes before and after each stage, and refuses late/foreign/malformed output or mutated plans. Synthetic JSON results can never become an `amd-worker` GLB.
- `amd_stage_contract.py` validates actual bounded JSON/safetensors bytes, exact host-owned tensor schemas, F32/F16/BF16 finiteness, sparse population ceilings and I32 coordinate payload hashes/order/uniqueness. Its 2,000,000-row readable ceiling is **not** a neural admission profile. Source-specific coordinate resolution and guide/mask linkage remain producer obligations.
- `amd_stage_process.py` creates private Windows Jobs with both 6 GiB caps and gates execution until attachment. `amd_stage_bootstrap.py` preserves exact nonce controls and transport status. The Linux broker uses a subreaper, new stage session, UID/start/boot identities, live broker ancestry and pidfds, reaping and an authenticated terminal acknowledgement. Windows client exit cannot prove Linux cleanup. The signalling path is source/mock qualified; it was not exercised on Linux here.
- The existing generation cancel route now delegates to flagged job owners, keeps status nonterminal while settlement is pending, and reports unverified/forced cleanup as an error. Legacy generators keep their previous path. The host adapter is unregistered. No new endpoint, manifest, ordinary fixture Generate option or alternate completion publisher was added.

Current `generated` plans fail closed before dispatch: this CPU milestone provides no path around inference qualification. Historical brick inputs, latents or GLB cannot satisfy a current request. Existing replay classification remains unchanged.

## Pipeline authentication and remaining limits

The [source audit](amd-job-stage-orchestrator-evidence/pipeline-audit.json) records 44 repository/source hashes, three installed original source hashes, original normalization comparisons and historical stage receipts. The accepted brick chain genuinely completed conditioning, occupancy/shape sampling, selectively BF16 geometry decoding, texture sampling, selectively F16 guided material decoding, CPU dual-grid extraction, world-position eight-neighbour colour sampling and portable GLB verification. Turret results establish separate input-specific successes, not arbitrary-image support.

Two concrete admission blockers remain:

1. The API Python has FastAPI but lacks Pillow. Actual alpha-image decoding passed ten tests in the **existing system** Python 3.12.10/Pillow 12.3.0; no packages were installed or borrowed into the API. The helper explicitly reports preprocessing incomplete; original crop/premultiplication and DINO remain real worker obligations. Preprocessing must be placed and independently qualified in the correct existing runtime before a product inference chain.
2. Existing larger-support geometry and guided F16 material profiles authenticate exact brick inputs/guides/hashes. They cannot be generalized by replacing hashes or assuming 3,743/4,430 rows. Current-run support, both C2S expansions, native maps, tiling, guide lineage and downstream memory must be admitted from authentic predecessor outputs.

Original real-stage GPU policies remain prerequisites, not CPU measurements: 18.5 GiB prelaunch/12 GiB live Windows RAM, 13 GiB initial free VRAM, 12 GiB Torch ceiling, 2 GiB device reserve, 1 GiB opaque allowance, 1.25 padding, 32,768 tile rows, original disk floors, native/CPU/GPU/host and I/O deadlines. The new executable backend is limited to the CPU qualification envelope; it is not a replacement GPU supervisor or permission to use a 180-second CPU plan for neural stages.

## Actual single CPU outcome

[Execution summary](amd-job-stage-orchestrator-evidence/execution-result.json). Fresh admission passed 12 GiB Windows RAM and 2 GiB destination-free floors. The admitted harness contained two standalone Windows controls followed by at most one Windows→WSL synthetic chain. It stopped after the controls; **actual orchestrated chains: zero**.

| Measurement | Observed |
|---|---:|
| Outer elapsed | 13.844 seconds |
| Outer peak Job commit | 102,535,168 bytes (97.8 MiB) |
| Outer peak process commit | 21,905,408 bytes |
| Minimum observed Windows available RAM | 18,769,424,384 bytes |
| Minimum observed destination free | 135,280,381,952 bytes |
| Outer processes accounted / remaining | 22 / 0 |
| Forced outer cleanup | false |

The target's ten-process tree reported exit 0, natural cooperative cancellation, matching contextual acknowledgement and zero active Job processes. The independent sentinel's supervisor recorded **`PermissionError: [WinError 5] Access is denied.`** Its later protocol says cancelled/0 without a cooperative cancellation acknowledgement; its receipt correctly retains `natural=false`, `forced=false`, `terminal=failed`, with zero remaining Job processes. No forced exit is labelled natural.

The sentinel error record lacks a traceback/native operation/PID. Its exact API failure origin is **unproved**. Do not infer that the target cancelled another job, or that an identity-query race is the cause. This failed isolation qualification is preserved unchanged; no guard was weakened and no probe was repeated.

Private evidence: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-job-stage-orchestrator-c5f7ccd\cpu-once`. The original attempt marker, process receipts, contexts, protocols, stderr and source pins remain there. `runs` and `chain-processes` were never created. No GLB/model output exists. Both registered WSL distributions were initially stopped; the Windows failure prevented any Linux launch.

Linux protection is explicitly different from Windows Job memory: inherited per-process 6 GiB `RLIMIT_AS` plus sampled aggregate RSS, not an atomic cgroup cap. No distro termination/shutdown, broad process signalling, permission changes or unrelated application termination was used. Natural cooperative cancellation does not qualify forced Linux signalling or genuine neural cancellation.

## Verification and preservation

- [Focused API/contracts](amd-job-stage-orchestrator-evidence/focused-tests.json): **152/152**, no skips; ownership OS boundaries are mocked in these tests.
- [Image/artifact tests](amd-job-stage-orchestrator-evidence/image-artifact-tests.json): **10/10**, existing system runtime, no package changes.
- [Independent contracts](amd-job-stage-orchestrator-evidence/independent-contract-review.json), [process review](amd-job-stage-orchestrator-evidence/independent-process-review.json) and [one-attempt admission](amd-job-stage-orchestrator-evidence/cpu-execution-admission.json) distinguish source/mocked checks from actual execution. Reproduced CPU findings were fixed before the attempt; no runtime repair followed it.
- [Independent outcome review](amd-job-stage-orchestrator-evidence/independent-outcome-review.json) corroborates the saved target, sentinel, host and source-pin observations without rerunning the probe. The precise sentinel native failure origin remains unproved.
- [Preservation AFTER](amd-job-stage-orchestrator-evidence/preservation-after.json): all 2,280 scoped file identities match BEFORE, including 100 protected files, app/master inputs, original profile, selected environments and source backups. All 118 pre-existing remote heads match; main is unchanged. Both registered WSL distributions remain stopped. Broader historical/checkpoint scopes retain their previously authenticated evidence; no model payload/header was loaded.
- Private assets and synthetic output arrays/images are excluded from Git publication. The existing four historical cache directories remain untouched.

The next justified action is a **separately approved CPU-only diagnostic qualification** of the sentinel supervisor's access-denied failure, capturing the precise failing native operation and process identity. Preserve the current failed attempt and require natural independent-job settlement. Do not proceed to a neural run, lower limits, suppress identity failures or rerun this harness automatically.
