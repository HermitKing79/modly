# Genuine brick guided F16 PBR attempt

Base report: `4f91b52c1ef0cb9bfd32d265eefe7b7784135e00`. Qualified implementation: `791d19b6ee3546489306d5b557b7b9b958187936`. Branch: `codex/amd-brick-pbr-gpu-attempt`.

At execution-harness publication: **PREPARED**. Actual outcome: **BLOCKED during Linux input admission, before Torch or any forward**. One host/worker invocation occurred; no retry followed.

All original 107 source bindings and their committed blobs were verified. New host/worker copies differ only by fresh experiment addresses; their 109-file closure includes all original bindings. The complete fresh production plan passed actual `reserve_attempt`, serialization/readback, command construction and genuine worker plan verification with publication redirected only to ordinary CPU fixture filenames. No genuine marker was created by this qualification.

Fresh production output: `C:\Users\wattz\Documents\ModlyArtifacts\amd-brick-pbr-gpu-4f91b52\field`. Actual Windows SID/ACL/file IDs, alias/reparse checks, exclusive fsync write/readback and empty field were verified. Previous destinations and failed markers remain intact. Two historical live symlink subcases lacked privilege and remain skipped; they are not counted as passing. Actual production canonical/reparse/hardlink admission remains mandatory.

Initial RAM readings: 22,306,140,160 / 22,313,517,056 / 22,316,158,976 bytes. D: free: 74,269,659,136 bytes. Original limits remain: 18.5 GiB prelaunch and 12 GiB live host RAM; 50 GiB D: and 2 GiB destination free; 13 GiB initially free VRAM; 12 GiB allocator ceiling; 2 GiB device reserve; 1 GiB opaque allowance; 32,768-row maximum tiles; native/CPU/GPU/host deadlines 60/120/240/480 seconds and original I/O limits. Host release barriers retain the historical remaining-RAM reference and recent downward-step reserve. Fresh final admission supersedes these saved readings.

Required genuine computation: original 284 checkpoint tensors, 208 F16 / 76 F32 runtime parameters, complete support 4,430 → 21,391 → 97,313 → 438,684 → 1,957,891; one parent forward, 40 convolutions, eight C2S operations, five native maps, 84 large-site tiles and 60 final tiles. Production persistence requires finite full-support F32 fields, original coordinates, independent Windows acknowledgement before GPU release and fresh readback after natural exit. Zero owned Torch allocated/reserved bytes and healthy monitoring remain mandatory.

Evidence: [original candidate verification](amd-brick-pbr-gpu-attempt-evidence/qualified-original-source-verification.json), [fresh plan](amd-brick-pbr-gpu-attempt-evidence/fresh-path-plan-qualification.json), [destination ownership/write admission](amd-brick-pbr-gpu-attempt-evidence/production-destination-preflight.json), [preservation BEFORE](amd-brick-pbr-gpu-attempt-evidence/preservation-before.json).

## Result

Published execution revision: `102b8b8ec01385388cc5132ad4176f6cc6a5cbd9`. The real Ubuntu worker accepted the complete plan and all 109 source identities, received binary GO, and qualified the fresh production output-ownership factory under actual UID/EUID/GID/EGID `[1000,1000,1000,1000]`.

It then raised `ValueError: Artifact paths must be owned by the current user` at:

`worker.execute_admitted → inputs.prepare → texture sampling artifact verify → approved_root → _plain(ancestor, directory=True)`.

The input reader checks current-user ownership while traversing every ancestor of the saved texture-latent directory. The traceback does not identify the first rejected ancestor or its numeric UID/GID; those were not recorded and are not inferred here. The Windows CPU plan proof did not exercise this POSIX ownership predicate. This is an input-admission blocker, not evidence of an AMD numerical or native failure. The qualified output factory succeeding does not qualify this separate input-reader policy.

**No Torch import, genuine FlexGEMM import, model load, GPU admission or forward occurred.** All required computation counts remain zero; no material field exists. Initial free VRAM, Torch peaks and zero-allocation release were not measured, so none is claimed. The field directory remains empty. Both the original failed marker and this new consumed marker/plan are preserved.

Measured results:

| Measurement | Actual result |
| --- | --- |
| Host invocation | 51.859 seconds; exit 1 |
| Worker admitted processing | 1.2555904 seconds; exit 1 |
| Pre-marker Windows RAM | 22,135,492,608 bytes |
| Lowest sampled Windows RAM, including cleanup | 19,273,228,288 bytes; original 12 GiB live floor held |
| Lowest sampled D: free | 74,236,104,704 bytes |
| Windows host/descendant sampled working-set sum peak | 194,121,728 bytes; shared pages may be counted repeatedly |
| Sampled vmmemWSL RSS peak | 2,447,282,176 bytes; separate from guest/process-tree figures |
| Guest self HWM | 82,305,024 bytes |
| Containment | Natural client exit; empty Job; no forced Job cleanup; named Modly distro termination succeeded |
| Windows observation | Healthy before, during and after; RX 9070 XT controller status OK |

Whole-host RAM changes cannot be attributed solely to WSL. No model/GPU resource measurement is substituted by these host/guest observations.

Evidence: [actual outcome and exact traceback](amd-brick-pbr-gpu-attempt-evidence/actual-outcome.json), [worker record](amd-brick-pbr-gpu-attempt-evidence/run/guest/results.json), [supervisor record](amd-brick-pbr-gpu-attempt-evidence/run/results.json), [sampled host measurements](amd-brick-pbr-gpu-attempt-evidence/measured-host-summary.json), [independent outcome review](amd-brick-pbr-gpu-attempt-evidence/independent-outcome-review.json). Raw guest/supervisor classification is `UNKNOWN`; the experiment wrapper correctly reports `BLOCKED` for this failed admission with zero forwards. The native, decoder and forward statuses are `NOT_ATTEMPTED`, `NOT_ATTEMPTED` and `0` respectively.

The [independent AFTER preservation audit](amd-brick-pbr-gpu-attempt-evidence/preservation-after.json) found zero issues across all 400 scoped identities, fresh hashes for all 100 protected files, the original 107 candidate bindings, environments, source edits/backups, prior reports and artifact trees. The output field is empty; the new production marker/plan and all failure evidence remain preserved. No private model output or image was published.

**Next:** separately approve one CPU-only WSL input-reader ownership qualification under the same non-root identity, recording the exact failed ancestor/UID/GID and exercising the complete genuine `inputs.prepare()` path before considering another GPU attempt. Preserve checksum, alias and read-only confinement checks; do not globally bypass ownership or change permissions. No patch, additional WSL probe, inference retry, geometry change, sampling, baking, rendering or export was performed after this failure.
