# Zombie memory reconciliation and one standalone decode handoff

Decision: **GO for one user-controlled, guarded decode launch.** This is permission to run the admission harness once, not a claim that live GPU/host admission has already passed. Do not repeat CPU preparation separately. The launcher obtains fresh admission and the original decoder repeats its necessary integrity checks as part of that single launch. No generation, WSL command, marker reservation or model load was performed while preparing this handoff.

## Proven observer defect

The completed local diagnostic was `CPU_PREPARATION_COMPLETE`, using implementation `d77c7ef199b961e49afc63f8041efa02a164738c`. Its Windows venv `Scripts/python.exe` was a redirector, which launched the actual interpreter as a second process. The observer refreshed and sampled the redirector throughout. The child's native self measurements belonged to the interpreter.

The private process snapshot at `imports/after` contains a second Python process whose working set **40,120,320 bytes** and private commit **532,762,624 bytes** match the phase record exactly. The observed launcher instead held 4,321,280 / 811,008 bytes. All 177 numeric observer samples referred to the launcher; no recorded measurement error explains the discrepancy. Maximum observed gap was 0.239 seconds. Summary maxima were calculated correctly for the wrong process. Both identities and original timestamps remain local in the unchanged evidence.

The repair sends an interpreter PID/image identity before the first phase permit. PowerShell validates native membership in its exact owned Job, image path and start time before binding the process. Launcher and interpreter are recorded separately, with process start identity. Missing/duplicate identity or a live memory measurement failure stops completion. Before binding, interpreter memory is unknown, not the launcher's memory. Exit measurements are explicitly marked exited.

An actual Windows regression launches the same qualified venv executable, holds a touched 64 MiB byte array, and compares native self counters with the production observer functions. The old implementation failed by tracking the redirector. After repair, private commit matched exactly and working set differed by 16 KiB in the initial verification. Foreign-process and wrong-image identities were rejected. This test loads only the stdlib diagnostic library; it does not run preparation, import models or launch WSL.

## What the completed diagnostic establishes

| Quantity | Recorded bytes |
| --- | ---: |
| Minimum available physical RAM | 25,494,872,064 |
| Margin above 22 GiB | 1,872,551,936 |
| Baseline-to-minimum host change | -108,765,184 |
| Imports working-set change | +15,855,616 |
| Imports private-commit change | +517,705,728 |
| Interpreter OS peak working set | 106,168,320 |
| Maximum interpreter private commit in independent snapshots | 556,314,624 |

Private commit is committed virtual memory, not equivalent physical residency. The 493.7 MiB import commit increase does not establish a 493.7 MiB physical RAM loss. Recorded host availability decreased by at most 103.7 MiB from the pre-Python baseline. Observer overhead, other host activity and filesystem cache also contribute. The host's 12-second post-exit available-RAM change was -4,321,280 bytes; this does not show exclusive recovery attributable to the child. The measurements do not reproduce the historical multi-GiB decline or justify an extra mandatory margin. **24 GiB remains preferred only.**

## Decoder readiness and retained policies

Successful guarded CPU completion authenticated the original sources, both saved-input lineage checks, configuration/header metadata and full opaque checkpoint stream. The unchanged decoder checkpoint contract requires 948,490,494 bytes and SHA-256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581`. The saved zombie source is the independently reviewed 1,000-row Shape-SLat result at `35a5b825e3e9128326897a2ce35c87b237fdfa8b`, bound to source job `913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf`. Coordinates and latent hashes remain enforced by the original reader. No new full checkpoint stream or full lineage replay was needed during this repair.

The handoff privately copies the existing zombie adapter solely for a fresh evidence route and run ID; original module globals, guest code, model/precision, tiling, readers and supervisor stay unchanged. The route has one fixed unused destination and exclusive reservation; there is no retry or product dispatch. It requests one shape-decoder forward, with the generic 1,842,968 final-row / 1,000,000 intermediate-row caps and 32,768-row tiles. The brick support extension is not selected.

- Initial admission: three fresh `GlobalMemoryStatusEx` readings, four seconds apart, each at least **22 GiB**. A failing reading stops without a runtime, marker or WSL. Receipt spacing/freshness uses the existing diagnostic validation rule.
- Immediately before exclusive neural reservation: another **22 GiB** check, after original CPU preflight. A failure records a rejection without reserving the marker.
- Live host: original **12 GiB** available-RAM and **50 GiB D:** floor; **2 GiB** host output-volume floor remains enforced.
- Guest: original **6,972,719,480-byte** CPU RAM requirement, **5 GiB** guest disk and **50 GiB D:** disk checks.
- GPU: original single **RX 9070 XT / gfx1201**, at least **13 GiB** free VRAM before admission, original **12 GiB** per-process allocation cap and unchanged live checks. No current guest/GPU admission was measured in this preparation session; it must pass during the independent run.
- Original **60/120/240/480-second** native/CPU/GPU/host deadlines, 450-second guest timeout, GO plus EOF transport, health observer, owned Windows Job, bounded named-distro cleanup and complete independent ten-tensor readback remain in the original supervisor. Only its owned `Modly-AMD-Ubuntu-24.04` route is used.

This decode recovers detailed geometry. Texture generation and subsequent stages are separate; this command does not automatically start them.

## Standalone invocation

Close Codex Desktop, ChatGPT Desktop and browsers. Use a normal, non-administrator PowerShell window. Run `Invoke-ZombieNeuralHandoff.ps1 -Run -ExpectedCommit <the exact published commit supplied with this report>` at its full checkout path. Without `-Run`, the PowerShell script creates no process or evidence and prints its plan. The Python default also only prints its plan. The published source and current HEAD are authenticated before admission. Python uses the original pinned environment with `-B -I -S` and explicitly adds its existing site-packages without executing startup hooks, matching the completed diagnostic's controlled startup.

New private admission/final-gate records are exclusively created in `artifacts/amd-zombie-neural-handoff-d77c7ef-v1`. Decoder evidence uses `docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-standalone-d77c7ef/decode`; the guest run is `/home/modly/run-zombie-standalone-d77c7ef-v1`. Existing or consumed destinations stop the handoff. Keep generated private evidence local. Preserve the console output on a rejection and stop for review; do not retry automatically.

Verification and final publication receipts are linked from the companion JSON report. Raw process inventories and diagnostic evidence are not published.
