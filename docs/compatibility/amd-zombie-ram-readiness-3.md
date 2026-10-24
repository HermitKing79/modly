# Zombie recovery — NO-GO at the final prelaunch RAM check

Approved decoder base `35b3c70`; parent `91b7455` retains both previous no-go reports.
Execution candidate `85ee9c6d5d2bcf7cf05ab749b6dd03f11faf7810` was independently
reviewed, pushed and GitHub-verified. Branch: `codex/amd-zombie-ram-readiness-3`.

## Measured admission

Three initial `GlobalMemoryStatusEx` available-physical-RAM readings passed the
additional 22 GiB (23,622,320,128-byte) target:

| UTC on 2026-10-24 | Available bytes | GiB |
| --- | ---: | ---: |
| 10:20:38.997600 | 23,797,481,472 | 22.1631 |
| 10:20:42.998216 | 23,797,092,352 | 22.1628 |
| 10:20:46.998316 | 23,799,910,400 | 22.1654 |

The readings were four seconds apart with 2,818,048-byte variation. These
established readiness for further admission, not a completion guarantee.

After original source/checkpoint preflight completed, the immediate pre-marker
measurement at 10:33:51.503470 UTC was **22,085,574,656 bytes (20.5688 GiB)**.
This was **1,536,745,472 bytes (1.431 GiB)** below 22 GiB. The original 12 GiB live
floor and 50 GiB D: floor passed; D: had 74,074,488,832 bytes free. The lowest saved
CPU-preflight host sample was 22,027,182,080 bytes. There was no neural execution
from which to report a runtime RAM minimum or VRAM peak.

The exclusive final guard raised before calling the original reservation
function. **No new neural marker, WSL client, owned Job, model or forward was
launched. No neural attempt was consumed.** The rejected final-gate record and
complete stdout/stderr are retained; no retry or new RAM assessment followed.

## Scope and results

The fresh standalone wrapper privately clones the accepted zombie host, changes
only new evidence/run-ID routing, authenticates the original three-reading
receipt and adds the experiment-only 22 GiB final guard. Eight focused CPU tests
passed, including denial before marker creation and exclusive single reservation.
Original input readers, checkpoint/precision, tiling mathematics, generic
1,842,968/1,000,000 row caps, 32,768-row tiles, GPU admission, 12 GiB live host floor
and 60/120/240/480-second deadlines remain unchanged. No brick extension or
product CPU-only authority route was used.

| Required result | Outcome |
| --- | --- |
| Source/latent/checkpoint admission | Original CPU preflight completed; no model loading |
| Native neural attempt performed | NO |
| Convolutions / C2S | 0 / 0; learned subdivision populations unknown |
| Actual VRAM peaks and GPU teardown | NOT MEASURED |
| Natural / forced worker settlement | NOT APPLICABLE; no worker launched |
| Complete ten-tensor artifact readback | NOT RUN |
| Conditional CPU mesh and inspection images | NOT RUN |
| Overall generation | NOT QUALIFIED |

Read-only normal-user process observations after rejection found significant
ChatGPT, Defender and Codex working sets. They cannot establish the cause of
the earlier decline: shared working sets are not exclusive physical accounting,
and the preflight process had exited. An initial sandbox-restricted process
view is retained and explicitly labelled incomplete. No process was terminated,
cache cleared, system setting changed, package installed or resource gate lowered.

## Evidence and preservation

- [Initial physical-RAM readiness](amd-zombie-ram-readiness-3-evidence/readiness-summary.json)
- [Focused CPU tests](amd-zombie-ram-readiness-3-evidence/cpu-tests.json)
- [Independent pre-execution review](amd-zombie-ram-readiness-3-evidence/preexecution-review.json)
- [Final prelaunch rejection and exact measurements](amd-zombie-ram-readiness-3-evidence/execution-summary.json)
- [Independent outcome review](amd-zombie-ram-readiness-3-evidence/outcome-review.json)
- [Normal-user process observations](amd-zombie-ram-readiness-3-evidence/memory-consumers-user-context.json)
- [Preservation after](amd-zombie-ram-readiness-3-evidence/preservation-after.json)
- [Remote preservation](amd-zombie-ram-readiness-3-evidence/remote-preservation.json)

BEFORE/AFTER inherited states and added evidence snapshots match: 100 protected
identities/78 bounded hashes, both prior no-go trees, original nine failed-run
files/consumed marker and five saved shape outputs. Private images and geometry
remain local. Private control evidence:
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-zombie-ram-readiness-91b7455`.

Next action: establish 22 GiB available RAM **through completion of CPU preflight**
before separately requesting a fresh launch. The earlier 6.24 GiB neural-startup
decline remains a planning observation; neither it nor these initial passing
readings guarantee future completion. No additional attempt is authorised here.
