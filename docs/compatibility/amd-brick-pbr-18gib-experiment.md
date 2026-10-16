# Brick guided PBR — one-time 18 GiB RAM exception

**BLOCKED — no WSL launch, model loading or GPU forward.** Base `28ab4c1bdd0342f4d3e57e6c70278319eb475660`; isolated outcome branch `codex/amd-brick-pbr-18gib-experiment`.

The user approved an experiment-only prelaunch threshold of **18.0 GiB / 19,327,352,832 bytes**, requiring three fresh readings strictly above it. The unchanged **12 GiB / 12,884,901,888-byte live physical-RAM floor** still applies. The exception was applied to this run's CPU admission comparison only; no worker override was activated because the independent adequacy review refused launch. The historical 18.5 GiB default and all source policies remain unchanged. [Exact exception](amd-brick-pbr-18gib-evidence/one-time-exception.json).

## Fresh readings: numeric exception passes

Existing `read_resources()` / Windows `GlobalMemoryStatusEx`, three seconds apart on 2026-10-16:

| UTC | Available physical RAM, bytes | GiB |
|---|---:|---:|
| 09:01:24.055 | 19,841,519,616 | 18.479 |
| 09:01:27.055 | 19,855,331,328 | 18.492 |
| 09:01:30.056 | 19,859,955,712 | 18.496 |

D: free space passed at **74,269,659,136 bytes** against the unchanged 50 GiB floor. Memory load was 42%; commit availability was approximately 31.98 GB. Commit headroom is not available physical RAM and cannot replace the live floor. [Full pressure/readings record](amd-brick-pbr-18gib-evidence/initial-host-readings.json).

## Independent review: adequate operating headroom is unproved

The closest successful genuine F16 guided PBR run used the same 284-entry, 208 F16 / 76 F32 model contract. Its actual stage-correlated **Windows available physical RAM** was:

| Historical stage | Bytes |
|---|---:|
| Pre-marker | 22,434,783,232 |
| Native-qualified CPU preparation entry | 17,921,138,688 |
| After strict CPU loading, GPU-stage entry | 15,996,715,008 |
| Lowest sampled availability, holding output before host acknowledgement | 14,853,283,840 |

The observed whole-host decline was **6,438,068,224 bytes before the GPU stage**, then another **1,143,431,168 bytes**, totaling **7,581,499,392 bytes (~7.06 GiB)**. At exactly 18 GiB, that pre-GPU decline alone leaves only **4,382,720 bytes** above the 12 GiB floor. If the full observed decline recurred from today's best reading, availability would be **12,278,456,320 bytes**, **606,445,568 bytes (~578 MiB) below the floor**.

This is a **counterfactual safety screen**, not a measurement of today's decoder, a prediction, or proof that WSL caused the entire historical decline. The saved guest CPU-load high-water RSS was 2,786,930,688 bytes; it does not bound WSL VM/kernel/cache memory, the host process tree or independent readback. No current WSL/vmmem commitment baseline or bound on the remaining exact brick host demand exists. The larger brick support and device-visible memory projections do not close that gap. Monitoring can detect a breach; it cannot prove the next allocation is safe.

The independent reviewer therefore did **not** qualify conditional WSL startup or model loading under the present headroom. No higher global threshold was invented. [Independent assessment, source hashes and measurement limits](amd-brick-pbr-18gib-evidence/host-memory-review.json). Its historical source hashes and arithmetic were independently rechecked by the coordinator.

## Stop, preservation and next action

No inference marker, output leaf, model load, native import, GPU query, forward, retry, cache clearing, application termination, installation or Windows configuration change occurred. There is **no new material field, GPU peak or runtime result**. Ownership/runtime/native/VRAM execution gates remain untested. No decoder/supervisor code or safety policy was modified, and no broad test suite was repeated.

All 16 CPU-qualified source identities match. [Preservation](amd-brick-pbr-18gib-evidence/preservation.json) covers the 100 protected files, prior artifacts, original caches, selected environment/native/source/edit/backup identities and checkpoint stat/header/configuration scope. Private model outputs and images remain local. The main and all pre-existing remote branches are preserved.

The read-only [composition note](amd-brick-pbr-18gib-evidence/composition-review.json) records remaining concrete adapter work, including nonblocking independent host acknowledgement before GPU release. None was implemented after the memory decision.

**Next:** obtain more available physical RAM with operating margin, then request fresh admission. Roughly 0.6 GiB additional availability would only cover this historical-screen deficit; it would not establish a safety margin or a new guaranteed threshold. Before any future forward, retain host-controlled checks after runtime startup and strict CPU preparation, all unchanged live floors/reserves/deadlines, and fresh source/ownership/native/output qualification. Do not lower another limit or automatically retry.
