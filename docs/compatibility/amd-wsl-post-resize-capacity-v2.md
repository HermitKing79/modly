# Phase 6C v2 — fresh held capacity check after Windows restart

30 October 2026. **Preparation only; execution HOLD.** This is one new handoff
using the reviewed 28ba485 capacity implementation. No second resize or
filesystem repair is proposed. The original v1 evidence is preserved in full.

## What v1 actually reached

All three prelaunch readings passed: **23.827076, 23.832069 and 23.837780 GiB**,
four seconds apart. Final-before-query RAM was 25,595,617,280 bytes, followed by
CAPACITY_START at **10:21:17.630684 UTC**. WSL and the ordinary-user guest Python
did start: the saved Linux traceback reached the guest's GO/EOF check.

The failed gate was the retained **22 GiB startup/live-query guard**, not the
three-reading prelaunch admission. Its first recorded breach was
**23,545,454,592 bytes / 21.928413 GiB at 10:21:21.231057 UTC**. Minimum recorded
available RAM was **23,247,003,648 bytes / 21.650459 GiB**. The observed decline
from final-before-query to minimum was **2.187317 GiB**. This is a host-wide
measurement during WSL startup, not an isolated measurement of VM consumption.
Concurrent application contributions were not separately measured.

The guest rejected input with `RuntimeError: Exact GO/EOF required`; no valid GO
reached the measurement section, and no Linux capacity result exists. The RAM
fault precedes the final `Capacity natural exit/guards failed` exception. The
inventory Job has natural/empty cleanup and the controller reports handles
closed. The old controller did not save a separate capacity-client numeric exit
or GO receipt, so those missing records are not invented. Source code supports
natural settlement but does not replace absent per-client evidence.

The second invocation's occupied-path refusal is consistent with the reviewed
guard. No separate second-attempt log was found within v1's ten files and its
empty `tmp` directory; consumed-path behaviour is verified again with a fixture.

## Minimal changes and provenance

The fixed new output is
`C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-wsl-post-resize-capacity-v2`.
It must not exist. No path parameter, fallback, overwrite or automatic retry
has been introduced. The PowerShell entrypoint, guest, geometry interpretation,
resource policy, query controller and cleanup behaviour remain unchanged.

A path-only change would encounter a second known refusal: Windows source
mtime advanced from **1793352760979960200** to **1793355757297511000** ns after
the v1 startup. Current exclusive read-only Windows handle inspection verified
the same file ID, volume, length, link count, canonical path, attributes, exact
64 GiB container and both table hashes. Only mtime differs. This is consistent
with normal WSL activity; timestamps alone do not prove the cause or unchanged
guest file contents.

The new private SHA-pinned authority therefore retains the old authority,
authenticates all ten v1 files and the retained empty directory, and binds this exact newly inspected source
snapshot. It requires every non-timestamp identity field and container metadata
to match the old 64 GiB state. **The future source must still match the new pin
exactly, including mtime.** Nothing is ignored or refreshed automatically.
Any subsequent change, including intervening WSL/Docker activity, stops for
review. Both original authority and resize evidence remain unchanged. No fresh
whole-VHDX hash or whole guest filesystem preservation claim is made.

The accepted E: backup retains its original identity/metadata/receipt checks
and read-only write/delete-denying handle at execution. Eight box source pins,
including the authenticated 3,888-row shape data, remain mandatory. The new
preparation did not access E: or change the VHDX; its Windows header inspection
was read-only and exclusive.

## Conditional operation after restart

**Separate execution approval is still required.** Then:

1. Save the exact final command before closing Codex. Perform a fresh Windows
   **Restart**. Do not reopen Codex, ChatGPT Desktop, browsers or unnecessary
   applications before this single check. A restart is an operational measure,
   not evidence that RAM admission will pass.
2. Keep Docker and WSL applications from starting. All distros/utility VM must
   already be stopped. The launcher only checks this; it never stops unrelated
   work. If they are running, stop for review instead of forcing shutdown.
3. Use a normal 64-bit Windows PowerShell console under the distro-owning
   account, with stable power and the console attended until cleanup.
4. Only after the approval and prerequisites, invoke the command below once.

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\wattz\Documents\ChatGPT\modly\proof\scripts\compatibility\Invoke-ModlyPostResizeCapacity.ps1' -Run -ApproveCapacityCheck -ExpectedCommit '<published-commit>'
```

The final delivery supplies the exact new commit. Default invocation stays
inert. Keep both old and new local authority/evidence directories available.
No installation, download, elevation or environment change is required.

All original requirements remain: **three >=22 GiB readings four seconds
apart**, final >=22 GiB before WSL launch and guest GO, >=22 GiB throughout the
query for acceptance, **12 GiB emergency floor**, 24 GiB preferred only, D:50 GiB,
C:/E:2 GiB, 512 MiB process bounds, 120-second preflight, 30-second controller
query checkpoint, guest 15-second alarm/10 CPU seconds, strict owned cleanup,
read-only audit and the unchanged **5 GiB decoder free-space gate**.

Full64, partial/unchanged expansion, unavailable exact geometry and valid usable
space remain separate classifications under the [reviewed measurement
criteria](amd-wsl-post-resize-capacity.md). PASS_SPACE_GATE is not decoder
approval and does not establish full64 unless exact geometry confirms it.
Any fault latches FAIL; preserve the new evidence and do not rerun. No resize,
repair, deletion, rollback, neural execution or decode follows automatically.

Verification: **151** scoped offline/synthetic tests passed with no skips,
including 22 focused capacity tests, plus four inert/approval dry-runs.
Independent review identified and verified the fix for v1's retained empty
directory; no blockers remain. All ten v1 files and that directory, 18 resize
receipts, 23 backup receipts, 3,346 historical files, 188 protected/retained
records, 1,199 box records, eight box pins and 191 old local refs passed
preservation checks. No full environment or current E: backup audit is claimed.
