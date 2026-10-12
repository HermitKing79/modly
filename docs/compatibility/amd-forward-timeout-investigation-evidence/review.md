# Independent read-only review

Reviewer: `/root/forward_timeout_review`, 2026-10-12.
Scope: diagnostic/cache-audit code, focused tests, saved evidence and report.
Reviewer performed source/evidence reads only; no harness, tests, AI imports,
model loads, GPU operations or cache execution.

Initial findings and verified corrections:

1. Synchronous journal fsync could block the supervisor deadline. Durable writes
   now run on a daemon with acknowledgments bounded by the smaller of one second
   and the remaining deadline. Shutdown is bounded; errors prevent PASS. Stalled
   and failed-write CPU tests pass.
2. Observer/monitor work could authorize GPU execution after the CPU deadline.
   Startup uses the remaining budget; the deadline is rechecked immediately
   before authorization. Fake observer and monitor expiry tests prove no stdin
   permit is written and the owned worker is reaped.
3. Root disappearance during descendant traversal could falsely invalidate
   monitoring. This ordinary exit is handled separately from permission failure
   and root PID reuse; the CPU race regression passes.
4. Cache-audit output previously could target historical evidence. Its resolved
   destination is now restricted to the new investigation evidence directory;
   negative path tests reject before reading any historical artifact.

Final verdict: **All four findings resolved; no remaining critical or important
code findings. Ready for publication once referenced evidence is saved.**
The reviewer inspected the saved 48-test passing results, without rerunning them.
The additions introduce no extra GPU query/synchronization, model call,
computation allowance, retry or persistent model/tensor reference. The original
execution limits and transfer contract remain unchanged.

The report distinguishes native compilation evidence from an established timeout
cause. Suggested timestamp wording was adopted to avoid implying measured
compiler CPU time. The referenced preservation file is saved.

Unjudged: actual GPU diagnostics overhead, root cause, copied-cache compatibility,
adequacy of a future 120-second deadline, successful output and model release.
The proposed future experiment requires separate approval and additional bounded
Windows monitoring; this review does not authorize it.
