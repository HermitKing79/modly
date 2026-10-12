# Independent read-only preparation review

Reviewer `/root/instrumented_forward_review`, 2026-10-12.
Read-only source/evidence review; no tests, AI imports, model/cache execution or
GPU calls performed by reviewer.

The reviewer independently verified all nine private cache file hashes/sizes,
the exact copied file set, four installed native DLL hashes/sizes and matching
passive environment inventory hashes. Selection excludes uncertain framing and
bitcode, preserves historical directories and native cache keys, and reattests
runtime/copies before launch. No important cache/provenance finding remained.

Periodic Windows checks use four-second subprocess bounds, five-second cadence,
durable readiness, nine-second freshness and 0.5-second parent polling; unhealthy
monitoring prevents authorization/PASS. Diagnostic UTC/monotonic/executable-path
metadata introduces no GPU query or additional clock call.

One important inherited cleanup gap was identified: root-only reaping could miss
a compiler child after root exit. The correction uses a private, non-inheritable
Windows job with kill-on-close and no breakaway permission. Assignment precedes
the worker ownership handshake, checkpoint preparation and Torch import.
Accounting/termination verifies zero active processes with a bounded cleanup;
forced or unverified cleanup prevents PASS.

Final verdict: **No remaining critical or important findings. Cleanup blocker
resolved. Preparation is defensible for the single authorized attempt after
exact commit/push verification.** Saved evidence records 65 passing CPU tests,
including real CPU orphan termination and corrupt database cases. The reviewer
did not rerun tests or the harness.

This review does not prove cache compatibility, GPU completion, output validity
or release. Those remain the subject of exactly one separately authorized run.
