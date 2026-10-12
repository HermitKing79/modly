# Independent evidence review

Reviewer `/root/forward_resume_review`, 2026-10-12. Read-only saved-evidence
review; no new executions, AI imports, tests or edits.

**PARTIAL is correct.** One strict local load and conditioning invocation,
one image/output transfer, two normalization constants, all 24 layer returns
and 24 SDPA dispatch counts are recorded. Expected F32 [1,1029,1024] finite
output and final synchronization were reported; conditioning took 29.703s.

Live-worker release retained 79,691,776 bytes allocated/reserved (76 MiB)
despite collected model/output weak references and a cleared RoPE cache.
Cause is unresolved. Exit 1 is a handled validation failure; no timeout or
native computation error was recorded. The owned tree exited without forced
cleanup. Seven healthy Windows samples, maximum sample gap 5.047s, and matching
preservation comparisons are recorded.

Do not claim successful in-process cleanup, zero VRAM after exit, a known leak
or runtime-workspace cause, independent recomputation of the output hash (full
feature values were not saved), per-kernel timing (ATEN counters are separate
from module entry/return markers), full-pipeline readiness or conditioning
quality/parity validation. The shared error's legacy "Decoder release incomplete"
label does not imply a decoder was executed.
