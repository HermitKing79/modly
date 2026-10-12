# Independent post-attempt read-only review

The independent reviewer verified the raw results, trace, stderr, allocator
snapshot and safetensors container/payload using CPU-only reads. No additional
GPU operation or conditioning attempt was run.

PARTIAL is correct: one strict load and conditioning invocation succeeded;
the genuine finite F32 [1,1029,1024] feature payload and complete provenance
match reported hashes. Strict release failed after successful snapshot capture
and history disable, with one active 79,691,776-byte block. Snapshot totals
reconcile exactly. All 2,288 observed weak wrappers were collected, without
claiming exhaustive ownership coverage. Seven Windows samples were healthy;
owned Job exit was verified and observers/journals/readers stopped cleanly.

The exact BLASLt workspace-size match, DINO q_proj/linear allocation origin
and installed workspace-map declarations strongly support a persistent HIP
BLASLt workspace hypothesis. Ownership is not proved because native C++ context
capture is unsupported on this Windows build. `_cuda_clearCublasWorkspaces`
is a source-supported future correction candidate; the native implementation
must first be checked for HIP BLASLt coverage. No workspace-clearing call,
backend/settings change, criterion relaxation or post-exit GPU query occurred.

The reviewer also checked the final report, CPU saved-evidence analysis and all
four installed source excerpts. No important correction or publication blocker
was found. Feature/input/snapshot copies and source hashes match; before/after
inventories and compressed metadata match, supporting preservation claims.
Future acceptance wording explicitly retains the shape/dtype/finite contract
without requiring bitwise equality. This review performed no additional tests,
worker execution, GPU query or inference.
