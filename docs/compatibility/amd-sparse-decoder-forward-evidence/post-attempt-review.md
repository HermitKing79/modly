# Saved single-attempt evidence review

Read-only review confirmed PARTIAL at input transfer: `aten.to.dtype_layout` was
rejected before the permit or root decoder invocation; both counts are zero.
Strict CPU loading passed once for 74 tensors and 74 payload hashes. Model
placement and GPU schema validation completed before the rejection.

Worker exit 1 after 8.438 seconds, reaping/output readers verified, no timeout or
forced cleanup. All three observer captures succeeded, controller status OK and
queried events empty. Checkpoint records are identical. No output or release
memory record exists. Process exit supports teardown; zero allocator usage and
complete VRAM recovery were not measured.

The failure is the harness transfer guard before inference. It does not establish
an AMD decoder-forward failure. The attempt remains consumed; narrowly reviewed
copy-overload changes and separate approval are required before another attempt.
The reviewer ran no imports, tests, observers, workers or model/GPU operations.
