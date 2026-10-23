# Evidence writer: precise failure captured, publication BLOCKED

Base `d784e4b89e0a05096d2503755c7e931895895d45`; branch `codex/amd-evidence-writer-preflight`.

**Error-reporting repair: CPU PASS. Tiny real writer: BLOCKED at `os.link`, errno 13 / WinError 5 (access denied).** Exactly one real filesystem probe ran, with the original API interpreter and the same default sandboxed launch route. No Job, lifecycle fixture, WSL, GPU, model or generation ran. No retry, alternate publisher or permission change was attempted.

## Repair

The diagnostic `windows_qualification.py` now annotates each publication operation and relevant read-only preflight operation. Host and explicit-preflight entry failures retain the original exception and traceback. Reports include errno/winerror when present, phase, attempted staging/final existence and separate operation-path state. Errors after exclusive directory reservation are written to an independently selected private diagnostics directory outside the consumed attempt. Delivery failure emits the full diagnostic to stderr. A secondary close or diagnostic failure cannot replace the primary exception.

The publication mechanism remains exclusive staging creation, original platform text-newline handling, JSON write, flush, fsync, close, `os.link` without replacement, then staging unlink. No atomicity or overwrite rule was relaxed. Lifecycle admission, ownership, cancellation, resource limits and deadlines were not changed. Historical admission/source pins were deliberately not refreshed to authorise another lifecycle run.

[Twenty focused mocked tests](amd-evidence-writer-evidence/cpu-tests.json) passed under the pinned API interpreter; an independent rerun passed all twenty. Tests include the actual host's reservation-to-publication route with Job/process creation forbidden, every publication failure, explicit preflight read errors, stderr fallback, primary/close-error preservation, refusal to overwrite, alias-safe cleanup and Windows newline/readback parity. [Independent preprobe review](amd-evidence-writer-evidence/independent-preprobe-review.json) bound the exact repaired source, probe, tests and interpreter.

## One real filesystem result

The 151-byte staging payload used a unique disposable private leaf. Durations below are measured using `QueryPerformanceCounter` at 100 ns clock resolution. These are whole-boundary elapsed times, including Python work and applicable ownership guards, not isolated native-call or physical-disk benchmarks.

| Boundary | Result | Elapsed ms |
| --- | --- | ---: |
| Exclusive staging creation | PASS | 0.3886 |
| JSON write | PASS | 0.0265 |
| Flush | PASS | 0.0585 |
| `os.fsync` | PASS | 0.5151 |
| File close | PASS | 0.0553 |
| No-overwrite hardlink publication | **BLOCKED: actual `os.link` raised PermissionError** | 7.2358 |
| Final-file readback | NOT RUN after first failure | — |
| Cleanup | PASS: only the newly owned staging file removed | 3.0840 |

The first exception records `errno=13`, `winerror=5`, both hardlink arguments and the complete traceback. Guard checks passed before the actual `os.link` call. The final name was not created. The scratch directory is retained empty. The full report was successfully written outside it using an ordinary exclusive diagnostic write; the original failure survived cleanup and reporting.

Private full report: `artifacts/amd-evidence-writer-d784e4b/probe-36fc408e1fc945f59e00b36d81a1f8c9-report.json`, SHA-256 `d657489a9510e324380c90f425c517b8ebd46760baff0c572f67f4639bfe8a0e`. Source-safe results and independent saved-observation review accompany this report. The full traceback, hardlink arguments and filesystem identities remain private.

## What this proves and the smallest next step

Access denial at **the actual hardlink publication call** is proved for this probe. Creation, writing, flushing, fsync and closing are successful in the tested context. This establishes this probe's failure; the historical exact failing operation and discarded exception remain unknown.

The test does **not** isolate an ACL, restricted-token or sandbox-policy root cause: only one execution context was tested. The current default route is not qualified for this hardlink publisher. A supported route must permit the unchanged same-volume, no-overwrite hardlink operation and verified cleanup. An explicitly authorised outside-sandbox run of the same tiny probe is the cheapest next context qualification; it is an untested candidate, not a demonstrated fix. No outside-sandbox probe or lifecycle retry occurred here.

Keep the existing publication mechanism until a context passes or a separately reviewed replacement establishes same-volume atomic publication, durable contents, exclusive ownership and refusal to overwrite. No timeout adjustment or lifecycle success follows from this filesystem result.

## Preservation

The historical 1,991-byte `cpu-once/attempt.json.writing` remains unchanged: SHA-256 `25ebe42f554489a23815c0e37ba09eae4dd143428481ce3344527cbf384fe6bd`. Its directory was not reused, renamed or cleaned. The original diagnostic source is preserved in the base commit and a private backup. Preservation receipts authenticate the prior full closure and record the narrower fresh checks used here, without repeating the entire 2.283 GB audit. Only the explicitly requested diagnostic source and new source-safe report/tests/evidence change. Private generated assets and all existing remote heads remain protected.

Stop at this result. No Windows control pair, WSL/GPU work or generation is authorised by the writer qualification.
