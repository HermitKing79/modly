# Outside-sandbox hardlink comparison: PASS

Base `e971fbca9e06c81648ade22270777c8ec5289c2c`; isolated branch `codex/amd-writer-outside-sandbox`.

Exactly one unchanged tiny writer probe succeeded through the supported `exec_command` request with `sandbox_permissions=require_escalated`. This requests unsandboxed execution, not Windows administrator elevation. The request was accepted and returned exit code 0. There was no sandboxed fallback, retry, alternate publisher, Job, fixture, WSL, GPU or generation execution.

## Context and authentication

The [reviewed launch command](amd-writer-outside-evidence/outside_command.ps1) directly invoked the original API interpreter with the same `-B -I -S` flags. Those Python flags remain interpreter isolation settings; they do not describe the Codex OS sandbox. Before invoking Python, the shell verified the unchanged probe/interpreter hashes, fresh C: namespace, matching Windows user name/SID, and disabled administrator role. SYSTEM and anonymous identities were rejected. No `RunAs`, administrator request, ACL/security change or alternate launch route was used.

Execution-mode evidence is the supported tool request and returned execution, corroborated by the same-shell identity record. The probe has no independent sandbox detector. Administrator-role-disabled does not mean the account lacks administrative group membership. Raw identity and tool receipts remain private.

Probe SHA-256: `08769da0449f6b82a2db8e18619cf7b78147e028bab09842782366b1c157622e`. Interpreter SHA-256: `5b9eb768cd084d4b7959bc83b1a5111a10f8aaa18e5f4f7ef4187baf85dcf2c3`. Both match the failed baseline. The original and new private roots report the same filesystem device on C:.

## Actual result

| Boundary | Outside-sandbox result | Elapsed ms |
| --- | --- | ---: |
| Exclusive staging creation | PASS | 0.3020 |
| JSON write | PASS | 0.0232 |
| Flush | PASS | 0.0589 |
| `os.fsync` | PASS | 0.5716 |
| Close | PASS | 0.0642 |
| No-overwrite `os.link` | PASS | 6.9478 |
| Exact final readback | PASS | 0.9004 |
| Owned-file cleanup | PASS | 2.4866 |

The measured Python process interval was 1.8248196 seconds. Boundary durations include Python work and ownership checks; they are not isolated syscall benchmarks. The published hardlink shared the staging identity, readback verified the expected JSON and exact Windows newline bytes, and cleanup verified link counts/identities before removing only the two newly owned names. The fresh scratch directory remains empty.

Payload: 151 bytes, SHA-256 `cdab4d6ba143e10fe8fd7bfb11c7127332720031cdeb5a89de5013ac9903b30c`.

Private full report SHA-256: `48468f2003f5314e1e1e5fa2326f7ca5fe5fb925ab12c2dac4d9acab8723972b`. The [independent outcome review](amd-writer-outside-evidence/independent-outcome-review.json) checks the actual saved observations and comparison.

## Interpretation and preservation

The earlier default-sandbox probe remains **BLOCKED at `os.link`, errno 13 / WinError 5**. The unchanged probe succeeds outside that context on the same C: volume. This demonstrates different outcomes in the two tested contexts; it does not identify the exact Windows policy responsible, prove the discarded historical exception, or qualify lifecycle supervision.

No existing tracked source, publication mechanism or installed environment was changed. Preservation receipts retain the original failed reports, exact 1,991-byte historical partial, protected100 metadata and 78 previously selected small-file hashes, old cache identities and all 124 prior remote heads. They authenticate the prior full closure without repeating a bulk model/environment rehash. Preparatory read-only path and receipt-line-ending assertions were corrected before the sole probe; neither launched a probe or consumed a production marker.

**Stop.** Any later Windows control-pair qualification needs separate approval and its own reviewed admission. No further filesystem test or lifecycle execution follows automatically.
