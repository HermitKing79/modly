# Job-owned AMD stage orchestrator v1 implementation plan

Base `c5f7ccd3cdb3de371405c09856a356ebc41a6ba3`; isolated branch `codex/amd-job-stage-orchestrator-v1`.

Goal: bind one product job's actual image and predecessor artifacts to an ordered, owned stage lifecycle. This implements the user's supplied design; genuine neural execution and the visible AMD capability remain disabled.

The existing completion context/authenticator/publisher remain authoritative. A new small orchestration unit owns run/stage receipts, a contract unit validates decoded images and actual bounded artifacts, and an OS-process unit owns Windows Jobs and Linux sessions. No old brick/turret output is a source for a new request.

## Units and tests

1. `api/services/amd_stage_contract.py`, `test_amd_stage_contract.py`: actual image decoding and canonical preprocessing metadata, strict tensor headers/layout/dtype/finite payloads, dynamic row ceilings and exact coordinate payload lineage. Negative tests precede implementation. No Torch/model import.
2. `api/services/amd_stage_process.py`, `amd_stage_bootstrap.py` (separate Linux broker only if necessary), `test_amd_stage_process.py`: host-owned process Job, GO gate, bounded log/control protocol, Linux subreaper/session/PID-start identity, exact descendant reaping and terminal acknowledgement. Mock-only unit checks first; actual OS claims require the single reviewed CPU execution.
3. `api/services/amd_stage_orchestrator.py`, `test_amd_stage_orchestrator.py`: immutable host-defined ordered plan, exclusive per-run root, exact image/parameter/model context, source pins, predecessor receipt hashes, live admission, timestamps and durable receipts. Cancellation cannot return a completed result; unverified or forced settlement blocks the chain. Synthetic results cannot become an `amd-worker` GLB.
4. Minimal `api/routers/generation.py` job-owner cancellation hook and focused route tests: request cancellation without killing only `wsl.exe`; settle the owned stages before a clean cancelled state. Legacy generators retain their existing route. An unregistered adapter uses the existing completed-result interface; no manifest or user-visible Generate registration.

## Safety and admission

- CPU evidence workers retain 6 GiB Windows process AND Job ceilings, 180-second workers, 12 GiB available Windows RAM and 2 GiB destination-free floors. No arbitrary child process is launched before Job attachment.
- WSL client exit is not Linux settlement. Only the exact named distribution/UID and a nonce-bound broker acknowledgement can establish Linux completion. No distro shutdown/termination, broad `pkill`, permission/mount changes or unrelated application termination.
- Linux per-process address-space limits and owned-RSS monitoring must be described separately from Windows Job limits; no cgroup hard-cap claim is made.
- Missing identities, unsupported capabilities, malformed receipts, stale/foreign outputs, failed admission and incomplete/forced cleanup fail closed.
- Real-stage original GPU/host budgets are recorded from source. CPU tests cannot qualify VRAM or GPU allocator enforcement.
- Existing larger-support geometry/PBR profiles authenticate the original brick only. No arbitrary-image inference is admitted merely by replacing hashes or historical row counts.

## Single execution and review

First independently review all contract/process/orchestration code and focused CPU regressions. Then admit at most one synthetic lifecycle execution, with its exact scenarios frozen in the execution harness. It must use real owned OS workers where ownership/cancellation is claimed, keep outputs synthetic/private and never publish a generated completion. Missing WSL prerequisites produce a separate BLOCKED WSL verdict, not a Windows mock replacement. Preserve any failed execution; do not retry it.

Finally review saved receipts independently, run relevant existing generation/workflow/completion tests, perform one preservation AFTER check, and publish only source/text evidence. The accepted 97 MB replay is not repeated. Fresh neural inference remains NOT RUN.
