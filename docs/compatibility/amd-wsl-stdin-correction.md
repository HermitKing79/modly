# AMD WSL stdin correction — UNKNOWN

Base: `codex/amd-wsl-rocdxg-smoke`, `1d9d5c9e13ba2509102e208d6bc5c649ccf0067d`.
Continuation: `codex/amd-wsl-stdin-correction`. CPU-only; no GPU attempt.

The authorization defect is confirmed and corrected. Overall acceptance remains
**UNKNOWN** because the corrected batch required forced cleanup of two remaining
owned Job processes. This is a process-release qualification, not evidence of a
ROCm numerical failure. Their identities were not recorded; no cause is inferred.

## Observed cause and correction

The original supervisor requested `GO\n` (`474f0a`). Its Windows UTF-8 text pipe
actually wrote `GO\r\n` (`474f0d0a`), and the Linux worker received those exact
bytes. The original production gate read `'GO\r\n'` and rejected it. EOF or a
lost token was not the observed cause.

Production now uses binary `Popen` stdin, writes `b"GO\n"`, and validates
`sys.stdin.buffer.read(4) == b"GO\n"`. The host closes stdin after the token:
the fourth-byte read requires EOF and rejects trailing data, including CRLF.
Missing authorization, incomplete tokens and EOF remain failures. Job ownership,
monitoring, guest deadline, GPU contract and original attempt marker are unchanged.

Corrected production SHA-256:
`19e1b7bc59c23f593408e57a995b7fa80d1313e92a5d3ca4704636d106209d69`.

## Actual CPU acceptance

The standard-library probe used the production WSL command builder, authorizer,
strict guest gate, health observer and owned Job implementation. Only the guest
script was replaced with a byte recorder; it never called the GPU worker. The
host recorded the actual pipe writes, and the guest recorded its actual stdin.

| Case | Host and guest bytes | Production gate |
| --- | --- | --- |
| GO + LF | `474f0a` | Accepted |
| Wrong token | `4e4f0a` | Rejected |
| CRLF | `474f0d0a` | Rejected |
| Missing newline | `474f` | Rejected |
| Extra data | `474f0a58` | Rejected |
| Empty line | `0a` | Rejected |
| EOF | empty | Rejected |

All seven guests exited 0 after recording the expected acceptance/rejection;
none imported Torch or Triton. The batch took 41.25 seconds. Windows monitoring
remained healthy (maximum observation-start gap 5.016 seconds). Job cleanup found
two active processes, forcibly terminated them and verified zero remaining.
Named-distro termination returned 0. Therefore clean unforced exit of **every**
owned process was not proved, and the saved aggregate stays UNKNOWN.

Focused host-only discovery passed **20 tests**. Tests cover the real production
writer/gate, bounded extra-data rejection and supervisor binary launch contract.
A cleanup regression proves healthy monitoring continues after generic Job/case
errors; reintroducing the old condition in memory fails that test. Independent
review reran all 20 tests and found no remaining blocking issue. This final
probe-only failure-monitoring correction was CPU-tested, without repeating WSL
cases. Exact executed probe snapshots and hashes are retained with both batches.

## Preservation and next action

The 61 original WSL evidence files and historical local/remote branches match
their starting identities. Linux package lists and all 26,617 venv metadata
entries match before/after (Python 3.12.3; kernel
6.18.33.2-microsoft-standard-WSL2). Both Windows AI environments, AMD source edits,
backup and selected native artifacts were independently checked. The AMD driver
remains 32.0.31041.1004. Both WSL distributions are stopped; `docker-desktop`
remains default. No packages, drivers, kernels, models or installed source were
changed. No GPU initialization, computation or native-extension execution occurred.

The binary handshake is ready, but this evidence does **not** fully qualify a new
GPU smoke attempt under the clean-exit requirement. The single recommended next
action is a separately approved, CPU-only single-worker teardown observation
recording remaining owned Job process identities before named-distro termination.
Do not retry GPU work or change cleanup policy on the strength of this report.
The old GPU attempt marker remains intact.

Evidence: [original transport](amd-wsl-stdin-evidence/original-text/results.json),
[corrected transport](amd-wsl-stdin-evidence/corrected-binary/results.json),
[focused tests](amd-wsl-stdin-evidence/cpu-final.stderr.txt),
[review](amd-wsl-stdin-evidence/independent-review.json),
[Git/history preservation](amd-wsl-stdin-evidence/preservation-after.json),
[Linux preservation](amd-wsl-stdin-evidence/linux-environment-after.json).
