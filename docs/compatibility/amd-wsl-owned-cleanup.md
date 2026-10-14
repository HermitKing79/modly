# WSL owned-directory cleanup correction — PASS

Base: `9a53e6b882818a29228f4236b9249c6fedeabf48`. Branch: `codex/amd-wsl-owned-cleanup-correction`. The exact reviewed candidate was GitHub-verified at `289c58c11bcefc7cc5db6af8d105bc7852631f4a` before the single installed-runtime acceptance.

**Actual CPython TemporaryDirectory/rmtree cleanup and genuine FileLock 4.1.1 import succeeded under the corrected production policy. All 13 unsafe installed-runtime requests were rejected; no prohibited library imports or unapproved write effects were observed. Containment and cleanup passed.** No FlexGEMM, Torch, Triton, GPU, model or native binding execution occurred. The historical failed import remains UNKNOWN and unchanged.

## Surgical contract

Only `os.remove` (the audit event for `os.unlink`) and `os.rmdir` gain a descriptor-relative cleanup path. It requires an exact nonnegative integer descriptor and a single safe relative basename. Empty/dot/dotdot names, absolute paths, both embedded separator types, NUL and malformed arguments are rejected.

The guard reads `/proc/self/fd/<fd>`, rejects deleted or nonabsolute targets, resolves the parent strictly inside an approved writable root, and compares directory `fstat` before/after with the resolved directory's metadata: device, inode, mode, UID and GID. Positive inode/link counts and an unchanged descriptor target are required. It checks the parent entry; unlinking an owned symlink removes the link without writing its foreign referent.

`_write_path` is unchanged. Every other existing method except `audit` is AST-identical. Nondefault descriptors remain denied for rename, link, symlink, mkdir, chmod, chown and utime. Network, process, tensor, GPU, native-function and Triton restrictions are unchanged. This adds 38 lines and changes one existing line in the policy, with directly related focused tests; no process-manager redesign occurred.

## Tests and real acceptance

| Boundary | Actual evidence |
|---|---|
| Before correction | Focused owned cleanup reproduced the exact prior policy rejection |
| Host focused tests | 7 cleanup test groups passed |
| Existing host policy regressions | All 16 passed |
| Installed WSL CPU acceptance | Exactly one; 19 checks: 6 accepted, 13 rejected |
| Real cleanup observation | 27 CPython remove/rmdir audit attempts, including denied requests |
| Real positive routes | Nested TemporaryDirectory cleanup, shutil.rmtree, owned FD unlink/rmdir, FileLock import, owned symlink-entry deletion |
| Negative routes | Foreign and symlink-escaped directories, nondirectory/closed/deleted/negative FDs, traversal, separators and unsafe basenames |
| Postconditions | TemporaryDirectory and FileLock probe files removed; foreign sentinel and retained owned marker unchanged |
| AI/native import attempts / modules | Both empty; ctypes loading was independently denied |
| Worker / outer exit | 0 / 0 |
| Supervisor / outer duration | 15.500 / 15.783 seconds |
| Windows monitoring | Four healthy observations |
| Owned Job | Naturally empty; no forced cleanup; five-second grace observed empty at 0 s |
| Distro stop | Only `Modly-AMD-Ubuntu-24.04`, exit 0 |

The genuine Python is 3.12.3 under `/opt/modly-wsl/venv`. FileLock loaded from its actual baseline package; `filelock._strict` SHA-256 is `28799c08bff96b810dc810601d6f046b8875e19b21986d5dc86177a4a0331403`. Torch `2.9.1+rocm7.2.1.lw.gitff65f5bc` and Triton `3.5.1+rocm7.2.1.gita272dfa8` identities were read from distribution metadata without importing either package. GPU memory was not queried.

[Guest results](amd-wsl-owned-cleanup-evidence/run/guest/results.json) include actual directory metadata and the complete checks; [durable journal](amd-wsl-owned-cleanup-evidence/run/guest/phases.jsonl), [CPU verdict](amd-wsl-owned-cleanup-evidence/run/CPU-verdict.json), [verified summary](amd-wsl-owned-cleanup-evidence/verified-result-summary.json) and [independent outcome review](amd-wsl-owned-cleanup-evidence/independent-outcome-review.json) preserve the observed result. The unchanged GPU supervisor's raw UNKNOWN/exit 1 is retained because this worker intentionally supplies no GPU computation result. It is not relabeled; the separate CPU PASS requires real checks and healthy/unforced containment.

## Preservation and limits

The [Windows after audit](amd-wsl-owned-cleanup-evidence/windows-preservation-after.json) matched both installed environments, all 12 selected native binaries, AMD edits/backups, four saved turret artifacts and decoder metadata/header. No checkpoint payload was read, hashed or loaded. Passive Linux [before](amd-wsl-owned-cleanup-evidence/linux-preservation-before.json)/[after](amd-wsl-owned-cleanup-evidence/linux-preservation-after.json) checks matched the baseline venv, package/dpkg/source identities, approved library metadata and all 922 staged payloads, including the original wheel. These metadata workers imported no AI/native/FileLock package and were not extra acceptances. Final named-distro stop and default Docker registration are recorded separately. All 133 historical evidence files, four unchanged control sources and four preexisting bytecode files hash-match; only the approved policy control changed.

Identity is attested at the audit check. Concurrent FD reuse or directory moves before the syscall cannot be made race-proof by a Python audit hook; this is the bounded trusted-worker cleanup contract, not a general filesystem sandbox. Unresolved attestation errors fail closed; a resolve-loop RuntimeError may be reported directly rather than as ImportPolicyError. Permission-repair paths through nondefault chmod/chown/utime remain unqualified and denied. No unrelated obstacle appeared in this one acceptance.

**Next: separately approve one fresh contained genuine FlexGEMM native import using this reviewed policy and the already-built wheel, preserving the previous import limits and guards.** Native loadability and all 18 binding resolutions remain unproved. Do not execute that import, load the decoder or run any forward automatically. Stop here.
