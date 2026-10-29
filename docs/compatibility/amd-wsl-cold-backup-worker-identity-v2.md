# Cold-backup v2 — demonstrated worker identity correction

Preparation only. Real backup, WSL startup, resize and neural decoding remain
unexecuted; one new backup requires separate execution approval.

## Cause and evidentiary limits

The preserved v1 owner recorded worker PID **8028**, controller PID **7412**
and the same canonical 32-character token as `contract.json`. Job attachment,
return code 1 and natural cleanup were recorded. The worker rejected before its
reporting/file-access section, so neither `worker-result.json` nor an observed
inner PID exists. Do not invent that missing historical PID.

Windows Python venv executables are redirectors.
[Python 3.12 documentation](https://docs.python.org/3.12/library/venv.html#venv.EnvBuilder.setup_scripts).
An identity-only probe with the exact SHA-authenticated venv executable and
`-B -I -S` reproduced **both** mismatches: `Popen.pid` named the launcher,
Python's PID named its child, and Python's parent named the launcher instead
of the controller. Tokens matched. Exact Job process lists contained both;
cleanup was natural. A second probe launched the already-installed base Python
directly and demonstrated exact PID and parent matches. Native process-image
observation agreed with that same base executable. No backup/WSL code ran in
these probes. This demonstrates the failure mechanism, rather than supplying
unrecorded values for the historical run.

## Small correction

The controller still starts through the qualified venv. Only its stdlib-only
worker now launches the same underlying base Python **directly**, with unchanged
`-B -I -S`. The fixed worker binary SHA-256 is:
`4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a`.
Its canonical path must also equal the controller's `_base_executable`.
No installation, environment or package changes are involved.

The fixed subprocess allowlist replaces the venv worker command; it does not
accept arbitrary descendants or guessed PIDs. Job attachment still precedes
GO/EOF. Exact worker PID, exact parent PID, canonical token and the complete
owner-record schema still must match. Rejections now include PID diagnostics
without printing the nonce. An unauthenticated worker still cannot write
receipts or open the source; its stderr and controller FAIL receipt preserve
the refusal. Exclusive source/directory/destination/volume checks, independent
full hashing, resource limits and owned cleanup are unchanged.

## New one-shot routing and acceptance

- Fresh backup: `E:\Modly-Cold-Backup-closed-solid-v2\ext4.vhdx`.
- Fresh evidence: workspace `artifacts/amd-wsl-cold-backup-v2`.
- Existing launcher: `scripts/compatibility/Invoke-ModlyColdBackup.ps1`, with the
  exact new published commit and explicit approval flags after separate approval.
- Reuse the unchanged authenticated private v1 preparation `authority.json`.
  Do not delete or rewrite either the consumed v1 case or its backup path.
- All prior admission/resource/ownership/file-integrity conditions from
  `amd-wsl-cold-backup-v1.md` remain unchanged; only the fixed paths above and
  worker executable change. CPU live floor 12 GiB; child Job aggregate/private
  commit 512 MiB; wall deadline 1800 s; D: floor 50 GiB; C:/E: reserves 2 GiB;
  E: initial requirement 33,880,539,136 bytes at the pinned source size.
  Neural 22 GiB gates and the decoder's 5 GiB guest floor are untouched.

Only v2 `final-result.json` top-level PASS accepts the future backup. Preserve
all failed/partial output. No retry, resume, overwrite, alternate destination,
resize or decoder follows automatically. Binary/path/authority changes or any
guard failure stop before proceeding. No live resource admission or new backup
has been run during this preparation.

## Verification

Regression tests reproduce the qualified venv's actual two-field mismatch
under a Windows Job, then pass the actual controller handshake using the
authenticated direct worker on a harmless C: probe. Negative tests reject wrong
PID, parent, token, schema, base path, binary hash and old/foreign commands.
Existing synthetic copy/hash/partial-file/native-lock/Job safety tests remain.
Independent review and private before/after evidence/ref manifests accompany
publication. Real source VHDX contents, filesystem health, actual backup hashes,
elapsed time and peak resources remain unmeasured until separate execution.
