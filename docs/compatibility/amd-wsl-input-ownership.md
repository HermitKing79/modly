# WSL brick input ownership qualification

**PASS — CPU-only input preparation.** Base: `48ad0813d12c2222f97e1357eb7e76503a90d385`.
Branch: `codex/amd-wsl-input-ownership`. No production attempt was reserved; no Torch,
FlexGEMM, model, checkpoint tensors or GPU computation was loaded/executed.

## Measured cause

The genuine worker is UID/EUID/GID/EGID **1000/1000/1000/1000**, using
`/opt/modly-wsl/venv/bin/python` 3.12.3. The first rejected directory is the actual
texture root, `/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688/latent`:
UID/GID **0/0**, mode **040777**, device **71**, inode **5066549582302538** in the
diagnostic process. The texture file is also 0/0, mode 0100777. `/mnt` and `/` are
0/0, mode 040755. Every ancestor fails the original UID-equality predicate.

`/mnt/c` is 9p/DrvFS, with `uid=0;gid=0` and no `metadata` mount option;
`/etc/wsl.conf` contains only `[boot] systemd=true`. These are synthetic Linux
ownership values, not evidence that Windows ownership is wrong. Windows reports
the current user and all five original file owners as
`S-1-5-21-3582664715-2766927308-497596791-1001`. All five files have one hard link,
stable NTFS identities and no reparse points. The other observed user has only
read/execute access. System ancestors retain SYSTEM/Administrators/TrustedInstaller
ownership. NTFS and DrvFS file IDs are recorded separately, never assumed equal.

The [diagnosis](amd-wsl-input-ownership-evidence/diagnose-confirmation/guest/results.json)
records every actual ancestor, exact original predicate, mount and rejection.
An earlier CPU probe aborted on a concurrent harness source change before inspecting
inputs. Its evidence is preserved under `diagnose/`; the frozen confirmation passed.

## Narrow correction

[readonly_input.py](amd-wsl-input-ownership-evidence/readonly_input.py) compiles the
unchanged, pinned `inputs.py` and calls its real `prepare()` with a private latent
reader. Only that reader's filesystem admission gains a read-only capability for
the five exact authenticated files, backed by a trusted Windows SID/ACL/file-ID
attestation. The capability binds the actual UID-1000 worker, diagnosed mount,
live guest identities, canonical ancestors, sizes and original hashes. Reads are
bounded and identities are checked across open/read. Its writer entry points reject.

Original container, sidecar, provenance, source, coordinate, guide and checkpoint
header checks all execute. Historical readers and protected sources are unchanged;
the original UID predicate still rejects the same root in a separate control.
Symlinks, reparse points, hard links, foreign paths, descriptor aliases, traversal,
changed ownership and hashes remain rejected. No ownership, ACL or mount was changed.

## Actual results

| Check | Primary | Independent rerun |
|---|---:|---:|
| Genuine `inputs.prepare()` | PASS | PASS |
| Worker elapsed | 6.019516195 s | 6.005626928 s |
| Runtime security/validation checks | 16 passed | 16 passed |
| Observed prepared arrays collected | 15/15 | 15/15 |
| Guest exit / active Job processes | 0 / 0 | 0 / 0 |
| Forced Job cleanup | false | false |
| Windows observer | healthy | healthy |
| Forbidden framework imports | none | none |

Both runs used genuine staged NumPy **2.2.6**, the authenticated 4,430-row texture
latent and full 1,957,891-row geometry. Original readers verified all ten geometry
tensors, four BF16 guides, exact child ordering through
**4430 → 21391 → 97313 → 438684 → 1957891**, and read-only array hashes. The original
decoder source/config and **26,800-byte checkpoint prefix/header** were verified;
no weight tensors were read. The separate host suite passed **22 tests**.

Evidence: [primary](amd-wsl-input-ownership-evidence/acceptance/guest/results.json),
[independent](amd-wsl-input-ownership-evidence/independent/guest/results.json),
[source review](amd-wsl-input-ownership-evidence/review.json),
[outcome review](amd-wsl-input-ownership-evidence/outcome-review.json), and
[summary](amd-wsl-input-ownership-evidence/qualification-summary.json).
The reused supervisor's GPU-result classification remains `UNKNOWN` because its GPU
contract was intentionally unused; separate CPU verdicts require successful guest
exit, healthy monitoring, empty Job, no forced cleanup and named-distro termination.

Fresh Windows verification brackets the full qualification. Original input hashes,
100 protected files, historical failed markers, evidence, selected environment/native
identities and source edits/backups are preserved; scope is detailed in the
[preservation receipt](amd-wsl-input-ownership-evidence/preservation-after.json).
Private Windows ACL evidence remains outside Git in
`artifacts/amd-wsl-input-ownership-48ad081/windows-input-attestation.json`.

**Next:** separately approve using this exact reviewed read-only adapter in the
owned guided-PBR worker and one genuine forward, retaining every existing RAM,
VRAM, deadline, precision and output-acknowledgement gate. This qualification does
not execute or claim completion of that forward. No GPU work follows automatically.
