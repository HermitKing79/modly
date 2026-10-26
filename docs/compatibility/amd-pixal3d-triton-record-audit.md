# Pixal3D V3 Triton binary authentication postmortem

**NO-GO for operators. Confirmed cause of the size-check failure: the approved Triton wheel's RECORD disagrees with its own ZIP payload, and V3 assumes RECORD describes those bytes exactly. Current installed-byte equality remains unverified because the separate read-only inspection failed RAM admission. No integrity exception or environment correction was applied.**

Base commit: `5f3e275e8d920bf03d6ce88dfa34004e3cc33376`. Preserved failed case: `amd-pixal3d-operators-v3-20261026T210723Z-f80ee65374ab48ea98dff9a44e3ab136`. New private audit evidence: `amd-pixal3d-triton-record-audit-20261026-v1`.

## Measured identities

The cached wheel `triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl` is **287,563,448 bytes**, SHA-256 `07787af1d28c273852f897bfeaa7bca29f2fa4a13ca0f28f535832b240ce7016`. It matches the approved pin. All 13 approved cached archives were independently size/hash verified read-only.

| Source for `triton/_C/libproton.so` | Bytes | SHA-256 |
|---|---:|---|
| Cached wheel's top-level RECORD claim |17,836,280|`395d87a54e76e49ab02374023528e9c134e5495023929fe2cc8a7c81ef7f70f3`|
| Actual uncompressed ZIP member, streamed and hashed |17,923,497|`6dd1972835cf57ab43a0dfbbbb5046e2f1deed159ac553b2a0272df68716737f`|
| Saved V3 installed-file metadata |17,923,497|NOT MEASURED|

The member is **87,217 bytes larger** than its RECORD claim; its hash also differs. ZIP metadata and actual streamed byte count agree. The top-level archive RECORD itself is 44,207 bytes, SHA-256 `2c428aae88db2c16def8d62e233a8a9ae6bee3f9b3a5922efc17d3d568fe0b21`.

V3's before/after environment snapshots are equal. The saved binary identity is inode 93791, mtime_ns 1793040721535629651, mode 33261 (regular executable file). The saved installed distribution RECORD is 45,120 bytes, inode 93869, mtime_ns 1793040721709707255; its contents/hash were not captured. These are saved metadata identities, not a measurement of current installed content or proof of bit-exact immutability.

## Why V3 stopped

`verify_installed_wheel()` in `scripts/compatibility/amd_pixal3d_operator_common_v3.py` takes expected size/hash from CSV RECORD and compares installed size before hashing the installed file. It therefore compares 17,923,497 with 17,836,280 and raises exactly the reported error. A normal extraction of the pinned ZIP payload can trigger this failure without any post-install binary transformation. The archive alone independently proves the metadata/content inconsistency; the producer build step responsible is not established.

V2 receipts confirm successful installation of all 13 pinned archives through hash-required, binary-only, no-index/no-deps pip, plus natural owned cleanup. Pip's archive-hash/install receipt authenticates the accepted wheel, not the accuracy of every embedded RECORD claim. V3 failed before operator GO; its result records `child_started:false`, all HIP operators NOT_RUN and unchanged before/after environment metadata.

No competing approved package claims this target: independent collision analysis of 13 wheels, 25,358 non-directory ZIP entries and 25,358 top-level RECORD rows found no duplicate or overlapping normalized installed destinations, including `.data` spreading. Only Triton claims this file. This rules out an overlapping file claim within the approved set; it does not establish all possible later activity in the guest.

## Installed-file inspection: safely not executed

A separately reviewed, private two-file stdlib probe was prepared for the exact installed binary and Triton distribution RECORD. It exposes no operator, provisioning, download or repair path. It checks symlink/parent identity, no-follow opens, lstat/fstat before/after, SHA-256 and the matching installed RECORD row; writes are confined to new private receipts. It retains the existing model-free three/final 20.198032 GiB prelaunch, 18 GiB startup/final, 12 GiB live floor, disk/VHD protections, 6 GiB process bounds, 120-second deadline and nonce-bound owned cleanup. It is not a V3 rerun or V4 launcher.

Its single admission attempt at **2026-10-26T21:25:32.197655 UTC** measured **20,226,510,848 bytes (18.837406 GiB)**, below the unchanged 21,687,472,128-byte requirement. It stopped on the first reading: **WSL_started:false**, no second reading, no guest process, no installed-file read, no retry. Case: `installed-file-inspection-f7c9d51a663844b8a515ebb1372c432c`.

Accordingly, current installed binary SHA-256, current symlink/resolved-path identity and installed distribution RECORD contents remain **NOT INSPECTED**. No unsupported attribution to Codex or another host application is made from this available-RAM reading.

## Decision and smallest justified next action

The pinned package's internal RECORD inconsistency is proven and explains the observed size failure. Unexpected installed modification is neither established nor ruled out at the byte level. **Retain NO-GO for operators and all new operator attempts.** Complete only the narrow read-only identity inspection in a later user-authorised session with sufficient RAM; do not rerun V3 or alter Triton.

If the installed file independently matches the authenticated ZIP member's full 17,923,497-byte/`6dd197...` identity, the smallest justified verifier correction is to derive expected installed size/hash from the **actual member of the already whole-wheel-hash-authenticated archive**, and require exact installed equality. Keep strict path/import inventory, provenance, resource gates and mismatch stops. Preserve/report RECORD discrepancies rather than adding a filename exemption, trusting installed RECORD, skipping the file or weakening hashes. No verifier change is implemented by this task.

Two additional archive-layout issues were observed, not exercised as operator attempts: Triton's ZIP has `triton/backends/amd/lib/libelf.so`, `libnuma.so`, `libtinfo.so`, while RECORD names `.so.1`, `.so.1`, `.so.6`; setuptools contains 13 RECORD suffix matches because of vendored nested distributions, while the verifier expects one. They are separate latent assumptions, not the cause of this observed stop. Any later correction needs correct top-level distribution RECORD selection and authenticated ZIP inventory handling; a one-file bypass would not qualify the environment.

Independent review confirmed the archive measurements, collision analysis and narrow-probe guards. Seven CPU fixture tests passed. During this audit there was no WSL launch, HIP operation, inference, installation, repair, binary replacement, V3 rerun or V4 launcher. Preservation checks cover recorded historical Windows file identities, protected assets, cached archives, saved failure evidence and old Git heads. Guest environment bytes remain unqualified; the audit did not write to that environment. Only reviewed text findings are published.
