# Pixal3D V3 authenticated payload correction

**GO to prepare one conditional operator-only handoff; HIP support remains unproven.** No execution, live RAM admission, WSL launch, installation, download or GPU work occurred during this preparation. This corrects the verifier, not the installed environment.

## Evidence and correction

The completed read-only inspection found the installed `triton/_C/libproton.so` to be a regular, unaliased file of **17,923,497 bytes**, SHA-256 `6dd1972835cf57ab43a0dfbbbb5046e2f1deed159ac553b2a0272df68716737f`. It exactly matches the payload in the approved whole-wheel-authenticated archive. Archive and installed RECORD instead claim 17,836,280 bytes and SHA-256 `395d87a54e76e49ab02374023528e9c134e5495023929fe2cc8a7c81ef7f70f3`. The installed RECORD itself hashes to `0d32fb4c9790e0d79cc472144469baccc06fbe83fa255933dc7be28f1b41adab`. The inspection reports natural cleanup, no environment writes and no operators.

The corrected verifier authenticates each complete wheel against its original pinned size and SHA-256 **before parsing ZIP contents on that same open handle**. It streams every actual file's SHA-256 and size, validates literal destinations including `.data` spreading, and rejects cross-package collisions before installed-file verification. Installed payloads must match exactly; canonical parent checks, no-follow opens, regular-file/hardlink checks and before/after identity checks remain enforced. Unknown installed runtime files remain rejected. No binary receives an exemption.

All 13 pinned archives passed offline: **25,358 files / 4,459,761,446 expanded bytes**, zero destination collisions. Triton has **13 metadata discrepancies**: seven size/hash disagreements, three RECORD-only names and three payload-only names. The other 12 archives have none. Full safe details are in `amd-pixal3d-payload-verifier-verification.json`. The actual `libelf.so`, `libnuma.so` and `libtinfo.so` are required and hashed; RECORD's `.so.1`/`.so.6` names are reported without substitutions or equivalence assumptions.

Only the exact top-level `<distribution>-<version>.dist-info/RECORD` is treated as pip-rewritten installation metadata. Its installed bytes are still hashed and reported separately. Setuptools 84.0.0's 12 nested vendored RECORDs remain ordinary payloads and must match exactly. Existing narrowly identified pip-generated metadata and ensurepip bootstrap handling are retained; operator imports of pip remain prohibited.

Each authenticated archive's metadata discrepancies are journaled before checking installed payloads, so a subsequent installed-file failure does not discard the archive evidence. Successful installed authentication also records installed RECORD hashes. The original cache and environment are never rewritten or repaired.

## Verification and preservation

The stale-RECORD regression reproduced the original size failure before the correction. The focused guard/cleanup/authentication suite completed **97 tests: 96 passed and one optional real-cache audit skipped**. That audit ran separately with an explicit immutable cache path: **18 tests passed**, including all 13 wheels. The sets overlap; this is **97 distinct tests passed overall**. Adversarial cases cover stale/missing metadata, unknown files, same-length tampering, missing installed files, whole-archive tampering before ZIP parsing, nested RECORD tampering, ZIP links/special files, installed hardlinks, path traversal, NUL truncation and cross-wheel/spread/file-directory collisions. Windows symlink-type/parent-redirection observations were injected without requiring privileged symlink creation; the real production checks were exercised. A real NTFS hardlink fixture was rejected.

No-work PowerShell, host and guest entrypoints passed dry-runs. The regression harness blocked child launches, network calls and live admission reads; no Torch, Triton or Kitchen runtime was imported. Independent review identified directory-claim and original-ZIP-name gaps; both received failing regressions and fixes before final review.

Preservation checks reconfirmed 3,346 historical file hashes, all 100 protected identities and 88 retained records, failed attempt evidence, the completed installed-file inspection, immutable cache identities and old Git heads. Only reviewed code, tests and text are published. Original evidence is untouched.

## Fresh one-shot handoff

Entrypoint: `scripts/compatibility/Invoke-Pixal3DOperatorsV3.ps1`. Use the exact published commit in the final handover with `-Run -ApproveOperatorProfileV3 -ExpectedCommit <reviewed-commit>` from a normal, non-administrator PowerShell after closing Codex/ChatGPT Desktop and browsers as appropriate. The checked-out source must match that commit. The switches are the deliberate approval; no execution occurs by default.

The existing V2 environment `/home/modly/pixal3d-native-proof-v2/.venv` and immutable `D:/AI/Modly-Pixal3D-Proof-v1` cache are reused read-only. Nothing is installed or downloaded. New exclusive paths are:

- Windows ownership root: `D:/AI/Modly-Pixal3D-Operators-v3-payload-auth`.
- Guest scratch: `/home/modly/pixal3d-operator-proof-v3-payload-auth`.
- Local evidence: `artifacts/amd-pixal3d-operators-v3-payload-auth-<UTC>-<nonce>`.

The Windows root was absent at preparation; guest exclusivity is checked by the controller without creating it before authentication/GO. Any occupied path, authentication error, resource failure, failed operator or cleanup error stops without retry. Historical V3 paths are not reused. The older V3 assessment describes the original attempt and is retained as history.

All resource policies are unchanged: three fresh **20.198032 GiB** readings four seconds apart and final prelaunch admission; **18 GiB** throughout controller startup/authentication and immediately before GO; **12 GiB** live host floor; existing 6 GiB process bounds, disk/VHD/VRAM/allocator limits, deadlines, nonce/PGID ownership and cleanup. Neural admission remains **22 GiB** and 24 GiB remains preferred only. The operator profile is an admission budget, not a measured GPU peak or hard total-VM bound.

The controller authenticates the full V2 environment before any operator GO. Only then may the original synthetic gfx1201, HIP INT8/ConvRot, SDPA, NAF, sparse and conversion checks run. No provisioning, model loading, inference or generation route is introduced. Inspect the new `REPORT.txt`, `summary.json`, archive journal, environment authentication, operator results and cleanup receipts after natural exit. A PASS ends at review; it does not authorize Pixal3D generation.

## Remaining limits

Only `libproton.so` has the separate live installed-byte inspection supporting this correction. The full current installed closure, controller authentication time/memory and HIP numerical/backend checks remain pending the conditional run. The 120-second authentication deadline and memory stops have not been raised; extra payload streaming can still fail those limits. A later failure retains NO-GO and its evidence. No broad Modly UI/integration work or full application test suite was part of this focused verifier change.
