# Kitchen authentication: minimal offline optimisation

Base: `8bb96247aa563823a037eb7827dc1b0f31d0b105`. **Offline patch only. Operator execution remains held.** No launcher, contract, admission threshold, deadline, ownership rule or cleanup limit changed. No execution is authorised.

## Removed work

`verify_manifest` now validates canonical `env` and `site` roots and their containment once per manifest, then uses the existing lexical installed-path mapping. The unchanged `readonly` reader still checks the full target ancestry at entry **and** exit, no-follow open, regular/unaliased file type and before/FD/after identity. No ancestor trust is cached across files.

For 25,358 installed members in 13 manifests, `plain` calls decrease from **101,432 to 50,742**: **50,690 eliminated calls**. This removes the redundant mapping-time target check and per-file environment check, replacing them with 26 root checks. Every archive, ZIP-member and installed-file SHA-256, collision check, RECORD rule and unknown-file inventory remains.

## Bounded progress

A callable in-memory tracker wraps the original guard without changing its invocation count or exception. It records package, file, action checkpoint, current-file hashed bytes, verified files/bytes by representation, monotonic elapsed and phase durations. Installed payload counters advance only after hash equality and successful reader exit. Rewritten top-level RECORDs are reported separately as hashed metadata.

Archive manifests, installed files, inventory, source integrity, startup checks and packaging-overlay validation have separate phase labels. Checkpoints also identify metadata failures before ZIP parsing, RECORD selection, budget checks and overlay validation. Existing controller traceback evidence remains available.

The capture controller enriches its **existing 250 ms resource samples** and final result; there is no per-file journal, extra process scan or new fsync. State contains current context and fixed counters rather than file history. File labels are capped at 4,096 characters, with a full-label SHA-256 when truncated. Failure freezes context/timing before cleanup. Success also freezes timing. Elapsed origin is tracker construction immediately before authentication; the unchanged controller deadline starts earlier and still includes snapshot work.

Byte counters describe the streamed archive/member/installed representations, not physical disk traffic or every small preflight/overlay metadata reread. Cross-file checks may have no individual file. Progress timing does not imply exclusive hashing CPU time.

## Offline verification

- **170 regression cases: 169 passed, one optional cached-wheel case skipped.** Existing admission, provenance, mutation-denial, pidfd, ownership, cleanup and operator-hold tests remain included.
- A separate bounded audit authenticated **all 13 cached archives**, 2,032,899,944 archive bytes and **25,358 payloads / 4,459,761,446 expanded bytes** using the tracker. It took 19.640 seconds on Windows; no archive was extracted or installed.
- A frozen copy of the reviewed original verifier matches its base-commit AST. Equivalence controls cover valid results, callback counts, hash mismatch, missing/directory payloads, unsafe members, observed parent/entry redirection, real fixture hardlinks and read-time identity change. Existing archive-tampering/collision and RECORD regressions pass.
- Progress tests cover chunk accounting before a failing guard, preservation of that exception, zero-byte files, metadata failures, RECORD separation, frozen phase timings and bounded state. The mocked controller produces the same three throttled samples and five process scans, then stops at 120.25 seconds without worker launch.
- AST comparison confirms the reader, identity, canonical-path mapping and collision helpers are unchanged. The source change optimises only redundant mapping checks; remaining changes provide requested diagnostics.

One ordered read-only Windows control on 16 existing first-party files (183,291 bytes) gave:

| Variant | Path validations | Elapsed |
| --- | ---: | ---: |
| Original | 64 | 0.393 s |
| Optimised | 34 | 0.231 s |
| Optimised with progress | 34 | 0.223 s |

All results/hashes matched. These small ordered measurements are affected by caching and scheduling. They do **not** establish WSL speed, large installed-file throughput or completion within 120 seconds.

## Preservation and decision

Read-only preservation checks passed for 3,346 historical file hashes, all 100 protected identities, 88 retained records, immutable cache identities, the 18 failed-capture files and 181 pre-existing local refs. No guest environment was opened or written. Saved V2 snapshots/evidence were preserved; no fresh live guest authentication is claimed. Private logs and benchmark details remain local.

Independent source/test review found no remaining blocking issue. Preparation performed no WSL launch, GPU/native-runtime execution, live admission, download, installation or neural operation. All original RAM/VRAM/disk/process guards and the 120-second authentication budget remain unchanged; the neural 22 GiB gate is untouched.

A future **separately approved Kitchen-import-only capture could be justified** because the original denial remains unobserved and incomplete authentication will now retain useful checkpoints. Target-Linux timing remains unknown and another 120-second timeout is possible. This patch provides no execution command, new launcher, automatic retry, operator release or neural authorisation. Stop for review.
