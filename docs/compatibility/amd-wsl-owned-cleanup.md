# WSL owned-directory cleanup correction — prepared CPU candidate

Base: `9a53e6b882818a29228f4236b9249c6fedeabf48`. Branch: `codex/amd-wsl-owned-cleanup-correction`.

The original FileLock/TemporaryDirectory failure is reproduced by the focused test before correction. The surgical change adds only an unlink/rmdir audit carveout for safe basename targets relative to attested live directories inside explicitly approved roots. It verifies `/proc/self/fd`, strict resolved containment and matching before/resolved/after `fstat` identity, mode and owner metadata. Deleted, foreign, unresolved, malformed or changing descriptors fail closed. `_write_path` and all other existing methods except `audit` are unchanged; rename/link/symlink and other directory-descriptor writes remain denied.

Seven focused CPU test groups and all 16 existing policy regressions pass. The host tests use real directory metadata with simulated `/proc` transport because Windows has no Linux directory-descriptor interface. One genuine installed-WSL CPU acceptance is prepared for actual TemporaryDirectory/rmtree cleanup, FileLock 4.1.1 import and negative descriptor/path requests. AI/native imports and ctypes library loading are independently blocked; no Torch, Triton or FlexGEMM executes.

The existing binary GO, owned Windows Job, periodic health observer, 90-second guest timeout, 120-second host deadline and configured five-second natural-exit grace are reused. Helpers remain anchored to scripts/compatibility independently of the guest path. New evidence/marker paths are exclusive.

Windows preservation and passive Linux package/wheel/source preservation passed; all 922 staged payloads remain verified. A successful CPU verdict would qualify this narrow cleanup contract, not native loadability. Audit checks cannot eliminate a concurrent directory move/fd reuse between verification and the syscall; this is a contained trusted-dependency contract, not a general filesystem sandbox. Any unresolved path exception fails closed.

[Evidence](amd-wsl-owned-cleanup-evidence/). The one installed-runtime acceptance has not run yet. No GPU or native import is authorized by this correction milestone.
