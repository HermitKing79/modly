# Genuine turret texture field: explicit Windows-owner admission

Base `067f668feb8784f163f006c1cb7708e43b86fc24`; branch `codex/amd-turret-texture-owner-admission`.

**CPU ownership acceptance PASS. Genuine texture decode not yet attempted.** A single real forward remains conditional on publication of this reviewed candidate and unchanged resource/source admission.

## Measured cause and narrow correction

The actual baseline WSL Python 3.12.3 worker has UID/EUID/GID/EGID **1000/1000/1000/1000**. The old exact Windows field leaf reports Linux UID/GID **0/0**, mode **040777**, device **0:71**, inode **7036874419322825**. Trusted `/home/modly` scratch reports UID/GID 1000/1000, mode 040755, ext4 device 8:48. `/mnt/c` is 9p/DrvFS with `path=C:\;uid=0;gid=0`, without metadata; `/etc/wsl.conf` only enables systemd, `/etc/fstab` is unconfigured. This is synthetic ownership reporting, not an observed loss of Windows ownership.

Windows owner equals current `HERMIT\wattz`, SID `S-1-5-21-3582664715-2766927308-497596791-1001`. The exact leaf/parent ACL grants writes only to that SID, SYSTEM and Administrators. SID ending1002 has ReadAndExecute only. NTFS; all inspected ancestors have no reparse point. Both namespaces hash the existing probe to `28ebf773ce72dac0c6ace8516d7914327660830ad7a231b1ab23ad317464c23e`. Numeric lstat metadata, interpreter SHA, mount/configuration text and full Windows SID/ACL records are preserved in [read-only diagnosis](amd-turret-texture-owner-evidence/diagnosis-run/guest/results.json) and [Windows diagnosis](amd-turret-texture-owner-evidence/windows-ownership-diagnosis-verified.json).

The selected correction is an explicit immutable ownership capability for the two freshly host-attested `field`/`probe` leaves, and only the two exact artifact filenames. The host-generated attestation is hash-pinned into the candidate. Windows checks actual current SID, root/parent ACL and NTFS identity before authorization; file SID/ACL and root identity are checked before and after independent readback. The guest checks the exact canonical drive mapping, diagnosed UID1000, current synthetic C:9p mount and live root device/inode. Device IDs are bound per worker, because WSL boots can change them. No assumed NTFS-to-9p inode arithmetic.

Default UID checks remain active everywhere else. Authority is threaded through root admission, file admission, pathname and opened-handle reads, exclusive writes, internal verification and final byte comparisons. Alias/reparse, outside-Git, single-link, exact-name, identity, bounds, exclusive xb/fsync and tensor/hash checks remain. No chown/chmod, ACL/mount changes, installation or baseline environment edit. WSL-owned staging/export was rejected as extra copying/receipt machinery when direct confined Windows authority is supported by the measured evidence.

## Actual CPU acceptance and persistence

**One real WSL CPU acceptance passed**, using the actual baseline interpreter and staged NumPy **2.2.6** from `/home/modly/native-decoder-preparation-6dd94a85/site-packages`. It called the shared production bootstrap/directory checks, wrote a two-row fixture with the real writer, and completed the standard safetensors reader plus every internal readback/owner check. Four negatives reject default unowned-root access, default unowned-file access, a foreign root capability and a foreign filename. Torch/model/Triton/FlexGEMM imports were prohibited; policy reports zero native calls, zero tensor operations and zero rejected audit operations.

Windows independently reopened the same file, verified its actual SID/ACL before and after reading, and compared every tensor against separately encoded little-endian CPU arithmetic. Fixture SHA-256 **`92dc27521f2adeb4394e80a6b361b162c3b7aadd9947cb53e66809579f989fe5`**. The fixture is synthetic, not learned output. Worker exit0, observer healthy, natural empty Job, no forced cleanup, named distro termination exit0. [CPU verdict](amd-turret-texture-owner-evidence/cold-run/CPU-verdict.json) · [CPU/containment evidence](amd-turret-texture-owner-evidence/cold-run/results.json) · [independent Windows readback](amd-turret-texture-owner-evidence/persistence-preflight.json).

23 focused CPU tests: **22 passed, one Windows symlink-privilege skip**. Seven tests cover the explicit owner context and whole writer path; the 16 existing writer regressions cover the directly changed code. The original 71 passing suite was not rerun. The initial sandbox denied temporary-file cleanup; ordinary-permission reruns and all diagnostics are retained. PowerShell ownership queries use the established observer's process-local Desktop module resolution; no global policy was changed.

The numerical implementation is unchanged: 284 checkpoint entries, 208 F16 / 76 F32 parameters, original saved texture latent and four exact guides, 40 convolutions, eight C2S operations, 57 final tiles and full 1,842,968 support. Original 13GiB free-VRAM / 12GiB host-RAM admission, 12GiB allocator ceiling, reserves, native/CPU/GPU 60/120/240s deadlines, guest/host 450/480s deadlines, Windows monitoring and Job/distro containment remain mandatory. Actual read-only model/source/output preflight passed; no real weights or GPU work were involved.

Permanent output root:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-owner-067f668`

Its `probe` contains the CPU fixture; `field` remains empty before inference. A real result must contain finite raw and unclamped `raw * .5 + .5` F32 `[1842968,6]` plus exact original I32 coordinates, with independent verification before model release and after exit. No real field, output statistics or peak GPU measurements are yet claimed.
