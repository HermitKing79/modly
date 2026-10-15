# Texture decoder: corrected bootstrap, actual WSL cold-start gate

Base `473b7bdfa9c35145e0d5c646d04ead30b515ee2d`; branch `codex/amd-turret-texture-bootstrap-correction`.

**UNKNOWN. One actual WSL CPU cold-start verification stopped at the output-root ownership guard. No inference attempt, checkpoint load, Torch/FlexGEMM import or GPU computation occurred.** The required conditional gate failed, so no genuine decoder marker or run directory was reserved. No retry or post-failure guard change.

## Correction and real observation

The new candidate uses one shared `prepare_bootstrap()` for production and the CPU probe. After the original binary GO, it creates exclusive scratch/HOME/TMP/cache directories, sets the private environment, activates the already-staged dependency paths, then imports the actual writer through `artifact_io.check_directory()`. The original missing NumPy failure no longer occurs: the NumPy-dependent writer reached its real directory validation.

The **one** cold worker used `Modly-AMD-Ubuntu-24.04`, `/opt/modly-wsl/venv/bin/python -B -I`, the existing monitored Windows Job and original filesystem/process/network audit. Torch, Triton, FlexGEMM and model imports were denied. It stopped at:

`prepare_bootstrap -> artifact_io.check_directory -> approved_root(FIELD) -> _plain()`

**`ValueError: Artifact root/files must be owned by the current user`**

The exact failed predicate is `lstat(FIELD).st_uid != os.getuid()`. This is an ownership-admission failure for the Windows-created `/mnt/c` output leaf. It does not demonstrate an AMD/native/checkpoint failure. Neither UID number nor mount ownership settings were recorded, so their actual values and the appropriate authority correction remain unproved. We did not start another WSL probe to investigate them.

NumPy 2.2.6 origin/version checks and the final excluded-module assertion were after the shared bootstrap call; the exception prevented those results from being recorded. They are **not** claimed as passing. The old 71 passing tests were not substituted for this installed-runtime qualification.

| Actual evidence | Result |
| --- | --- |
| Focused bootstrap/source regression checks | **2 passed**; expected pre-correction failure preserved |
| Real WSL CPU attempts | **1** |
| Genuine neural attempts / forwards | **0 / 0** |
| Cold worker exit / supervised time | **1 / 10.953 s** |
| Windows observer | Healthy |
| Job processes before / after cleanup | **0 / 0** |
| Forced cleanup | **False** |
| Termination of named Modly Ubuntu | Exit 0 |
| Real field output / GPU memory measurements | None / not reached |

The four computational modules (`neural`, `runtime`, `admission`, `chunked_final`) are byte-identical to the previous candidate. Original 284-tensor loading, 208 F16 / 76 F32 policy, four guides, 40 convolutions, eight C2S calls, 57 final tiles and all resource/deadline gates remain unchanged. The inference preflight now explicitly requires a passing actual cold-run record before it can reserve a real attempt.

## Persistence and preservation

Fresh outside-Git output root:

`C:\Users\wattz\Documents\ModlyArtifacts\amd-turret-texture-bootstrap-473b7bd`

Its `field` directory is empty. **No real texture file or texture-field hash exists.** The separate small synthetic CPU `probe` passed the unchanged real writer/reader and has SHA-256 `28ebf773ce72dac0c6ace8516d7914327660830ad7a231b1ab23ad317464c23e`; it is not learned material output. Host write permission does not establish WSL ownership admission.

Fresh hashes of the original saved texture latent and geometry/guides remain exactly `5506c4d0199bef64423b41b8bed560e2b7def9d29c5bb1c3072c0ea31ee1ddc4` and `dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22`. The previous failed marker, evidence and branch remain intact. All 65 prior remote heads, including main, are unchanged. Post-run preservation matched 51 protected files, 17 AMD source/configuration files, five saved inputs, five previews, the existing AMD backup and both Windows environments' selected package/venv/native identities. Driver remains `32.0.31041.1004`, controller OK; both WSL distributions are stopped, docker-desktop remains default. No package, source, driver or mount change was made. No fresh comprehensive Linux environment inventory is claimed.

[Actual CPU worker/error](amd-turret-texture-bootstrap-evidence/cold-run/guest/results.json) · [monitoring and release](amd-turret-texture-bootstrap-evidence/cold-run/results.json) · [verified summary](amd-turret-texture-bootstrap-evidence/verified-result-summary.json) · [independent outcome review](amd-turret-texture-bootstrap-evidence/independent-outcome-review.json) · [Windows preservation](amd-turret-texture-bootstrap-evidence/windows-preservation-after.json).

**Smallest next action, requiring separate approval:** inspect the exact field leaf's WSL UID, worker UID and Windows/DrvFS ownership authority read-only, then choose the narrow ownership-admission correction with that evidence. Do not globally allow foreign owners or change mount/environment settings merely to pass. Actual cold-start acceptance must succeed before a fresh guided decode can be authorized. No inference retry, alternative algorithm, installation, geometry change or texture export was performed.
