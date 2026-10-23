# User-launchable AMD Image Test Preview

**PASS — separate, usable developer preview.** Resumed the saved `codex/amd-image-test-preview` working tree after the interrupted session, retaining its plan, API wrapper, tests and preservation records. Base: `63332ca5d415de8eaaff15fdaf193bc7d9935a72`. No original product source or accepted historical evidence was changed.

## What the user can open

The prepared private `preview-desktop-fixed/Launch AMD Image Test.cmd` opens a separate Electron application. Select **Generate → AMD Image Test → Choose test image…**. It displays the original image, checkerboard, filename, dimensions, size, preliminary browser checks and the definitive original CPU input-contract result. Input eligibility does not enable AMD generation or execute image preprocessing/models.

Use a single-frame RGBA PNG/WebP of at most 2 MiB with a genuinely transparent background, foreground alpha above 204 and sufficient extent. Ordinary JPEG photographs are rejected with corrective guidance; the app does not fabricate transparency. Close the preview window's X and let its launcher finish. Normal user sessions have no smoke-test deadline.

The local guide gives the exact machine-specific launcher path. Keep the prepared directory in place. This is a **noncanonical developer preview**, not an installer or release build. Canonical Vite/Tailwind/compiler dependencies were unavailable. The build reuses the previously qualified renderer, preload, CSS, Electron and installed esbuild; only guarded main-process sources are compiled. Styling retains that build's limitations and unused OBJLoader alias. Other Modly features are outside this preview's scope.

## Isolation and runtime

The launcher authenticates 45 critical files, the complete 6,113-file runtime tree, and both existing interpreter binaries and runtime probes. API Python 3.11.9 has no Pillow; validation explicitly uses the already qualified Python 3.12.10/Pillow 12.3.0 CPU environment. No interpreter discovery, installation or fallback occurs. The reused CPU validation code and panel remain unchanged.

Settings, userData, workspace, logs, temporary data and caches are confined to the new preview profile. Default-deny guards block updates, setup, syncing, downloads, extension discovery, storage redirection, generation/model routes, external navigation and new network destinations. Only local health and CPU image validation are exposed. Port 8765 must be free; an occupied port fails admission without terminating its owner. The owned Job retains 6 GiB process/aggregate limits, 12 GiB available Windows RAM and 2 GiB destination disk floors. No WSL, GPU/model execution or installed-environment changes occurred.

## Actual application evidence

The first 180-second smoke opened successfully but the native picker reported a missing private `home/Desktop` directory. It reached its explicit deadline and required owned Job cleanup (7 → 0); that run remains **BLOCKED**, preserved unchanged. The narrow correction adds Desktop to the existing confined directory map before startup. Its regression failed before the correction and passed afterward. A fresh build preserves the first preview entirely.

The corrected launcher was then exercised outside the sandbox in the supported ordinary-user context:

| Check | Observed result |
| --- | --- |
| Transparent synthetic RGBA, 320×240 / 2,274 bytes | Original CPU contract eligible; foreground 211×171 |
| Replacement with synthetic RGB JPEG, 5,760 bytes | Rejected: original mode RGB, no conversion; transparent-cutout guidance shown |
| Replacement back to RGBA | Prior result cleared; new eligible result and image displayed |
| HTTP validation | Three real `POST /amd/image-test` responses, all HTTP 200 |
| Close through window X | App return 0; launcher `NORMAL_APP_CLOSE`; Job empty before cleanup |
| Elapsed / samples | 112.14 seconds / 112 resource observations |
| Peak process / aggregate Job memory | 112,033,792 / 310,435,840 bytes |
| Minimum available Windows RAM / destination free disk | 18,615,406,592 / 139,435,016,192 bytes |
| Forced Job cleanup | False; no unrelated process terminated |

The API is deliberately stopped through its exact owned child handle when the window closes; its log reports a null exit code. This is not a claim of a graceful ASGI shutdown. The Electron application and Job settled without forced containment. Private screenshots and accessibility readbacks show both validation outcomes and the generation-disabled notice.

## Focused verification and preservation

Nine launcher tests and ten builder/bootstrap tests pass. Seven CPU API tests pass in the qualified Pillow runtime; the API-only interpreter independently has six pass/one intentional Pillow-dependent skip. No skipped case is counted as a pass. Checks cover configuration/runtime tampering, busy-port refusal, resource floors, isolated paths, guarded IPC/network routes, exact runtime probes, endpoint restrictions and the Desktop regression. The ASGI unit driver inlines threadpool scheduling; the application smoke separately proves actual HTTP handling. Previously completed panel tests were reused, not rerun as a broad suite. See [focused test evidence](amd-image-test-preview-evidence/focused-tests.json) and [actual UI measurements](amd-image-test-preview-evidence/ui-smoke.json).

See [independent prelaunch review](amd-image-test-preview-evidence/independent-prelaunch-review.json), [Desktop correction review](amd-image-test-preview-evidence/independent-desktop-correction-review.json), and the independent outcome receipt in the same evidence directory. Public evidence contains source-safe measurements and hashes; copied runtimes, images, logs and private paths remain local.

The preservation receipts verify 100 protected identities, 78 bounded content hashes, inherited historical evidence/caches, installed interpreter identity, accepted-master metadata and saved full-hash provenance, and all 135 pre-existing remote heads including main. Large historical payloads were not unnecessarily reread. A previously recorded remote-list representation correction is disclosed in the preservation receipt; no historical file was overwritten by this task.

**Scope ends here.** Genuine AMD image-to-3D generation remains disabled and NOT QUALIFIED. This preview qualifies local image eligibility testing only.
