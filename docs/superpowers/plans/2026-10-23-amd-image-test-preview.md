# User-launchable AMD Image Test Preview

**Goal:** Deliver one separately launchable, noncanonical Windows preview using the accepted image panel and existing runtimes. The user's detailed execution request supplies the design and publication approval.

**Architecture:** Reuse the successful Electron/dependency copy and image panel; a reproducible developer builder writes only a fresh private destination. Preview-only guards select isolated persistent directories, disable installation/update/sync/downloads, and expose the unchanged CPU image-validation endpoint through a restricted local API. One Windows command invokes a verified interpreter and a small launcher. The installed application and historical smoke builds remain unchanged.

**Tooling decision:** Prefer canonical tooling only if already complete. Existing checks find no repository node_modules and no installed Vite/Tailwind/compiler chain. Do not run build-builtins/package/installer or acquire dependencies. Reuse qualified esbuild, automatic JSX and installed CSS; disclose styling/packaging limitations.

## Work

- [x] Authenticate the prior UI PASS, source/runtime identities, protected assets and remote branch baseline; create a new branch and exclusive preview directory.
- [x] Add the reproducible preview builder and guards. Preserve existing picker/panel/validator. Keep all profile, logs, temporary data and workspace private to the preview. Block settings redirection and all model/generation actions.
- [x] Add one Windows launcher with source/runtime verification, explicit existing API/CPU Python configuration, free-port refusal, bounded owned-process monitoring and normal-close evidence. Normal user operation has no five-minute smoke TTL or debugger dependency.
- [x] Run focused CPU tests for isolation, busy-port refusal, tampered configuration/runtime rejection and restricted CPU endpoint behavior. Independently review before app launch.
- [x] Run the actual preview through its user launcher: valid cutout, invalid image, replacement and visible results; close normally and record resources. No GPU, WSL or model execution.
- [ ] Publish source-safe tests/review/report and verify GitHub plus prior heads. Give a short guide with the exact existing launcher path and limitations.

## Review focus

Busy ports must fail without terminating their owners. Persistent settings cannot redirect writes into the installed application's directories. Missing or changed CPU runtimes must fail without fallback or installation. Only original bytes reach the CPU validator. Input eligibility must not imply enabled generation. Historical proof files and private images must not be overwritten or published.

## Resume checkpoint

The interrupted working tree and saved API tests were preserved. The missing launcher/build was completed and independently reviewed. An initial bounded UI attempt exposed the missing private Desktop directory; its failure and forced owned cleanup remain recorded. A one-line confined-directory correction passed a RED/GREEN regression. A fresh preview then completed the actual RGBA → JPEG → RGBA sequence and normal close in 112.14 seconds, without forced Job cleanup. Exact launcher instructions are saved privately in `OPEN-AMD-IMAGE-TEST.txt`; publication verification follows the final commit.
