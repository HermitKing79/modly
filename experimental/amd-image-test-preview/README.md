# AMD Image Test developer preview

This separate Windows preview reuses Modly's accepted image-test panel. It is **not a release installer** and cannot generate a 3D model. No package acquisition, Python installation or environment update is part of building or launching it.

## Use the prepared preview

Double-click **Launch AMD Image Test.cmd** in the prepared preview directory. Leave its console open while using the app. Select **Generate → AMD Image Test → Choose test image…**. Changing the image clears the previous result and starts local validation for the new selection.

Use a single-frame, true **RGBA PNG or WebP**, at most **2 MiB**, with fully transparent background pixels. Dimensions must be at most 4096 pixels per side and 16 Mi pixels total. Foreground alpha must be **above 204** and span at least 3 pixels on both axes, including after the original 1024-pixel resize. Ordinary JPEG photographs, opaque images and palette-mode images do not qualify; the preview does not remove backgrounds or convert them into admissible cutouts.

An eligible CPU input result is not full preprocessing or generation readiness. Browser checks are labelled preliminary. Images are handled locally; the restricted API exposes only health and the original CPU image-validation endpoint.

Close the **preview window's X** normally. Its launcher then closes and records the result under `profile/launches/`. Do not close the launcher console first: the owned-process safety boundary may stop the preview. If port 8765 is occupied, the launcher refuses startup and leaves its owner untouched; close the other application yourself if appropriate. Low RAM/disk or changed runtime files also stop admission.

Normal launches have no smoke-test auto-close timer. The launcher retains 6 GiB process/Job caps, a 12 GiB available-RAM floor and a 2 GiB destination disk floor. The optional fixed `--smoke-test` launcher mode adds a 180-second developer test deadline; it does not relax validation.

Keep this local preview in its prepared location: its paths and runtime identities are fixed. Its `profile/` contains separate settings, cache, workspace, temporary files and logs. Your installed Modly is unchanged. Updates, syncing, setup, downloads, model/generation routes, external navigation and storage-setting changes are blocked. Other Modly functions are outside this preview's scope.

## Reproduce without installing anything

The builder takes one absolute-path JSON file with `repo`, `seedRuntime`, `destination`, `apiPython` and `cpuPython`. `seedRuntime` is the previously qualified corrected image-panel smoke runtime. `destination` must not exist and must be outside the source and seed trees. The builder pins the already-qualified interpreter binaries; it does not discover a substitute.

Run `node experimental/amd-image-test-preview/build_preview.mjs <absolute-build-config.json>` using the existing Node installation. It copies the qualified local runtime/dependencies, preserves renderer/preload/CSS bytes, compiles only guarded main-process sources with the existing esbuild, copies the unchanged CPU validator closure, probes the existing interpreters and writes the launcher, `preview.json` and `build-receipt.json`. It never launches the app.

Canonical Electron/Vite/Tailwind/TypeScript tooling was unavailable. The preview therefore uses existing compiled CSS, with the previously documented unused OBJLoader alias. New Tailwind utilities are not rebuilt. No canonical packaging, installer quality, unrelated 3D viewing or model execution is qualified by this preview.
