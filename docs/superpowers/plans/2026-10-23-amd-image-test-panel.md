# Experimental AMD Image Test implementation plan

**Goal:** Let people inspect their own local image and distinguish browser checks, authenticated CPU input admission, and unavailable generation.

**Design:** Add a clearly labelled test tab to the existing Generate workflow panel. Reuse the existing Electron image picker and byte reader, sharing their image-loading helper with workflow image nodes. Start with the current workflow image when available. Keep test selection local to the panel; it creates no job and changes no source image. Use existing dark controls and a transparency checkerboard.

**Approach:** A browser-only checker cannot establish original mode, animation or Lanczos-resized foreground validity. A new local upload route therefore calls the existing `admitted_image` contract without changing it. The API environment lacks Pillow; an explicitly host-configured, already-installed Windows CPU Python may run the same service worker. Never discover or install a runtime automatically. No extension process, model, Torch or WSL is needed. The result remains input admission, with original preprocessing and generation explicitly not executed.

**Constraints:** Existing two MiB input limit, 4096 dimension/16M pixel limits, genuine RGBA, alpha-zero background, strict alpha >204 and minimum three-pixel foreground extent including the original resize. No conversion or background fabrication. Keep images and paths private. Generate and mock dispatch remain untouched.

## Tasks

- [x] Backend: focused real-image tests, bounded local upload/service and explicit CPU-runtime bridge; valid RGBA, JPEG, invalid/resized alpha, animation, oversize, corruption and unavailable-runtime checks. Only the new route was wired into the API.
- [x] Frontend: tested shared local-image loading, browser alpha, response validation and latest-selection cancellation; integrated the panel with the existing picker and workflow reader.
- [x] UI: actual private source build, native picker and CPU API tested with five input cases. No dependencies installed. Canonical packaging remains unverified; the earlier private JSX build failure is preserved.
- [x] Review: independent source, tests, UI outcome and preservation checks completed. Source-safe publication set prepared; GitHub publication is verified separately in the final commit receipt.

## Review focus

Changing images during a pending read/validation must never display stale success. Corrupt/oversize files must not enter large browser or Pillow allocations. A browser-decoded opaque image must never be described as an original RGBA cutout. Missing backend/runtime must remain unavailable, rather than become eligible. A valid CPU input must not enable generation or create a product job.
