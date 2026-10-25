# Zombie CPU mesh extraction and visual verification

> **For agentic workers:** Use superpowers:executing-plans. Obtain an independent final review.

**Goal:** Convert the first successful standalone zombie decode into its genuine unchanged polygon mesh and inspect eight views plus a silhouette.

**Architecture:** A CPU-only, GO-gated Windows Job authenticates the completed output and original contract, invokes the unchanged `amd_generation_mesh.extract`, then runs a separate source-derived mesh analysis. Blender Cycles CPU produces local previews from the verified PLY with no modifiers or geometry edits.

**Tech Stack:** Existing Windows CPython/Numpy, existing dual-grid converter and checker, Blender 5.2 Cycles CPU.

**Spec:** User request of 2026-10-25 in this conversation.

## Constraints and review focus

- Never launch WSL, generation/model code, inference or texture generation.
- Preserve original decoded tensor bytes, input lineage, run contract, all prior evidence and branch tips.
- Use the existing CPU mesh precedent: 6 GiB process commitment cap, 180-second stage deadline, 12 GiB live available host RAM and 2 GiB output-disk floors.
- Every output is created exclusively in a fresh local directory. No repair, smoothing, vertex removal or silent coordinate transformation.
- Verify all ten tensor hashes and payload linkage, independent PLY/NPY readback, source vertices/splits/winding, zero-area/duplicate faces and edge incidence. Measure connected components and unused vertices separately.
- Rendering must use CPU only and unchanged counts/bounds; eight cameras and silhouette with recorded source hash and camera transforms.

## Tasks

- [x] Snapshot protected state, branch tips, successful neural tree and private control records.
- [x] Write focused failing tests for closure validation, admission and components; implement CPU owner/authentication/analysis.
- [x] Run existing reader/converter tests and harmless owned-process policy fixture.
- [x] Execute one bounded extraction and one separate readback/topology analysis; stop on failure without changing safeguards.
- [x] Render eight views and silhouette under the existing CPU limits; inspect source image and actual results.
- [x] Verify preservation, independently review code/outcome and publish only code/textual evidence on the new branch.

Ruling: retain the path-bound checkout and create `codex/amd-zombie-cpu-mesh-inspection`, leaving the neural execution branch unchanged. User authorised this concrete extraction; no further design approval round is needed.

Ruling: use the already qualified CPU extraction limits, not the separate 22 GiB pre-neural requirement. No neural stage is being launched.

Final: fixed Important renderer attachment-failure cleanup finding — negative render-GO and mocked unattached-process/cleanup-failure tests RED→GREEN; final focused suite 20/20. Successful assets/evidence retained unchanged, with no extraction/render repeat. Independent outcome/code review is bound to ca68dc1; the cleanup follow-up is author-verified. No other actionable findings were reported.

Metadata clarification: preserve first analysis file and append a hash-bound correction for an inherited false components flag. Actual component counts were already computed; no geometry or measurement was silently changed.
