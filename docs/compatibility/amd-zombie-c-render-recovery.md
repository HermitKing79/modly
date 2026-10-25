# Zombie C CPU comparison recovery

Decision: **GO for one standalone CPU render recovery, subject to its unchanged live admission.** Generation and extraction already passed. The saved overall pipeline/comparison FAIL receipts remain unchanged: they correctly record the later rendering failure.

## Proven cause and minimal fix

The saved Blender 5.2.1 traceback stops at NumPy import, when importlib attempts `os.mkdir` for NumPy's installation-directory `__pycache__`. The existing filesystem audit correctly rejects it. The original CPU owner supplies `PYTHONDONTWRITEBYTECODE=1`, but a harmless real embedded-interpreter probe measured `sys.flags.ignore_environment == 1` and `sys.dont_write_bytecode == False` with that environment variable present. The same prefix reproduced the denied mkdir.

The renderer now sets `sys.dont_write_bytecode = True` immediately after importing `sys`, before runtime imports. A second real probe imported bundled NumPy 2.3.4 with **zero bootstrap write attempts**. Deliberate external file creation and directory creation still failed under the identical audit policy. No installation permission or environment was changed. No rendering or geometry processing ran in that probe.

The reusable opt-in regression executes the actual renderer AST prefix through NumPy import, stopping before any scene or geometry code. The original Windows CPU owner enforces its Job, GO/EOF, memory, resource and deadline safeguards for the probe.

## Recovered source authority

All successful neural provenance is bound to the original execution commit `4ab4616c4019705d5f53802b053156ef65cae150`, separately from the new recovery publication commit. The recovery contract pins the 14 existing comparison files, including empty successful-extraction logs and failed render stdout/stderr. It never rewrites their receipts or copies a mesh over them.

| Item | Authenticated result |
| --- | --- |
| Source PNG SHA-256 | `60862722c989772ad973870890601a6a95ed9a5cd5a9150a378e989db35ad437` |
| Ten-tensor neural artifact SHA-256 | `7cff6a60c2c51e118f7d7994fce7644b4a91a3f735f79b4622d4412517d9e507` |
| Raw PLY SHA-256 | `a2d015fbcd54f6b1232674c0fef239040501629460ab177dcf826f4b34d3809d` |
| Existing raw mesh | 325,074 vertices; 665,232 triangles |
| Readback | PLY and saved NPY position/index payloads bit-exact, including order; bounds unchanged; zero zero-area triangles |
| Original reference PLY SHA-256 | `5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c` |

A separate bounded read-only authentication checked all ten actual tensor payloads, learned subdivision linkage, source image/conditioning/shape/decoder contracts, extraction receipt, PLY/NPY correspondence and all 13 original mesh/view files. It completed naturally in approximately 7.23 seconds with 49,950,720 bytes peak process commit. It did not extract, construct, alter or render geometry; no neural model or GPU module was imported.

## One exclusive recovery

Run `scripts/compatibility/Invoke-ZombieCRenderRecovery.ps1 -Run -ExpectedCommit <exact published recovery commit>` from a normal, non-administrator PowerShell after closing Codex, ChatGPT Desktop and browsers. The final delivery supplies the exact command. Without `-Run`, PowerShell launches no child and Python's default is a stdlib-only no-work dry run.

Only these stages exist: **authenticate saved data, CPU render, assemble review**. There is no generation, extraction, WSL or neural-attempt route. Authentication reads existing tensors/PLY/NPY; rendering opens the saved mesh directly. The current CPU code, launcher, tests and recovery contract must match the exact supplied Git commit. Blender executable SHA-256 is pinned in the contract.

Exclusive private output, absent at delivery:

`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-zombie-c-render-recovery-60862722-v1`

On success:

- `previews/view-000.png` through `view-315.png`: eight new flat-shaded C views, every 45 degrees, using the original camera locations, orthographic scales, target, world coordinates, neutral material and lighting.
- `review/original/` and `review/candidate-C/`: eight authenticated pairs; original images are copied byte-for-byte.
- `review/paired-eight-view-contact-sheet.png` and `review/paired-review.html`: local visual assessment.
- `authentication-recovery.json`, `previews/render-metadata.json`, stage `host-result.json` files and `recovery-final.json`: hashes, unchanged geometry, CPU measurements and natural owned cleanup.

CPU limits remain **6 GiB process commit, 12 GiB available host RAM, 2 GiB free evidence-volume disk, 180 seconds per stage and four rendering threads**. The original owner samples RAM/disk every approximately 0.2 seconds; it retains deadlines, owned Job attachment, exact GO/EOF, cleanup and peak-commit reporting. No neural gates are changed: 22 GiB admission/final marker protection and 24 GiB preferred headroom remain in their existing generation code.

Any failure stops all later stages. There is no automatic retry, overwrite, alternate output path or change of limits. Preserve the new failure evidence for review rather than rerunning it. The old comparison and neural artifacts remain immutable throughout.

## Visual limitation

Real recovery rendering is reserved for the independent user command. The launcher can prove matched rendering and immutable geometry; it cannot declare that C has better hands, toes, wounds or surfaces. Its successful final classification is `PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED`, with five explicit human review criteria. No quality winner is inferred from triangle counts. All meshes, reference images, previews and private telemetry stay local.
