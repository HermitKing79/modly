# Repeatable AMD Generation v1: work in progress

> Draft discovery record, not a final milestone verdict. The CPU admission blockage below is confirmed. Subsequent independent review established a bounded parameterization route through the existing workers; that implementation and the genuine new-image proof are now in progress. Final classification will follow actual execution evidence.

**The primary acceptance goal is not met. No new-image mesh, learned material field or GLB was generated, and Generate-button inference was not tested.** Discovery found working model stages and portable turret assets, but no repeatable image-to-GLB inference service. A bounded CPU check reproduced the actual extension's rejection of a non-mock request. This is missing orchestration, not an AMD numerical/native failure.

## Preservation and actual working state

Baseline: GitHub-verified `6ea0abb20b3391ff3dd4daceeb2e1a9db903afa2`, not `main`. Development branch: `codex/amd-repeatable-generation-v1`. The only listed Modly checkout is `proof`; tracked files were clean, with four existing untracked bytecode-cache directories. No running Modly/Electron process was found. The installed application data/environment is separate from this source checkout.

All **76 prior remote heads** were recorded. One hundred protected artifact/source/native/backup files matched their published hashes before and after the probe. Twenty local-only files were backed up and independently read back; private backup and full path-bearing receipts remain outside Git in `artifacts/amd-repeatable-generation-v1-6ea0abb`. Existing AMD source HEAD remains `3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`, with its modified status and backups unchanged. No application or environment code was edited.

Current read-only metadata: RX 9070 XT, Windows driver `32.0.31041.1004`, controller status OK. Installed AMD Python **3.12.10**, Torch distribution **2.13.0+rocm10.0.0**, Pillow **12.3.0**. Installed Modly retains Torch **2.11.0+rocm7.14.0**. Named Ubuntu WSL2 and default `docker-desktop` were both stopped and remained untouched. WSL decoder runtime was last verified as Python3.12.3/Torch2.9.1/ROCm7.2.1; its packages/GPU were not requalified here. Windows RAM available during discovery was **21,160,427,520 bytes**; D: free space **75,019,395,072 bytes**. VRAM admission was not attempted.

## Existing architecture and exact gaps

The functional application route is:

`WorkflowPanel.handleGenerate` → `workflowRunStore.executeExtensionNode` → HTTP `/generate/from-image` → registry → `ExtensionProcess` → `api/runner.py` → generator → workspace GLB URL → `Viewer3D/useGLTF` → GLB download.

The existing viewport and export route can consume a real newly returned GLB. They are not the missing inference engine. Electron starts the API; model requests use HTTP rather than Electron's process-node IPC.

The experimental AMD extension's `AMDScaffold.generate()` requires `mock=true` and emits JSON receipts. Geometry preparation creates unloaded pipeline structure and explicitly disables generation. No weights or models are connected to Generate. Its runtime schema only supports those modes, its offline guard prohibits computation/subprocesses, and its generation timeout is **30 seconds**.

Two additional integration mismatches were found: normal frontend/API requests add `remesh`, `enable_texture`, and `texture_resolution`, which the mock parameter allowlist rejects; mesh inputs arrive as `mesh_path`, while mock Refine expects `existing_mesh_path`. Fixing these alone would still produce no geometry, so no cosmetic/mock integration repair was made. The checkout also lacks local Node dependencies/build output; no installation or broad build was attempted before the inference gate.

The genuine successful turret chain remains separate proofs: Windows DINO/sparse/shape sampling; WSL chunked shape decode; CPU flexible dual-grid extraction; Windows texture sampling; WSL guided texture decode; CPU learned-colour GLB export. [Neural/material success](amd-turret-texture-decode-resume.md) and [portable GLB success](amd-turret-compact-colour-glb.md) are preserved. They do not demonstrate a previous Generate-button run or a single reusable generation command.

Those workers pin turret image/artifact hashes, **3,743 input rows**, fourth subdivision **378,891→1,842,968**, final support and **57 tiles**. Texture guides and export also refer to that exact lineage. Changing filenames cannot admit a new image; replacing the checks or directly invoking underlying models would bypass existing safety evidence. Three independent read-only reviews found no narrower complete invocation.

## Cheapest actual check

The previously selection-only, vine-covered brick image is materially different from the turret: **1018² RGBA**,184,424bytes, SHA-256 `a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293`. Existing alpha validation accepted foreground bounds `[198,46,820,973]`; no RMBG is needed. Unreadable and empty inputs were rejected.

One owned subprocess ran with the real Windows AMD interpreter:

`$WINDOWS_AMD_PYTHON -B admission_probe.py $OWNED_OUTPUT`

Calling the **unchanged production method** with the image and `{"mock":false,"image_mode":"alpha-only"}` returned exactly:

`AMD scaffold requires mock=true; inference is not implemented`

This was direct-method CPU admission, **not application IPC, verified worker identity, model loading or genuine inference**. No readiness was fabricated. AI/native imports were prohibited and none were attempted. The probe took **2.422 seconds supervised**, peak committed memory **38,776,832 bytes**, exit0 with a naturally empty owned Job, under 30seconds/512MiB. There were no neural forwards or GPU measurements. [Exact result](amd-repeatable-generation-v1-evidence/admission.json), [containment](amd-repeatable-generation-v1-evidence/cpu-host.json), [preservation](amd-repeatable-generation-v1-evidence/preservation-after-summary.json).

No GPU run was launched against a route known to be unavailable. The missing engine is not a one-line repair; frontend integration was withheld as requested. No build or full workflow test passed or is claimed.

## Shortest supported next action

Parameterize the **existing staged runner**, before adding frontend behavior: one job-local image/seed/settings manifest; hashed predecessor artifacts; bounded run-derived coordinate/subdivision counts; dynamic admission using the existing allocation formulas, unchanged precision, global neighbour lookup and row tiling; durable outputs with independent readback. Keep one GPU job at a time, owned process cancellation/settlement,13GiB initial VRAM,12GiB allocator cap, live reserves and stage deadlines. Never relax fixed checks without replacing them with a reviewed general contract.

Then execute one CLI new-image proof through geometry, same-job learned materials and independently reopened learned-colour GLB. Only after that succeeds, connect a real AMD extension mode to the existing HTTP/result/viewport route, replace the mock deadline with stage-aware supervision, reconcile parameters, and test Generate plus sequential distinct jobs/cancellation. The initial portable GLB can reuse genuine RGB vertex colours with documented constant metallic/roughness; it must not claim a complete UV PBR bake or game-ready topology.

**Stop here:** the required new-image inference gate remains unproved. No turret replay, occupancy-only success claim, frontend mock success claim, new model, install, repair, decoder rerun or geometry alteration was used to manufacture progress.
