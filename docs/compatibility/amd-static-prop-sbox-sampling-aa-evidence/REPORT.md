# MODLY Phase 22B — texture sampling / anti-aliasing diagnosis

## Result and stop decision

**READ-ONLY DIAGNOSIS; NO NEW CAPTURE.** The identified CMAA2 mechanism has no qualified public camera-local disable in the inspected installed capture path. The public `SceneCamera.AntiAliasing` flag is insufficient: it changes an MSAA render attribute, while CMAA2 is selected separately from global settings and does not test that flag or `EnablePostProcessing`. The existing CameraComponent's SceneCamera accessor is internal. Changing global graphics settings, using reflection, changing debug rendering or substituting an unqualified render path would not satisfy the authorized comparison.

**Historical Phase 22 remains NO-GO**, exactly as published: 829,648/829,668 eligible samples match; top view 132,230/132,250; original zero-based GLB triangle 659 scores 34/38 (89.473684%), below its frozen 95% gate. No historical criterion, mask, classification, negative control, image or result was changed. There is no Phase 22B image-sampling PASS and no retroactive default/gameplay rendering qualification.

The installed code and saved mixed pixels support **CMAA2 finished-image filtering as a concrete candidate explanation**, without proving that it caused the twenty historical discrepancies. The historical capture did not record its effective AA/filter/resolve state. This limitation cannot be replaced by today's settings or release notes. The conditional single-capture stage and new-image comparison were not executed. New captures: **zero**; camera/global setting writes: **zero**; project or historical-evidence writes: **zero**.

Phase 11's original high-fidelity benchmark remains permanently **NO-GO**. **USER_APPEARANCE_APPROVAL_PENDING** is unchanged. No production promotion or further experiment is started.

## Authenticated parent, process and preserved inputs

Parent: [56de6d96772ee83eb8edb5248262980d2a80f5ae](https://github.com/HermitKing79/modly/commit/56de6d96772ee83eb8edb5248262980d2a80f5ae), tree `cc13cc63543ce305fbff1d44ee52481a0f164a78`, sole parent `b88200257f672bb8713c8f0936bf6595bf9164c5`. All six published Phase 22 code/text files were fetched again at the exact commit and matched local UTF-8 contents. A separate reviewer independently refreshed those contents and their Git blob identities.

The installed skill `C:\Users\wattz\.agents\skills\sbox\SKILL.md`, current first-party `sbox-cli --skill sbox-cli` instructions and run-C# reference were read. Discovery attached only to the already-running Greyfield editor through `--connect/--call`. Both root and an independent reviewer inspected process ownership: the sole `sbox-dev.exe` PID **7656**, owned by **HERMIT\wattz**, started **2026-10-31 15:17:12.818088 UTC**, with the exact existing Greyfield project command line. No editor was launched or replaced.

The active project is `C:\Users\wattz\Documents\S&box Projects\greyfield\greyfield.sbproj`, engine **26.10.27c**. The active scene is the existing `modly_tests/phase22_uv_pbr/modly_phase22_uv_pbr_test.scene`. All three open scenes were saved, Play was off and the compiler was idle with its last compile successful and zero errors.

| Identity | SHA-256 |
|---|---|
| Original Phase 20C GLB | `afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201` |
| Existing Phase 21 model source | `113d688135246d71cbd85c9c74dbd65af9f75160cce8efddce921c5964c8547c` |
| Existing compiled model | `9a83938757967a81cb7f37d9a505ae14dc491b109ba19df42c4d2b830cfd7a11` |
| Diagnostic shader source | `b712ac40cc8445e9830cdddca141d72ee06e4d7b1c2fd258a1fd4a3033ab7982` |
| Compiled diagnostic shader | `ce3da90fab295ade7b6a169721f4b7befeaf5dec5b0b78de3e2a070f2d9f59d2` |
| Diagnostic PNG | `35161147faf014e2069660a8681af66762f3d3ad4cc859904087becbb328a6d3` |
| Original aggregate capture metadata | `b0372a7aa3edd03f3d19872c80050a1474215e0f38bfb3fd8a55f49453465e5b` |
| Original frozen oracle | `b0aadae4d6c3ebb859aeada7f135621d219a8cbe26d80124d1296a30e3aab1ae` |
| Original sampling result | `e988366234ea888028083fffd030ab2f57afdfb08a9efdf2db8a857b5cb4e718` |
| Existing Phase 22 saved scene | `280f17fcc78c4db7c6085bb10e6f7cc2cc8b43459a303eb85168428e68890e20` |
| Inspected installed Sandbox.Engine.dll | `930d55e1a69ea58883fc1266aef447142d7e6e48eb210bce68900a754467f583` |

All five native PNGs and per-view metadata were authenticated against the original capture manifest. All 22 Phase 22 project files matched their final private asset manifest. The preservation baseline covers **6,821 files**, including all 6,460 previously protected originals and **335 historical Phase 22 evidence files**, all authored Greyfield files and the inspected engine assemblies/docs. Mutable `.sbox` caches, `.git` internals and existing Python bytecode are excluded. Existing local HEAD/refs/status were retained. The verifier checks changed/missing files, new project files and additions to the historical evidence directory. An independent reviewer identified and closed the last fileset-coverage gap before final verification.

## Actual observations versus unrecorded historical state

An independently reviewed read-only C# probe used only public getters through `Editor.EditorUtility.RenderSettings` and the existing top CameraComponent. It did not render, access the internal SceneCamera, invoke setters, save files or change graphics settings. Its exact script SHA-256 is `9d7ad7c842f672b0e9bce56d9afeaaa1daca5c84495b45abb36ccf39aac4e93f`.

| Setting | Current measured observation | Historical Phase 22 bitmap qualification |
|---|---|---|
| User AntiAliasMode | `Cmaa2` | Not recorded at historical capture time |
| User AntiAliasQuality | `Multisample8x` | Preference does not establish the bitmap's effective native sample count |
| RenderScale | `1` (100%) | Effective historical intermediate render resolution not measured |
| UpscalerMode / UpscalerRenderScale | `Off` / `1` | Historical per-view upscaler state not measured |
| Dormant FSR preferences | FSR3 Performance; FSR1 sharpness .25; FSR3 sharpness .5 | Inactive under current Off preference; no inference that FSR ran |
| Existing top camera | Position `(32,-38,110)`, scale one, FOV 42 horizontal, near/far 10/10000, full viewport, perspective, no custom projection | Public pose/projection inputs match the saved original contract |
| Top camera EnablePostProcessing | `false` | Also recorded false historically; does not establish AA/CMAA2 disabled |
| Top camera CustomSize | `null` | Historical capture temporarily used 1280×960 and restored null |
| Effective historical native AA/filter/resolve state | Not exposed by this read-only public camera observation | **Unobserved**, not reconstructed as a measured fact |

The separate `perf_rendering` tool reported the **current last native frame** at **1936×1272**, 8× MSAA, Cmaa2 and upscaling Off, with one render-target resolve. This is viewport/last-frame diagnostic evidence, not the original 1280×960 top bitmap. No GPU event trace, per-view target sample count, CMAA2 invocation counter or historical output-resolve metadata was recorded in Phase 22. A current preference or API default cannot fill that gap.

## Installed API and mechanism qualification

Installed XML describes SceneCamera AA and post-processing independently, RenderSettings' CMAA2 selection separately from hardware MSAA, and render scale independently from upscaler choice. The public [SceneCamera documentation](https://sbox.game/api/Sandbox.SceneCamera) and [RenderSettings documentation](https://sbox.game/api/Sandbox.Engine.Settings.RenderSettings) corroborate these API concepts. The [27 October 2026 first-party release notes](https://sbox.game/release-notes/) describe the introduction of CMAA2 and changes to presets/render-scale controls. These are context and API leads; installed implementation evidence and the read-only observations determine this diagnosis.

Independent inspection of the authenticated installed assembly establishes:

- `CameraComponent.SceneCamera` has an internal getter. The existing component exposes no qualified public local CMAA2 switch through its camera settings.
- `SceneCamera.AntiAliasing` is public but operates on the `msaa` render attribute: false writes 0 and true writes 4. Its getter compares the attribute (default 4) with 4. Restoring only the Boolean would not necessarily restore a previous absent attribute or another raw value.
- The separate CMAA2 application reads global `RenderSettings.Instance.AntiAliasMode` and the tools-visualization mode. It does not test SceneCamera's AA flag or EnablePostProcessing. Consequently the local MSAA Boolean is not a CMAA2 bypass.
- The reviewed public current-settings getter reads the existing video-cookie settings without setter/Apply/Save/ConVar operations. Dormant preferences are reported as dormant, not effective rendering stages.

The exact installed managed bitmap path is public `CameraComponent.RenderToBitmap` → internal initialization and its privately owned SceneCamera → internal `SceneCamera.RenderToBitmap` with a default ViewSetup → internal `CameraRenderer.Configure` → native `CCameraRenderer.RenderToBitmap(pointer,width,height,bytesPerPixel)`. ViewSetup exposes no public AA/CMAA/MSAA/upscaler override. No explicit managed bitmap AA bypass is present in this inspected chain. The native boundary's effective target dimensions, sample count, filtering stages and stage invocation remain unmeasured; the presence of the managed CMAA2 stage is not a per-capture GPU trace proving its execution on a historical bitmap.

No supported public local control was qualified that disables this identified finished-image mechanism while retaining the original bitmap path and all source/camera/projection/output invariants. Internal reflection, guessed attributes, tools/debug-mode changes, alternative camera/texture pipelines and global-setting changes are outside this proof. No such fallback was attempted. The absence of a qualified reversible control is the reason the conditional capture was not run.

## Independent saved-pixel diagnosis

The original frozen oracle was imported unchanged in memory. Its complete historical report was reproduced exactly. All **20** saved mismatch positions, exact RGBA values and source-category records were independently checked, together with all **38** triangle-659 samples: **54 unique screen positions** because four points are shared. Original geometry, source texture and public camera metadata supply the reference. Independent ray/plane interpolation differs from the frozen UV result by at most **1.22e-15**; independent screen-center reconstruction differs by at most **2.28e-13 pixels**.

All twenty mismatches have nonbinary RGB. Nineteen classify as unknown under the frozen classifier; one expected-black sample classifies as white. Triangle 659's failures remain two expected-magenta pixels `(223,101,223)` and two expected-green pixels `(101,223,101)`. Exact pixel locations, all twenty values, source UVs and mapped category-boundary witnesses are retained privately in `stage1-private-exact-witnesses.json`.

Nearby source-category boundary examples were independently mapped inside the same triangles for all twenty mismatches. Their reported 1.50–1.98-pixel screen distances are examples selected by texture-space distance, **not bounds on the nearest screen-space edge or an inferred AA kernel radius**. The prior half-pixel sensitivity grids were independently reproduced. These observations support image-footprint sensitivity; they do not identify an MSAA sample pattern, CMAA2 kernel, upscaler or resolve stage.

The original point-center UV/projection uncertainty envelope still preserves the expected categories at the four face-gate failures, as established in Phase 22. Nothing in this diagnosis changes its three-texel plateau margin or any other eligibility/acceptance rule.

The falsifiable candidate is: **if the historical bitmap executed this stage, finished-image CMAA2 filtering contributed to its mixed pixels; a qualified comparison that changes only this filter would change the original discrepancy locations without changing geometry, source sampling or projection.** No filter coefficient was fitted to the image. Because the historical effective state is unrecorded and the local bypass is unqualified, the candidate remains **unresolved**, rather than a confirmed root cause.

## Frozen comparison populations and omitted capture

The unchanged reference reconstruction authenticated these original populations:

| Population | Count | Array SHA-256 |
|---|---:|---|
| Original eligible top mask | 132,250 | `abfc695ea14c7dde63fe5ee29e2f199376f916d440afed5b2ab89592b6b87f8c` |
| Original triangle-659 positions | 38 | `afd334a5118c73fedb7d29662dc67c2f66fa3c15da0846112b706fb3ec409d67` |
| Original mismatch positions | 20 | `5387df32004b5849d20e03476bac6758c1b0546fa140a002c8eaa08f748031b8` |

Array hashing includes dtype/shape and contiguous bytes as defined in the reviewed analysis code. The expected-category array hash is `4d8f19fc0431452f8c7fdd2d3899f20ffd0d2d5ff6318f6180a6434b9cea2084`.

No conditional image exists. Therefore new-image per-face/overall agreement, repaired/new discrepancies and before/after rendering-configuration causality are **NOT TESTED**. No restoration operation was needed because no camera or settings value was changed. The original classifier, 95% face gate, all other numeric gates and all negative controls remain intact. The local conditional comparison plan is a deferred plan, not a completed experiment or permission for a second attempt.

## Preservation, publication and required next step

All authored evidence outputs of this investigation are confined to the new private directory `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-static-prop-sbox-sampling-aa-20261031-v1`. Raw API/implementation diagnostics, exact pixel/UV/geometry witnesses, captures and existing assets remain local. Final verification found **all 6,821 protected files unchanged**, no new project files, no historical evidence additions and unchanged local Git state. The 335-file historical evidence set is intact. Final editor checks show all three scenes saved, Play off, compiler idle with zero errors, and no new Warn-or-higher console entries after cursor 83. The final preservation/process review and publication receipt authenticate these findings. Public code requires the withheld local evidence to replay the historical analysis; it contains no embedded private image/mesh payloads. The generic `sbox_call.py` file is an attachment/logger helper; every live call still requires its own authorized scope and independently reviewed arguments. Only the four recorded read-only call batches were used here.

The next sampling proof requires a **first-party supported per-capture CMAA2 bypass and effective AA/resolve metadata**, independently verified to preserve the original path, source and projection. A comparison requiring global graphics changes needs separate scope and authorization; it is not performed here. Any future result would qualify only its explicitly measured configuration. Production/default gameplay sampling additionally needs its own configuration/coverage proof.

Neutral PBR rendering, intermediate sRGB transfer, filtering/mip/LOD behavior and BRDF/output equivalence remain unqualified from Phase 22. They require a separately authorized measured proof after the sampling path is settled. Source-normal fidelity, broader surface/texel/seam coverage and user appearance approval remain outstanding. No normal bake, geometry modification, dependency installation, neural work, gameplay integration or promotion was performed.

Only independently reviewed code/text is published on a new branch descending directly from the authenticated parent. The separate publication receipt records the commit, exact additions-only allowlist, sole parent, remote UTF-8 readback and unchanged prior branch heads. **STOP after this reviewed result; no automatic second experiment.**
