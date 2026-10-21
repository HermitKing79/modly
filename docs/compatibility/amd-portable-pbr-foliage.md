# Portable PBR: brick Blender PASS; foliage qualification BLOCKED

Base `ecb15aa0a2855b9c4f9d3a45aea4192cd92a3900`; isolated branch
`codex/amd-portable-pbr-foliage`. One real Blender import/two CPU views and one
CPU foliage-admission worker. No retry, rebake, alternate selection or parameter
search. No neural/GPU generation or changes to accepted assets.

## Revised brick GLB — PASS

Authenticated **638,700-byte** private `patch.glb`, SHA256
`dc772cd55fc523c5427ab4771da27e62cd6905d5109f00b2fca47403704da857`.
One fresh **Blender5.2.1 LTS** import verified all781 oriented triangles with
bit-exact original positions/winding/multiplicity, UV correspondence, original
scale/orientation and both embedded PNGs. All packed image channels, including
retained alpha, match the accepted bake. Imported shader uses **sRGB RGB**,
**Non-Color G→roughness/B→metallic**, unit factors, two-sided geometry and opaque
alpha; no vertex-colour substitute, new normals, emission or recolouring.

Exactly one scene/import produced near/far640² CyclesCPU renders,32samples,
CPU OpenImageDenoise. Only orthographic scale changes, by4×. Camera position,
orientation, neutral studio lights, material, colour management and+2EV stay
identical. Image hashes/readback, geometry and material were verified after
each render and independently reopened after process exit.

**Viewer distinction:** glTF retains LINEAR_MIPMAP_LINEAR/CLAMP_TO_EDGE, while
actual Blender nodes report **Linear/EXTEND**. The installed importer explicitly
discards the mipmapping distinction in `set_filtering()`; source SHA256
`656b49e03b84f3aac4e829e713d6386d14e63f1c1f7623bb9ddf26a33c810e7d`.
These lit Cycles pictures do not reproduce or revoke the separately qualified
F64 box/trilinear filter model. Near appearance retains the brown mottling and
original openings; far presentation loses fine detail under actual minification.
No universal viewer/engine parity or visual-quality improvement is claimed.

Worker-measured time **52.453s**; supervisor child lifetime **58.406s**; full host
**58.531s**. Peak process/Job commit **905,465,856B (863.52MiB)**. Minimum sampled
available RAM **17,183,649,792B**, destination free **103,062,007,808B**. Exit0,
naturally empty Job, no forced cleanup, all watched source identities unchanged.
The stdout warning `CUEW initialization failed: Error opening the library`
records a failed incidental CUDA-library probe; verified rendering used CPU.
Do not infer that no native device-library discovery was attempted.

[Pre-execution review](amd-pbr-foliage-evidence/independent-brick-review.json)
and [independent actual outcome](amd-pbr-foliage-evidence/independent-brick-outcome.json)
bind the exact candidate, inputs, one import, images, material and cleanup.

## Foliage transfer — BLOCKED before UVs or bake

The predeclared original +X ROI **[332,384]–[400,448]** was nominated visually
to include an orange leaf and thin curved vines in the prior B_vines region.
The algorithm selects complete original visible faces with a bounded centre-
depth raster, then one original-edge neighbour ring inside that same ROI.
No face is clipped, resampled, reconstructed or removed from the master.

The genuine selection produced **8,650 original triangles**, exceeding the
existing qualified **6,000-triangle** UV/overlap/raster bound by2,650 (44.17%).
This is the tested implementation's input bound, not a demonstrated RAM limit
or proof that8,650 faces are intrinsically unsafe.
It stopped exactly at `foliage_select.visible_faces()`:

```text
ValueError: Original ROI face budget: 8650
```

The failure precedes selected-geometry persistence, actual chart construction,
spatial material sampling, UV rasterisation, atlas padding, foliage GLB export
and filtering. There is **no foliage material-quality result**, no proven atlas
space requirement for this real selection, and no claim that foliage cannot
fit a1024² atlas in general. Per-face IDs/UVs were not saved because selection
admission failed first. Full stderr, exact executed source/marker hashes and
host source-identity ledger remain preserved privately.

Prepared CPU-only chart code handles curved surfaces using dominant-normal-axis
projections and source-edge collision seams, rather than assuming YZ projection.
Fixed projected density4096√3(1+1e-6) supports a4096 tangent-density floor;
one32-aligned shelf pass and analytical support protect LOD0–5. Positive area,
overlap, coverage, portableF32 and source-ID checks fail closed. **These are
synthetic qualifications only, not measurements of this real foliage patch.**
All original RGB/scalar/contamination gates and6GiB/180s limits remain unchanged.

Foliage worker exited1 at the admission refusal. Host time **6.906s**, peak
process commit **260,034,560B (248MiB)**, minimum available RAM
**18,403,254,272B**, free disk **103,081,906,176B**. Owned Job emptied naturally;
no forced termination or source mutation. This is a bounded-input rejection,
not a crash, resource exhaustion or demonstrated texture/native defect.
[Independent foliage outcome](amd-pbr-foliage-evidence/independent-foliage-outcome.json)
verified the exact failure/source ordering without rerunning selection. No actual
foliage UV, mip or material-accuracy numbers are manufactured from synthetic tests.

**Cheapest next action:** separately approve one source-connected vine/leaf
selection that fits the existing6,000-face bound, with its IDs/count admitted
before UV work. Do not increase the bound or retry this ROI automatically.
Its anatomical completeness and1024² mip-safe capacity still require proof.

## Private visual evidence and preservation

Root: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-pbr-foliage-ecb15aa`

- `brick\near.png`:330,436B, SHA256
  `401eac4c1568497b6e8977a6bff7ed18e43c5a6e29804474722a4011fef9dbfb`.
- `brick\far.png`:305,247B, SHA256
  `5cb93c8f7b6aba276039cce690c83c02683f13243b080abfe27a632a6c9f0ea2`.
- `foliage\run\brick-import-comparison.png`:401,195B,1280×690, SHA256
  `221db77e2896aec3a60f7491e5604d09f1cfa27222478791aa6a7aad271341dd`.
- `brick\result.json`, `brick-host.json`, `brick-attempt.json` and
  `foliage\host-result.json`/`attempt.json`/`run\stderr.txt`: actual raw evidence.

The comparison simply assembles the two authenticated original Blender PNGs;
it is not another render or a foliage-bake image. Generated assets/images stay
outside publicGit. [24 new focused CPU tests](amd-pbr-foliage-evidence/focused-tests.txt)
passed (13import/supervision,11selection/chart). Source-safe numeric evidence
and independent reviews are published. No full-model bake, WSL, Torch/models,
geometry changes, repair, package installs or altered environment settings.
The first AFTER audit correctly reported five new Python bytecode cache files
from synthetic tests in three older evidence folders. All original bytes matched.
Only those proven-new cache directories were moved intact into this experiment's
`generated-cache-quarantine`; all five hashes were verified, no files deleted.
The original failed AFTER receipts remain unchanged. A separate reconciliation
records the final preservation state; do not describe the initial audit as PASS.
[Final reconciled preservation](amd-pbr-foliage-evidence/preservation-reconciled.json)
passes the exact BEFORE scope:100 protected files,30 frozen trees/828files/
13reports and selected environment/edit/backup/cache/source identities. The
relocation ledger records pre-move hashes; the failed audit recorded additions'
names. Reconciliation independently verifies the five retained cache payloads
against that ledger. Three older disposable copies retain labelled stat-only
checks; this is not a whole-installation byte audit.
Stop for review; retain the accepted brick GLB and all original material data.
