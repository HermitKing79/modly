# Portable import then one representative foliage patch

Approved base ecb15aa0a2855b9c4f9d3a45aea4192cd92a3900. One fresh Blender5.2.1
CPU import of the638700-byte revised GLB. Verify complete oriented geometry/UVs,
packed image pixels, original unit-factor RGB/MR node routing and sampler metadata.
Render two sizes from that one scene under identical CPU lighting/materials;
record native Blender interpretation separately from earlier CPU box-mip proof.
Stop before foliage if import/material/resource checks fail.

One foliage nomination only: original +X image ROI[x332..400,y384..448], chosen
visually before geometry measurement to include an orange leaf and thin curved
vines in the established B_vines region. Use complete original faces, never clip
or retessellate. Bound nomination to6000 retained faces. Select the front visible
surface by a bounded ROI depth raster, then one original-edge neighbour ring
within the same ROI. This is a visible cropped surface qualification, not a
closed botanical part or proof for all rear foliage. Preserve source IDs.

Cheapest geometry/UV admission first: dominant-normal-axis projection per face,
source-edge connected collision seams via established island code, no YZ-only
assumption. Fixed projected density4096*sqrt(3)*(1+1e-6); actual tangent minimum
must remain>=4096, at least one covered texel per original triangle. Single
deterministic32-aligned shelf pack; no size/density/gutter parameter search.
Each chart starts32base pixels inside its owner; derive rectangle dimensions
analytically from the complete box/bilinear ancestry through LOD5. Fail closed
for overlap, zero coverage, capacity, orientation, source or admission problems.
Never describe one failed shelf pack as proof no alternative atlas can exist.

Only independently admitted foliage proceeds to one genuine sparse eight-neighbour
CPU PBR bake with original six-channel semantics/quantization and per-island
padding, portable GLB and original point/seam and footprint-matched mip gates.
RGB RMSE/p95/max .02/.05/.15; scalar RMSE.02; foreign/unowned tolerance1e-7;
LOD0,.5,...,5. No change to source geometry/materials or original assets.

Every actual worker:6GiB/180s,12GiB available host RAM/2GiB destination free,
owned Windows Job before binaryGO, natural emptyJob settlement. NoGPU/WSL/Torch/
models, installs, repairs, full-model baking or automatic retries. Private assets
and images outsideGit; only code, textual metrics and reviews published.
