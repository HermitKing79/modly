# One connected-chart feasibility candidate, no bake/export

Base bad7ba4e9e8c3b14539d02a0c3930080891e0c56, isolated branch
codex/amd-connected-foliage-uv. Reuse only the saved4531-face/3177-vertex selection.
No ROI, visibility selection, source geometry modification or material generation.
Authenticate all selected original IDs/corner bits and full learned coordinate-row
correspondence against unchanged originals. No WSL/models/GPU computation.

Capacity first: total original triangle area times4096² is an unavoidable content
area floor; >=83 source-edge components is a chart-count floor.219 nonmanifold
edges do not imply219 extra charts. Show floors separately, combine with max,
never add overlapping floors. Distinguish the previous1127-island overhead from
intrinsic surface area. Owner bounds based on64x64 minima are explicitly
conditional on this candidate's conservative prescribed padding policy.

One deterministic seam construction: maximum-length dual spanning forest over
only incidence-two, oppositely directed original shared edges; stable original
edge/face-ID tie breaks. Cut all other adjacencies. Glue disposable triangle
corners only across retained forest edges; each tree forms an abstract disk.
Original triangle positions, order, winding, multiplicity and source IDs remain
bit-exact. No disconnected surfaces bridged. Check <=256 charts before unwrap.

If capacity supports this candidate, invoke installed Blender5.2.1 CONFORMAL
exactly once on the explicit seam mesh. No UV alternative, unwrap iteration
search or repair. Background/factory startup/disabled autoexec, all faces selected,
identity transforms, no modifiers, original triangle topology. fill_holes=False,
correct_aspect=False,use_subsurf_data=False,margin=0. Operator FINISHED alone is
not success; preserve warnings and actual UVs. Blender's internal layout may
translate/rotate/scale charts; final density/packing are independently calculated.

Normalize each chart's uniform scale so its minimum actual tangent singular
value meets4096 with1e-5 numerical margin. Require every oriented UV area positive;
whole-chart reflection may correct a consistently negative chart only. Mixed
flips, degeneracy, anisotropy>4, nonfinite values or any interior overlap reject
the sole candidate; never split again or rerun the unwrap after a failure.

One deterministic rectangle pack with no parameter search, allowing90-degree
chart rotations. Conservative32-aligned owner dimensions
32*(ceil(content_span/32+.5+1e-5)+1), content offset32px, protects complete positive
LOD0..5 box/trilinear ancestry, including half/intermediate levels. Use original
pinned footprint code; no foreign/unowned influence above1e-7 (strict positive
ownership test is stronger). Fixed1024²,256islands,6000faces,250000clip work and
64Mi raster bbox work per verification phase. Report actual UV area, owner area,
padding waste and distortion. Every original triangle must have a covered texel.
Both F64 and portableF32 saved UVs independently checked; no atlas fit theorem.

Per actual worker:6GiB process commit/180s,12GiB available WindowsRAM and2GiB
destination free, monitored original ownedJob/binaryGO/natural process settlement.
No original writes or package/environment changes; -B and bytecode disabled before
imports. Private arrays/images outsideGit, source/text evidence only published.
Independent review binds each stage's exact sources and actual result. Stop after
one candidate or earlier refusal. Any future multiple-atlas/material-group route
requires separate approval; no bake/export even after qualificationPASS.
