# One bounded no-UV export decision

Base7f1b1675913dcdd91175caa8169383236f8ed7eb; isolated branch
codex/amd-foliage-export-decision. Reuse exact4531 triangles/3177 vertices,
original source IDs and learned material rows; authenticate all private selection,
original full geometry/material coordinate correspondence and815-disk evidence.
No ROI selection, chart reconstruction, unwrap, bake, export, model, WSL orGPU.

Use pinned SparseSampler without changes: original F32 encoded RGB/six-channel
field, eight-neighbour order/renormalisation/denominator floor, no clamp before
spatial interpolation. Reference appearance clamps only the final derived sample
to[0,1]. Preserve raw samples/ranges/outlier/support counts separately.

Conventional glTF candidate: original vertex encoded RGB, clamp only derived
attributes, exact IEC sRGB decode then F32 linear COLOR_0. Barycentric linear-light
interpolation and exact encode give encoded-RGB errors. Unit baseColorFactor,
no texture, no RGB2.2 substitution, no display exposure and no byte quantization.
This is an idealized F32 direct-row vertex-colour candidate without byte
quantization or client losses, not a rendered export or an optimized colour fit.

Every face centroid plus10000 deterministic area-weighted surface locations.
PCG64 seed20261021; draw10000 uniform pairs then face choices, sqrt-barycentric
sampling. Preserve positions/faces/barycentrics. Include all shared source-edge
face pairs cut by the saved disk forest, including its cycle cuts and nonmanifold
multiplicity. For each pair use t0,.25,.5,.75,1 at the edge and0.005 inset toward
each face's third corner; near endpoints nudge to.005/.995. No dropped pairs.
Expected44420 seam queries+14531 required queries=58951, below65536 batch cap.
Reference zero support, if present, remains recorded genuine six-zero sampler
semantics; never exclude it from errors. It blocks a qualified material verdict
rather than silently treating unsupported black as valid material evidence.

Gates unchanged: encoded RGB pooled-component RMSE<=.02,p95<=.05,max<=.15,
for centroids, weighted points, combined required surface and edge/near-seam
cohorts separately. Each metallic,roughness,alpha scalar RMSE<=.02 independently.
Fit one constant per scalar from mean(clipped weighted source samples), then
validate all cohorts. Also compute per-cohort optimum clipped means and minimum
RMSE floors, so no fixed constant is claimed feasible when even the optimum fails.
Report raw/clipped global field, selected vertex and queried material ranges.
OPAQUE ignores alpha; a constant alpha does not establish genuine transparency.
No learned emission/AO/normal channel is invented.

Multi-atlas costs are conditional only:815 charts need>=4groups under256/island
limit and64² owner minima. Four1024² maps:28MiB base/37.3333MiB mip chain for
RGBA8+RGB8, or32/42.6667MiB if both GPU resourcesRGBA8. No compression/packing
performance or actual waste measured. Keep optional high-fidelity, notgame-ready.

One owned CPU worker,6GiB process commit/180s,12GiB Windows availableRAM and2GiB
destination-free floors; original WindowsJob/binaryGO/live monitoring/natural
settlement. -I -B and bytecode disabled before imports; historical files unchanged.
Private numerical arrays outsideGit. Source/text metrics and independent reviews
only published. Quality failure recommends a separately approved disposable
target-mesh simplification/retopology proof; no implementation in this mission.
