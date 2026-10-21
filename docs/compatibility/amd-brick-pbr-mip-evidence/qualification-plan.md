# Portable PBR mip/seam probe — predeclared contract

Approved base: `0f2fd9250e8a2825df240a1f948afed0b07d31fa`.
This is one disposable CPU filtering experiment on the accepted GLB, not another
UV unwrap, material-field sample, texture bake, neural stage or geometry change.

- Authenticate the accepted 493,124-byte GLB, both embedded maps, all prior
  producer hashes, four-island ownership/coverage/UVs, original PBR field and
  original geometry identities. Preserve the accepted portable asset unchanged.
- Model its actual `LINEAR_MIPMAP_LINEAR` / `CLAMP_TO_EDGE` sampler using an
  explicitly declared **floating-point, linear-light 2×2 box mip pyramid**.
  Decode RGB with IEC sRGB; metallic, roughness and retained alpha are linear.
  Test LOD 0, 0.5, 1, 1.5, …, 5, not an unspecified renderer-selected kernel.
  Mip-level storage rounding and all-engine compatibility are outside this proof.
- Reuse every one of the 10,961 saved query positions and the 45 paired seam
  samples. Add the saved covered texel centre nearest each of every triangle's
  three edges (minimum opposite barycentric coordinate, deterministic tie order).
  These are exact positions of existing genuine samples, not new nearest-neighbour
  material sampling. Retain every triangle and island ID; no failing query is dropped.
- Propagate four island indicator planes and unowned space with the same filter.
  Any foreign-island or unowned contribution **above 1e-7** at tested queries is
  a contamination/admission failure, even if neighbouring colours look similar.
  This tolerance covers numerical noise, not visible bleed permission.
- Compare A (full accepted PNG atlas) with B (same PNG values, restricted to the
  query's own padding rectangle) and C (genuine saved F32 spatial samples only
  at covered texels of that island). B/C normalize numerator and mass **after**
  the complete bilinear plus between-level blend. A−B isolates contamination;
  B−C identifies padding/quantization error; A−C is total footprint-matched error.
  Require positive C support; never substitute zero or silently discard a query.
- Keep the established encoded-RGB RMSE/p95/max gates **0.02/0.05/0.15** and
  scalar RMSE **0.02**, separately for interiors, island/triangle edges and
  seams. For mip seam pairs score `(Aleft−Aright)−(Cleft−Cright)`; also report raw
  gaps, since unequal mip phase can create legitimate source-footprint gaps.
  Preserve the previous continuous-point comparisons at LOD0 only. Do not use
  the point reference as a minified-pixel reference or weaken any original gate.
- Measure ordinary source blur/mottling loss separately from A−B contamination.
  It is an observation, not a new relaxed accuracy threshold.
- Produce a private +X software surface diagnostic at actual projected object
  heights 512, 128 and 32 pixels. Derive isotropic LOD from each triangle's UV
  screen derivatives. Match the C reference to each exact footprint. Label this
  **unlit learned base-colour filtering**, not a Blender PBR lighting/engine test.
  Record metallic/roughness numerically. Any enlarged inset uses labelled
  nearest display enlargement after the actual distance projection/filtering.
- One owned worker, 6 GiB process memory / 180 seconds, 12 GiB live available
  Windows RAM / 2 GiB destination free floors, exact binary GO, no automatic
  retries and natural empty-Job cleanup. Focused synthetic tests and independent
  pre-execution/outcome review. Old artifacts and source closures remain frozen.

PASS requires every declared numerical/support/contamination gate and process
check. NO-GO means measured contamination or material loss under this explicit
filter requires a revised strategy; it does not revoke the accepted LOD0 proof.
BLOCKED means the filter/reference/containment cannot be qualified. Save complete
negative results and comparison images rather than attempting a padding repair.
