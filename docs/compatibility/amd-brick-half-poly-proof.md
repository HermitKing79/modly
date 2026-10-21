# Full brick-and-vines 50% reduction: NO-GO

Base `5705f9d325771b8b2595b6385a630879f9ff700e`.
Branch `codex/amd-brick-half-poly-proof`.

**One genuine CPU simplification produced a candidate, but it failed the frozen
topology, material and local foliage gates. No Blender render was admitted.**
The accepted full-resolution asset remains the reference and is unchanged.

## Actual candidate

The already-installed meshoptimizer0.22 stable `simplify` API used original
indices/positions, target50%, error0.005 and exactly `LockBorder`. No weld,
Prune, sloppy/experimental mode, ratio retry or alternative method ran.
The native error scalar is approximate; independent checks decide acceptance.

| Measurement | Original | Candidate |
|---|---:|---:|
| Referenced vertices | 1,952,142 | 975,633 |
| Oriented triangles | 4,173,584 | 2,083,776 |
| Face-bearing vertex-connected components | 207 | 207 |
| Boundary edges | 267,652 | 261,381 |
| Non-manifold edges | 387,068 | 270,987 |
| Maximum edge/face incidence | 4 | **9** |
| Duplicate oriented triangles | 0 | **3,642** |
| Zero-area/nonfinite faces | 0 | 0 |

Triangle reduction **50.0723%**; referenced-vertex reduction50.0224%.
All original components survived without cross-component triangles. Retained
positions and native colour bytes match their exact original source IDs.
Boundary length fell1.98%. However, **116,786 candidate non-manifold endpoint
pairs were not in the original non-manifold set**. Fewer total bad edges does
not imply no new junctions. Another675 output triples match reverse winding
of pre-existing triples; collapse ancestry was not tracked, so this is not a
claim that675 specific original faces were individually flipped.

All conservative nearest-referenced-vertex distance bounds passed the frozen
0.005/0.01/0.03 normalized RMSE/p95/max gates, including source interior,
known foliage, component and separate reverse samples. This is bounded evidence,
not a Hausdorff, global normal or self-intersection guarantee.

## Shape and learned colour

| Frozen640px view | Overall silhouette IoU | Known visible foliage retained |
|---|---:|---:|
| +X | .997758 | 598/599 (.998331) |
| +Y | .997660 | **46/47 (.978723)** |
| +Z | .998166 | **197/200 (.985000)** |
| Isometric A | .997322 | 422/422 (1.000000) |

Overall silhouettes passed0.98. The +Y/+Z foliage cohorts failed0.99.
These are small pixel-centre cohorts:1 and3 pixels failed coverage/depth
retention, respectively. This does not establish global foliage destruction.
No antialiasing, smoothing or reframing hid differences.

Each of16 original source-ID-derived regions received128 deterministic
candidate-interior samples against the genuine sparse eight-neighbour PBR
field. RGB gates remain RMSE0.02/p950.05/max0.15:

| Failed colour region | RMSE | p95 | Maximum |
|---|---:|---:|---:|
| Brick | **.020100** | .044335 | .100612 |
| High variation | **.028577** | **.067177** | **.157972** |
| Octant2 | **.027585** | **.059100** | **.242992** |

All metallic/roughness/OPAQUE-alpha scalar gates passed; no sampled field support
was missing. Exact retained vertex colours did not prevent changed face-interior
interpolation. Regions overlap and are not semantic segmentation. No shaded
appearance claim is made because the predeclared render gate correctly stopped.

## Resources, artifact and preservation

| Corrected owned stage | Seconds to natural empty Job | Peak process / Job bytes |
|---|---:|---:|
| Read-only baseline | 6.625 | 184,205,312 / 185,049,088 |
| Single simplification | 3.985 | 1,160,708,096 / 1,160,708,096 |
| Full readback and quality assessment | 21.297 | 687,964,160 / 688,803,840 |

The simplifier call itself took2.503139s. Minimum observed available RAM was
19,824,422,912bytes; destination free space112,070,000,640bytes. All workers
stayed inside unchanged6GiB process AND Job/180s/12GiB RAM/2GiB disk limits;
all Jobs settled naturally, with no forced cleanup. Runtime2GiB WASM and1.5GiB
JS-heap caps held; projections were not presented as measured peaks.

Private **rejected** candidate, independently byte-read back with its arrays:
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-half-poly-5705f9d-corrected/assess/candidate.ply`

41,723,823bytes; SHA-256
`5024daaaa716a1bbf00d316f89ff47e5e6d30653a370b15a9aff7cb4fbf8fc99`.
Source-ID mapping, colour bytes, vertices/faces, four-view masks/depths and
quality samples remain private alongside it. Estimated compact F32-COLOR0 GLB
size48,424,600bytes is a buffer estimate only; no GLB was exported.

One earlier Node startup used the wrong named-export syntax and stopped before
any simplify call. Its marker, error, original code and all46 retained files are
unchanged. A narrow direct-CommonJS adapter correction and fresh exclusive paths
preceded the **only** actual simplification; no method/ratio/error retry followed.
Twenty-one focused tests and independent source reviews passed. The static
review initially missed that adapter mismatch; the exact limitation is retained.

Scoped preservation retains protected100,400 original identities,52 historical
trees/1,366files/24reports, selected environments/source edits/backups and cache/
quarantine evidence unchanged. No checkpoint weights, WSL, GPU inference, new
packages, baking, repair, extra reduction or application integration occurred.
All104 prior remote heads, including main, are verified again at publication.

[Frozen criteria](amd-brick-half-poly-evidence/qualification-plan.md) ·
[All measurements and artifact hashes](amd-brick-half-poly-evidence/assessment-metrics.json) ·
[Outcome](amd-brick-half-poly-evidence/outcome.json) ·
[Independent final review](amd-brick-half-poly-evidence/independent-outcome.json).

**Next:** keep the master and separately inspect one newly formed high-incidence
junction in this rejected candidate before choosing further reduction or a
part-aware target-mesh approach. No new experiment is started. Stop for review.
