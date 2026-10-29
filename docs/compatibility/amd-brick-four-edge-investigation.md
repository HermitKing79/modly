# Brick panel: four-edge investigation

**Investigation PASS; local geometry correction NO-GO.** No mesh variant,
repair, UV generation, baking or export was performed. This is not a
texture-ready complete-asset pass.

The accepted 29 October geometry baseline was reused, not repeated. This
investigation authenticates its exact 457-vertex, 781-triangle panel and traces
only its four excessive-incidence edges into the original source. Source IDs
below are zero-based original-mesh row numbers, not GLB corner indices.

## Authenticated source and existing material evidence

Private panel root: `artifacts/amd-brick-seam-uv-bake-87f3755`.
Pinned preparation receipt SHA-256:
`b527aee6a88c08f34b32dac755bddadba2f22fc6c32430ef39a134444e788c63`.
Qualified UV receipt:
`86d0c01d54966df891b1ebe027c4bcdbcbc976a3e22bd4411f341e1bc8bd7229`.
PBR bake receipt:
`7f06d75e55c0435bea68539b4b7d10194f0677486839d0cc7960d193a325e481`.
All their named artifacts were SHA-256 authenticated before and after queries.

Original full brick PLY identity remains
`ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd`.
Both original NPY arrays were independently authenticated. Every panel vertex
and face was checked against its original source row. The saved decoded-output
file was authenticated as
`d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861`;
only sparse coordinate/feature rows were inspected, without neural imports or
evaluation. Original paired triangles recover the flagged grid quads below.

## Individual diagnoses

| Original edge | Incident original face rows | Parent quad rows | Source grid arrangement | Pairwise normal angles |
| --- | --- | --- | --- | --- |
| 1796162–1796174 | 3809831, 3809837, 3809838 | 1904915, 1904918, 1904919 | Grid-Y edge; two X-plane quads and one Z-plane quad | 1.209–10.200° |
| 1796174–1796176 | 3809856, 3809863, 3809864 | 1904928, 1904931, 1904932 | Grid-Y edge; two X-plane quads and one Z-plane quad | 1.555–9.515° |
| 1796175–1796176 | 3809862, 3809868, 3809871 | 1904931, 1904934, 1904935 | Grid-X edge; two Z-plane quads and one Y-plane quad | 0.237–15.377° |
| 1810755–1810760 | 3843348, 3843359, 3843370 | 1921674, 1921679, 1921685 | Grid-X edge; two Y-plane quads and one Z-plane quad | 6.064–12.311° |

Here X/Y/Z-plane means constant grid X/Y/Z before vertex displacement.

Each edge has exactly the same three incident faces in the **full source** as
in the panel. All eleven implicated parent quads have positive corresponding
decoded edge logits and the expected original grid offsets. These are three
distinct quad perimeters meeting, not extra triangulation diagonals. Spatial
dual-vertex displacements make the resulting fans nearly parallel, despite
their different lattice planes. Saved positions, individual normals, winding
signs and full source quad coordinates are in private `analysis-qualified/details.json`.

All twelve triangle-pair relations within these fans are **noncoplanar,
shared-edge-only**. There are no duplicate vertex positions in the panel.
Exact rational tests of the saved F32 coordinates found zero nontrivial
contacts, coincident triangles, positive-area overlaps or interior crossings
for the 20 implicated faces and their in-panel quad mates against all 781
panel faces: 15,410 unique pair queries, 164 surviving bounding-box rejection.
This is a local test, not an exhaustive full-brick intersection certificate.

**Demonstrated cause:** genuine branched/excessive-incidence connectivity in
the original decoded FDG quad graph. Duplicate geometry, UV seams and triangle
orientation are not the cause of these four incidences. This does not establish
which branch is unwanted, nor attribute a model-versus-decoder implementation
fault. Distinct nearly coincident-looking sheets are not proven duplicate faces.

## Why an edit is not justified

Each fan has two possible opposite-direction face pairings. Winding therefore
cannot identify a unique sheet; flipping cannot change three-face incidence.
The closest-normal pairing also changes along the first three-edge chain and
does not establish intended surface membership.

Ignoring only the four non-manifold adjacency edges for a **read-only graph
query** yields face groups of 770, 5, 5 and 1. The first two groups both belong
to UV chart 0; the other five-face group spans charts 1 and 2. In particular,
an ordinary two-face source edge connects charts 1 and 2. UV islands are not
physical sheet labels. No geometry was split for this query.

Original quad mates 3809869 and 3843371 lie outside the panel crop. Complete
source vertex fans and intersections with the full surrounding brick were not
certified. Selecting a branch to disconnect or remove would assert surface
intent that these measurements do not establish. Reindexing alone could hide
the counter without proving a useful clean surface.

These four edges are junctions, **not holes needing caps**. The accepted
baseline identifies 138 crop boundary edges and nine original-source boundary
edges among the panel's 147. No boundary is reclassified as an unwanted opening
here; an intentionally open panel need not be watertight. No correction is
admitted, including Modly's default face-deleting non-manifold repair.

## Preservation and UV/PBR readback

| Metric | Immutable panel after investigation |
| --- | --- |
| Vertices / triangles | 457 / 781 |
| Vertex-connected components | 1 |
| Boundary / non-manifold edges | 147 / 4 |
| Two-face winding conflicts | 0 |
| Geometry, bounds, face order, UVs, maps modified | None |

The existing independent CPU GLB reader confirms exact oriented triangle
geometry against the qualified UV mesh, 2,343 exported corners and 781
triangles. Corner UV roundtrip error remains **2.9802322387695312e-8**. Base
colour pixels match; both embedded PNG byte hashes and UV/material references
were checked independently:

- GLB: `e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21`.
- Base colour: `f294c9006b286747fb76ddb2f637e02ecfb0710410acfc2db68937a741d8891d`.
- Metallic/roughness: `2900c8e6124ffbb992eb863e3a2a10bd337b4851b509645f9a6a266dfc8fea1e`.

Opaque, double-sided PBR remains unchanged. The accepted four-island,
1024-square atlas and its 586,848 covered texels / minimum 74 per face remain
the original qualified evidence, not a new bake or coverage experiment.
Unchanged bytes preserve silhouettes, local detail and material output; no
before/after variant renders are warranted without an edit.

## Execution, tests and independent review

The owned CPU query passed in **1.172 seconds**, peak private committed memory
**50,712,576 bytes**, peak working set **61,575,168 bytes**. Minimum sampled
Windows available RAM was **19,836,616,704 bytes**; free disk exceeded 140 GB.
The existing 6 GiB process cap, 180-second deadline, 12 GiB host floor, 2 GiB
disk floor, pre-GO Job attachment and natural owned cleanup were retained.
Host samples do not attribute unrelated concurrent activity to this diagnostic.

An initial harness run stopped before geometry inspection because receipt keys
used Windows backslashes while lookups used forward slashes. Its logs and empty
output directory are preserved. A tested `as_posix()` correction and fresh
exclusive paths were used for the successful run. No guard was relaxed.

Thirteen focused tests pass, covering exact shared-edge/vertex contacts,
interior crossings, coplanar overlaps, duplicates, degeneracy, tiny separations,
large dyadic coordinates, corner/winding permutations and Windows receipt keys.
Three existing offline supervisor admission/GO/resource tests also pass.
An independent reviewer checked code, containment and final evidence and ran
432 additional tiny permutation/symmetry controls. Review agrees with
**NO-GO for geometry correction** and with the declared intersection scope.

Preservation checks passed for 3,346 historical file hashes, all 100 protected
identities and 88 retained records, 25 scoped original inputs and all 182
pre-existing local Git refs. Selected environment records remain unchanged;
this is not a whole-installation byte audit. No original asset was written.

Private evidence remains under
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-four-edge-investigation-20261029-v1`.
No raw geometry, tensors, textures or screenshots are published.

## One next decision

Stop local patch fixes. Separately review qualification of the **existing
upstream geometry-finalisation/export path on a complete simpler prop**, using
the preserved brick data and proven CPU PBR transfer after geometry acceptance.
The native AMD finalisation path is not yet qualified: this recommendation is
not permission to enable its cleanup or run it. Declare intended brick/foliage
parts and openings first, preserve the raw source, and reject lost features,
unexplained junctions or new intersections before UV/PBR promotion. A complete
asset must pass those gates; a cleaner cropped counter is insufficient.

No follow-up experiment or execution handoff is initiated by this report.
