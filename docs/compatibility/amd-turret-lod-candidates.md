# Turret 100K / 30K LOD qualification — BLOCKED

Base `1f947fa794084657c0ff1dd5793a6b9fd4944dd9`; branch `codex/amd-turret-lod-candidates`.

**NO-GO for Blender Collapse decimation of the unchanged turret topology.** The single small fixture exposed protected non-manifold faces. A read-only census of the exact coloured PLY establishes a **1,097,383-triangle protected lower bound**, exceeding both requested targets. The stop condition was applied before a full-resolution decimation. No LOD candidate, comparison render or replacement GLB was created; the accepted92.3MB GLB remains intact.

## Actual probe

One installed Blender **5.2.1 LTS**, CPU-only worker used a curved open grid plus four distinct triangles sharing an edge (incidence4). Synthetic affine RGB used the same `Col / POINT / FLOAT_COLOR` representation as the turret. The ordinary **COLLAPSE** modifier had triangulation enabled, symmetry disabled and no vertex group. Its ratio matched the turret's intended100K ratio exactly: `100000 / 4012360 = 0.02492298796718141`.

| Fixture measurement | Before | After |
|---|---:|---:|
| Vertices | 87 | 8 |
| Triangles | 132 | **4**, versus requested3 |
| >2-face edges | 1 | 1 |
| Zero-area triangles | 0 | 0 |

The four protected faces retained exact positions, winding and RGBA. All colours remained finite and within the source range. Two newly interpolated colour rows were on unused vertices; **this does not demonstrate interpolated colour fidelity on surviving reduced surface triangles**. The ordinary grid disappeared under the aggressive ratio; the protected junction remained. This is a synthetic qualification, not a reduced turret or a colour-quality success claim.

## Exact source and decisive floor

Source: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-learned-colour-fa3a98c\turret-learned-colour.ply`, **79,805,490 bytes**, SHA-256 `d79f6fb38d23e2f0c7bcc492153290d67767466411f71dc19fcc5c6540ceeb72`.

Read-only PLY parsing revalidated complete position, index and RGB payload hashes against the published original before computing edge incidence:

| Source measurement | Count |
|---|---:|
| Vertices / triangles | 1,842,968 / 4,012,360 |
| Boundary / >2-face edges | 164,154 / **338,449** |
| Maximum face incidence | 4 |
| Vertices touching >2-face edges | **331,921** |
| Triangles directly incident to these edges | **994,310** |
| Triangles with at least two protected vertices | **1,097,383** |

The [Blender5.2.1 collapse implementation](https://github.com/blender/blender/blob/v5.2.1/source/blender/bmesh/tools/bmesh_decimate_collapse.cc#L788-L820) rejects a collapse touching a vertex incident to an edge with more than two faces. A triangle with two such protected vertices has no collapsible edge. Those faces preserve the original bad-edge incidences, so the protected set persists by induction. Triangle-only input with triangulation enabled does not invoke a separate repair pass. Thus the counted set is a **source-derived lower bound**, not a measured final full-decimator count. This protects face count, not necessarily every face position: a third unlocked vertex could still move. It rules out100K/30K under this method; it does not rule out every possible optimisation algorithm.

## Preservation and next decision

The complete bounded worker took **5.140s**, peak committed memory **698,236,928 bytes**, inside a **2GiB / 60s** owned Windows Job. It exited0 naturally; Job counts were zero before cleanup and no forced termination occurred. This successful qualification execution does **not** mean LOD acceptance passed. No retry, full turret modifier call, render, export, topology change or environment change followed.

Local evidence only: `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-lods-1f947fa` (`fixture.json`, `result.json`, `host.json`). [Fixture](amd-turret-lod-candidates-evidence/fixture.json) · [Source census / stop](amd-turret-lod-candidates-evidence/result.json) · [Containment and ten preserved inputs](amd-turret-lod-candidates-evidence/host.json) · [Independent review](amd-turret-lod-candidates-evidence/independent-review.json).

Original coloured/uncoloured meshes, both Blender projects, both portable GLBs, neural output, complete learned material field and original front/side PNGs retain their published hashes. Main and all73 prior remote heads remain preserved; unrelated untracked caches remain untouched. No Torch, ROCm, WSL, model execution, install, repair or baking occurred.

**Decision:** automatic Collapse alone is not viable for these targets on the preserved topology. The next useful approval would be a small topology-preparation test on a derived copy, preserving the master and checking silhouette/colours, before attempting further decimation. Do not invest in UV/normal baking for nonexistent low-poly candidates. Stop here.
