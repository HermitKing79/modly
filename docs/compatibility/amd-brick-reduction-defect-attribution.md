# Rejected brick reduction: attribution PASS

Approved base `da37443fa7569ee155482c258a705704d78ade1c`; isolated branch
`codex/amd-brick-reduction-defect-attribution`.

**The diagnostic reproduces concrete connectivity, material and projection
failures. The 50% candidate remains NO-GO.** No new simplification, repair,
render, field query or full quality run occurred. The accepted master is unchanged.

## Four reproducible witnesses

All IDs below are zero-based saved source/candidate rows. Exact positions,
normals, oriented triples, adjacency, native colour bytes and source material
values are retained privately, not published as model data.

| Witness | Exact correspondence | Demonstrated finding |
|---|---|---|
| Maximum junction | Original source edge **1456609–1456613**; original faces3033048,3033067,3033078,3033080 | **4→9** incident face occurrences; seven distinct oriented triples and two duplicate pairs. This is a worsened **existing** non-manifold edge; no incidence-nine edge was found in the newly non-manifold pair set. |
| Oriented duplicate | Candidate faces **1411401 and1411403**; exact source triple **[1456609,1456611,1456613]** | Same positions, corner order and winding appear twice. That oriented triple does not exist in the original. Its three original pair incidences are0,3,4. |
| Worst saved colour | Octant2 query **1404**, candidate face **855137**, source IDs **[790131,921005,921020]** | Encoded RGB maximum error **0.2429918264** in red. Saved prediction reproduces with **zero difference** from retained native bytes and barycentrics. No original forward or reverse triple matches this new triangle. |
| Known foliage pixel | **+Y, column221,row260**, centre(221.5,260.5); original face **3905033**, source IDs **[1839209,1839773,1839266]** | Original nearest hit reproduces its saved depth; **zero candidate containing triangles**, despite nine covering bounding boxes. This is lost pixel-centre coverage, not dark RGB or a farther/nearer occluder. |

The maximum junction's complete candidate incidence is faces1411092,1411401,
1411403,1411407,1411412,1411416,1411419,1411421,1411530. The other duplicate
pair is1411419/1411421. Both endpoint positions are unchanged; neither endpoint
has an original boundary edge, but they touch six and five original non-manifold
edges respectively. Six incident occurrences have no original oriented-triple
match. This demonstrates changed local adjacency and repeated surface coverage.
It does **not** identify individual collapses or prove which sheets were merged.

The foliage hit has barycentrics(.486098,.500899,.013003), near one triangle
edge. The exact frozen opaque two-sided,1e-9 edge-tolerance query agrees with
the saved mask/depth. This single pixel is not proof of a large hole, an entire
missing leaf or a continuous silhouette error. The original +Y cohort was47
pixels with one failure; that remains the scope of the observation.

## Recurrence and material explanation

Eight additional pairs were frozen at evenly spaced ranks of the saved116,786
newly non-manifold pairs. **Six were absent original edges; two originally had
incidence2. Six now have incidence3 and two incidence4.** All eight touch at
least one endpoint with original non-manifold incidence; seven also touch an
original boundary vertex. These eight have distinct incident triples, so the
maximum junction's duplicate mechanism alone cannot explain even this small
sample. New adjacency near existing junctions recurs; a single collapse cause
for the complete population is unproved.

The installed stable meshoptimizer0.22 call received positions and indices,
target50%, error.005 and `LockBorder`; it received **no colour attributes or
field error objective**. Retaining a border vertex's position does not freeze
its incident face connectivity. Global component retention likewise does not
preserve sheet or semantic-part interfaces inside a component.

For colours, each new face interpolates decoded scene-linear endpoint values
and converts the result back to encoded RGB. Exact endpoint bytes cannot
guarantee that this interpolation matches a varying spatial field inside a
new triangle. The selected residual is reproduced without colour replacement,
gamma changes or sampler changes. However, no paired original-surface material
evaluation exists at this exact new point. **The whole residual is not proved
to have been introduced by collapse**; original approximation, native-byte
quantisation and spatial variation remain contributors. No new field positions
were sampled to conceal that limitation.

## Bounded next experiment justified, implementation not yet qualified

There is sufficient local evidence to justify **one separately approved small
original real-patch proof around this source junction**, retaining complete
incident triangles and an explicit crop ring. A credible method must protect
original junction/interface connectivity and crop/foliage boundaries, operate
only within explicitly manifold-consistent sheet/part membership, preserve
components and oriented multiplicity, and reject new bad pairs/duplicates.
It must also test fixed face-interior colours against the genuine field; exact
vertex-byte transfer is insufficient.

The current stable0.22 API does not expose arbitrary vertex locks or colour
weights. No replacement API, sheet construction or semantic part classifier
has been qualified here. The cheapest next step is to qualify an already
installed constraint-capable method on that single patch, with original
junctions held fixed and unchanged numeric gates. Do not repeat the full-model
index-only call or automatically repeat the previously fragmenting fan split.
No promise of50% reduction, full-model feasibility or game readiness follows.

## Resources, evidence and preservation

One owned diagnostic worker completed in **8.797s**; supervised completion and
post-checks **11.484s**. Peak committed process/Job memory was
**252,321,792 /253,161,472bytes**. Minimum observed available Windows RAM was
**19,512,467,456bytes**; destination free space111,638,130,688bytes. Unchanged
6GiB process AND Job/180s/12GiB live RAM/2GiB disk gates passed; exact binary
GO followed attachment, and the Job emptied naturally with no forced cleanup.
Ten focused synthetic checks and independent source review preceded execution.

Private diagnostics:
`C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-reduction-attribution-da37443/diagnose/`

- `topology-witnesses.json`: complete primary/eight additional stars and duplicate.
- `material-witness.json` and `projection-witness.json`: exact rows, geometry,
  native values, source references, local edge incidence and pixel arithmetic.
- `junction-wireframe.png` and `projection-witness-mask-crop.png`: bounded
  wireframe/saved-buffer schematics, **not new shaded renders**.

The original/rejected mesh hashes, exact source-ID map and accepted source
closure were authenticated. Scoped BEFORE/AFTER audits preserve protected100,
400 original identities,55 frozen trees/1,518files/25reports, selected installed
environments/source edits/backups and all cache/quarantine evidence. No checkpoint
payload was read. All105 prior remote heads are unchanged; final publication
verifies the added diagnostic branch without modifying them.

[Plan](amd-brick-reduction-attribution-evidence/diagnostic-plan.md) ·
[Numeric outcome and private artifact hashes](amd-brick-reduction-attribution-evidence/outcome.json) ·
[Independent review](amd-brick-reduction-attribution-evidence/independent-outcome.json) ·
[Preservation](amd-brick-reduction-attribution-evidence/preservation-after.json).

Stop for review. The rejected candidate does not replace the master.
