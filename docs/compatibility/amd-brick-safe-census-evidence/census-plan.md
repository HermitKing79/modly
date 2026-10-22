# Original full-mesh safe-simplification census

Approved basee262ad7c37f108e1cea456775b9b8c4b11cc9ed8; isolated branch
codex/amd-brick-safe-simplification-census. Read-only CPU feasibility; no
simplifier, renderer, field query, UV/bake or model execution exists here.

Reuse authenticated original F32 positions/I32 faces/accepted colour rows and
sorted global edge/incidence/vertex-component caches. The rejected candidate
is preservation-only. Derived directed occurrence/face-fan indices are needed
because old caches have no face IDs/directions; never overwrite those caches.

1. In32768-face batches derive edge-cache slots and assert exact key identity,
   cached multiplicity, finite positions, valid source IDs and source component
   correspondence. Build one derived sorted occurrence index (`face*3+corner`).
2. Classify boundary/non-manifold vertices and faces, incidence-two winding,
   oriented edge-connected sheets, and local vertex fans. Face-fan graph nodes
   are original corners; across a consistent edge join the SAME source vertex
   on its two faces at both endpoints. Exclude unused vertices from fan/sheet
   min/max reductions. Separate all three concepts from source components.
3. Exact equal-position classes cover referenced rows only, including signed
   zero aliases. No welding occurs. Freeze faces incident to junction/boundary,
   sheet/fan, alias or winding interfaces; freeze their vertices and complete
   one-ring neighbourhood, closing alias lock classes. Report immutable face
   counts, guard-affected interior faces and fully-unlocked interior faces
   separately. A guarded corner can leave its opposite edge eligible; only
   actual endpoint and full-link checks establish eligible edges.
4. Only consistent edges with free endpoints and no immutable face can enter
   full link certification. Use exact neighbour CSR intersections to require
   shared neighbours exactly the two distinct opposite vertices; reject a
   shared link EDGE using unordered source-triangle membership (tetrahedra).
   Degree bound128; padded comparison budget4,000,000,000; batch maximum8192
   and boolean intersection maximum8MiB. Exhaustion is incomplete BLOCKED,
   never silent edge exclusion. Check worker time between phases/batches.
5. Distribute every face stage over source face-bearing vertex components and
   eight fixed AABB-centroid octants (>= midpoint selects high side). Vertex
   distributions use vertex positions; edge counts use endpoint midpoints.
   Preserve these different denominators and5749 unused original vertices.

Bounds: N<=1,957,891,F<=4,173,584,E<=6,167,053; int32 index capacities checked.
Mapped originals/caches about150MiB; occurrence/order/offset/direction and
endpoint temporaries budget450MiB; staged SciPy float64 CSR/transpose/DFS,
including corner graph12,520,752nodes and<=11,024,666links, budget1GiB; masks,
sheet labels, alias sort and source/neighbor/link indices budget700MiB. Staged
known peak projects below2.5GiB; admit5GiB total including padding/opaque
allowance, beneath unchanged6GiB process AND Job hard caps. This is a CPU
projection, not a measured peak. Disk projection<=512MiB new private output.
The worker stops work at150s to leave30s for durable receipts/cleanup; owned
supervisor180s,12GiB live available RAM and2GiB destination disk remain fixed.
One separate independently authored readback verifier uses the same owned
Job/resource envelope and its own exclusive marker. It verifies saved masks,
indices, distributions and applicable link witnesses without rerunning the
global fan graph, census or any simplifier.

Only if a meaningful globally link-certified population exists, separately
review one positive-only patch selection. Largest eligible-face component
(not global sheet), ties lowest source face; lower-median eligible key selects
one seed. Complete endpoint stars plus exactly3 all-incidence face rings, hard
6000 whole faces, no alternate seed/rings/truncation. Inherit global masks,
add actual crop guard, recheck original full endpoint stars and link tests.
Report protected floor, optimistic upper bounds and disjoint-initial-collapse
capacity separately, without guaranteed achievable reduction. One private
PIL diagnostic only if this one patch qualifies. No reduction after PASS.
