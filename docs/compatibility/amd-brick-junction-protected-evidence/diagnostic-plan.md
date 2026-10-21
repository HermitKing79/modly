# Original junction patch: frozen feasibility probe

Base4a033425d55947e080eee27df2757139ba7bec31; branch
codex/amd-brick-junction-protected-proof. This is the user-approved bounded
feasibility spike, before any separately reviewed conditional candidate.

Input is only original vertices/faces/native accepted source-colour rows.
The rejected candidate is hashed for preservation, never selected as geometry.
Seed is the complete two endpoint vertex stars of edge1456609–1456613; include
all four original incident faces. Add exactly three original edge-adjacent
face rings, recording each addition before counting. Hard cap6000 original
triangles; no truncation or alternate patch/ring search. Last ring is the
explicit frozen crop buffer. Save complete original face and vertex IDs.

Classify actual global versus patch incidence, directed winding, two-face
consistently oriented sheet connections, non-manifold junctions, source and
artificial boundaries, separate sheet-component memberships and exact equal
position classes. Never infer global manifold incidence from a cropped patch.
Protect every patch face incident to a global junction or source-boundary
vertex, a winding/interface or crop edge, multi-sheet vertex, duplicate-position
interface, and the entire outer ring. A protected face's three original vertex
IDs are immutable. Its complete one-ring vertex neighbourhood is additionally
locked against incoming collapses, with exact position-class lock closure.

Strict eligible edges have two opposite-wound faces in the same identified
sheet, two free endpoints, no protected incident face, and the original
manifold link condition. Count eligible faces and a deterministic non-touching
edge certificate; do not collapse or move anything. A feasible target requires
at least one certified edge and enough non-touching edges to remove20% of
the eligible face population (two triangles per collapse, rounded upward).
These are a topological admission estimate, not a geometric/material PASS.

Zero eligible/certified edges is BLOCKED and stops Phase2. A positive estimate
alone does not authorise a call: exact installed API/guard implementation,
fixed target/error, frozen local samples/views, source-spatial material and
independent candidate review still must pass. No arbitrary fallback or retry.

Reuse one owned CPU feasibility worker:6GiB process AND Job,180s,12GiB live
Windows available RAM,2GiB destination disk, binary GO after Job attachment,
fresh exclusive outputs and natural settlement. Preserve previous scopes and
private arrays; publish only source-safe counts/hashes/report/reviews.
