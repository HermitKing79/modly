# MeshLib counter binding micro-proof

**PASS — installed MeshLib 3.1.4.297 counter binding and one synthetic triangle only.**

Base: `8ff054951fc02f1d56de375c3ee9949379611036`. Experimental branch: `codex/amd-meshlib-counter-proof`. No real-patch arrays, production-worker modification, remeshing or target construction occurred. The previous failed real-patch probe remains unchanged.

The installed declarations expose a vector-backed `int_output`; `.value` requires exactly one element. Their unnamed constructor arguments did not establish initialization semantics. The independently reviewed worker observed these exact installed forms:

| Construction | Stored integers | `.value` |
| --- | --- | --- |
| `mr.int_output()` | `[0]` | `0` |
| `mr.int_output(0)` | `[]` | `RuntimeError: Expected exactly one value.` |
| `mr.int_output(1)` | `[0]` | `0` |
| `mr.int_output(2)` | `[0, 0]` | same rejection |
| `mr.int_output(1, 7)` | `[7]` | `7` |

The same expression in the unchanged installed build therefore creates zero output slots, rather than one counter initialized to zero. This reproduces the earlier readback error. The previous process did not record its storage length; its original BLOCKED outcome is retained. Explicit zero- and two-element vector controls also rejected scalar access.

The selected holder used `int_output()`, `values.resize(1)`, then `.value = -12345`. After binding it to `settings.skippedFaceCount`, an in-place `holder.values[0] = -23456` changed the settings getter from `-12345` to `-23456`. This proves a live mutable output binding for this retained holder. The settings getter itself returns an `int`; it is not another output wrapper. The holder was never resized after binding.

Exactly one `meshFromFacesVerts` call processed synthetic F32 vertices `[[0,0,0],[1,0,0],[0,1,0]]` and I32 face `[[0,1,2]]`, with non-manifold-edge permission, face shift zero and vertex duplication disabled. Both pre-call region views contained face 0. The same bound counter changed **`-23456 → 0`**; both post-call region views were empty. Readback retained exactly three vertices, one face, dense valid IDs, exact promoted position bits and original `[0,1,2]` winding. Consequently zero is an observed native output, corroborated by empty failed regions and one accepted face from one input face, rather than an unmodified default.

Thirty pure positive/negative contract tests passed. An independent source review admitted the exact candidate. A separate stdlib-only audit reopened all 17 saved scalar/journal snapshots, verified the synthetic geometry without a second native call, and freshly checked all 92 watched source/library identities. See [tests](amd-meshlib-counter-proof-evidence/focused-tests.txt), [source admission](amd-meshlib-counter-proof-evidence/independent-counter-review.json), [observations](amd-meshlib-counter-proof-evidence/counter-result.json), [host receipt](amd-meshlib-counter-proof-evidence/counter-host.json) and [independent outcome](amd-meshlib-counter-proof-evidence/independent-outcome.json).

Measured worker time was **1.266 s**, supervised elapsed **3.172 s**. The constructor was below the observed timer resolution; its recorded duration is `0.0`, not a claim of zero computational cost. Peak committed process memory was **168,603,648 bytes** and Job memory **169,439,232 bytes** (160.79/161.59 MiB), under the unchanged separate 6 GiB caps. Minimum sampled available RAM was **18,248,466,432 bytes**, destination free space **107,339,198,464 bytes**; all 12 GiB/2 GiB admissions passed. The 180-second deadline remained unchanged. Return code was zero, the Job was naturally empty before cleanup, and no forced termination occurred. stdout/stderr were empty.

Private observations remain at `C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-meshlib-counter-proof-8ff0549\counter\result.json`, SHA-256 `81755459b9f13e8dbc586ef3d0b3908d4acdccb2854ad67e6dde9928c4966c9e`. Public evidence contains only code, text and this synthetic numerical fixture.

[Before](amd-meshlib-counter-proof-evidence/preservation-before.json) and [after](amd-meshlib-counter-proof-evidence/preservation-after.json) checks retained all 100 protected files, 67 historical trees/1,921 files/31 reports, selected environment/source/backup identities, caches and quarantine. All 78 SDK files matched size, full hash and this mission's captured timestamps. All 111 pre-existing remote heads were unchanged before publication. Checkpoint preservation used stat/header checks with historical full-hash provenance; no weight payload was read. The final publication check separately verifies that only this new branch was added.

The minimal **future** worker correction is `mr.int_output(0)` → `mr.int_output(1)`, with a one-element assertion before binding/readback and the holder kept alive without resizing. The explicitly tested resize/set path is also valid. No such production patch was applied here. `.value` is safe only for the verified single-element state. Nonzero skipped-face cases and real non-manifold patch transport remain unqualified; this proves neither remeshing nor original sheet pairing.

The single next step, requiring separate approval, is the narrowly corrected real-patch constructor roundtrip with independent topology/readback checks. No additional native operation is authorized by this PASS.
