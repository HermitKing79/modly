# Amended full Blender import: qualification BLOCKED, visual BLOCKED

Base `a258c4eb97c239e3ec766648e71ca59b51ce1de9`.
Branch `codex/amd-brick-full-glb-acceptance`.

**One genuine Blender 5.2.1 import completed.** It preserved exactly
1,952,142 referenced vertices and all 4,173,584 original triangles. Every
referenced position was F32 bit-exact, and triangle order, corner order,
winding and multiplicity were exact under the sorted original-index remap.
Only the approved 5,749 triangle-unreferenced vertices were omitted.

The actual imported CORNER/BYTE_COLOR association was exact at all 12,520,752
corners against the installed native colour setter oracle. The original
vertex-colour shader routing, constant metallic/roughness, opaque alpha and
double-sided material passed readback. There was no colour replacement,
welding, geometry repair, extra import or original-file modification.

**Independent acceptance stopped at an additional ideal-IEC quantisation
ceiling introduced by this task's harness. No render was started.**
The fresh verifier reached that assertion only after its complete exact
geometry and native corner-byte comparisons. Its `NO-GO` failure and the
supervisor's `BLOCKED` receipt remain unchanged.

The declared bound was `0.5/255 + 0.000002 = 0.00196278431372549` encoded RGB.
A separate bounded read-only diagnostic of the already-saved arrays measured
a maximum of **0.002073088847329929**, exceeding it by
**0.00011030453360443915**. There were 18,128 exceeding components across
18,080 referenced vertices. Unique-referenced-vertex RGB RMSE was
0.0011318668515622864; this is not a corner-weighted RMSE.

That ideal-rounding bound was not specified by the user and had not been
qualified against the installed native quantizer. Adding it was a harness
mistake. Exact native-byte preservation is established; this failure does
not demonstrate GLB corruption, an AMD defect or incorrect colour association.
We did not weaken the failed gate after observing its result or retry the
import/production verifier. Native colour conversion quality and visual
acceptance are not claimed as a complete PASS.

## Measured execution and preservation

| Stage | Owned execution | Peak process / Job committed bytes | Settlement |
|---|---:|---:|---|
| Single Blender import, oracle and full readback | 31.937 s | 1,684,721,664 / 1,684,721,664 | Natural, empty Job |
| Independent acceptance, stopped at quantisation gate | 5.407 s | 170,225,664 / 171,061,248 | Natural, empty Job |

Blender logged 5.76 seconds for its importer. The script measured 6.6203444
seconds including the durable pre-call journal and 29.1266082 seconds for
its complete workload. Minimum observed available Windows RAM across these
stages was 19,137,646,592 bytes; free destination disk was at least
112,423,342,080 bytes. The unchanged 6 GiB process AND Job caps, 180-second
deadlines, 12 GiB RAM floor and 2 GiB disk floor all held. No forced cleanup.

The untouched 97,073,940-byte GLB still has SHA-256
`1970bbcc5740dc56c8c9e6e20d97e8afd3176126c57b87d886a034f876eb38ae`.
The private saved `inspected.blend` is 267,217,765 bytes, SHA-256
`67d9a9439417eebfa8ab294d955e8056f413fea28866f6006f8298a105e19588`.
Its location is `artifacts/amd-brick-full-glb-acceptance-a258c4e/import/`
under the local Modly workspace, outside the proof Git worktree.
There are **no new PNGs** or visual-quality findings from this task.

Eleven focused CPU checks passed without skips. The independent source,
supervisor and outcome reviews are retained below. Scoped preservation found
all protected100, original400 identities, selected environment/source/backups,
48 historical trees, 1,261 files, 22 reports and cache/quarantine evidence
unchanged. Checkpoint payloads were not loaded. Only source-safe code,
measurements and hashes are published; models, arrays and images stay private.

- [Candidate review](amd-brick-full-glb-acceptance-evidence/independent-candidate-review.json)
- [Supervisor review](amd-brick-full-glb-acceptance-evidence/independent-host-review.json)
- [Actual outcome](amd-brick-full-glb-acceptance-evidence/outcome.json)
- [Saved-array quantisation diagnostic](amd-brick-full-glb-acceptance-evidence/quantization-diagnostic.json)
- [Independent outcome review](amd-brick-full-glb-acceptance-evidence/independent-outcome.json)
- [Preservation](amd-brick-full-glb-acceptance-evidence/preservation-after.json)

**Next approval:** qualify the measured native BYTE_COLOR conversion using
these saved readbacks, then render the existing imported scene. A second GLB
import, asset re-export, geometry change or neural execution is unnecessary.
