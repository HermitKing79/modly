# Brick final-support qualification — CPU PASS

Starting revision: `a0f9b74beca6d6bbdc1c818e53b577ee677ba9a3`, branch `codex/amd-repeatable-generation-v1`.

**PASS for CPU qualification only.** The opt-in `brick-1957891-v1` contract supports exactly **1,957,891 final rows** from the authenticated 4,430-row brick shape input. No new neural forward, WSL execution, GPU initialization, checkpoint loading or mesh extraction occurred. The previous genuine run remains PARTIAL: 39/40 convolutions, 84 large tiles and 6/8 C2S operations; its final selected mask was not saved. No authentic final output has been recovered or fabricated.

## Narrow implementation

The historical **1,842,968-row default is unchanged**. The optional profile requires the exact source-job and two input SHA-256 identities, large-convolution tiling, and the measured parent sequence `4430 → 21391 → 97313 → 438684 → 1957891`. Its maximum is included in the authenticated run contract; the host also pins the published measurement receipt. Different jobs, support counts, hashes, missing opt-in and larger unsupported support fail closed.

The same validated contract now reaches both fourth-stage C2S admissions, final sparse/tiled-convolution bounds, complete output validation, strict child-order reconstruction, ten-tensor writer/readback, Windows verification and the CPU mesh reader. The mesh-reader CLI accepts an explicit `--run-contract`; it does not silently expand its default. The original CPU dual-grid converter already permits 2,000,000 rows, leaving 42,109 rows of capacity. Its arithmetic and limit were not changed.

There are no changes to model operations, weights, precision, native kernels or installed environments. The final convolution still uses the full global native neighbour map, ratio-2 hash and original row order with at most 32,768 rows per tile. Acceptance requires all 40 convolutions, eight linked C2S calls, five native maps, the six correct 438,684-row large-convolution receipts, **60 complete final tiles**, finite output, independent readback, healthy observation, natural process settlement and zero owned Torch allocation/reservation.

## Resource qualification

These are **CPU projections from the unchanged formulas**, not new GPU measurements. Admission remains `max(allocated,reserved) + ceil(1.25 × visible) + 1 GiB ≤ 12 GiB`, with `free ≥ ceil(1.25 × visible) + 2 GiB`. Initial free VRAM remains 13 GiB; runtime host RAM/disk floors and all deadlines are unchanged.

| Operation | Visible bytes | Padded bytes | Required device-free bytes |
| --- | ---: | ---: | ---: |
| Each fourth-stage C2S check | 2,941,417,584 | 3,676,771,980 | 5,824,255,628 |
| Final convolution before native map | 3,245,216,244 | 4,056,520,305 | 6,204,003,953 |
| Final full output/first tile after map | 3,002,437,760 | 3,753,047,200 | 5,900,530,848 |
| Each tile after full output allocation | 2,751,827,712 | 3,439,784,640 | 5,587,268,288 |

The genuine pre-fourth-stage sample was allocated **1,684,097,024**, reserved **3,451,912,192**, device-free **11,531,587,584** bytes. It fits the first projected C2S check. Both expansions and every later allocation still require fresh live checks; this sample cannot prove their eventual memory usage or runtime. The final map is 211,452,228 bytes, ratio-2 hash 31,326,256 bytes and full BF16 64-channel output 250,610,048 bytes. The 60th tile has 24,579 rows.

All row, hash, neighbour and grid-key indices fit the original integer representations. The complete ten-tensor payload is **104,125,380 bytes**, below the unchanged 256 MiB limits. The host reader's stricter 64 KiB header limit was exercised using the production metadata builder: actual synthetic-file header **23,200 bytes**, complete file **104,148,588 bytes**.

For a later CPU mesh gate, at most six triangles per row implies **11,747,346 triangles**; face arrays plus concatenation alone can require 281,936,304 bytes. Worst-case PLY payload is 176,210,190 bytes. The converter has no separate face-count cap. Actual flags, topology, mesh counts and process peak remain unknown; extraction needs its own bounded RAM/disk admission. See the [source and memory audit](amd-brick-final-support-evidence/memory-audit.json).

## CPU verification and findings

**79 distinct focused checks passed**: 48 decoder/admission/tiling/contract checks, 17 existing host regressions, seven profile host checks and seven full-sized consumer checks. Three affected entry/acceptance checks were repeated after the host-import correction and passed. No broader application suite ran.

The full-sized fixture used synthetic coordinates/logits/features with the measured counts, explicitly labelled synthetic. It exercised the actual production provenance builder, finite/chain validator, safetensors writer, byte-exact readback and independent mesh reader. Only the Linux-owned output-directory boundary was replaced with a disposable Windows test directory. Tests rejected changed provenance, missing opt-in, altered support/order, nonfinite features and foreign module bindings; failed reads closed all ten mappings. A scalar-only hook fixture separately proved that the second C2S remeasures memory and stops on failed admission. Small CPU calculations compared final 64-channel tiled arithmetic with the pinned original pure-Torch reference, including cross-tile neighbours and missing-neighbour sentinels.

Final consumer test time was **21.144 s**; actual finite/chain validation **7.825 s**, writer/readback **0.334 s**, peak Windows process working set **1,099,055,104 bytes**. This is a CPU test measurement, not a WSL decoder or GPU-memory measurement. The temporary 104 MB fixture was removed.

CPU findings corrected within scope:

- The actual Windows writer exposed `fsync` failing on a read-only handle. Reopening the just-created owned artifact as `r+b` preserves durability and exact readback without truncation.
- Host regression exposed late helper imports after the host restored its import path. Pure CPU helper references are now bound during the existing guest-module import; no AI imports were added.
- Independent review found that profile acceptance also needed to bind large-convolution receipt rows to 438,684. A negative regression now covers this.
- Test metrics are optional and exclusive, so future tests cannot overwrite the published receipt.

[Test results and source identities](amd-brick-final-support-evidence/qualification.json), [final full-sized CPU evidence](amd-brick-final-support-evidence/consumer-final.json), [independent review](amd-brick-final-support-evidence/independent-review.json), [before preservation](amd-brick-final-support-evidence/preservation-before.json), [after preservation](amd-brick-final-support-evidence/preservation-after.json).

## One next step

The harness is CPU-qualified for a **separately authorised single neural-only GPU run**, selecting `--tile-large-convolutions --final-support-profile brick-1957891-v1` with the new published harness commit and fresh exclusive paths. Preserve the complete saved brick input. Stop on any changed learned count, admission failure, timeout, numerical or cleanup fault. Actual post-expansion VRAM, final native map/convolution execution, the 240-second deadline and a genuine saved final output remain unproved. No run is authorised by this report; mesh extraction, materials and frontend integration remain separate gates.
