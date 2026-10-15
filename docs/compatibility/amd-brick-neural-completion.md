# Genuine brick neural completion — PASS

**One genuine RX 9070 XT parent neural forward completed and its full ten-tensor output is permanently saved and independently verified.** No decoder code, model precision, memory contract or installed environment was changed. No retry occurred.

Execution harness: GitHub-verified `c9447ddcba283eaa365defc74704a22442906579`, branch `codex/amd-repeatable-generation-v1`, with `--tile-large-convolutions --final-support-profile brick-1957891-v1`. Fresh run ID: `run-brick-final-support-c9447dd`. The published 79-check CPU qualification was reused after source-identity verification; tests and earlier model stages were not rerun.

## Genuine result

| Required operation | Measured result |
| --- | --- |
| Parent neural forwards | **1** |
| Original convolutions | **40/40** |
| Six large-support convolutions | **84/84 tiles**, 14 each, complete 438,684-row support |
| Original C2S operations | **8/8**, both expansions at each stage |
| Authentic global native neighbour maps | **5** |
| Final 64-channel convolution | **60/60 tiles**, last tile **24,579 rows** |
| Learned support lineage | **4,430 → 21,391 → 97,313 → 438,684 → 1,957,891** |
| Final features / coordinates | Finite **F32[1957891,7] / I32[1957891,4]** |

The complete artifact preserves original coordinate order and four pairs of original BF16 subdivision logits and I32 coordinates. Strict loading verified all 292 genuine checkpoint entries, followed by the reviewed 216 BF16 / 76 F32 runtime configuration. All three genuine TunableOp flags remained false. Source-pinned native operations, global maps, original weights, learned support and exact profile remained intact.

## Saved output

Local-only bundle, outside the proof checkout:

```text
C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-brick-neural-completion-c9447dd\neural-output\shape-decoded.safetensors
```

**SHA-256:** `d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861`

**File:** 104,148,612 bytes; payload 104,125,380 bytes; header 23,224 bytes.

The bundle also retains exact `run-contract.json` and `provenance.json` copies. Serialization and byte-exact guest readback completed before GPU release. The original Windows supervisor independently verified all ten tensors after worker termination. A further CPU-only read of the permanent copy streamed all ten payload hashes, checked finiteness, uniqueness, full original C2S child order and correspondence with the authenticated input coordinates. All source/input/copy hashes remained unchanged and mappings closed. [Artifact verification](amd-brick-neural-completion-evidence/artifact-verification.json).

## Actual resources and cleanup

| Measurement | Actual |
| --- | ---: |
| Synchronized parent forward | **79.366 s** |
| Entire GPU stage | **87.774 s** |
| CPU decoder preparation / loading high-water | 11.650 s / 2,787,229,696 bytes |
| Worker / supervisor elapsed | 114.028 s / 140.640 s, excluding separate host preflight |
| Initial device-free VRAM | 14,772,154,368 bytes |
| Peak Torch allocated / reserved | **3,354,263,040 / 4,389,339,136 bytes** |
| Lowest recorded available Windows RAM | **13,792,145,408 bytes** |
| Lowest D: free space | 74,264,416,256 bytes |
| Measured post-release Torch allocated / reserved | **0 / 0 bytes** |

All 200 live allocation admissions and 137 host RAM/disk admissions passed; all 29 Windows observation captures were healthy. No admission-cache release was needed. The 12 GiB allocator ceiling, 13 GiB initial VRAM requirement, 2 GiB device-free reserve, 1 GiB opaque headroom, 1.25 padding, 32,768-row tile cap and original deadlines were unchanged.

Worker exit was **0**. Both method patches were restored and owned tensor/model references collected. The Windows Job was already empty, with **no forced cleanup**. Only the named Modly Ubuntu distro was terminated after completion; `docker-desktop` and its default status were preserved. Device-free memory after release was 14,637,944,832 bytes; zero owned Torch memory does not imply zero device-global usage.

## Evidence and preservation

[Exact execution result and raw-file hashes](amd-brick-neural-completion-evidence/execution-result.json), [full durable GPU journal](amd-brick-neural-completion-evidence/execution-phases.jsonl), [resource measurements](amd-brick-neural-completion-evidence/resource-admission.jsonl), [independent outcome review](amd-brick-neural-completion-evidence/outcome-review.json).

[Before](amd-brick-neural-completion-evidence/preservation-before.json)/[after](amd-brick-neural-completion-evidence/preservation-after.json) checks cover all 100 protected files, source edits/backups, saved brick inputs, prior artifacts/raw evidence, archives and selected environment/native identities. All 21 qualified source bindings remain exact. No conditioning, occupancy, shape sampling, mesh extraction, material work, GLB conversion or frontend integration occurred. Publication preserves all 76 preexisting branch heads, including main. Original raw evidence remains locally under `docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/final-support-c9447dd/decode`; the 104 MB neural artifact is not committed to Git.

**Single next step:** separately approve a bounded, source-faithful **CPU dual-grid mesh extraction and readback from this saved artifact**, followed by visual inspection. The measured support fits the existing 2,000,000-row converter bound. Actual topology and visual fidelity remain unproved; no additional neural forward is needed for that test.
