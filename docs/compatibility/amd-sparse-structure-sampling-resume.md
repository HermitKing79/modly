# First saved image-conditioned sparse-structure latent — PASS

2026-10-13. Same branch `codex/amd-sparse-structure-sampling`; executed published
harness `98d1d29f750236580c7853091dfd949a077b531a`. The
[previous stopped launch](amd-sparse-structure-sampling.md) and all its evidence
remain unchanged. This was the **first actual sampling attempt**, with no retry.

## Actual execution

`WORKSPACE_DIR` was set to the exact proof checkout in the parent PowerShell
process. The actual `amd_blaslt_teardown.verify_workspace()` passed in an
AI-import/checkpoint-read-blocked CPU process, followed by fresh source, asset,
RAM, inventory and Windows checks. The same PowerShell then invoked the unchanged
published launcher; its supervised worker inherited the setting.
[Workspace/preflight evidence](amd-sparse-structure-sampling-evidence/resume-preflight.json)
and the [exclusive marker](amd-sparse-structure-sampling-evidence/attempt-started.json)
record the admissions and exact execution commit.

Only the audited `ss_flow_img_dit_1_3B_64_bf16` checkpoint was strictly loaded:
640 complete entries, no missing/unexpected parameters, 220 explicitly verified
reversible BF16-to-F32 widenings, and the pinned procedural RoPE buffer.
Checkpoint SHA256:
`ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`.
The saved DINO feature file SHA256 remained
`3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a`;
DINO was not rerun. Runtime was Python3.12.10 / PyTorch2.13.0+rocm10.0.0 /
HIP7.15.26333 on RX9070XT, gfx1201. CUDA build metadata was `None`.

Genuine `FlowEulerGuidanceIntervalSampler`: sigma_min=1e-5, steps12,
rescale_t5, guidance_strength7.5, guidance_rescale0.7, interval[0.6,1],
dino_lock0. CPU float32 Gaussian noise `[1,8,16,16,16]` used seed20261013.
Positive conditioning was the saved F32 `[1,1029,1024]` tensor; negative
conditioning was source-faithful `zeros_like(cond)`.

| Completed step | Complete forwards | Synchronized step seconds |
|---:|---:|---:|
| 1 | 2 | 3.781 |
| 2 | 2 | 0.422 |
| 3 | 2 | 0.437 |
| 4 | 2 | 0.422 |
| 5 | 2 | 0.438 |
| 6 | 2 | 0.421 |
| 7 | 2 | 0.438 |
| 8 | 2 | 0.422 |
| 9 | 2 | 0.437 |
| 10 | 2 | 0.438 |
| 11 | 1 | 0.203 |
| 12 | 1 | 0.219 |

Exactly **22 forwards: 12 positive and10 zero-negative**. Every invocation passed
its timestep/conditioning identities, 91 selected module counters and one
frequency-copy guard. Actual transfers: x1, cond1, neg_cond1, timestep22,
frequency22, output1. All12 durable completion records are finite and explicitly
synchronized. Worker diagnostics used90 events under the192-event cap.

[Raw results](amd-sparse-structure-sampling-evidence/run/results.json):
CPU strict loading **10.328s**, synchronized sampling **8.078s**, measured GPU
metadata/placement/sampling/release stage **9.578s**, whole owned worker **28.750s**.
The first step was slower; sampled process records identify the underlying Python
child, with no native compiler process identified. This does not establish the
cause of that first-step delay.

## Artifact, memory and independent verification

Published identical copy:
[sparse-structure-latent.safetensors](amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors).
Exact local path:
`C:\Users\wattz\Documents\ChatGPT\modly\proof\docs\compatibility\amd-sparse-structure-sampling-evidence\sparse-structure-latent.safetensors`.
The original is retained under `run\scratch\sparse-structure-latent.safetensors`.

- File: **134,688 bytes**, below256KiB; one finite F32 `[1,8,16,16,16]` tensor.
- File SHA256: `8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2`.
- Tensor payload:131,072 bytes; SHA256 `68930e793a5ec752852db45ab9985ab012a0dd32b6e7234f2ebb1f3177d08e56`.
- Values: min−3.083783388, max2.361266136, mean0.006999060, standard deviation0.154177151.

[Independent stdlib-only readback](amd-sparse-structure-sampling-evidence/resume-result-analysis.json)
validated the bounded container, duplicate-free header, raw32768 F32 values,
all finiteness, representative values, payload/file hashes and all provenance;
no framework import or model/GPU work. Metadata retains the execution commit,
seed/noise hash, complete sampler settings, model/config/source hashes, runtime
and original DINO/image provenance, including source PNG SHA256
`9abc7b8f89e5b001e863fc3ef2fbd6813b2e83f63c0a6b0a45203c2ef1edfd8d`.

Fresh RAM before marker:17,432,297,472 bytes; before strict load:16,976,908,288,
both above11,411,181,632. Available VRAM before placement:16,937,582,592 bytes,
above13GiB. Peak allocated: **2,918,161,408 bytes (2.718GiB)**; peak reserved:
**3,038,773,248 bytes (2.830GiB)**, below the unchanged12GiB cap.

All model/tensor/history references were collected. Synchronization, exactly
one approved BLAS workspace clear and allocator cleanup left **zero allocated
and reserved PyTorch bytes** in the live worker. Available VRAM then measured
16,807,886,848 bytes. Worker exit0, verified empty Job tree, no forced cleanup.
Three healthy periodic Windows samples had a maximum5.031s gap; before/gate/after
checks were healthy. No driver-reset evidence, timeout, guard or native error.

## Preservation and stop

[Before/after inventory reconciliation](amd-sparse-structure-sampling-evidence/resume-preservation-summary.json)
confirms both installations and model/cache directories unchanged, with
byte-identical raw metadata. AMD source edits/backup and historical branches/main
are preserved. [3,496 historical files retain their hashes](amd-sparse-structure-sampling-evidence/resume-historical-preservation.json),
including the failed-launch report and evidence. No execution-harness or
production-worker change, suite repetition, VM action, download, installation,
environment repair, other model, decoder, mesh, texturing or GLB operation.

**PASS applies to this one finite, saved sparse-structure latent.** Occupancy,
geometry and image fidelity remain unvalidated. The shortest next separately
approved experiment is one strict load of the already-proven sparse decoder and
one decode of this exact saved latent to an occupancy volume, with bounded
memory/monitoring and cleanup, stopping before any mesh or shape model.

Stopped after this sampling sequence, independent artifact validation and
publication. No automatic progression or second sampling attempt.
