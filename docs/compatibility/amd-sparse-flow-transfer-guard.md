# Sparse-flow frequency-transfer guard — CPU-only correction

2026-10-13. Branch `codex/amd-sparse-flow-transfer-guard-correction`, based on
`6c1b882a4c130a78e87a2bfa16959f262362cba7`. **PASS for the CPU guard correction**;
no GPU initialization, real weights, model forward or sampler execution occurred.
The successful computation and overall **PARTIAL** classification in the
[previous report](amd-sparse-flow-forward-resume.md) remain unchanged.

## Exact correction and evidence

The installed Python 3.12.10 / PyTorch 2.13.0+rocm10.0.0 runtime reported:

```text
aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None,
 Layout? layout=None, Device? device=None, bool? pin_memory=None,
 bool non_blocking=False, bool copy=False,
 MemoryFormat? memory_format=None) -> Tensor(a)
```

Only `self` is positional. [Schema/provenance evidence](amd-sparse-flow-transfer-guard-evidence/schema.json)
records every argument, HIP 7.15.26333, CUDA build metadata `None`, Torch source
identity and GPU initialization false before/after. No positional layout was guessed.

`BoundedDispatch` now binds the exact CPU `exp` result constructed inside the
genuine timestep module, verifies F32 `[128]` and all values against the pinned
frequency formula, then permits its `aten.to.dtype_layout` transfer only inside
that scope in the forward phase. It requires explicit CUDA:0, unchanged dtype
and strided layout, no pinning/nonblocking/forced copy, and default or preserved
memory format. Missing/extra/foreign arguments, equal-valued clones, mutations,
wrong devices and repeat construction/transfer fail before backend execution.
The existing `_to_copy` route shares the identity check and single-use counter.
Other CPU-to-GPU `to` overloads and CPU-source `copy_` are rejected. Existing
internal FP32/BF16 `_to_copy` and observed `to.dtype` conversion policies remain.
The classifier still requires `frequency == 1`; no historical counter was edited.

[11 installed-runtime regressions](amd-sparse-flow-transfer-guard-evidence/green-final.json)
exercise the **actual production dispatcher body**, installed ATen overload and
real CPU tensors under inference mode. A separate CPU sink prevents every GPU
backend copy. [23 host regressions](amd-sparse-flow-transfer-guard-evidence/focused-final.json)
cover dispatch, lifecycle, strict contracts, deadlines, shutdown, casts and
acceptance with AI imports blocked. All **34** passed.
[Independent review](amd-sparse-flow-transfer-guard-evidence/independent-review.json)
found no actionable concerns, checked the source/config contract and separately
passed all 11 installed-runtime tests with GPU initialization still false.

The first CPU replay was blocked by Torch's lazy Dynamo/Triton import wrapper;
that error is retained. Tests unwrap only the dispatch method's Dynamo decorator,
keeping its production body intact and compiler imports prohibited. A separate
`Tensor.to(cuda)` facade call was blocked by the GPU initialization safeguard;
the passing replay invokes the real ATen overload directly. Red contract/copy
failures are preserved alongside green results. This demonstrates guard behavior,
not a newly executed transfer on AMD. No GPU attempt was consumed.

## First bounded sampling experiment — recommendation only

[Read-only source/config audit](amd-sparse-flow-transfer-guard-evidence/sampler-contract.json)
verifies the actual local `pipeline.json` against its previous audit, SHA256
`222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6`.
It records pinned source hashes/excerpts and a rational schedule without importing
or implementing the sampler.

Use genuine `FlowEulerGuidanceIntervalSampler` with `sigma_min=1e-5`, `steps=12`,
`rescale_t=5.0`, `guidance_strength=7.5`, `guidance_rescale=0.7`,
`guidance_interval=[0.6,1.0]`, and explicitly `dino_lock=0.0`. Its optional lock
path would add model calls/substeps; keep that path disabled. Reuse the verified
saved F32 `[1,1029,1024]` DINO features, with source-faithful `zeros_like(cond)`
negative conditioning. No DINO rerun or other checkpoint is needed.

The Euler grid is `u_i=1-i/12`, `t_i=5*u_i/(1+4*u_i)`. Each model receives F32
`[1000*t_i]`; latent/noise/output are F32 `[1,8,16,16,16]` per actual model config.
Use one recorded seed for standard Gaussian noise. Each update is
`x_next=x-(t_i-t_next)*v`. Guided prediction is `7.5*v_pos-6.5*v_neg`, followed
by the genuine x-start standard-deviation rescale and conversion back to velocity.
Rescaling adds arithmetic, not another model call.

| Steps | Starting times | Model calls per step |
|---|---|---:|
| 1–10 | 1, 55/56, 25/26, 15/16, 10/11, 7/8, 5/6, 25/32, 5/7, 5/8 | 2: positive then zero-negative |
| 11–12 | 1/2, 5/16 | 1: positive |

This is **22 complete forwards**, ending at time zero. The current single-forward
guard intentionally cannot authorize it. Before a separately approved experiment,
review an explicit 22-call contract with per-call timestep, conditioning identity,
module-count and frequency-copy validation. Test actual sampler/control paths
with CPU fixtures; do not simply raise the old counter or reuse a consumed marker.

The prior synchronized forward took 4.250s with 2.82GiB peak reserved VRAM.
`22*4.250=93.5s` is only a planning estimate: new timesteps, CFG, reductions and
allocator reuse are unmeasured. Recommend retaining CPU/GPU/overall limits
**180/240/480s**, the **11,411,181,632-byte host RAM admission**, **13GiB available
VRAM gate**, **12GiB allocator cap**, periodic healthy Windows checks (at most
10s apart), owned Job containment and zero allocated/reserved cleanup.
Stop without retry on any guard/native/finite-value/observer/deadline failure.

The model does not multiply in size across steps. Negative conditioning adds
4,214,784 bytes; each latent is 131,072 bytes. The source retains 12 intermediate
latents plus 12 x-start predictions (about 3MiB of F32 payload), with additional
unmeasured CFG temporaries/workspaces. These estimates do not establish sampler
VRAM stability. Clear the complete sampler result/history as well as tensors,
model and BLAS workspaces before enforcing the unchanged zero-allocation rule.

Save only `result.samples` after CPU copy as one finite F32 `[1,8,16,16,16]`
safetensors tensor in exclusive owned scratch, file cap **256KiB**. Record seed,
complete sampling parameters, model/config/source hashes, feature-file hash and
original source-image provenance. Write exclusively/atomically, re-open offline,
validate key/shape/dtype/finiteness/payload integrity, and record final SHA256.
This is the dense sparse-structure **latent**, not occupancy, coordinates or a mesh.
Stop before decoder loading. Do not call the pipeline's `sample_sparse_structure`,
which immediately continues into decoding and optional geometry processing.

## Preservation

[Before/after preservation](amd-sparse-flow-transfer-guard-evidence/preservation-summary.json)
confirms identical installed dependency inventories and raw metadata structures,
including both venvs, model/cache directories and existing AMD source edits/backup.
Main and all prior branches/evidence remain preserved. All **3,424** historical
evidence files retain their hashes. Production experimental mock routes, strict
loader, deadlines, cache policies and historical results were
not changed. No installations, downloads, native rebuilding or model execution.
