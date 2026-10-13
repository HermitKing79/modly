# Approved bounded sampling contract

Starting commit38e73a5e02476a597ab0b91bd4f2fc58754baf4d. One12-step
FlowEulerGuidanceIntervalSampler run,22 forwards (12positive/10zero-negative),
sigma_min1e-5,rescale_t5,guidance7.5,rescale0.7,interval[0.6,1],dino_lock0.
Seed20261013,CPU torch.Generator,Gaussian F32[1,8,16,16,16]. Reuse verified
DINO features only; strict640-entry mixed-precision flow checkpoint.

Reuse existing worker/supervisor/Job/observer,with optional sampling control.
Retain existing standalone single-forward behavior and historical evidence.
Bind each step/input and role/timestep factory to its exact next ticket.
Stage the genuine sampler's timestep constructor through the established
explicit CPU-to-GPU copy operation; no new unvalidated GPU factory allowance.
Reset frequency source identity/once state only at an admitted root invocation.
Retain91selected module counters per invocation; use compact step/root diagnostics
with workerTrace cap192 (original validation/journal caps remain).

CPU regression fixtures exercise the real pinned sampler and actual worker
dispatch/module/factory functions,using CPU tensors and a backend-blocking sink.
Synthetic output writer tests cover exclusivity,corruption,finiteness/provenance.
Independent review and exact remote harness verification precede execution.

180sCPU/240sGPU+release/480soverall; available hostRAM11411181632before marker
and strict load;13GiB available VRAM;12GiB allocator cap. Healthy periodicWindows
monitoring at<=10s,contained owned process; final zeroallocated/reserved bytes.
One exclusive marker. No retry,warm-up,decoder,mesh,CuMesh,texture,GLB,otherweights,
download,installation or source/environment change. Save one bounded256KiB
verified F32 sparse latent; retain history until cleanup,then collect all refs.
