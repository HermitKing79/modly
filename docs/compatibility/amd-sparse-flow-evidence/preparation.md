# Reviewed single sparse-flow attempt

Gate1 CPU preflight passed; Gate2 has not run in this preparation commit.
Base: codex/amd-blaslt-workspace-teardown at065bc79b8c2db97eea3892c34481dbe21e6a118d.
New branch: codex/amd-sparse-flow-forward.

Genuine source3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5; model source SHA256
858d9b076cc71f6955bb292ea394d620ce6f6f53cebb53bff2116a8c5cec026e.
Exact config resolution16,width1536,30blocks,12heads,MLP8192,shared modulation,
cross image attention, RoPE, BF16 transformer linears. No remote factory.

Checkpoint ss_flow_img_dit_1_3B_64_bf16.safetensors:
ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6.
640 BF16 entries /1,292,179,976 parameters /2,584,359,952 payload bytes.
The unchanged strict loader verifies every key,shape,dtype and raw payload byte.
This proof-only factory temporarily maps precisely220 meta parameters to BF16
storage, then restores their genuine FP32 runtime dtype with finite-value and
BF16 roundtrip payload-hash checks. All420 remaining parameters stay BF16.
Runtime parameter bytes2,619,359,264. No global loader/source policy changes.

Only generated rope_phases is explicitly nonpersistent in this proof, because
it is absent from disk. Genuine CPU construction matches an independent source
reference exactly: complex64[4096,64],SHA256
31612f9475a55a4ba9d03f9450b99841b92e4ef4a7fd0985b9e93b3243c090f6,
2,097,152 bytes. Its named-buffer/device/dtype contract remains mandatory.
Model placement changes device only; no blanket dtype cast.

Saved DINO artifact is untouched:
3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a.
Full original provenance and payload-integrity/finiteness are validated by the
existing writer verifier. No DINO model is instantiated. CPU inputs:
x=(arange32768%257-128)/128 asF32[1,8,16,16,16],t=F32[500],
cond=verifiedF32[1,1029,1024]. Genuine source explicitly converts hidden state,
modulation and cond to BF16; norm/RMS/RoPE useFP32 intermediates; output returns
F32[1,8,16,16,16]. Autocast/gradients must be off; inference mode on.

Explicit sdpa selects an existing genuine backend. No Triton compilation,
other models, samplers, CuMesh, pipelines, mesh export or CPU fallback. The
pinned inert pipelines package initializer is permitted because trellis2 eagerly
imports it; its submodules are forbidden.

Actual CPU-to-meta Module.to and frequency Tensor.to observations include
unchanged dtype and strided layout defaults. Placement accepts only those
defaults and the exact owned source IDs/device. Internal casts are onlyF32↔BF16;
complex64 stayscomplex64. Direct input/output copies remain device-only permits.
CPU harness failures and their precise corrections are retained in separate
directories. No GPU run occurred during those tests.

69 focused host tests passed:18new,51existing diagnostic/control/monitor/teardown.
No AI framework imports in those regressions. The owned AMD meta/synthetic probe
passed in7.578s, no real weights loaded/GPU initialized, exit0, Job empty.
Independent source/candidate review passed and separately ran18new tests.
Passive observer healthy. Exact harness must be committed/pushed/verified first.

Bounds:CPU180s,GPU placement+forward+release240s,overall ownedworker480s.
Larger checkpoint and4096-token30-block attention justify these ceilings;
first-use native SDPA/BLAS may consume substantial time. Fresh scratch caches;
no historical cache reuse for this different workload. Estimated resident model
~2.62GB, plus tens/hundreds ofMB activations and potentially~1GB attention matrices.
Cap PyTorch allocator at12GiB, require13GiB available before placement, check
block memory counters and preserve runtime OOM/timeout evidence without retry.
The allocator cap does not bound every native driver allocation; observer/
deadline/owned Job remain necessary. No numerical/quality parity claim.

Exactly1root invocation,30block,self-attention andcross-attention invocations
each, one copy ofx/t/cond/output andsource frequency. Durable markers distinguish
host returns from post-forward synchronization. Windows checks every5s with9s
staleness rejection; process/childCPU every2s. On success clear owned refs,
synchronize,clear reviewed BLAS workspaces once,empty allocatorcache,synchronize,
require0allocated/0reserved,weakref collection andcleanprocess/treeexit. Errors
never trigger a recovery GPU operation, warm-up, second input or retry.
