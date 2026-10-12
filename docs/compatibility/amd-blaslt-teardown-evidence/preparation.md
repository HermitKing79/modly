# Bounded BLASLt teardown preparation

2026-10-13 (Europe/London). Base `62409edb5545857012116cb64fc2335acd938f54`;
branch `codex/amd-blaslt-workspace-teardown`. No GPU attempt has yet occurred.

Independent installed-PE review traces the no-argument Python method table and
callback through the torch_hip.dll import/export to both distinct workspace
maps. Each map is exclusively locked, erased, reset and unlocked; tree nodes
destroy their c10::DataPtr values. Bounded raw disassembly, PE rows, exact
commands, machine-byte checks and binary hashes are saved. This matches the
relevant upstream v2.13.0 behavior, without assuming that version metadata
establishes binary identity. The clear routine contains no handle/device
initialization or workspace-size override operation. Its scope is both
process-wide maps, requiring completed GPU work and an exclusive owned worker.

The actual AMD Python runtime exposes the expected builtin from torch._C.
Windows loaded-module queries locate the independently hash-pinned
torch_python.dll and torch_hip.dll in the intended venv. That CPU-only check
did not invoke the native binding, initialize the GPU or read model weights.
Two preliminary CPU smoke captures are preserved: missing WORKSPACE_DIR before
Torch import, then a correctly rejected evidence write outside owned scratch.
The final smoke sets the workspace explicitly and returns stdout for the
parent to save; the original write/network/subprocess guards remain intact.

Only the new evidence profile opts into workspace teardown. Existing mock
application behavior and prior proof modes remain unchanged. The actual worker
releases the model/output/pipeline/extractor and bound-method references,
clears the genuine RoPE cache, runs GC, synchronizes device 0, calls
_cuda_clearCublasWorkspaces once with no arguments, then performs the established
empty_cache and synchronization. Durable entry/return/error markers surround
the native call; a failure propagates to the existing owned-job supervisor with
no GPU recovery or retry. Zero allocated/reserved acceptance remains unchanged.

The previous bounded allocator history/weak registry is reused. Snapshot is
captured only if allocated or reserved bytes remain after cleanup. On zero
release, history is disabled without a snapshot. No new feature file is written
or the previous artifact republished; its SHA-256 remains
3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a.

CPU-only regressions: 65 existing diagnostic/control tests, 11 allocation tests
and 10 new actual-worker teardown/binding/classification tests pass. Red captures
demonstrate the missing clear/history/classifier behavior before implementation.
The tests exercise actual AST teardown and logging functions with native/GPU
boundaries stubbed; they check ordering, single invocation, failure propagation,
no recovery, conditional snapshot, default-mode preservation and the unchanged
F32 finite [1,1029,1024] one-forward contract. AI imports are blocked in fixtures.
The explicit CLI requires WORKSPACE_DIR to resolve to the proof checkout before
consuming an attempt marker; missing/wrong values fail in CPU-only preflight.

Before inventories and compressed environment/model-cache metadata are
byte-identical to the previous proof's after records. The existing source edits,
audited DINO checkpoint/source hashes and saved features are preserved. Fourteen
installed source/binary identities are pinned. Independent complete-candidate
review, exact harness commit/push verification and fresh healthy Windows
observation are required before the one approved attempt.

Execution scope remains exactly one strict DINO load and genuine 512-resolution
conditioning pass on the same deterministic RGBA fixture. F32 throughout;
CPU120/GPU180/overall360 seconds, healthy periodic Windows monitoring and
pre-preparation Job Object attachment. Fresh private scratch/cache only; no
historical artifact execution, warm-up, second pass, other model, geometry,
texture, GLB, downloads, installations, source edits or native rebuilds.

Exact launch (only after review/publication/healthy observation):

```powershell
$env:WORKSPACE_DIR='C:\Users\wattz\Documents\ChatGPT\modly\proof'
python -B -I scripts/compatibility/amd_blaslt_teardown.py --output docs/compatibility/amd-blaslt-teardown-evidence/run --harness-commit <verified-published-commit>
```
