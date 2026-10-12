# DINO allocation diagnosis and feature checkpoint preparation

Base `dfc01cb2f233e2d2cbabfcec6a27ac4c42f23668`; branch
`codex/amd-dinov3-allocation-diagnosis`. One new conditioning invocation is
authorized only after this exact candidate is reviewed, committed and published.

Read the original report and release evidence. No obvious surviving reference
was found in the cleanup sequence: model/feature/pipeline/cond/output cells and
the bound original_features method are cleared; module/placement maps contain
IDs rather than objects; diagnostics contain primitive descriptions; the real
RoPE coordinate cache is explicitly cleared. Previous weak references reported
model/output collection, but did not establish collection of every tensor wrapper
or identify native workspace owners.

Installed Torch 2.13 source documents bounded allocator-history recording and
active_allocated/active_awaiting_free/inactive snapshot states. Its wrapper uses
_cuda_memorySnapshot(None), even when a device argument is supplied. The harness
therefore requires exactly one visible GPU, current device 0, verifies every
segment device and uses augment_with_fx_traces=False. History uses enabled=all,
context=state, stacks=all, max_entries=512, clear_history=True; no FX/compile
annotations. Active allocation stacks survive independently of the 512-entry
history ring. CPU fixtures replay the actual installed Python wrappers with
native stubs; no Torch import/GPU initialization occurred in those tests.

The native call remains bounded by the unchanged 180s GPU/360s overall supervisor
and healthy periodic monitoring. Snapshot bounds: 4096 segments, 16384 blocks,
512 history entries, 256 frames per allocation, 16 MiB serialized evidence;
overflow/malformed data or inconsistency with measured allocated/reserved bytes
fails clearly. Actual workload is the unchanged single DINO/512 fixture. A
weak-only registry observes up to 8192 GPU tensor wrappers from dispatch and
placed model state without retaining them. Matched storage addresses indicate
observed Python owners; allocation stack frames indicate origins, not ownership.
Absence of an observed wrapper does not prove native ownership.

Installed PyTorch's cudagraph_trees source documents persistent BLAS workspaces,
and its backends module declares read-only size queries. Capture those two
settings. **Do not clear BLAS workspaces, reset their sizes, change backend or
relax the zero allocated/reserved release criterion in this diagnostic run.**
The snapshot should determine whether this candidate explains the 76 MiB residue.

The explicit local DINO checkpoint and seven prior source pins are unchanged;
full hash dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179.
Reuse the verified strict loader, genuine architecture, alpha-only preprocessing
and float32 [1,1029,1024] output contract. No old proof/GPU test is repeated.

Only after finite output validation, save the existing CPU copy through the
genuine safetensors.numpy serializer. The writer accepts only a contiguous,
little-endian F32 CPU NumPy array of exact shape [1,1029,1024], 4,214,784 bytes;
total artifact is bounded by payload plus 16 KiB. Exclusive write+fsync preserves
existing files. The actual header, sole cond tensor, shape/dtype/provenance,
finite payload, output SHA and full file SHA are independently read back.
Metadata pins checkpoint/config, source image pixel/PNG, genuine extractor/
pipeline source, runtime and execution harness commit. No dtype conversion.
Audit allows only scratch/conditioning.safetensors during feature-save, with
GPU tensor operations prohibited in that phase. This is a feature artifact,
not a second AI model checkpoint. No production worker changes.

Original supervisor/Job/observer/guards remain in use. CPU120/GPU180/overall360,
one strict DINO load, one conditioning pass, no warmup/retry. All temp/home/cache
paths are fresh owned scratch; no historical cache is executed or modified.
After the existing model/output/cache cleanup, capture reconciled allocation
data, disable history and apply the unchanged strict release criterion. On any
failure, preserve evidence/features and terminate only the owned process tree.

CPU verification: 65 focused existing regressions plus 11 new tests cover actual
audit/logging expressions, installed allocator-wrapper replay, authorization,
single snapshot, state/size/bound rejection, weak non-retention and genuine CPU
safetensors integrity/corruption/dtype/shape/path/provenance/overwrite handling.
An AST replay ordering bug in an intermediate test was corrected; no GPU run
was attempted. Two initial missing-helper assertion groups demonstrated red
tests before the helpers were implemented. Independent complete-candidate
review and exact harness publication are still required before execution.
