# AMD cached-kernel-only convolution proof

Status: CPU preflight GO; GPU result pending. Starting commit `d0a7cd6b81abb14d502fe8604d0ec34bca385ce6`. One conditional launch only; previous full-wrapper failures remain unchanged.

The genuine installed FlexGEMM SplitK leaf is used directly. No full wrapper, native neighbor generator or postprocessor is invoked. Fixed N2049/Ci=Co1024/V27/F16, center-only identity weights, zero bias, B1=64/B2=128/BK32/SPLITK2 and grid `(264,2)` remain unchanged.

CPU metadata was derived from installed source and independently checked by tuple membership and source review: 35,353 valid neighbors, 599 valid-kernel indices, 34 segment offsets. Sentinel `0xffffffff`, center13, sorted Gray/binary codes, per-block unions, padded-row accesses and pointer bounds are verified. Stable tie order is valid because equal binary codes have identical masks. Principal GPU buffers total 77,847,128 bytes.

Six focused host tests pass. The owned AMD CPU worker passes 13 genuine constructor regressions, cached-leaf policy checks and all nine scratch-artifact readbacks; GPU/native execution and weights remain false. Exit0, 9.891 seconds, unforced owned Job cleanup. An earlier CPU metadata-comparison failure and NumPy sentinel-conversion failure are preserved. They are harness failures and establish no GPU defect.

The actual installed binder produces the historical kernel cache key and exact compilation options for gfx1201. Complete cache container, HSACO, helper/source/options/runtime identities are pinned. All 17 inherited rejection seams remain. The admitted genuine AST-derived JIT leaf must consume exact owned, contiguous, offset-zero, aligned storages; compilation, builds, cache writes/misses, autotuning and alternative launches are forbidden.

One prospective attempt retains 45/120/180-second CPU/GPU/overall limits, healthy Windows monitoring, owned Job containment, 13GiB available-VRAM admission, 12GiB allocator ceiling and 1GiB observed current/peak allocation/reservation limit. Driver/native allocation visibility is limited; available-memory loss is also capped at1GiB. Cached kernel metadata declares no global/profile scratch.

Output starts as NaNs. Both raw F32 SplitK planes must be copied to CPU and saved inside owned scratch, finite in every element, and their CPU F32 sum must exactly equal the original F16 features converted to F32 before casting back. Required counters: convolution/JIT/binary/launch each1; neighbor/post1/post2/outer/inner each0. Clean synchronized release requires collected tensor references and zero allocated/reserved bytes. Parent preservation/integrity evidence and actual outcome will follow.

Evidence: [CPU oracle and identities](amd-cached-kernel-convolution-evidence/cpu-oracle-audit.json), [installed acceptance](amd-cached-kernel-convolution-evidence/cpu-acceptance/results.json), [independent candidate review](amd-cached-kernel-convolution-evidence/independent-review.json), [preservation baseline](amd-cached-kernel-convolution-evidence/git-before.json).

A successful result would prove only this cached leaf and fixture. It would not validate the complete convolution wrapper, native postprocessors or TRELLIS.2 decoder. No AI checkpoint, geometry, mesh or texture work is authorized here.
