# AMD cached-kernel-only convolution proof

**PASS â€” one genuine cached kernel launch, exact identity output and clean owned release.** This covers only the N2049/C1024 centre-identity fixture, not the full convolution wrapper or TRELLIS.2 decoder. Starting commit: `d0a7cd6b81abb14d502fe8604d0ec34bca385ce6`. Executed harness: `ee419ff3887bcf601b0e00529c9e04310b6cd0bc`, committed, pushed and GitHub-ref verified before execution.

## Actual result

| Measurement | Observed |
|---|---|
| Runtime/GPU | Python 3.12.10, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, Triton 3.8.0; RX 9070 XT, gfx1201 |
| Selected kernel | Genuine installed FlexGEMM F16 masked-implicit-GEMM SplitK; N2049, Ci=Co1024, V27, SPLITK2; B1=64/B2=128/BK32; grid(264,2) |
| Raw output | F32[2,2049,1024], all 4,196,352 values finite |
| CPU reduction/reference | All 2,098,176 values exactly equal in F32 before F16 conversion; maximum error 0; coordinates unchanged |
| Calls | One convolution/JIT/HSACO/launch and one cached HIP-helper load; neighbor/post1/post2/outer/inner all 0; forbidden calls[] |
| Synchronized elapsed | 0.016 seconds for host cached-leaf call plus synchronization; owned worker supervision 7.828 seconds |
| Peak PyTorch allocated/reserved | 77,848,576 / 98,566,144 bytes |
| Release | Allocated 0 / reserved 0; all owned tensors collected; worker exit 0, owned Job exited unforced |
| Monitoring | Before, CPU gate, one periodic sample and after-exit healthy; no selected System events; controller OK |

No warmup or retry occurred. The immutable exclusive attempt marker and durable trace establish the single attempt.

## CPU/source admission

The CPU neighbor metadata uses source-derived ordering `v=(dx+1)*9+(dy+1)*3+(dz+1)`, centre13 and UInt32 sentinel `0xffffffff`. Independent tuple membership confirms 35,353 valid neighbors. Source-faithful Gray-to-binary sorting and per-64-row block unions yield 599 kernel indices and 34 segment offsets. Stable sorting of ties is valid because equal binary codes have identical masks. Padded-row and invalid-neighbor pointer masking were independently reviewed. Principal GPU buffers total 77,847,128 bytes.

The actual installed binder matched the existing complete cache container, source key, compilation options, gfx1201 HSACO and exact cached HIP helper. All 17 inherited compile/build/cache/autotune/JIT/launch safeguards remain. Direct invocation uses exact owned Tensor/Storage identities, aligned offset-zero contiguous buffers, fixed arguments and one launch. No full Python convolution wrapper, native neighbor generation or native postprocessors ran.

Six focused host regressions and 13 genuine installed constructor tests pass. GPU initialization/native execution/weights were blocked in the owned CPU worker, which verified all nine scratch artifacts by readback. CPU-only NumPy sentinel and metadata-comparison failures are preserved. The restricted observer query lacked System-log/WMI permissions; the unchanged observer passed with those permissions before execution. All GPU-stage observations used that working context.

The 45/120/180-second CPU/GPU/overall deadlines,13GiB free-VRAM admission, 12 GiB allocator ceiling and 1 GiB current/peak allocation/reservation cap remained unchanged. GPU admission observed 16,937,582,592 free bytes. Available-memory loss at kernel was 109,838,336 bytes; after release 13,369,344 bytes. These stayed below the 1 GiB native-memory stop limit. PyTorch metrics do not enumerate every native/driver allocation; the remaining global difference has no proven owner and is not claimed to be zero device usage. Cached metadata declares no global/profile scratch.

## Preserved outputs and independent verification

Local artifact directory: `C:\Users\wattz\Documents\ChatGPT\modly\proof\docs\compatibility\amd-cached-kernel-convolution-evidence\artifacts`.

| File | Bytes | SHA256 |
|---|---:|---|
| [raw_splits.npy](amd-cached-kernel-convolution-evidence/artifacts/raw_splits.npy) | 16785536 | `db45c447b1bb35aa148ed9f0dbd19c0d50f83bf2906f2767162cbbc200c25e9a` |
| [input_features.npy](amd-cached-kernel-convolution-evidence/artifacts/input_features.npy) | 4196480 | `05ea4d32e669734812d24510e18bdf81abf99f87964cc5b540c1f06e5fed9c90` |
| [output_features.npy](amd-cached-kernel-convolution-evidence/artifacts/output_features.npy) | 4196480 | `05ea4d32e669734812d24510e18bdf81abf99f87964cc5b540c1f06e5fed9c90` |
| [coordinates.npy](amd-cached-kernel-convolution-evidence/artifacts/coordinates.npy) | 32912 | `635d2189e1dc178e14bf454932ac3dfb3f5c5a7a0e4bfdeb88f24e8901ce7cc7` |

The worker exclusively saved/read back outputs in owned scratch. The parent promoted byte-identical files after verified clean exit. Independent CPU addition and a separate standard-library NPY reader both reproduced every reference value and coordinate. Reviewers independently rehashed all artifacts and original scratch files. The executed 37-file hash map exactly matches the independently reviewed candidate.

[Run evidence](amd-cached-kernel-convolution-evidence/run/results.json), [trace](amd-cached-kernel-convolution-evidence/run/trace.jsonl), [CPU oracle/cache identities](amd-cached-kernel-convolution-evidence/cpu-oracle-audit.json), [independent artifact validation](amd-cached-kernel-convolution-evidence/independent-artifact-validation.json) and [outcome review](amd-cached-kernel-convolution-evidence/outcome-review.json) contain the complete records.

## Preservation and stop

[Postcheck](amd-cached-kernel-convolution-evidence/git-after.json) verifies all 5,865 historical files and all prior local/remote branches, including main and the starting branch. Both installed-environment inventories are byte-identical before/after (SHA256 `0be055c97659c345754515298c56b8a707d7303a7e30159af00e8a142b9cf475`). AMD source HEAD/status, existing nodes.py modifications/backup, model/cache metadata and native binaries are unchanged. The only old harness change returns the already-admitted JIT leaf when explicitly requested; production extensions are untouched. No AI weights, decoder, mesh or texture operations were used.

Smallest next numerical experiment, only with separate approval: one non-identity neighbor-mixing stencil through this same cached leaf, with a complete CPU reference and unchanged limits. It would test noncentral neighbor contributions without exposing the full decoder. General convolution and dynamically expanding decoder support remain unproved. No further execution is planned under this approval.
