# Genuine WSL turret neural decode — UNKNOWN

Base `a9c9ad7bcdd164836d7f44244f32ad82a4ea2839`; branch `codex/amd-wsl-turret-neural-decode`. Reviewed execution harness published and GitHub-verified before the single attempt at **`72c190324805d0930983682d9d619ef686f998d3`**. [Four candidate hashes](amd-wsl-turret-neural-decode-evidence/candidate-identities.json) remain unchanged.

**UNKNOWN:** strict CPU loading and selective BF16 runtime configuration succeeded, but the worker stopped at `decode.py:325` with **`RuntimeError: Unexpected Torch BLAS tuning policy`**. The genuine parent forward was never entered: **forward count 0**, no GPU model placement, convolution, learned subdivision or output artifact. This is an admission-policy boundary, not demonstrated AMD numerical/native incompatibility. One worker attempt, no retry or post-attempt harness correction.

## Exact observed boundary

The rejected expression was:

```python
if torch.cuda.tunable.is_enabled() or torch.cuda.tunable.tuning_is_enabled():
    raise RuntimeError('Unexpected Torch BLAS tuning policy')
```

At least one predicate evaluated true. The worker did not record their individual values; neither their actual state nor an active benchmark can be inferred. [Read-only installed WSL source](amd-wsl-turret-neural-decode-evidence/installed-tunable.py.txt) documents feature enable and tuning enable as separate controls. An armed tuning flag can be different from active TunableOp routing, but that possibility is **not an observation of this worker**. The source was read without Torch import, runtime flag queries or changes.

[Raw worker](amd-wsl-turret-neural-decode-evidence/run/guest/results.json), [supervisor](amd-wsl-turret-neural-decode-evidence/run/results.json), [durable phases](amd-wsl-turret-neural-decode-evidence/run/guest/phases.jsonl), [outer capture](amd-wsl-turret-neural-decode-evidence/execution-host-capture.json) and [independent outcome review](amd-wsl-turret-neural-decode-evidence/independent-outcome-review.json) agree on that boundary. The last journaled successful stage is native qualification/CPU preparation; results additionally prove completed CPU loading and conversion before rejection. There is no `gpu-stage-entry` or neural-operation journal entry.

| Measurement | Actual result |
| --- | ---: |
| Genuine FlexGEMM import + synchronization | PASS, 18 bindings; 0.269 s |
| Strict decoder CPU loading | PASS, 292 entries; 10.785 s |
| Original precision contract | 216 F16 / 76 F32; exact byte checks and all 76 round trips |
| Recorded runtime conversion | 216 BF16 / 76 F32; full affected-key manifest |
| CPU RSS / high-water at strict-load completion | 1,761,386,496 / 2,700,746,752 bytes |
| Initial free VRAM | 13,979,987,968 bytes (13.02 GiB) |
| Torch allocated/reserved: baseline, peak, release | **0 / 0 bytes** |
| Device free VRAM after release | 13,888,884,736 bytes |
| Genuine neural forwards / native binding calls | **0 / 0** |
| Guest-after-GO / supervisor / outer duration | 23.593 / 40.531 / 43.114 s |
| Worker exit | 1, controlled policy rejection |

CPU timing/RSS were sampled before BF16 conversion; a whole-attempt CPU peak or conversion duration was not measured. Device-level free-memory variation is distinct from owned Torch allocation and is not attributed to a leak. Model/component/all parameter weak references were collected. Nine periodic Windows observations were healthy; Job active counts 0→0, naturally empty, no force; only Modly Ubuntu was terminated, exit 0. No timeout, reset or native crash was recorded.

## Source-faithful candidate and unexercised GPU protections

The authentic input remains F32 `[3743,32]`, SHA-256 `90eaf72030f303b209a4d02ae9adc7d4ecd39662344f35f0adc8120bc1e871b9`, and I32 `[3743,4]` coordinates, SHA-256 `949a9e6a2548abcad66354384502c06834d4d56ea3f2fa86b619c25926379ed4`. Original checkpoint SHA-256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581` and the unchanged loaders were verified. No latent regeneration or model-file change occurred. Inherited metadata-only labels do not replace the current `weights_loaded=true` strict loading diagnostics.

[Reference assessment](amd-wsl-turret-neural-decode-evidence/reference-runtime-policy.json) pins `kr4ckhe4d/trellis2-9070xt@0220d73971fd0be242b2b311ac00d590e1b26301`: after strict original loading, `blocks.float()` then only `MIX_PRECISION_MODULES` body tensors to BF16; norms/projections and input/output handling remain F32. Every runtime dtype change is recorded. Both TRELLIS and FlexGEMM select `explicit_gemm`, ratio 2.0, TF32 disabled, with original compiled neighbour-map construction and Torch im2col/addmm. No fallback, JIT, helper build or autotuning is admitted. The reference's vendored decoder is unpinned and HANDOVER reports nonfinite raw vertices, so policy correspondence cannot promise numerical parity or geometry quality.

CPU-tested active hooks cover 40 original sparse convolutions and eight original C2S calls in four learned stages, checking live memory, buffers/caches, actual learned masks and child counts before expansion. They preserve arithmetic, masks and row order. **These hooks and native execution boundaries were not reached by this worker.** The old diagnostic-only planner was not represented as an execution gate.

Preserved budgets: 13 GiB initial free VRAM, 12 GiB allocator cap, 2 GiB operational free reserve, 1 GiB opaque-workspace allocator headroom, 25% visible padding, one-million-row maximum, 256 MiB artifact payload limit; 60/120/240/450/430/480-second native/CPU/GPU/guest/authorised/host deadlines. Conservative source bounds and containment do not mathematically bound every opaque HIP/BLAS allocation. Six focused runtime CPU tests, 13 admission tests and marker-free read-only preflight passed; [pre-execution review](amd-wsl-turret-neural-decode-evidence/independent-preexecution-review.json) gave conditional GO before publication. Metadata native-result simulation proved no GPU computation.

## Preservation and one next action

[Final preservation](amd-wsl-turret-neural-decode-evidence/preservation-verification.json): both Windows environments/binaries, baseline WSL venv/packages, 922 original staged payloads plus 23 packaging files, 80 native build-source files, AMD edits/backups, all turret assets and checkpoint unchanged. Reverified 368 historical files, five controls, two loaders, four original bytecode files and all 53 previous local/remote heads, including main. Additional generated bytecode remains untracked. Both WSL distros are stopped; docker-desktop remains the default. No package, driver, kernel or native build change occurred. The exclusive attempt marker and every error are preserved. No neural tensor file or mesh exists.

**Exactly one recommended next experiment:** a separately approved, bounded CPU-only qualification of the installed WSL TunableOp gate. Record both actual flags and relevant environment settings before any model load, with GPU-entry denial; compare those values with the installed source's feature/tuning semantics and test the existing production gate. Do not change flags, load weights or retry a forward during that diagnostic. This is the smallest missing observation needed before reviewing a narrowly justified gate correction and any fresh real turret attempt. Stop here; no decoder retry, mesh extraction or texturing is authorised.
