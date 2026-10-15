# Genuine full turret neural computation — UNKNOWN: artifact not saved

Base: `a28e16ee569a168ac8c7df09bacd583eccff0ec7`. Branch: `codex/amd-wsl-turret-chunked-decode`. Independently reviewed candidate published at `043153e2e787afc3ffd24a439cedd300fcbe7d91`; exact harness and qualified readiness were GitHub-verified before the **single attempt** at **`e14489cc17a240a8eff922580596d35a7db94353`**.

**UNKNOWN overall. The complete genuine neural forward returned and synchronized, but no output artifact was saved.** This is a demonstrated output-directory preparation defect, not a native convolution, subdivision, numerical or VRAM-admission failure. The worker released its data; it cannot be recovered from this attempt. No repair or retry followed.

## What executed

| Measurement | Actual result |
| --- | --- |
| Genuine parent forwards / original convolutions | **1 / 40** |
| Learned support growth | **3,743 → 17,015 → 79,169 → 378,891 → 1,842,968** |
| Original C2S calls / native global neighbour maps | **8 / 5** |
| Final explicit-GEMM tiles | **57**, tile size 32,768; final tile 7,960 |
| Complete-forward GPU synchronized time | **10.966630407 s** |
| Actual initial free VRAM | 14,560,526,336 bytes; unchanged 13 GiB gate passed |
| Peak Torch allocated / reserved bytes | **5,588,655,616 / 7,291,797,504** |
| Post-release allocated / reserved bytes | **0 / 0** |
| Worker / supervisor / outer duration | 39.906 / 55.672 / 60.215 s |

Both fourth-stage expansions and every memory admission passed against live measurements. All 292 genuine checkpoint tensors loaded strictly; the reviewed 216 BF16 / 76 F32 runtime configuration and actual false TunableOp flags were unchanged. The full original latent/coordinate hashes and row order were retained.

Only `blocks.3.4.conv2` used row tiling. Its original full native neighbour map remained global; each tile gathered from the complete feature tensor using unchanged global indices, sentinel handling, kernel-slot order, weight view and original mm/addmm bias path. The temporary static-method patch was restored. Earlier convolutions, all learned masks, both C2S expansions, native/cache/build guards and original F32 finalization remained intact. The million-row cap still applies elsewhere; the exception admits only the exact verified final support and widths.

Eleven focused CPU tests passed, including comparison with the exact pinned original Python source and an independent coordinate-tuple reference. Tested BF16/F32 maximum differences were zero. This does not prove bitwise HIP equivalence for arbitrary inputs. Independent source and memory reviews approved the exact candidate; the conservative allocation plan fit the unchanged limits and then passed live admission.

## Exact persistence failure

After synchronization, the worker copied all ten tensors to CPU. The preserved traceback reaches `preserve_output()` at:

`ARTIFACTS.mkdir(parents=True, exist_ok=False)`

The intended leaf was `/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-wsl-turret-chunked-a28e16e`. Its **parent `artifacts` directory did not exist**. Pathlib attempted to create that parent; the unchanged policy permits writes only within the exact leaf, scratch and evidence roots, so it correctly rejected:

`ImportPolicyError: filesystem_write: /mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts`

The host preflight and reviews missed this deterministic prerequisite. The executed writer calls `validate_output()` before `mkdir`; reaching this line therefore supports that F32 `[1842968,7]` / I32 `[1842968,4]` shape, coordinate bounds/uniqueness, subdivision linkage and finiteness validation returned. **That is control-flow evidence:** its summary/statistics were not journaled, no safetensors bytes or hash exist, and independent artifact readback was impossible. The estimated ten-tensor payload was 96,412,768 bytes, below the unchanged 256 MiB limit.

[Raw worker](amd-wsl-turret-chunked-decode-evidence/run/guest/results.json), [operation journal](amd-wsl-turret-chunked-decode-evidence/run/guest/phases.jsonl), [verified findings](amd-wsl-turret-chunked-decode-evidence/verified-result.json) and [independent outcome review](amd-wsl-turret-chunked-decode-evidence/independent-outcome-review.json) preserve these distinctions.

## Cleanup, preservation and next action

Twelve Windows observations were healthy. Synchronization/cleanup succeeded, all tracked model/input/output references were collected, workspace clearing occurred once, and owned Torch counters reached zero. Worker exit 1 was handled; the Windows Job emptied naturally without forced cleanup and only named Modly Ubuntu was terminated. Both distros are stopped; Docker remains default.

Before/after audits matched both Windows environments, source edits/backups, turret inputs, checkpoint/configuration, baseline WSL packages/venv, 922 staged payloads, 23 packaging files, 80 native source files and protected ROCm dependencies. **643 protected historical files, main and 58 previous local/remote branches** remain unchanged. No installation, compilation, precision fallback, resolution change, mesh extraction or texturing occurred.

The smallest correction is **host-side preparation/checking of the exact artifact parent before any attempt marker**, followed by a CPU-only real writer/policy acceptance with a fresh leaf. Keep the worker policy narrow. A separately approved fresh decode is necessary to obtain the lost tensor data; no earlier model stage needs repeating.

Once data is saved, the shortest mesh route is a separately qualified O-Voxel extraction from that file, applying the pinned sigmoid vertex transform, `channels 3:6 > 0` flags and softplus split weights at grid 512/AABB ±0.5, then untextured OBJ/PLY export. It should reuse saved neural output rather than invoke the decoder. No O-Voxel binary/build or extraction was qualified or executed here. Stop for approval.
