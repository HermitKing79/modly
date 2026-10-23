# Preview image → genuine DINOv3: BLOCKED before launch

Base: `7d11b360cbd53db9ef0869daaa65f6bdf7d1fa3a`. Branch: `codex/amd-preview-dinov3-conditioning`.

**No neural attempt ran.** The user explicitly selected Windows-native conditioning, with WSL NOT RUN. The accepted preview remains running with its original image and generation disabled. No production decoder, authority, supervisor or installed environment was changed.

| Boundary | Verdict |
| --- | --- |
| Selected image identification and fresh exact-byte CPU snapshot | PASS within the scope below |
| Original CPU image contract and request-hash binding | PASS; CPU-only context, not a production stage ticket |
| Complete conditioning request/authority integration | BLOCKED |
| Genuine DINOv3 model load / preprocessing / forward | BLOCKED — NOT RUN |
| Neural artifact persistence and independent readback | NOT RUN; no conditioning artifact exists |
| Windows lifecycle | Prior synthetic authority PASS preserved; new neural lifecycle NOT RUN |
| WSL lifecycle | NOT RUN, per clarified scope |
| Downstream stages | Disabled / NOT RUN |
| Full generation / fresh image-to-3D | NOT QUALIFIED / NOT QUALIFIED |

## Image evidence

The running original picker identified the selected file's existing Downloads location; it was cancelled without replacement. The visible CPU result matched the fresh snapshot: 642,414 bytes, RGBA PNG, 1216×1294, alpha foreground 1186×1240 and resized foreground 937×981. The original CPU validator was called without Torch or models, and the source/copy were read back byte-exactly.

Original image SHA-256: `db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e`.

The snapshot, original path/file identity, canonical parameters, new CPU request context, screenshots and complete CPU response remain private under `artifacts/amd-preview-dinov3-7d11b36/`. The preview keeps its response in memory; this work does not claim that its earlier response was exported as a durable host authority receipt. The new receipt establishes fresh CPU input identity, not completed inference. An initial relative output argument was correctly rejected before writing; the corrected absolute-path CPU command succeeded. That first failure is retained.

## Concrete execution gap

The reusable real compute boundary is `amd_dinov3_conditioning.worker()` with the reviewed BLAS-workspace teardown profile and feature saving. The current real-image CLI in `amd_generation_occupancy.py` delegates a three-stage DINO/sampling/decoder sequence, so it was not invoked.

The direct legacy worker command uses the D: interpreter with `-B -I`. The production authority requires its exact host interpreter with `-B -I -S` and a host-pinned entrypoint. The actual bind guard rejects direct substitution. A wrapper must also preserve the legacy supervisor's true attachment, CPU-ready observation, binary GPU authorization, monitoring and release; supplying fake stdin authorization is not a solution.

A wrapper-only composition is **not qualified**. The legacy host waits at the CPU gate without a parent cancellation reader, sends GPU authorization and closes worker stdin. The worker has no production JSON-CANCEL polling during synchronous placement/forward/release. Existing forced Job containment remains a valid failure boundary; it cannot be relabelled as natural cancellation or successful settlement. The proposed nesting also adds short Git/PowerShell descendants that must all satisfy the production observer's exact process-accounting rule. No such composition was built or executed.

The production whole-stage ceiling is 180 seconds, including settlement. Legacy CPU/GPU/overall limits are 120/180/360 seconds after its preflight. Historical speed does not prove full-range compatibility. Likewise, the unchanged outer 6 GiB cap is not a qualified bound for a newly nested whole process tree. These are integration gaps, not evidence of DINO numerical or AMD incompatibility.

## CPU evidence and limits

Six focused checks passed once in 1.116 seconds using the pinned API interpreter. They exercise real `UpstreamJob`/authority binding on labelled synthetic requests: incompatible interpreter/isolation refusal, valid stdlib binding without authority, worker-PASS rejection, sticky cancellation denial and the unchanged 180-second/6 GiB limits. Native process/Job calls were deny-only mocks and remained uncalled. This is not native lifecycle proof.

Fourteen relevant production sources match the base commit; seven installed conditioning sources and the config match their pins. The checkpoint header, size, timestamp and 415-F32-tensor layout match the audit. The saved full-checkpoint SHA is retained as prior provenance; the full 1.21 GB payload was not rehashed, loaded or newly runtime-qualified after the launch gate blocked.

The original DINO arithmetic requires 5,923,815,424 available host bytes and 3,360,002,048 free device bytes, with zero owned baseline GPU allocation. The product RAM floor remains 12 GiB. These are source-level thresholds, not live admission or measured usage for this image. No GPU memory, neural runtime or release measurements are claimed.

Source-safe receipts are in [the evidence directory](amd-preview-dinov3-evidence/). Private images, raw context/nonces and model artifacts are excluded from GitHub. Preservation and independent review record the exact checked scope.

**Next step:** qualify one conditioning-specific production control adapter that preserves the genuine CPU-ready/GPU-GO exchange, cancellation, descendant accounting, whole-stage budget and host-owned feature readback. Then seek one fresh neural attempt; do not bypass those gaps or enable Generate.
