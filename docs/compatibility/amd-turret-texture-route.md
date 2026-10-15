# Turret texture route: assets present, execution unqualified

Base: `0a1cdee3aa63c8ab2adfd0206dc87f7b7f5b0be0`. Branch: `codex/amd-turret-texture-route-audit`.

**PARTIAL readiness: a source-supported route exists without regenerating geometry, but there is no qualified end-to-end texture/export path yet.** No inference, AI-runtime import, model construction, WSL startup, installation or environment change occurred. This audit used source/configuration reads, bounded checkpoint headers, streaming hashes and CPU inspection of saved data.

## Verified reusable assets

[Exact paths, complete hashes and checks](amd-turret-texture-route-evidence/local-assets.json):

| Asset | Verified contents |
| --- | --- |
| Original `selected-input.png` | Official Microsoft turret image; 1042×1042 RGBA, real transparent foreground; file SHA-256 `07557b7b9f726aad3b291f4902498e52423efadbf746fb2c5759e3e2c6adeef0` |
| Saved `conditioning.safetensors` | Original DINO F32 `[1,1029,1024]`, hash matches sampling provenance |
| Saved shape sample | Complete raw and denormalized F32 `[3743,32]`; original I32 `[3743,4]` coordinates; finite, provenance-matched; denormalization reproduced byte-exactly |
| `turret-neural-output.safetensors` | Original hash `dc466749…bd6e22`; four finite subdivision-logit records and parent coordinates, expanding 3743→17015→79169→378891→1842968 rows |
| `slat_flow_imgshape2tex_dit_1_3B_512_bf16` | Local configuration and complete 2,584,672,728-byte checkpoint; 640 BF16 entries; SHA-256 `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d` |
| `tex_dec_next_dc_f16c32_fp16` | Local configuration and complete 948,458,812-byte checkpoint; 284 F16 entries; SHA-256 `97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c` |

Both checkpoint/configuration pairs are under `D:\AI\Trellis2AMD\ComfyUI\models\microsoft\TRELLIS.2-4B\ckpts`. Payload coverage and full file hashes were checked without deserializing weights; this does not prove numerical validity or successful loading.

## Original learned-texture path

The [source audit](amd-turret-texture-route-evidence/independent-texture-source-audit.json) pins installed files and line references:

1. `sample_tex_slat` consumes the **denormalized** saved shape latent, normalizes it with **shape** statistics, and concatenates fresh 32-channel noise with that 32-channel condition. The genuine model requires **64 input / 32 output channels**, plus saved DINO conditioning. Existing shape-flow guards pin 32 input channels and cannot be reused unchanged. Canonical texture defaults are 12 Euler steps, sigma minimum `0.00001`, guidance `1.0`, rescale `0.0`, interval `[0.6,0.9]`, `rescale_t=3.0`, DINO lock `0`: **12 positive forwards**, no negative CFG forwards. Output uses distinct **texture** denormalization statistics.
2. `decode_tex_slat` clears the sparse spatial cache, calls the genuine texture decoder with `guide_subs=saved_subdivisions`, then applies `output * 0.5 + 0.5`. No shape-decoder reload is required. C2S consumes guide features by **row position**, ignoring guide coordinates: exact live parent-coordinate order must be asserted at every stage. The initial order already matches. Final texture support must match the original 1,842,968-row geometry support.
3. The six learned channels are **RGB, metallic, roughness and alpha**. They can provide image-conditioned dark metal and coloured details; fidelity remains untested. There is **no emissive channel** in this inference/export contract. Blue albedo is not glowing emission. True blue glow would require a separately identified, explicitly authored material extension; it must not be presented as learned TRELLIS output. Existing seven-channel geometry features contain no PBR textures.

## Existing AMD code and concrete gaps

Existing Windows AMD attention/flow code and the pinned Linux reference (`kr4ckhe4d/trellis2-9070xt`, `0220d739…`) already provide useful paths: explicit GEMM, Torch SDPA, staged offload, HIP rasterizer/export fixes. Linux reports a working textured GLB, but that is not a qualification of this turret or our WSL packages. Its selective BF16 fix targets **shape**, not texture. Explicit sparse SDPA selection must be retained; the environment-variable whitelist alone does not select it reliably.

Two later blockers remain:

- Texture decoding is configured **F16**, whereas our proven final-convolution tiling admits **BF16/F32**. Texture precision, strict parameter/buffer matching and the tiling contract need explicit qualification; shape success does not authorize silent conversion. The four guides still expand to 1.84 million rows.
- [Export inspection](amd-turret-texture-route-evidence/export-route-audit.json) found stock O-Voxel `to_glb` repairs/fills/simplifies even with remeshing disabled. CuMesh UV unwrapping also removes degenerate faces. Neither is an unchanged-mesh exporter. Windows has native components, but actual baking remains unqualified, including the historical CuMesh 719 failure. WSL's established stage has FlexGEMM, not a qualified CuMesh/nvdiffrast/O-Voxel bake stack. The existing vertex-colour branch omits spatial metallic/roughness; it is only a possible learned-colour preview, not complete PBR. Do not reuse Modly's `metallicFactor=0` workaround for learned metal.

## Resources and minimal next approval

**Estimates, not new measurements:** texture flow is almost identical to the proven shape flow except its input projection. Budget roughly **4–6 GiB GPU / 6–8 GiB host working RAM**, with at least **12 GiB available host RAM** admission. Prior shape sampling measured 2.91 GB allocated GPU and 5.67 GB host working memory. Current read-only host availability was 19.16 GB; VRAM was not queried.

Texture decoding is larger: the final untiled 64-channel im2col alone is **6,369,297,408 bytes**, with a potentially equal gather plus maps, indices and retained features. It cannot be admitted unchanged under the 12 GiB cap. Proven shape tiling peaked at 5.59 GB allocated / 7.29 GB reserved GPU; use **8–10 GiB GPU and ≥12 GiB available host RAM** only as provisional planning for texture, pending its precision/workspace admission. Six F32 output channels plus I32 coordinates require about **73.72 MB** on disk. Unload flow before decoder residency.

**Recommend one separately approved 512 texture-sampling sequence only**, preferably in the existing Windows AMD flow runtime already proven with this architecture:

- CPU-verify the explicit 64-channel model contract, complete 640-entry strict load/dtype policy, normalization, coordinate identity, saved DINO provenance and deterministic fresh noise. Reuse existing loader/SDPA/containment code; avoid factories with remote fallback.
- Run the canonical **12 positive forwards**; preserve raw and denormalized finite F32 `[3743,32]` texture latents, unchanged coordinates, hashes, counters and provenance. Retain 13 GiB free-VRAM admission, 12 GiB allocator ceiling, existing monitoring/cleanup and 180/240/480-second CPU/GPU/overall limits. Require zero owned allocated/reserved bytes and clean exit; no retry.
- Stop before texture decoding/export. Next, qualify the guided texture decoder's precise dtype/row-tiling contract; then save its six-channel field before any separately approved UV/PBR bake on an export copy. Preserve the original mesh throughout. No DINO, shape sampling, geometry decoding or repairs are needed for this route.

Preservation checks and independent review are attached in the evidence directory. No textures were generated during this audit.
