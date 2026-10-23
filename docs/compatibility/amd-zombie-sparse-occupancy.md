# Genuine zombie sparse structure and occupancy

PASS for the two authorised numerical stages and faithful local voxel previews.
One fresh sparse-sampling sequence completed, followed by independent readback
and review, then one occupancy-decoder forward. No neural stage was retried.

| Boundary | Result |
| --- | --- |
| Reused genuine zombie DINO artifact identity | PASS; original source and saved success evidence reauthenticated |
| Fresh sparse sampling and latent readback | PASS; 12 Euler steps, 22 forwards (12 positive / 10 negative) |
| Fresh occupancy decoder and tensor readback | PASS; one forward, original threshold and ordered coordinates |
| Windows Job settlement and GPU teardown | PASS; natural exit, empty owned Jobs, measured Torch allocation/reservation zero |
| Real local voxel previews | PASS; exact exposed-voxel OBJ and four independently reopened PNGs |
| Visual shape quality | Observational: recognisable coarse T-pose humanoid; detailed zombie likeness remains unproved |
| Detailed mesh, shape-SLat, textures, GLB and full generation | NOT RUN |

WSL was NOT RUN. Production generation remains NOT QUALIFIED and Generate was
not enabled. The running experimental image preview was left untouched.

## Authenticated execution

Base/report ancestry: `c570c2e3141e7371b24943fa9de65e16c5d8e429`.
Both new stages executed the published candidate
`e6b6330ba2f9e51abfb74c726729e0be4ade7cac` on
`codex/amd-zombie-sparse-occupancy`. The reused DINO artifact was genuinely
computed at **`93269412602a338acd9fbde41798901ddb5f707d`**, a different execution
commit. No fresh DINO receipt or inference was substituted.

The original zombie image SHA-256 is
`db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e`.
The reused finite F32 `[1,1029,1024]` conditioning file SHA-256 is
`e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60`.
Original image/crop, feature payload, checkpoint/config, old execution and saved
success receipts were checked together; historical brick/turret features were
not used.

Only experiment-specific input routing, host acceptance and bounded additional
preview views were added. The original sampler, mathematical forward,
occupancy decoder, supervisors and original preview writer remain unchanged.
Their 65-file source closure was checked against the exact execution commit.
The real owned-Job handshake precedes preparation and the original binary GPU
authorisation remains required. CPU qualification included 25 focused passing
tests and independent review. An accidentally broad earlier test selection
included a historical hardlink test blocked by the known sandbox WinError 5;
its capture is preserved and was not counted as passing qualification.

The unchanged installed Windows runtime was Python 3.12.10 at
`D:\AI\Trellis2AMD\venv\Scripts\python.exe`, Torch `2.13.0+rocm10.0.0`,
HIP `7.15.26333`, RX 9070 XT / gfx1201, NumPy 2.5.3 and safetensors 0.8.0.
Both neural invocations used the supported non-elevated outside-sandbox context.

Sampler checkpoint SHA-256:
`ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`.
Decoder checkpoint SHA-256:
`1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a`.
Full identities, config/header hashes and parameter sizes are in the committed
[CPU admission summary](amd-zombie-occupancy-evidence/cpu-admission-summary.json).
Its header-hash domain is the eight-byte length prefix plus padded JSON header.

Original precision was retained: sampler 640 BF16 checkpoint entries with 220
source-faithful reversible F32 runtime widenings and original procedural RoPE;
decoder 38 F32 / 36 F16 parameters with the original `use_fp16` contract.
Sampling retained seed 20261013, sigma minimum 1e-5, rescale_t 5, guidance 7.5,
rescale 0.7, interval `[0.6,1.0]`, dino-lock 0 and genuine SDPA. The initial
noise SHA-256 was `f76e94b25a4798a5da0e9de56667b5dda38e75bf12ccbc373521fbcd7af1a7b3`.

## Actual measurements

| Measurement | Sparse sampling | Occupancy decoder |
| --- | ---: | ---: |
| CPU model load | 13.515 s | 2.110 s |
| Synchronized neural computation | 9.969 s | 27.625 s |
| Supervised worker elapsed | 39.516 s | 40.484 s |
| Peak Torch allocated | 2,918,161,408 B | 1,127,093,760 B |
| Peak Torch reserved | 3,038,773,248 B | 1,648,361,472 B |
| Worker peak working set | 5,680,148,480 B | 2,007,670,784 B |
| Worker peak private commit | 6,803,759,104 B | 2,316,353,536 B |
| Available host RAM before strict load | 17,512,071,168 B | 18,148,216,832 B |
| Original required host RAM | 11,411,181,632 B | 4,294,967,296 B |
| Free VRAM before placement | 16,937,582,592 B | 16,937,582,592 B |
| Torch allocated / reserved at release | 0 / 0 B | 0 / 0 B |
| Healthy periodic Windows observations | 3 | 7 |

The original 13 GiB free-VRAM admission, 12 GiB allocator ceiling, per-operation
checks and BLAS workspace cleanup remained enforced. CPU/GPU/overall deadlines
remained 180/240/480 seconds for sampling and 120/180/360 seconds for decoding.
Both workers returned zero naturally, observers settled, models/tensors were
collected and owned Job active counts were zero before and after cleanup.
Neither forced cleanup nor timeout occurred.

These are the original standalone neural proof contracts, not the synthetic
production CPU supervisor contract. Their kill-on-close Jobs have no aggregate
or per-process RAM byte cap. Continuous whole-host available-RAM minima and
aggregate Job peak RAM were not measured. Sampled global GPU usage at release
was 288,096,256 / 273,022,976 bytes respectively; zero Torch bytes does not mean
zero driver/context usage. No additional post-exit GPU query was performed.

## Persisted private artifacts

All paths below are relative to this **private local directory**, which is
excluded locally from Git. Nothing in it is published:

`C:\Users\wattz\Documents\ChatGPT\modly\proof\artifacts\amd-zombie-occupancy-c570c2e`

| Relative file | Bytes | SHA-256 |
| --- | ---: | --- |
| `sampling\run\scratch\sparse-structure-latent.safetensors` | 134,688 | `ecf57d3a0db0bec99b79712fc3626c5dd224cff03e33c60e0297500cde761930` |
| `decoder\run\scratch\occupancy-logits.npy` | 1,048,704 | `46f1c152b3752dc13cdc7e02a0226c3935fe329b3559daa5484883476294b325` |
| `decoder\run\scratch\occupancy-mask.npy` | 262,272 | `bf9c0d82ae063b3bc213edc73b8ab959c08260b77c558abfa4eb4f8bf35b1025` |
| `decoder\run\scratch\occupancy-coordinates.npy` | 68,384 | `dcce90480523ea69af66515dfface9bf3a5e1417c6d3093ba56efbe06fa90b40` |
| `preview\voxel-preview.obj` | 237,741 | `de45549ddf3f3aa684c596942152ba0873ce1a15013e1e39ff6cf1bc2efc940b` |
| `preview\voxel-preview.png` (original projection) | 27,974 | `0190f319bc71ebdb859b59910635d2dce4adcee38bdc80a6ec756ffdc4f5b4de` |
| `preview\voxel-preview-isometric-z-up.png` | 27,778 | `0b8eb486d82174e9c95d6bd03e0b2b5813baa53fcfa19a412b27e6f2a9130ca8` |
| `preview\voxel-preview-plus-y.png` | 14,411 | `c4f848e712f30c6771f866726fd59ef52fedef5bc77657169ae43e61a79e23c3` |
| `preview\voxel-preview-plus-x.png` | 14,126 | `13e0c012ec26228b50e78f35a53091dec816c34735819fa6027d18204b5e1aea` |

The latent is finite F32 `[1,8,16,16,16]`. Independently reopened logits are
finite F32 `[1,1,64,64,64]`; the bool mask is exactly `logits[0,0] > 0` and I32
`[4266,4]` coordinates are exactly zero batch plus original ordered D/H/W indices.
Full receipts/provenance and both consumed exclusive markers remain intact.
Independent raw readback and sampler outcome review passed **before** the
decoder was launched. Decoder outputs were independently reopened after exit.

## Real voxel shape and limitations

The native 64³ mask contains **4,266 occupied voxels**, one six-connected
component and inclusive D/H/W spans **64 × 14 × 64**. Its bounding coordinates
are `[0,25,0]` to `[63,38,63]`. The exact exposed unit-voxel surface has 7,373
vertices and 7,370 outward quads. Independent verification compared original
surface ordering/winding and independently checked every occupied-owner / vacant
neighbour face. No threshold, support, scale or geometry improvement was applied.

The original bounded CPU writer ran first, retaining its 50,000-face, 12 MiB OBJ
and 2 MiB PNG limits. Added views use that same surface. Preview generation took
0.375 seconds within a 60-second total limit; its peak process RAM was not sampled.
All four 900×900 PNGs were independently decoded and hashed. Native axes are
OBJ x=D, y=H, z=W; the three added views explicitly label W as up and include
verified camera bases. The original legacy projection has no embedded camera
metadata and is preserved as such.

Actual inspection shows a coherent T-pose humanoid with head/neck, outstretched
arms, torso/hips, two separated legs and feet. Isometric and side views show
actual depth. The +Y projection is broad/front-like and +X is the side profile;
the source photograph's camera is not independently calibrated. Blocky limbs
and faceted contours are expected at this resolution. Facial, finger, clothing
and zombie-specific detail cannot be judged here. Neutral diagnostic shading is
not learned material.

There are 166 unique occupied voxels on volume boundaries: D−/D+ counts 8/8,
H−/H+ 0/0, W−/W+ 126/24. Hands and top/bottom extremities therefore have no
margin at some grid boundaries. This is a limitation/potential clipping risk;
contact alone does not prove that source geometry was clipped. There is no
claim of watertightness, game readiness or detailed reconstruction quality.

## Verification and preservation

- [Focused CPU tests](amd-zombie-occupancy-evidence/focused-tests.json) and
  [pre-execution independent review](amd-zombie-occupancy-evidence/pre-execution-review.json).
- Actual measurements in [measured results](amd-zombie-occupancy-evidence/measured-results.json).
- Independent actual [sampling review](amd-zombie-occupancy-evidence/sampling-outcome-review.json),
  [decoder review](amd-zombie-occupancy-evidence/decoder-outcome-review.json), and
  [saved-preview review](amd-zombie-occupancy-evidence/preview-review.json).
- [Preservation after](amd-zombie-occupancy-evidence/preservation-after.json) equals
  before: 100 protected identities, 78 bounded content hashes, interpreter,
  historical evidence/caches, original image, saved DINO features and receipts.
  Large accepted-master preservation uses exact metadata and previously verified
  full-hash provenance; its large payload and historical scratch cache payloads
  were not unnecessarily rehashed. These bounded limitations are explicit.
- [Remote preservation](amd-zombie-occupancy-evidence/publication-preservation.json)
  records comparison against the fresh 138-head baseline. Main and all baseline
  heads are preserved; only this new experimental branch is published.

Only code and source-safe textual evidence are committed. Original photographs,
features, latents, occupancy arrays, OBJ, PNGs and raw process identities stay
private. Installed environments and accepted historical assets were unchanged.

Stop reached. The next decision is visual review of these coarse previews;
shape-SLat and detailed neural geometry require a separate approval.
