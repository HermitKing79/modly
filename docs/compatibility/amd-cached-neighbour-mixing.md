# AMD cached non-identity neighbour-mixing proof

**PASS â€” one genuine cached kernel launch; both SplitK planes and every reduced F32 value match independent references exactly. Synthetic convolution testing is finished.**

Starting branch/commit: `codex/amd-cached-kernel-convolution-proof`, `92cdb3fccdf6ccc60f3a241684aafe323c8c34fd`. Exact reviewed harness `6fd7345e0cb4e5b11c3aa2da1c9b1e5a8006e08d` was published and GitHub-ref verified before execution. This result covers the fixed signed two-axis, diagonal-channel stencil; native wrappers, learned subdivision and real geometry decoding remain unproved.

## Numerical contract and actual result

The original N2049/C1024 F16 features, unique int32 coordinates, zero bias, native metadata, kernel/source/cache/helper/tuning and launch arguments were retained. **Only synthetic weights changed:** diagonal channel taps centre (v13)=1, offset(0,0,+1) (v14)=1/2, offset(-1,0,0) (v4)=-1/4; every other entry 0. Stored coordinate order is [batch,axis1,axis2,axis3], avoiding ambiguous native W/H/D labels. Weight bytes SHA256 `cc6f8dd4e83d7e55ed0db313423eadde64216ba46b8fefd08b834099536ce9cb`.

| Check | Actual result |
|---|---|
| Full output | F32[2049,1024]; all 2,098,176 values exact; maximum error 0 |
| Raw SplitK output | F32[2,2049,1024]; all 4,196,352 values finite and exact per-plane |
| Noncentral coverage | 3,081,216 valid terms; 2,064,894 outputs differ from centre identity |
| Valid tap rows | Centre 2,049; axis3+1: 1,984; axis1-1: 1,025 |
| Missing neighbours | Boundary 0/64/1024; in-bounds sparse 0/1/0, including row 2048 |
| Native calls | Convolution/JIT/HSACO/launch each 1; cached helper 1; neighbour/post1/post2/outer/inner all 0; forbidden[] |
| Host call plus synchronized completion | 0.031s; owned-worker supervision 9.891s |
| Peak PyTorch allocated/reserved | 77,848,576/98,566,144bytes |
| Release/exit | Allocated 0, reserved 0; owned tensors collected; exit 0;unforced verified owned Job exit |
| Windows observation | Healthy before, CPU gate, one periodic sample and after exit; controller OK; selected events [] |

The CPU full reference uses original coordinate tuples, actual tap coefficients and integer feature units U=F16_feature*1024. It computes (4Ucentre+2Uaxis3plus-Uaxis1minus)/4096, missing terms 0. Actual numerator range [-6119,6085]; the conservative absolute bound 7,168 and power-of-two scaling make F32 products/partial sums exact. It never uses GPU output or precomputed neighbour arrays. A separate source-derived SplitK routing reference also matches each raw plane, rejecting equal-and-opposite errors hidden by reduction. References and final output remain F32; no F16 rounding or tolerance relaxation was used(atol=rtol=0).

13 focused host regressions (7 new, 6 original identity route) and 13 genuine installed constructor tests pass. The owned CPU acceptance exited 0 in 13.125s, GPU/native/AI-weight execution false, with 11 scratch artifacts verified. An independent standard-library NPY reader then recomputed every full/partial result and coordinate without production-oracle or ML/GPU imports. All40 reviewed dependency hashes match the executed manifest.

## Safety and saved evidence

The 17 existing native/cache safeguards, 13 GiB free-VRAM admission, 12 GiB allocator ceiling, 1 GiB observed current/peak smoke budget, 45/120/180-second CPU/GPU/overall limits, Windows monitoring and Job containment were unchanged. One kernel launch, no warmup/retry, no full wrapper/native neighbour/postprocessing or real model loads occurred.

Available VRAM was 16,937,582,592bytes before inputs. Observed availability loss at kernel was 109,838,336 bytes; after release 13,369,344 bytes. Both stayed below 1 GiB. Zero ownership refers to PyTorch allocations/reservations and collected tensors; the remaining global difference has no proven owner. These measurements cannot enumerate every native/driver allocation.

Artifacts were exclusively saved/read back inside owned scratch, then promoted byte-identically only after verified clean exit. Local directory: `C:\Users\wattz\Documents\ChatGPT\modly\proof\docs\compatibility\amd-cached-neighbour-mixing-evidence\artifacts`.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| [raw_splits.npy](amd-cached-neighbour-mixing-evidence/artifacts/raw_splits.npy) | 16,785,536 | `e93033470aa6d67cb1f23c0f56eb4b7a7846b368decfb50ce215bd4846bbb8be` |
| [input_features.npy](amd-cached-neighbour-mixing-evidence/artifacts/input_features.npy) | 4,196,480 | `05ea4d32e669734812d24510e18bdf81abf99f87964cc5b540c1f06e5fed9c90` |
| [coordinates.npy](amd-cached-neighbour-mixing-evidence/artifacts/coordinates.npy) | 32,912 | `635d2189e1dc178e14bf454932ac3dfb3f5c5a7a0e4bfdeb88f24e8901ce7cc7` |
| [cpu_reference.npy](amd-cached-neighbour-mixing-evidence/artifacts/cpu_reference.npy) | 8,392,832 | `6c97d66b798f7ba34719739c742b09314cbc60a01ae21b1dadcbf435d1e35a01` |
| [cpu_split_reference.npy](amd-cached-neighbour-mixing-evidence/artifacts/cpu_split_reference.npy) | 16,785,536 | `e93033470aa6d67cb1f23c0f56eb4b7a7846b368decfb50ce215bd4846bbb8be` |
| [output_features.npy](amd-cached-neighbour-mixing-evidence/artifacts/output_features.npy) | 8,392,832 | `6c97d66b798f7ba34719739c742b09314cbc60a01ae21b1dadcbf435d1e35a01` |

[Run](amd-cached-neighbour-mixing-evidence/run/results.json), [trace](amd-cached-neighbour-mixing-evidence/run/trace.jsonl), [CPU numerical/cache audit](amd-cached-neighbour-mixing-evidence/cpu-numerical-audit.json), [independent artifact validation](amd-cached-neighbour-mixing-evidence/independent-artifact-validation.json) and [outcome review](amd-cached-neighbour-mixing-evidence/outcome-review.json) preserve the complete evidence.

## Preservation and route decision

[Postcheck](amd-cached-neighbour-mixing-evidence/git-after.json) verifies all 5,951 historical evidence files, all prior local/remote branches (including main and the successful starting proof), AMD source HEAD/status/modifications/backup, model/cache metadata and native identities. Both installed inventories are byte-identical before/after, SHA256 `0be055c97659c345754515298c56b8a707d7303a7e30159af00e8a142b9cf475`. The approved small numerical hook in the repository proof worker is explicitly recorded; its original identity mode remains tested. Installed Modly/ComfyUI and production mock routes are untouched.

The separate [Linux comparison](amd-linux-route-comparison.md) finds a stronger author-reported route to a textured 512Â³ asset on the same GPU, using explicit GEMM and a BF16 decoder. It also exposes raw nonfinite geometry/export processing and safety/import-order qualifications. This does not establish local Linux readiness or a drop-in Windows replacement.

**Recommendation:** stop synthetic tests. Separately approve one offline neural-only decode of the saved turret latent using the Linux reference's declared explicit-GEMM/BF16 path, conditional on identifying an existing compatible runtime and proving strict loading, memory/support admission, containment and finite-output acceptance. If no runtime exists, seek a separate setup decision instead of rebuilding automatically. Continued Windows decoding still needs the real wrapper, dynamic specialization and subdivision/resource contracts; this kernel PASS does not remove those gaps. Extraction, textures and GLB quality require subsequent approvals.
