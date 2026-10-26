# Pixal3D HIP proof: stopped at Windows admission

26 October 2026. Base `d9191fda03aa9e79da4da2ee42db1c7ca71c9906`. **NO-GO: the approved isolated proof was not admitted. Native compatibility remains untested.** No environment provisioning or native operator test followed the failure.

The original qualified `GlobalMemoryStatusEx` reader measured **18.031502 GiB available** at `2026-10-26T00:14:37.287760Z`, below the unchanged **22 GiB** requirement. The original `fresh_admission` function rejects a failing reading immediately, so only one reading was taken. It did not take another reading, wait for recovery or retry. Three passing readings four seconds apart remain required; **24 GiB remains preferred only**. The 12 GiB live-host floor and every other existing protection were left unchanged.

The admission process reported approximately 20.46 MiB working set and 12.70 MiB private commit. These are admission-process measurements, not native-test peaks. Output-volume free space was approximately 131.37 GiB and D: free space 68.36 GiB. The 2 GiB artifact floor passed; guest-disk and complete provisioning admission were not reached. No GPU allocation or native-test execution time was measured.

| Required check | Result |
| --- | --- |
| Windows RAM admission | **FAIL — below 22 GiB** |
| Windows isolation path unused | PASS — `D:\AI\Modly-Pixal3D-Proof-v1` absent |
| Guest path exclusivity, guest resource/disk checks | NOT RUN — no WSL launch |
| Exact runtime dependency availability/download | NOT RUN |
| Wheel contents / HIP extension loading | NOT RUN |
| Single visible gfx1201 GPU | NOT RUN |
| Actual HIP INT8/ConvRot dispatch and CPU-reference accuracy | NOT RUN |
| PyTorch attention / NAF | NOT RUN |
| Native sparse convolution / geometry conversion | NOT RUN |
| Actual GPU execution, allocation and release | NOT RUN |
| Native worker natural exit/cleanup | NOT RUN — no worker was started |
| TRELLIS protected-file and historical-asset preservation | PASS |

The user's additional published-wheel hash/build-workflow verification remains packaging evidence. No wheel was downloaded or locally inspected in this run, and no execution claim is inferred from that evidence. There was no package/model download, new environment, shared site-packages, existing package replacement, kernel build, fallback implementation, neural workload or generation-attempt reservation.

An initial private admission wrapper referenced a nonexistent hashing helper and exited before sampling. Its source and error are retained locally. The wrapper was corrected to use stdlib hashing; the sole subsequent live admission failed as reported above. This was not a native compatibility retry. The corrected admission logic is published as an **admission-only recorder** with a no-work default; even a future PASS cannot launch WSL or continue to setup/testing. Five mock-only regression tests passed, covering immediate rejection, later rejection, exact-22 admission/four-second wait requests and measurement errors. No real admission was repeated during verification.

Read-only closeout rehashed **3,197 historical private files** and verified all **100 protected identities / 88 retained records**, including the inherited protected hashes and identity metadata. Original tracked sources remain unchanged; the Windows isolated path is still absent. WSL was not launched to inspect guest paths or inventory the whole installation anew. No writes to the existing TRELLIS environments/installations were performed.

Only new code/tests and source-safe textual evidence belong on `codex/amd-pixal3d-hip-admission-stop`. Raw memory records, preparation error, source copies, test logs and preservation inventories remain under local `artifacts/amd-pixal3d-hip-proof-20261026-v1`.

Independent read-only review approved the four new files with no Critical, Important or Minor findings. It checked the raw admission arithmetic, source hashes, no-work/no-continuation paths, unchanged tracked files and all 171 recorded refs. It did not rerun live readings, tests or operators, or independently rehash the historical asset closure.

**Next decision: stop for review.** Preparing a generation attempt is NO-GO until the model-free proof actually passes. A future request to resume must start with fresh passing admission; there is no automatic retry or resource-threshold change. TRELLIS.2 remains the primary working engine.
