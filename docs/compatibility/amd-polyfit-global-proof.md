# Phase 12 — Official PolyFit release qualification

30 October 2026. Base `eaa84c66f00c2d0eecd1b99ad7d7b14e55e3b85d`.

**NO-GO for installing or executing the official PolyFit 1.6.0 Windows CPython 3.12 wheel in this proof.** Its actual native solver binary identifies an older, academically licensed SoPlex dependency. An applicable grant has not been established for this project. This triggers the requested licence qualification gate before the official example; it is not a solver or geometry-quality failure. Reconstruction quality remains **INCONCLUSIVE / NOT TESTED**.

## Actual release and binary evidence

The [official release](https://github.com/LiangliangNan/PolyFit/releases/tag/v1.6.0) identifies pinned source commit `c7d669cde09521d0099d868d4aee6a4ecd3b9fe3`. The exact official HTTPS asset downloaded was `polyfit-1.6.0-cp312-cp312-win_amd64.whl`, **6,980,207 bytes**, local SHA-256 **`b16d456c39785886832f0d419a4d6e1f876a26c657d44c5cd1be02ba66e4424f`**. All 15 ZIP members and embedded RECORD hashes/sizes verified; expanded payload is 16,679,804 bytes. No package was installed or imported.

The release API provides no whole-wheel publisher digest, signature or build attestation. The audit's signature/digest availability field refers to this archive, not the signing status of every bundled PE file. Official HTTPS origin, pinned source Git-blob verification and a locally pinned whole-archive SHA-256 establish the recorded provenance/integrity scope; they do not prove binary/source build equivalence. Downloaded archives and third-party sample/source files remain private.

The wheel contains `PyPolyFit.pyd`, `basic.dll`, `math.dll`, `method.dll`, `model.dll`, `gmp-10.dll` and `python312.dll`. Its metadata declares no Python runtime dependencies. The filename targets CPython 3.12/Windows x64, while the internal WHEEL incorrectly advertises `py3-none-any` and purelib. The available base interpreter is Windows x64 CPython 3.12.10; actual DLL loading/runtime compatibility was **not tested**. No existing Python environment was changed.

Read-only inspection of the actual `polyfit/math.dll` (8,302,592 bytes; SHA-256 `22f240b2d1f43fefaaff272e6099cb47a7b891b717714374d30878cf23a77004`) finds **`Soplex 3.1.1`** and a build-source path ending in `code\3rd_party\soplex\src\clufactor.cpp`. Pinned source corroborates version 311, its academic licence header and static library build:

- [SoPlex version/licence header](https://github.com/LiangliangNan/PolyFit/blob/c7d669cde09521d0099d868d4aee6a4ecd3b9fe3/code/3rd_party/soplex/src/spxdefines.h)
- [SoPlex build](https://github.com/LiangliangNan/PolyFit/blob/c7d669cde09521d0099d868d4aee6a4ecd3b9fe3/code/3rd_party/soplex/CMakeLists.txt)

[SoPlex's official licence page](https://soplex.zib.de/#license) explicitly leaves versions through 6.0.2 under the ZIB Academic Licence; Apache starts at 6.0.3. The [published ZIB terms, section 1](https://www.scipopt.org/doc-5.0.1/html/LICENSE.php) limit applicability to members of noncommercial academic institutions. No qualifying institutional status or separate permission is evidenced here. A free download, modern SCIP's Apache licence, or PolyFit's top-level GPL declaration does not establish a different grant for this old dependency. No paid licence, contact with maintainers or purchase occurred.

## Runtime and integration findings

The pinned [official step-by-step example](https://github.com/LiangliangNan/PolyFit/blob/c7d669cde09521d0099d868d4aee6a4ecd3b9fe3/code/python/Examples/Example_2_reconstruction_with_intermediate_steps.py), unchanged original `toy_data.bvg` and their Git-blob identities were obtained, but **not executed**. The bindings expose SCIP and default weights 0.43/0.27/0.30, without solver-timeout, memory, gap or termination-status controls. Any future cleared execution needs external owned-process bounds, including hypothesis generation and optimisation.

Source review also prevents two false success claims: face selection can log solver failure while leaving a nonempty candidate mesh; and projected support/alpha-shape coverage contributes a soft objective rather than forbidding unsupported completion faces. A returned/exported OBJ cannot replace solver evidence, finite-footprint validation or the frozen independent geometry gates.

PolyFit's [GPLv3-or-later declaration](https://github.com/LiangliangNan/PolyFit/blob/c7d669cde09521d0099d868d4aee6a4ecd3b9fe3/ReadMe.md#license) permits private running under GPL; conveying covered binaries requires the applicable notices and corresponding source. Output geometry is not automatically GPL merely because the program produced it. These GPL observations do not clear separate dependency terms. The wheel lacks a complete third-party licence manifest; Modly distribution/integration is not qualified.

## Measured result and preservation

| Stage | Result |
| --- | --- |
| Original ten tensors, 3,888-row provenance, raw PLY/NPY and comparison hashes | PASS reauthentication |
| Frozen Phase 11 criteria | PASS; SHA `397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0` unchanged |
| Official wheel origin, local SHA and embedded RECORD | PASS within the stated provenance scope |
| Dependency licence qualification | **NO-GO** |
| Disposable environment / installation | NOT CREATED / NOT RUN |
| Original official example / SCIP optimisation | NOT RUN |
| AI segmentation, candidate generation and optimisation | NOT RUN |
| Candidate topology, intersections, fidelity, bevels and eight views | NOT RUN; no candidate |

Authentication took **7.218 seconds / 102,146,048 peak private-commit bytes**; release byte inspection took **1.031 seconds / 40,042,496 bytes**. Both bounded CPU Jobs exited naturally and empty under unchanged 6 GiB process, 180-second stage, 12 GiB live-host and 2 GiB output-disk limits. These are qualification resources, not PolyFit solver measurements. Neural 22 GiB requirements remain untouched.

An initial authentication helper failed before source imports because isolated Python omitted its script directory; that receipt is preserved. The corrected import path ran in a new exclusive stage subtree. No native solver attempt or automatic retry occurred. Six offline tests cover archive/RECORD tampering, unsafe/colliding paths, actual-version marker reporting, fail-closed licence decisions and inert/owned-GO entry behaviour. Independent review checked the source, actual wheel evidence, tests and stop decision.

Private evidence: workspace `artifacts/amd-polyfit-global-proof-20261030-v1/`. Original AI hashes, protected file records and historical refs were rechecked; no geometry variant was created, and no original mesh, neural data, installed environment, WSL disk or E: backup was modified. No GPU, neural or WSL execution occurred.

**Single next action:** obtain an authenticated official build with a documented dependency licence set applicable to Modly—preferably using modern Apache-licensed SoPlex—or explicit applicable permission for this exact stack. Do not silently replace its static solver or initiate a custom port. A cleared build could support a separately accepted original-example-then-one-AI proof under unchanged Phase 11 gates; none is launched automatically. No further box-fitting campaign is recommended, and no integration or organic-asset quality claim is justified.
