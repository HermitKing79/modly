# Pixal3D V3: missing Kitchen runtime dependency

**Conditional GO for one new model-free operator proof with a read-only ZIP overlay.** Linux Kitchen/HIP success remains unproven. No WSL launch, live admission, GPU work, installation or download occurred during preparation.

## Confirmed cause and required dependency

The latest `feec959` proof passed complete V2 installed authentication (25,345 immutable payload files), RAM/startup and RX 9070 XT `gfx1201` detection. Kitchen import then failed with `ModuleNotFoundError: No module named 'packaging'`; the requested HIP numerical operators were NOT_RUN. Its saved result reports natural cleanup and unchanged V2 environment snapshots.

The exact cached Kitchen 0.2.40 wheel, SHA-256 `eb726664c772189e6ece742876487f1349965808b854742790e9001dbb7c0f9d`, contains this chain: CUDA backend import -> eager `quantization.py:27` -> `scaled_mm_v2.py:4`, `from packaging import version`. The next lines parse Torch's version. Kitchen METADATA declares only optional cublas/dev/build extras; both Torch (already qualified) and packaging are undeclared runtime requirements.

**Add only `packaging==25.0`**, wheel `packaging-25.0-py3-none-any.whl`, **66,469 bytes**, SHA-256:

`29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484`

The existing local wheel in `docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/` matches the official [PyPI 25.0 file checksum](https://pypi.org/project/packaging/25.0/#files). It requires Python >=3.8, declares no dependencies, and is a pure `py3-none-any` wheel with 23 payload files / 238,344 expanded bytes. No newer release or resolver is necessary for this narrowly tested version API.

## Offline closure evidence

All **50 Kitchen + 16 packaging Python sources** were statically inspected. Required external roots are Torch and packaging. Kitchen's Triton, Ascend `torch_npu` and NVIDIA imports belong to guarded optional backend routes; packaging's `_manylinux` platform hook is caught and is outside the version-check path. No additional proven mandatory package was found.

Independent inspection also followed Kitchen's `tensor/base.py:13` import of `torch._dynamo`. The qualified Torch sources gate/fall back for absent optree, tqdm, Triton and Z3. No required requests, psutil or YAML dependency was identified on that reviewed path. This static assessment cannot prove all future dynamic/native imports; a later failure must stop, not trigger speculative dependency installation.

The offline suite completed **106 cases: 105 passed and one optional cached-wheel case skipped**. The explicit 13-wheel audit separately passed all 18 cases, so **106 distinct tests passed overall**. Nine new cases exercise the real packaging ZIP, every packaging Python module, requirements/specifiers, Kitchen's exact PEP 440 expression with the literal qualified Torch version, missing/tampered archives, preloaded/foreign origins, unchanged policies and an actual Windows deny-write handle on a new fixture copy. No fake Torch/Kitchen modules are used. Whole Kitchen import under Linux remains pending.

Original 13-wheel payload authentication, path/collision/tampering tests and all retained admission/ownership/cleanup regressions pass. The harness blocks child execution, network calls and live RAM readers. Source and test results were independently reviewed. Private raw evidence, meshes and images are not published.

## Minimal overlay and fresh handoff

No installation, extraction, `.pth` hook, new site-packages or V2 edit is needed. The host extends its existing deny-write/delete file lease to the original packaging wheel. The controller authenticates the complete wheel and its payload before GO; the worker reauthenticates it, inserts that exact ZIP into its own `sys.path` after GO and before Torch/Kitchen imports, and verifies actual module origins and version. Preloaded packaging is rejected. Loaded origins are checked at exit and the controller reauthenticates the wheel. V2's 13-wheel installed inventory and critical-file checks remain unchanged.

The standalone entry remains `scripts/compatibility/Invoke-Pixal3DOperatorsV3.ps1`; the final handover supplies its exact reviewed commit and command. Run once from normal non-administrator PowerShell after closing Codex/ChatGPT Desktop. Both the immutable original dependency cache and the historical packaging wheel must remain present. Nothing will be installed or downloaded. New exclusive paths:

- Windows ownership root: `D:/AI/Modly-Pixal3D-Operators-v3-kitchen-overlay`.
- Guest scratch: `/home/modly/pixal3d-operator-proof-v3-kitchen-overlay`.
- Evidence: `artifacts/amd-pixal3d-operators-v3-kitchen-overlay-<UTC>-<nonce>`.

Windows exclusivity was checked offline; guest exclusivity is checked by the controller before work. Occupied paths or any failure stop without retry. The previous attempt and every original environment/artifact remain unchanged; preservation checks cover 3,346 historical hashes, 100 protected identities, 88 retained records and the 26 files of the latest failed proof.

The existing operator-only policy remains three fresh readings four seconds apart plus final prelaunch >=20.198032 GiB, startup/authentication/final GO >=18 GiB, live host floor 12 GiB. Neural admission remains 22 GiB and 24 GiB remains preferred only. All GPU/VRAM/allocator, disk/VHD, process-memory, CPU/GPU/overall deadlines and owned cleanup protections remain unchanged. No imports or numerical checks are disabled by this fix; no eager/CPU execution may satisfy a required HIP witness.

Inspect the fresh `REPORT.txt`, `summary.json`, `dependency-overlay.json`, environment authentication, per-operator results and cleanup receipts after exit. Stop for review regardless of result. No neural inference, full Pixal3D generation or automatic retry is authorised by a model-free PASS.
