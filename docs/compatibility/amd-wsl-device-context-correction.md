# WSL DeviceContext guard correction — UNKNOWN

Starting GitHub commit: `4e32bea4baab709d90d42217f1ac8ab83c70a557`. Branch: `codex/amd-wsl-device-context-correction`.

**The surgical policy correction passes 16 host unit tests, but the sole host acceptance launch failed before creating a WSL worker. Genuine installed-runtime acceptance is NOT RUN. No native import, GPU initialization/computation, real weights, neural forward or installation occurred.** The mandated stop was honored; no launcher repair or retry followed.

## Correction and inspected source

The installed Torch 2.9.1 sources were read without importing Torch and preserved byte-for-byte in [installed-source-before.json](amd-wsl-device-context-correction-evidence/installed-source-before.json). `_device.py`, `overrides.py` and the relevant `get_default_device`/`set_default_device` code establish that DeviceContext injects the implicit device through the active TorchFunctionMode stack. The indexless path in `get_default_device()` creates a tensor, so the guard never uses it.

The existing policy now captures the genuine `_get_current_function_mode_stack` and DeviceContext/device classes, preserves the original cached constructor identities, and inspects the active stack before calling an original factory. It rejects foreign modes, subclasses and malformed active device metadata. For genuine cached constructors with no explicit destination, the last DeviceContext supplies the implicit device; explicit CPU/meta destinations override it. Nonconstructor factories retain their input-device semantics. An unresolvable stack fails closed.

The directly related Tensor.to(other_tensor) parsing now checks Tensor before its `.type` method, preserving the genuine other tensor's CPU/meta destination. Existing tensor-dispatch, compiler/build/autotune/native-call guards and restoration remain in place. Only the existing policy and its focused unit tests changed outside documentation/evidence.

A new unit regression first failed because active CUDA was not rejected before the original callback. After correction, **16 tests passed** (0.460 seconds), covering pre-callback denial, explicit overrides, malformed/foreign/subclass modes, constructor identities, restoration and existing safety regressions. [Focused unit record](amd-wsl-device-context-correction-evidence/focused-unit-result.json). These host tests use test metadata, not installed Linux Torch execution.

## Sole acceptance launch and exact blocker

The independently reviewed [acceptance script](amd-wsl-device-context-correction-evidence/acceptance.py) proposed explicit/implicit CPU/meta, explicit/implicit CUDA, nested contexts, set_default_device, tensor transfers, callback observation and restoration checks. GPU-entry probes would record any entry as a failure; an exception from a substitute `_lazy_init` would never count as successful frontend protection.

The host command exited **1** after **0.5196 seconds** with:

`FileNotFoundError: .../amd-wsl-device-context-correction-evidence/amd_bounded_gpu_proof.py`

The adapter patched `smoke.SCRIPT` to select the new guest script in the evidence directory. The shared `load_sibling()` also derives helper paths from SCRIPT, redirecting lookup away from `scripts/compatibility`. `supervise()` reserves its attempt marker and run directory before that lookup, which occurs before HealthMonitor construction, WSL Popen or Job attachment. Consequently **zero installed-runtime route checks ran**, no GO was sent, and containment/monitoring acceptance was not exercised.

The [host capture](amd-wsl-device-context-correction-evidence/host-launch-capture.json), transcribed [returned traceback](amd-wsl-device-context-correction-evidence/host-launch.stderr.txt), original marker and local empty run directory are preserved. No CPU-verdict or guest result exists. The empty directory itself is not representable in Git; its state is recorded in [historical preservation](amd-wsl-device-context-correction-evidence/historical-preservation-after.json). [Independent outcome review](amd-wsl-device-context-correction-evidence/outcome-independent-review.json) corroborates the failure ordering. The prior static GO does not qualify the runtime result.

No demonstrated DeviceContext bypass remains in the inspected correction, but installed-runtime protection is **unverified**. This is a launch-harness blocker, not an AMD compatibility result or a passing real-runtime test.

## Preservation and next action

[Windows preservation](amd-wsl-device-context-correction-evidence/Windows-preservation.json) found zero changes in both Windows environments, 12 native binaries, 169/13/7 source groups, AMD HEAD/edits/backups, four turret artifacts and decoder metadata/header. No checkpoint payload was loaded or re-hashed. [Linux preservation](amd-wsl-device-context-correction-evidence/linux-preservation-after.json) confirms unchanged baseline venv metadata, package versions, dpkg inventory and the inspected Torch source. The baseline remains Torch 2.9.1 / Triton 3.5.1 / ROCm 7.2.1; no packages were installed. Only the named Modly Ubuntu distro was terminated after read-only preservation; Docker was untouched. Previous evidence and existing untracked bytecode are preserved.

**One next action, requiring separate approval:** correct only the acceptance adapter's helper lookup so it stays anchored to the original compatibility-script directory while selecting the guest script independently. Then independently review and run one fresh owned CPU acceptance with a new marker, preserving this failed launch. No further DeviceContext policy change is currently justified by installed-runtime evidence. Stop if that acceptance fails. Native import, GPU work and genuine model loading remain outside this milestone.
