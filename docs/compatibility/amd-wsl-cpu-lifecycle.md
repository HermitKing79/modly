# Native WSL CPU lifecycle qualification v1

**PASS — the single nonrecursive target/sentinel pair settled naturally in Linux
and Windows.** No production source changed.

Base: `8f175de809f3be304cfc92179cb42905a01b8f12`.
Branch: `codex/amd-wsl-cpu-lifecycle`.

| Boundary | Result |
|---|---|
| Real WSL broker readiness | PASS |
| Linux process/session ownership | PASS |
| Target cooperative cancellation | PASS |
| Sentinel independence and natural release | PASS |
| Linux reaping and empty settlement | PASS |
| Windows containment and cleanup | PASS |
| Complete CPU WSL control pair | PASS, one pair only |
| Windows UpstreamJob authority | Previously PASS; unchanged |
| WSL UpstreamJob production authority | NOT QUALIFIED |
| Neural GPU execution | NOT RUN |
| Fresh image-to-3D | NOT RUN |

## Qualified execution

The actual `OwnedStageRunner.run()` WSL path used the original Windows bootstrap,
named `Modly-AMD-Ubuntu-24.04` distribution, explicit `-u modly`, original
`/usr/bin/python3` and unchanged Linux broker. The registered default UID remains
0; it was not used or changed. Both fixtures measured UID/EUID 1000, user `modly`,
canonical `/usr/bin/python3.12` and inherited 6 GiB soft/hard `RLIMIT_AS`.

Both stages established separate host contexts/nonces, Linux leaders/sessions and
private Windows Jobs. Readiness required the trusted broker's boot ID, parent,
PID/start-tick/session/group identities, matching fixture runtime facts, and a
fresh native Windows observation of the exact original `System32/wsl.exe` image
with verified creation identity and Job membership. The fixtures spawned no
children; each broker's only child was its fixture leader.

After both readiness checks, the host sent CANCEL only to the target. The target
published and reopened its exact cooperative acknowledgement, exited 0 and was
reaped by the broker. Linux owned-process accounting and its second postscan were
empty; its Windows client/bootstrap also exited 0 and the private Job closed empty.

Only after target settlement did the host publish an unpredictable sentinel
challenge. The still-running sentinel answered with the original Linux identity,
then accepted a separate exact release and exited 0. Its broker reaped the leader
and reported empty settlement. Sentinel controls contained no CANCEL. This combines
a fresh pinned-fixture response with trusted broker lifecycle evidence; the
response is not a new trusted live-broker query or production authority.

Each saved protocol contains GO observation, broker readiness, broker terminal and
final successful Linux-client exit. Target terminal is `cancelled`; sentinel
terminal is `succeeded`. Both are natural and unforced, with no diagnostic errors.
Both stage Jobs and the outer Job ended empty without forced cleanup.

## Measured limits and tests

| Measurement | Actual value |
|---|---:|
| Host / coordinator / outside wrapper intervals | 19.657 / 16.547 / 21.794 s |
| Target / sentinel runner receipt intervals | 14.313 / 16.438 s |
| Target / sentinel broker intervals | 0.179 / 1.742 s |
| Outer peak Job / process memory | 77,615,104 / 58,028,032 bytes |
| Target / sentinel stage peak Job memory | 20,881,408 / 21,139,456 bytes |
| Target / sentinel Linux sampled RSS peaks | 28,069,888 / 27,811,840 bytes |
| Minimum Windows available RAM | 16,528,228,352 bytes (15.393 GiB) |
| Minimum destination free disk | 135,603,793,920 bytes (126.291 GiB) |
| Health readings, including initial admission | 1,131 |

The 25-second stage and 90-second outer deadlines, 6 GiB Windows process/Job
limits, 12 GiB host-RAM floor and 2 GiB disk floor were unchanged. Linux address
space is limited per process; aggregate owned RSS is sampled, **not an atomic
cgroup limit**. Windows Job memory does not cap Linux guest memory. Broker and
Windows monotonic clocks were not compared. Receipt intervals are their saved
measurement scopes, not estimates of model execution time.

All **48 focused CPU tests passed**, and the same 48 were independently rerun once:
22 existing runner/broker/bootstrap checks, 13 fixture protocol checks and 13
harness admission/settlement checks. Mock tests launched no WSL worker. Their
fixture hardlink and Linux runtime boundaries were substituted explicitly;
actual publication and Linux runtime evidence come from the single native pair.
Review findings about exact stage identity and original WSL executable/registration
checks were addressed before admission. No source changed after execution.

## Evidence and preservation

- [Frozen admission and source identities](amd-wsl-cpu-lifecycle-evidence/admission.json)
- [CPU tests](amd-wsl-cpu-lifecycle-evidence/cpu-tests.json)
- [Independent preexecution review](amd-wsl-cpu-lifecycle-evidence/preexecution-review.json)
- [Measured execution and private-evidence hashes](amd-wsl-cpu-lifecycle-evidence/execution-summary.json)
- [Independent outcome review](amd-wsl-cpu-lifecycle-evidence/outcome-review.json)
- [Preservation after execution](amd-wsl-cpu-lifecycle-evidence/preservation-after.json)

Raw Linux/Windows identities, controls, acknowledgements, challenges and traces
remain private under `artifacts/amd-wsl-cpu-lifecycle-8f175de/cpu-once` in the outer
Modly workspace. Published evidence contains source-safe text and hashes only.

Preservation covers all 100 protected identities, the same 78 bounded payload
hashes, interpreter, accepted-master metadata/prior full-hash provenance, historical
evidence and caches. Large model/master payloads were not rehashed. Original
registration and `wsl.exe` identities were checked before launch and finally.
Main and all 132 preceding remote heads are retained. The read-only preservation
checker's no-native flag describes that checker, not the separate qualification.

This PASS covers only this pinned, non-adversarial, nonrecursive CPU pair. Unbound
WSL file protocol and fixture controls retain the existing trusted-user
assumptions. WSL UpstreamJob authority, broader descendant behavior and neural
execution are not qualified by this result. No distro termination, retry, GPU
work, production Generate enablement or follow-on experiment occurred.
