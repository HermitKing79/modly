# Minimal native Windows supervisor: PASS

Base `df30cb32e87ab528d9c9961ea0953ec4798f6529`; branch
`codex/amd-minimal-windows-supervisor`.

Exactly one nonrecursive synthetic target/sentinel pair ran through the unchanged
production `OwnedStageRunner` and bootstrap. The target acknowledged its exact
cancellation and settled naturally. The independently owned sentinel survived,
received its separate authenticated success release, and settled naturally. Both
stage Jobs and the outer containment Job ended with zero active processes and no
forced cleanup. There was no retry.

| Qualification | Verdict |
| --- | --- |
| Windows production runner ownership | PASS for this executed CPU pair |
| Target cancellation and acknowledgement | PASS |
| Sentinel independence | PASS |
| Sentinel natural successful exit | PASS |
| Complete nonrecursive Windows CPU pair | PASS |
| Ability to supply authenticated host observations | PASS for the recorded Windows identities, membership, controls and settlement; no production authority issued |
| UpstreamJob production integration | NOT YET QUALIFIED |
| WSL ownership | NOT RUN |
| Genuine GPU inference | NOT RUN |

## Actual evidence

The target protocol records GO, CANCEL, child exit0, cooperative acknowledgement
accepted, then natural cancelled terminal. The sentinel records GO and natural
successful exit0 with `cancellation_sent=false`; it received no CANCEL. Both
bootstrap return codes are0. Stage/bootstrap stderr files are empty.

Readiness was admitted only after the actual runner had authenticated the fixture
PID, creation FILETIME, image, current Job membership and enumeration. Distinct
contexts, nonces and native private Jobs were required. After target settlement,
two sequential sentinel observations established a native-query ordering fence;
the later observation retains the same PID/creation/Job/image identity. Only then
did the host publish the exact sentinel release. Raw result dictionaries equal
their original production receipts. They have not been converted to
`HostStageObservation` or used to complete a product job.

| Measurement | Actual result |
| --- | ---: |
| Host measured execution interval | 0.735s |
| Shell-measured host process interval, including startup | 0.899s |
| Target stage / sentinel stage | 0.328s / 0.422s |
| Both authenticated ready after pair launch | 0.250s |
| Target natural settlement after cancellation | 0.078s |
| Outer peak Job / peak process memory | 65,417,216 / 17,428,480 bytes |
| Target / sentinel peak Job memory | 23,699,456 / 23,633,920 bytes |
| Minimum available Windows RAM,36 health records | 18,230,636,544 bytes (16.9786GiB) |
| Minimum destination free disk | 136,117,997,568 bytes |
| Final active processes: target / sentinel / outer | 0 / 0 / 0 |
| Native attempt evidence directory size | 569,629 bytes |

Monotonic measurements have the runtime's clock granularity; displayed decimals
are not sub-millisecond timing claims. The outer Job counted15 processes:
10 `python.exe` and5 `conhost.exe` identities. Each stage counted6. These include
the interpreter/console processes, not15 fixtures. The fixture source starts no
processes. Parent PIDs were not recorded, so this report does not reconstruct an
unobserved parent graph. Empty Job accounting proves process settlement; no
separate zero-byte allocator-release measurement is claimed.

## Scope, review and preservation

The supported `require_escalated` execution request ran the authenticated API
interpreter under the same qualified Windows user with Administrator role
disabled. No RunAs or permission changes occurred. This is supported-tool and
same-shell token evidence, not an independent child sandbox/token detector.

All original6GiB process AND Job caps,12GiB RAM and2GiB disk floors,25-second
stages and90-second outer bound remained intact. The fixture has no recursive
descendants or separate five-second readiness deadline. The production observer
and accepted retired-process identity handling are unchanged. No production
source files were modified.

The [independent source review](amd-minimal-windows-supervisor-evidence/preexecution-review.json)
binds10 source files and the private expected-user identity hash. All17 focused
mocked tests passed independently. Filesystem publication is mocked in those
tests after the known sandbox `os.link` denial; the real native run uses actual
exclusive hardlink publication. A premature static preflight failed because the
review receipt had not yet been saved, before any attempt reservation or native
launch. Its exact traceback is preserved. Once the receipt existed, static
preflight passed without source changes. Neither preparatory event is hidden or
counted as an additional native pair.

The [independent outcome review](amd-minimal-windows-supervisor-evidence/independent-outcome-review.json)
examines saved protocol, identities, acknowledgements, source pins, timing and
resources and provides raw-evidence hashes. The
[numerical summary](amd-minimal-windows-supervisor-evidence/resultsummary.json)
binds that review. Raw identities, nonces and traces remain private at
`artifacts/amd-minimal-windows-supervisor-df30cb3/cpu-once` outside the proof tree.

Before/after preservation states match:100 protected metadata/file identities,
the same78 small hashes, API interpreter, accepted-master metadata and prior
full-hash provenance, caches, historical evidence and129 pre-existing remote
branch heads including main. Large model/assets were not redundantly bulk
rehashed. The previously independently verified device-width representation is
used only for the exact recorded volume values; original receipts remain intact.
The preservation receipts' no-execution flag describes their read-only scan,
not this task's separately recorded native test.

The older recursive-fixture failures remain BLOCKED. This result neither explains
those failures completely nor qualifies WSL, neural cancellation or AMD Generate.
The next separately scoped step is to bind these production Windows observations
to host-owned UpstreamJob invocation authority, with immutable context/source and
deadline/cancellation validation. No such adapter or further execution was begun.
