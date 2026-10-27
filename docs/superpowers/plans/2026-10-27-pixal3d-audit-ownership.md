# V3 audit denial and process ownership

User scope: saved evidence plus offline tests only. No WSL, live admission, GPU, package/environment changes or operator retry.

1. Preserve and reconstruct the 27-file failed case. Separate worker mutation denial, later controller ownership-read failure and subsequent empty-group cleanup. Do not infer an unlogged event/path or terminal state.
2. Instrument filesystem denials using an exclusive preopened private evidence descriptor: event, arguments, timestamp, phase, process identity and caller frames. Keep all original mutation prohibitions; test real harmless audit events on new fixture files and forbidden paths.
3. Add a V3-only proc reader using pinned directory descriptors and before/after PGID/start identity. Permission errors require a same-descriptor terminal/disappearance check plus whole-process pidfd exit proof; active, ambiguous or reused identities stop. Use verified per-process pidfds for cleanup; no numeric process-group signalling or unsupported fallback.
4. Preserve controller failures while harvesting separate bounded worker diagnostics. Test disappearance, permission failures, PID replacement, nonce/leader identity, pidfd binding and refusal-to-signal cases.
5. Keep execution on hold because the mutation trigger is unproven. Independently review, verify historical preservation and publish code/tests/safe text. No new operator command is qualified by this investigation.
