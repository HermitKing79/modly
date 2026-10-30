# Attended native resize handoff

Implement the accepted b909cb1 runbook and explicitly accepted non-cancellable
storage policy. Preparation only; never run the live path in this session.

1. Test fault latching/natural settlement, exact command/geometry and inert gates.
2. Reuse qualified cold-backup file locks, identities, metadata, inventory,
   receipt validation, GlobalMemoryStatusEx and box input authentication. Hash
   source and accepted E: backup freshly; keep E: shared-read locked throughout.
3. Admit three fresh >=22 GiB readings four seconds apart and final22 before
   mutation. Native call exactly once, no Job/kill/terminate/cancel path. Durable
   logs and RAM/disk observer precede launch. Every fault latches FAIL while
   observing natural settlement, including deadline and interruption faults.
4. After fault-free natural success, verify container64 and one read-only Linux
   query for exact device geometry, ext4 rw and normal-user free>=5GiB; native
   resize2fs block count proves filesystem64. No neural/decoder follow-up.
5. Independently review, run mocked and tiny CPU fixtures/dry-runs, verify
   preservation, publish safe code/text and exact held command. New exclusive
   case only; no retry, alternative path, environment change or actual resize.
