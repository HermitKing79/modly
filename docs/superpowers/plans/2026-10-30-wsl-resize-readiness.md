# Phase 6B preparation plan

Scope: no startup, shutdown, resize, restore or decoder. Publish a minimal
operator runbook and pure offline receipt/acceptance checks, not a new executor.

1. Pin and cross-check the completed v2 backup receipts, fixed three-way hash,
   natural Job cleanup and unchanged source/backup metadata. Revalidate Windows
   registration, installed WSL file versions, ACLs and host disk capacity only.
2. Inspect Microsoft's versioned native sequence and privilege path. Preserve
   existing numeric gates; expose the service-owned non-cancellable critical
   section as a separate storage-approval condition rather than promising
   unsafe cancellation or reusing a kill-on-close diagnostic controller.
3. Write failing pure receipt/acceptance tests, implement minimal predicates,
   and test corrupt receipts, incomplete filesystem growth and latched faults.
4. Independently review; preserve accepted backup/evidence, original box inputs
   and Git refs. Publish only new code/tests/text on the isolated branch. Stop
   with conditional native command reference and explicit execution HOLD.
