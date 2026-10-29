# Cold-backup worker identity correction

**Scope:** Preparation/offline tests only. Preserve the consumed v1 case and every safety gate.

The authenticated Windows venv executable redirects to a child process, so its
`Popen.pid` cannot satisfy the exact worker PID/parent check. Identity-only Job
probes reproduce the mismatch and show direct base Python satisfies the check.
The worker is stdlib-only with `-B -I -S`; launch that same authenticated base
binary directly. Keep the qualified venv for the controller. Do not permit
arbitrary descendants or remove PID, parent, token or Job validation.

1. Add failing native launcher/direct-base handshake regressions, negative
   identity/token cases and executable-tamper rejection.
2. Pin the base executable SHA-256, retain exact owner comparison, and use new
   exclusive v2 backup/evidence paths. Improve stderr identity diagnostics only;
   do not let unauthenticated workers write failure receipts.
3. Run the full scoped offline safety suite and inert dry-runs. Independently
   review, verify failed-case/asset/ref preservation, publish safe changes and
   return one conditional command. No real VHDX access, WSL query, backup or E:
   copy occurs during preparation; actual execution needs separate approval.
