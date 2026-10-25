# Zombie telemetry repair and standalone handoff implementation plan

> **For agentic workers:** Use superpowers:executing-plans, with one independent final review.

**Goal:** Reconcile the completed CPU run and deliver one guarded, independent neural decode handoff without executing it.

**Architecture:** Bind external diagnostic telemetry to the interpreter's startup identity, validated against the owned Windows Job. Keep the venv launcher separate. A new, default-dry-run handoff obtains fresh admission and privately routes the unchanged zombie decoder and supervisor to a single unused destination.

**Tech Stack:** Windows PowerShell 5.1, CPython 3.12 stdlib, existing qualified decoder.

**Spec:** User request of 2026-10-25 in this conversation.

## Global constraints

- Three fresh 22 GiB readings, four seconds apart; 24 GiB preferred only.
- 22 GiB immediately before reservation; original 12 GiB live floor, disk/VRAM, deadlines and owned cleanup unchanged.
- No real preparation repeat, WSL launch, marker reservation, model load or inference during implementation.
- Preserve all previous evidence, protected identities, environments and branch tips; publish only source-safe material.

## Review focus

- A venv redirector PID must not be mistaken for the interpreter.
- An identity outside the owned Job or with a mismatched image must fail closed.
- Missing identity and measurement errors must not produce successful telemetry.
- A stale or low admission must not reach reservation.
- The single destination must not permit replay; original decoder safety policies must stay intact.

## Tasks

- [x] Snapshot protected state, refs and both real diagnostic trees; inspect recorded telemetry and source.
- [x] Add a failing actual-Windows allocation regression, then minimally repair interpreter binding and verify native memory agreement.
- [x] Add mocked handoff tests before implementation: fresh admission, final gate, exclusive output, routing and no-work dry run.
- [x] Implement standalone handoff, retaining original supervisor and decoder; no execution in this session.
- [x] Run focused tests, dry runs, preservation checks and independent review. Publish the isolated branch and exact commit.

Ruling: use a new branch in the existing checkout, retaining the established absolute-path environment and every old branch. The user requested branch isolation; duplicating this path-bound checkout would not improve the intended handoff.

Ruling: execute the explicit user specification inline without another design approval round. This task already authorises the minimal fix and standalone handoff.

Final review: no newly introduced Critical/Important/Minor findings. The inherited diagnostic start/attach race is documented without changing the original launch/cleanup mechanism. All focused checks passed; full project tests were not run because that would exceed the permitted CPU-only test scope and could import model modules.
