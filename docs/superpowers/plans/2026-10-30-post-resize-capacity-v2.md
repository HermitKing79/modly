# Fresh Phase 6C v2 handoff

Use the accepted 28ba485 implementation, with no WSL/live capacity execution.

1. Preserve all ten v1 files by hash and identity; establish the exact failed
   startup gate and guest progress. Do not rewrite or complete v1 evidence.
2. Test a fixed exclusive v2 path and refusal of any consumed path before live
   admission/dispatch. Change only the path and necessary provenance pin.
3. The v1 guest did start. Current Windows-only exclusive read-only inspection
   shows the backing file's timestamp advanced; every other identity field and
   64 GiB container table matches. Preserve the old authority; pin this exact
   new snapshot in a separately authenticated v2 authority and require exact
   equality at future execution. Never ignore timestamps or refresh on failure.
   No whole guest filesystem content-preservation claim follows from headers.
4. Keep original RAM/disk/process/ownership/backup/query/geometry rules, probe,
   entrypoint and no-retry behaviour unchanged. Test invalid provenance chains.
5. Run offline suites/dry-runs, independent review and preservation checks.
   Publish held command for one separately approved run after a Windows restart,
   with Codex/unnecessary applications closed and no automatic continuation.
