# Bounded final-convolution plan

User specification: pasted request continuing `a28e16ee569a168ac8c7df09bacd583eccff0ec7`.
Goal: preserve the genuine 1,842,968-child support and canonical 512 neural computation.

- [x] Verify starting GitHub identity; preserve historical files, branches and Windows environments.
- [x] Read pinned explicit-GEMM, original C2S and parent finalization source; independently audit equivalence and memory.
- [x] Write CPU tests first: exact global-index/slot/weight/bias equivalence, cross-tile neighbours, sentinels, short tail, BF16/F32; fixed-support admission and failure cases.
- [x] Implement one temporary static-method replacement activated only inside the final module call. Retain original global native map; tile only output rows, gather from full input and preserve mm/addmm and row order. Restore on all exits.
- [x] Add exact fourth-support admission exceptions only at both final C2S calls, final convolution and final output validation. Retain earlier row cap, reserves and deadlines. Test resource formulas, per-tile lifetime and rollback.
- [ ] Independently review all candidate changes. Preserve fresh read-only Linux inventory and live readiness; publish exact harness before execution.
- [ ] Only if equivalence/resource/preflight pass: one original parent forward, then complete finite output/readback and unchanged cleanup. No retry.
- [ ] Independently review actual result; verify preservation and publish. Stop before native geometry extraction.

Files: fresh copies of `decode.py`, `neural.py`, `admission.py`, `runtime.py`, `readiness.py`, `passive_linux.py`; new `chunked_final.py` and `cpu-tests.py`, all in this evidence directory. Installed sources and strict loader remain untouched. No dependency changes or broad tests.
