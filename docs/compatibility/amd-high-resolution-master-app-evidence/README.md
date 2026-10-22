# Saved installed-app acceptance evidence

These are diagnostic controls and observations for one installed Modly 0.4.3 session, based on source checkout `64d07f5c5a10a48a1cbff028cb6143d2a444ae89`. They do not constitute a new production importer or proof that the installed application is an exact build of that checkout.

`execution-summary.json` is derived from the saved private receipts by `summarize_session.py`. It deliberately reports overall BLOCKED: five actual app acceptance areas passed, while camera interaction and in-session model resource disposal were not established.

The consumed `renderer_observer.js` has a known cold-renderer startup dependency: it waits for `Page.enable` before the first navigation creates the renderer. Four mocked tests passed but missed that real lifecycle condition. The file is preserved exactly as executed, **not recommended for unattended reuse**. A separately reviewed single `about:blank` navigation allowed the pending registration and original app navigation to complete in the same bounded session. See `startup-observer-continuation-review.json`. This was diagnostic intervention, not a product fix. No second full-master import occurred.

The restricted-context launch failed before any asset import. Its receipt and exact source copies remain private and unchanged. The subsequent trusted desktop-context session has its own exclusive marker, profile, outputs and receipts.

Private model files, complete session logs and screenshots remain outside Git. Public JSON contains identities, scalar observations and limitations only. `master-orbit.jpg` is a misleading historical filename: its pixels show the export Save As dialog, and it provides **no camera/orbit evidence**.
