# Kitchen runtime dependency overlay

Implement the user's narrowly authorised correction; stop before live execution.

1. Authenticate Kitchen 0.2.40 and the existing official packaging 25.0 wheel. Audit every Kitchen/packaging Python source import and inspect the indirect Torch Dynamo path. Record mandatory versus guarded optional requirements and limits of static analysis.
2. Test first: reproduce the missing packaging import with no fake modules; test a genuine pure-Python wheel overlay, PEP 440 parsing, provenance, tampering, extra files and isolation. Use fresh private test scratch.
3. Reuse the unmodified wheel directly through Python ZIP import, after full archive/payload authentication. Add no site-packages, .pth hooks, extraction, pip installs or dependency resolver. Hold the original wheel read-only; verify loaded packaging origins and version. Keep original V2 closure, GPU checks, RAM/VRAM/disk/deadline/cleanup policies.
4. Prepare new exclusive Windows/guest/evidence paths, dry-run, run focused offline guard and closure tests, independently review and publish only code/tests/text on the isolated branch. Preserve all original evidence, assets, environments and heads. No WSL or live admission during preparation.

Review focus: shadowing existing packages, archive mutability, child ZIP import under -I, indirect optional dependencies, undeclared requirements and honest limits of offline proof. Full Kitchen/native import success remains pending the guarded Linux run.
