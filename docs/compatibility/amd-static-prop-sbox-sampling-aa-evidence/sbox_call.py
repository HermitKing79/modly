"""Log one supported call against the existing matching editor; never launches one."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent
cli = r"C:\Program Files (x86)\Steam\steamapps\common\sbox\bin\managed\sbox-cli.exe"
project = Path(r"C:\Users\wattz\Documents\S&box Projects\greyfield\greyfield.sbproj")
label, tool = sys.argv[1:3]
assert re.fullmatch(r"[a-zA-Z0-9_-]+", label)
arguments = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
out = base / "mcp" / label
out.mkdir(parents=True, exist_ok=False)
instances = json.loads(subprocess.check_output([cli, "--list"], text=True, timeout=15))
matches = [x for x in instances if x["kind"] == "editor" and Path(x["project"]) == project]
assert len(matches) == 1, "Greyfield editor must already be running and uniquely discoverable"
command = [cli, "--connect", str(matches[0]["pid"]), "--request-timeout", "90", "--call", tool,
           json.dumps(arguments, separators=(",", ":"))]
with (out / "request.json").open("x", encoding="utf-8") as stream:
    json.dump(dict(instance=matches[0], tool=tool, arguments=arguments), stream, indent=2)
started = time.monotonic()
result = subprocess.run(command, cwd=project.parent, capture_output=True, text=True,
                        timeout=100, stdin=subprocess.DEVNULL)
for name, text in (("stdout.txt", result.stdout), ("stderr.txt", result.stderr)):
    with (out / name).open("x", encoding="utf-8") as stream:
        stream.write(text)
with (out / "result.json").open("x", encoding="utf-8") as stream:
    json.dump(dict(exit_code=result.returncode, elapsed_seconds=time.monotonic() - started), stream)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
raise SystemExit(result.returncode)
