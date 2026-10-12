"""Capture only the affected CPU contract and synthetic-loader test results."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
rows = []
for name in ('test_amd_decoder_transfer', 'test_amd_decoder_forward', 'test_amd_decoder_contract', 'test_amd_strict_loader'):
    command = ['D:/AI/Trellis2AMD/venv/Scripts/python.exe', '-B', '-I', str(root/'scripts/compatibility'/f'{name}.py')]
    result = subprocess.run(command, capture_output=True, timeout=20)
    rows.append({'command': command, 'exit': result.returncode, 'timeout': False,
                 'stdout': result.stdout.decode(), 'stderr': result.stderr.decode()})
    print(name, result.returncode, result.stderr.decode().strip())
with (Path(__file__).parent/sys.argv[1]).open('x', encoding='utf-8') as stream:
    json.dump(rows, stream, indent=2)
raise SystemExit(0 if all(row['exit']==0 for row in rows) else 1)
