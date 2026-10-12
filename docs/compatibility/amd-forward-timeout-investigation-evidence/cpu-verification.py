"""Capture bounded CPU-only diagnostics and affected contract tests."""
import json
from pathlib import Path
import subprocess
import sys

p=Path(__file__).parent;root=p.parents[2];rows=[]
for name in ('test_amd_forward_diagnostics','test_amd_decoder_forward','test_amd_decoder_transfer','test_amd_decoder_contract'):
    command=[sys.executable,'-B','-I',str(root/'scripts/compatibility'/f'{name}.py')]
    result=subprocess.run(command,capture_output=True,timeout=15)
    rows.append({'command':command,'exit':result.returncode,'timeout':False,'stdout':result.stdout.decode(),'stderr':result.stderr.decode()})
    print(name,result.returncode,result.stderr.decode().strip())
with (p/sys.argv[1]).open('x',encoding='utf-8') as stream:json.dump(rows,stream,indent=2)
raise SystemExit(0 if all(row['exit']==0 for row in rows) else 1)
