"""Bounded focused CPU tests. Block AI imports, retain actual lifecycle records."""
import json
from pathlib import Path
import subprocess
import sys

p=Path(__file__).parent;root=p.parents[2]
names=('test_amd_sparse_flow','test_amd_sparse_flow_control','test_amd_dinov3_control','test_amd_diagnostic_payloads','test_amd_forward_diagnostics','test_amd_instrumented_forward','test_amd_blaslt_teardown')
code='''
import importlib.abc,io,json,sys,unittest
class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','triton','trellis2','safetensors'):
            raise RuntimeError('CPU-only verification prohibits AI import: '+fullname)
sys.meta_path.insert(0,NoAI())
sys.path.insert(0,sys.argv[1])
module=__import__(sys.argv[2]);log=io.StringIO()
result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),'log':log.getvalue(),
    'ai_imports':[name for name in sys.modules if name.split('.')[0] in ('torch','triton','trellis2','safetensors')],
    'replay':getattr(module,'REPLAY_ROWS',[]),'replay_shutdown':getattr(module,'REPLAY_SHUTDOWN',{})}))
sys.exit(0 if result.wasSuccessful() else 1)
'''
rows=[]
for name in names:
    command=[sys.executable,'-B','-I','-c',code,str(root/'scripts/compatibility'),name]
    run=subprocess.run(command,capture_output=True,timeout=15)
    row={'module':name,'exit':run.returncode,'timeout':False,'stdout':run.stdout.decode(),'stderr':run.stderr.decode()}
    if run.stdout:
        lines=run.stdout.decode().splitlines();data=json.loads(lines[-1]);row.update(data)
        row['diagnostic_stdout']='\n'.join(lines[:-1]);row.pop('stdout')
    rows.append(row)
    print(name,run.returncode,'tests',row.get('tests'),row.get('log',row['stderr']) if run.returncode else 'OK')
with (p/sys.argv[1]).open('x',encoding='utf-8') as stream:json.dump(rows,stream,indent=2)
raise SystemExit(0 if all(row['exit']==0 for row in rows) else 1)
