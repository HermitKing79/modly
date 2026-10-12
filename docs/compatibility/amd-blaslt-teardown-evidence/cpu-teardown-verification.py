"""Bounded CPU-only execution of actual teardown regression fixtures."""
import json
from pathlib import Path
import subprocess
import sys
p=Path(__file__).parent;root=p.parents[2]
code='''
import importlib.abc,io,json,sys,unittest
class NoAI(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','triton','trellis2','transformers','safetensors'):raise RuntimeError('CPU-only teardown tests forbid AI import')
sys.meta_path.insert(0,NoAI());sys.path.insert(0,sys.argv[1])
module=__import__('test_amd_blaslt_teardown');log=io.StringIO()
result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),'log':log.getvalue(),
 'ai_imports':[k for k in sys.modules if k.split('.')[0] in ('torch','triton','trellis2','transformers','safetensors')]}))
sys.exit(0 if result.wasSuccessful() else 1)
'''
run=subprocess.run([sys.executable,'-B','-I','-c',code,str(root/'scripts/compatibility')],capture_output=True,timeout=30)
record=dict(exit=run.returncode,timeout=False,stdout=run.stdout.decode(),stderr=run.stderr.decode())
if run.stdout:record.update(json.loads(run.stdout.decode().splitlines()[-1]))
with (p/sys.argv[1]).open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record));raise SystemExit(run.returncode)
