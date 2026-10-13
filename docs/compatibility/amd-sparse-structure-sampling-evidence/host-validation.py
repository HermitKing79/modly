"""Focused host regressions, bounded subprocess, with AI framework imports blocked."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
label=sys.argv[2] if '--child' in sys.argv else (sys.argv[1] if len(sys.argv)>1 else 'focused-host-final')
if '--child' not in sys.argv:
    result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--child',label],cwd=root,
        capture_output=True,timeout=45)
    with (evidence/(label+'.stdout.txt')).open('xb') as f:f.write(result.stdout)
    with (evidence/(label+'.stderr.txt')).open('xb') as f:f.write(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());raise SystemExit(result.returncode)

def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Host tests prohibit AI/native imports')
sys.addaudithook(audit)
sys.path.insert(0,str(root/'scripts/compatibility'))
names=('test_amd_sparse_sampling_contract','test_amd_sparse_sampling_supervision',
       'test_amd_sparse_flow','test_amd_sparse_flow_resume','test_amd_sparse_flow_control')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(__import__(n)) for n in names)
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
record=dict(tests=result.testsRun,passed=result.wasSuccessful(),log=log.getvalue(),
    torch_imported='torch' in sys.modules,gpu_initialized=False,real_weights_loaded=False,
    production_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'scripts/compatibility').glob('amd_sparse*.py'))})
with (evidence/(label+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='production_sha256'}))
raise SystemExit(0 if result.wasSuccessful() else 1)
