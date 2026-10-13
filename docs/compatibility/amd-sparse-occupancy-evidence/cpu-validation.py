"""Bounded focused tests with AI imports blocked; no models or GPU context."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
if '--child' not in sys.argv:
    result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--child'],cwd=root,capture_output=True,timeout=45)
    for name,blob in (('cpu-tests.stdout.txt',result.stdout),('cpu-tests.stderr.txt',result.stderr)):
        with (evidence/name).open('xb') as f:f.write(blob)
    print(result.stdout.decode());print(result.stderr.decode());raise SystemExit(result.returncode)
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel','nvdiffrast'):
        raise RuntimeError('CPU tests prohibit AI/native GPU imports')
sys.addaudithook(audit);sys.path.insert(0,str(root/'scripts/compatibility'))
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(__import__(n)) for n in ('test_amd_sparse_occupancy','test_amd_decoder_transfer'))
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
files=['amd_sparse_occupancy.py','amd_sparse_occupancy_contract.py','amd_sparse_occupancy_data.py','amd_sparse_decoder_forward.py','amd_decoder_forward_contract.py']
row=dict(tests=result.testsRun,passed=result.wasSuccessful(),log=log.getvalue(),torch_imported='torch' in sys.modules,
    model_weights_loaded=False,gpu_initialized=False,production_sha256={name:hashlib.sha256((root/'scripts/compatibility'/name).read_bytes()).hexdigest() for name in files})
with (evidence/'cpu-tests.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps(row));raise SystemExit(0 if result.wasSuccessful() else 1)
