"""Bounded entry point for schema inspection / actual-dispatch CPU regressions."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

root=Path(__file__).resolve().parents[3]
evidence=Path(__file__).parent
scratch=evidence/'scratch'/sys.argv[1]
scratch.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_bounded_gpu_proof as proof
os.environ.update(proof.scratch_environment(scratch),WORKSPACE_DIR=str(root))
for key in proof.CACHE_KEYS:
    if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'api'))
from services.amd_offline_guard import block_external_actions
block_external_actions(scratch_dir=scratch)
def forbidden(*args,**kwargs):raise RuntimeError('CPU correction prohibits GPU/model/compile operation')
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('triton','trellis2','safetensors'):
        raise RuntimeError('No model/native pipeline imports in CPU correction')
    if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.pt','.pth','.bin','.safetensors','.ckpt','.gguf'):
        raise RuntimeError('CPU correction prohibits checkpoint access')
sys.addaudithook(audit)
import torch
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
torch.nn.Module.__call__=forbidden
torch.set_num_threads(1)
record={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
    'torch_path':torch.__file__,'hip':torch.version.hip,'cuda':torch.version.cuda,
    'git_version':torch.version.git_version,'gpu_initialized_before':False}
assert sys.version.split()[0]=='3.12.10' and str(torch.__version__)=='2.13.0+rocm10.0.0'
assert torch.version.hip=='7.15.26333' and torch.version.cuda is None
schema=torch.ops.aten.to.dtype_layout._schema
record['schema']=str(schema)
record['arguments']=[{'name':a.name,'type':str(a.type),'keyword_only':a.kwarg_only,
                     'has_default':a.has_default_value(),'default':str(a.default_value)} for a in schema.arguments]
result=None
if sys.argv[1]!='schema':
    module=__import__('test_amd_sparse_flow_transfer_guard')
    log=io.StringIO()
    with torch.inference_mode():
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record.update(tests=result.testsRun,passed=result.wasSuccessful(),log=log.getvalue(),
        intercepted_cpu_copies=module.COPIES,production_guard_sha256=hashlib.sha256(Path(module.forward.__file__).read_bytes()).hexdigest())
record['gpu_initialized_after']=torch.cuda.is_initialized()
assert record['gpu_initialized_after'] is False
record.update(weights_loaded=False,model_forwards=0,network_allowed=False,
              gpu_backend_execution=False,prohibited_imports=[n for n in sys.modules if n.split('.')[0] in ('triton','trellis2','safetensors')])
with (evidence/(sys.argv[1]+'.json')).open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
raise SystemExit(0 if result is None or result.wasSuccessful() else 1)
