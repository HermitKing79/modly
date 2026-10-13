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
import amd_sparse_sampling_contract as cpu_policy
def forbidden(*args,**kwargs):raise RuntimeError('CPU correction prohibits GPU/model/compile operation')
def audit(event,args):
    if event=='import' and (str(args[0]).split('.')[0]=='triton' or cpu_policy.prohibited_import(str(args[0]))):
        raise RuntimeError('No model/native pipeline imports in CPU correction')
    if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.pt','.pth','.bin','.safetensors','.ckpt','.gguf'):
        if not Path(os.fsdecode(args[0])).resolve().is_relative_to(scratch.resolve()):raise RuntimeError('CPU preparation prohibits real checkpoint access')
sys.addaudithook(audit)
import torch
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
original_module_call=torch.nn.Module.__call__
def fixture_call(instance,*args,**kwargs):
    if type(instance).__module__!='test_amd_sparse_sampling_runtime':return forbidden()
    return original_module_call(instance,*args,**kwargs)
torch.nn.Module.__call__=fixture_call
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
    module=__import__('test_amd_sparse_sampling_runtime')
    names=('test_amd_sparse_sampling_runtime','test_amd_sparse_flow_transfer_guard','test_amd_sparse_latent_checkpoint')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(__import__(n)) for n in names)
    log=io.StringIO()
    with torch.inference_mode():
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    record.update(tests=result.testsRun,passed=result.wasSuccessful(),log=log.getvalue(),
        compact_trace=module.ROWS,production_guard_sha256=hashlib.sha256(Path(module.forward.__file__).read_bytes()).hexdigest())
record['gpu_initialized_after']=torch.cuda.is_initialized()
assert record['gpu_initialized_after'] is False
record.update(weights_loaded=False,model_forwards=0,network_allowed=False,
              gpu_backend_execution=False,prohibited_imports=[n for n in sys.modules if n.split('.')[0]=='triton' or cpu_policy.prohibited_import(n)])
with (evidence/(sys.argv[1]+'.json')).open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
raise SystemExit(0 if result is None or result.wasSuccessful() else 1)
