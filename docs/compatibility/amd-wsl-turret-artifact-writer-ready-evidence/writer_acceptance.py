"""Exactly one contained CPU-only acceptance of the real production writer."""
import argparse
import gc
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

SCRIPT=Path(__file__).resolve();E=SCRIPT.parent;ROOT=SCRIPT.parents[3]
SCRATCH=Path('/home/modly/writer-acceptance-4870856')
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
sys.path[:0]=[str(E),str(ROOT/'scripts/compatibility')]
import amd_wsl_rocdxg_smoke as base
from artifact_paths import leaf,check_host,check_guest
from artifact_io import write_output,HostReceipt

def qualify(r):
    base.validate_runtime_identity(r['runtime'])
    if (r['classification']!='QUALIFIED' or r['GPU_entry_calls'] or r['GPU_initialized']
        or r['forbidden_imports'] or not r['output']['verified'] or not r['output']['independently_verified_before_release']
        or r['policy']['rejected_calls'] or r['policy']['blocked'] or r['policy']['native_calls']
        or r['safetensors_version']!='0.6.2' or r['triton_version']!='3.5.1+rocm7.2.1.gita272dfa8'):
        raise RuntimeError('CPU writer acceptance incomplete')

def worker(output):
    output.mkdir(exist_ok=False);began=time.monotonic();torch=policy=None;patches=[]
    r={'classification':'UNKNOWN','GPU_entry_calls':[],'forbidden_imports':[],'neural_generation':False,'checkpoint_loading':False}
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','triton','trellis2','cumesh','o_voxel','nvdiffrast','spconv','torchsparse'}:
            r['forbidden_imports'].append(str(args[0]));raise RuntimeError('Writer acceptance forbids model/GPU-native import')
        if event=='open' and isinstance(args[0],(str,bytes)) and '/models/' in str(args[0]).replace('\\','/'):
            raise RuntimeError('Writer acceptance forbids checkpoint access')
    try:
        base.await_go(sys.stdin.buffer)
        r['artifact_directory']=check_guest('fixture')
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home','tmp'):(SCRATCH/name).mkdir()
        os.environ.update(HOME=str(SCRATCH/'home'),TMPDIR=str(SCRATCH/'tmp'),
            PYTORCH_TUNABLEOP_ENABLED='0',PYTORCH_TUNABLEOP_TUNING='0',PYTORCH_TUNABLEOP_RECORD_UNTUNED='0')
        sys.path.insert(0,str(STAGE))
        policy_path=ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py'
        r['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest()
        if r['policy_sha256']!='8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd':
            raise RuntimeError('Proven ImportPolicy identity changed')
        from amd_wsl_native_import_policy import ImportPolicy
        policy=ImportPolicy(SCRATCH,writable_roots=(output,leaf('fixture')))
        policy.install_audit_hook();sys.addaudithook(audit)
        import torch
        r['runtime']={'torch':str(torch.__version__),'distribution_version':md.version('torch'),
            'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
            'python':platform.python_version(),'prefix':sys.prefix,'gpu_initialized_before':torch.cuda.is_initialized()}
        base.validate_runtime_identity(r['runtime'])
        if torch.cuda.is_initialized():raise RuntimeError('GPU unexpectedly initialized at import')
        for owner,names in [(torch.cuda,('init','_lazy_init')),(torch._C,('_cuda_init','_cuda_getDeviceCount','_cuda_getDeviceProperties','_cuda_setDevice','_cuda_synchronize'))]:
            for name in names:
                original=getattr(owner,name,None)
                if original is not None:
                    def deny(*a,_name=name,**k):
                        r['GPU_entry_calls'].append(_name);raise RuntimeError('GPU entry forbidden: '+_name)
                    patches.append((owner,name,original));setattr(owner,name,deny)
        policy.set_phase('cpu_meta');policy.install_tensor_guards(torch);policy.install_profile()
        with policy.tensor_mode(torch):
            r['safetensors_version']=md.version('safetensors');r['triton_version']=md.version('triton')
            tensors={'features':torch.arange(14,dtype=torch.float32,device='cpu').reshape(2,7)/8,
                     'coordinates':torch.tensor([[0,0,0,0],[0,1,0,0]],dtype=torch.int32,device='cpu')}
            for i in range(4):
                tensors[f'subdivision_{i}_logits']=(torch.arange(16,dtype=torch.float32,device='cpu').reshape(2,8)/16).to(torch.bfloat16)
                tensors[f'subdivision_{i}_coordinates']=tensors['coordinates'].clone()
            provenance={'fixture_kind':'CPU_WRITER_ACCEPTANCE','neural_generation':False,'checkpoint_loading':False,
                        'runtime':r['runtime'],'tensor_count':10}
            summary={'fixture':True,'rows':2,'tensor_count':10,'payload_bytes':sum(t.numel()*t.element_size() for t in tensors.values())}
            r['output']=write_output(tensors,torch,provenance,summary,'fixture',output)
            tensors=None;gc.collect()
        r['GPU_initialized']=torch.cuda.is_initialized();r['policy']=policy.snapshot()
        r['classification']='QUALIFIED';qualify(r)
    except BaseException as exc:
        r.update(classification='UNKNOWN',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if policy is not None:
            r['policy']=policy.snapshot();policy.restore_profile();policy.restore_tensor_guards()
        for owner,name,original in reversed(patches):setattr(owner,name,original)
        if torch is not None:r['GPU_initialized']=torch.cuda.is_initialized()
        r['elapsed_seconds']=time.monotonic()-began
        base.save_json(output/'results.json',r)
    return 0 if r['classification']=='QUALIFIED' else 1

def host(output):
    if output.resolve()!=(E/'writer-run').resolve() or output.exists() or (E/'writer-acceptance.marker').exists():
        raise RuntimeError('Exclusive CPU writer attempt already reserved')
    if (E/'neural-attempt.marker').exists():raise RuntimeError('CPU acceptance must precede neural marker')
    directory=check_host('fixture');check_host('neural')
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        if Path(base.load_sibling(name).__file__).resolve()!=ROOT/'scripts/compatibility'/(name+'.py'):raise RuntimeError('Wrong helper')
    spec=importlib.util.spec_from_file_location('writer_runner',E.parent/'amd-wsl-packaging-decoder-loading-evidence/loading.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.SCRIPT=SCRIPT;runner.HOST_DEADLINE=120;runner.NATIVE_DEADLINE=60
    runner.worker_command=base.worker_command;runner.validate_qualified=qualify;runner.preflight=lambda:None
    receipt=HostReceipt(output,'fixture',base)
    def reserve(path):
        base.save_json(E/'writer-acceptance.marker',{'script_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            'CPU_only':True,'native_or_neural_authorized':False,'directory_preflight':directory,'started_utc':base.utc_now()})
        path.mkdir(exist_ok=False)
    runner.reserve_attempt=reserve
    def deadlines(now,began,go):
        receipt.poll()
        if now-began>=120 or (go is not None and now-go>=60):raise TimeoutError('CPU writer deadline')
    runner.check_deadlines=deadlines
    return runner.supervise(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker is not None:raise SystemExit(worker(a.worker))
    if a.output is None:p.error('--output required')
    raise SystemExit(host(a.output))
