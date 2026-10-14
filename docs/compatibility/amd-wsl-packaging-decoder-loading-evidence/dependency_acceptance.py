"""One installed-WSL CPU dependency import; reuse owned loading supervisor."""
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
import threading
import time
import traceback

SCRIPT=Path(__file__).resolve(); E=SCRIPT.parent; ROOT=SCRIPT.parents[3]
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
SCRATCH=Path('/home/modly/packaging-cpu-e24808e')
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
import amd_wsl_rocdxg_smoke as base

def qualify(record):
    if not {'torch.cuda._lazy_init','torch._C._cuda_init'}.issubset(record.get('gpu_entry_functions_instrumented',[])):
        raise RuntimeError('Genuine GPU-entry denial probe not installed')
    if (record.get('classification')!='QUALIFIED' or record.get('gpu_entry_calls')!=[]
        or record.get('gpu_initialized_before') is not False or record.get('gpu_initialized_after') is not False
        or record.get('packaging_version')!='25.0' or record.get('safetensors_version')!='0.6.2'
        or record.get('load_file_callable') is not True or record.get('checkpoint_loads')!=0
        or record.get('forbidden_imports')!=[] or record.get('policy',{}).get('rejected_calls')!=0
        or record.get('policy',{}).get('native_calls')!=0 or 'error' in record):
        raise RuntimeError('Incomplete CPU dependency acceptance')
    base.validate_runtime_identity(record['runtime'])

def worker(output):
    output.mkdir(exist_ok=False)
    record={'classification':'UNKNOWN','gpu_entry_calls':[],'forbidden_imports':[],
        'checkpoint_loads':0,'started_utc':base.utc_now()}
    torch=policy=None; patches=[]; began=time.monotonic()
    def denied(label):
        def callback(*args,**kwargs):
            record['gpu_entry_calls'].append(label)
            raise RuntimeError('CPU import reached GPU entry: '+label)
        return callback
    c_entries={'_cuda_init','_cuda_getDeviceCount','_cuda_getDevice','_cuda_setDevice','_cuda_getDeviceProperties','_cuda_synchronize'}
    def profile(frame,event,arg):
        if event=='call' and frame.f_globals.get('__name__')=='torch.cuda' and frame.f_code.co_name in {'_lazy_init','init','device_count','get_device_properties','current_device','set_device','synchronize'}:
            denied('torch.cuda.'+frame.f_code.co_name)()
        if event=='c_call' and getattr(arg,'__module__','')=='torch._C' and getattr(arg,'__name__','') in c_entries:
            denied('torch._C.'+arg.__name__)()
        if event=='call' and frame.f_globals.get('__name__')=='safetensors.torch' and frame.f_code.co_name=='load_file':
            record['checkpoint_loads']+=1;raise RuntimeError('Checkpoint loading forbidden in import acceptance')
        if policy is not None: policy.profile(frame,event,arg)
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','triton','trellis2','cumesh','o_voxel','nvdiffrast','spconv','torchsparse'}:
            record['forbidden_imports'].append(str(args[0]));raise RuntimeError('Unrelated native/model import prohibited')
        if event.startswith('socket.') or event.startswith('subprocess.') or event in ('os.system','os.posix_spawn'):
            raise RuntimeError('Network/process creation prohibited in CPU acceptance')
        if event=='open' and isinstance(args[0],(str,bytes)) and '/models/' in str(args[0]).replace('\\','/'):
            raise RuntimeError('No model file access in CPU acceptance')
    try:
        base.await_go(sys.stdin.buffer)
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home','tmp','triton-cache','torch-extensions'): (SCRATCH/name).mkdir()
        os.environ.update(HOME=str(SCRATCH/'home'),TMPDIR=str(SCRATCH/'tmp'),
            TRITON_CACHE_DIR=str(SCRATCH/'triton-cache'),TORCH_EXTENSIONS_DIR=str(SCRATCH/'torch-extensions'))
        sys.path.insert(0,str(STAGE))
        sys.addaudithook(audit);sys.setprofile(profile);threading.setprofile(profile)
        import torch
        record['gpu_initialized_before']=torch.cuda.is_initialized()
        if record['gpu_initialized_before']:raise RuntimeError('Torch initialized GPU during CPU import')
        if not callable(getattr(torch.cuda,'_lazy_init',None)) or not callable(getattr(torch._C,'_cuda_init',None)):
            raise RuntimeError('Cannot instrument genuine installed GPU-entry functions')
        for owner,names in ((torch.cuda,('_lazy_init','init','device_count','get_device_properties','current_device','set_device','synchronize')),
                            (torch._C,tuple(sorted(c_entries)))):
            for name in names:
                if hasattr(owner,name):
                    patches.append((owner,name,getattr(owner,name)));setattr(owner,name,denied(str(owner.__name__)+'.'+name))
        record['gpu_entry_functions_instrumented']=[owner.__name__+'.'+name for owner,name,_ in patches]
        from amd_wsl_native_import_policy import ImportPolicy
        policy=ImportPolicy(SCRATCH,writable_roots=(output,));policy.set_phase('cpu_meta')
        policy.install_tensor_guards(torch);policy.install_audit_hook()
        with policy.tensor_mode(torch):
            import packaging
            from packaging.version import Version
            import safetensors
            from safetensors.torch import load_file
            if Path(packaging.__file__).resolve()!=STAGE/'packaging/__init__.py' or Path(safetensors.__file__).resolve()!=STAGE/'safetensors/__init__.py':
                raise RuntimeError('Dependency resolved outside intended stage')
            record.update(packaging_version=packaging.__version__,packaging_path=packaging.__file__,
                safetensors_version=safetensors.__version__,safetensors_path=safetensors.__file__,
                load_file_callable=callable(load_file),load_file_module=load_file.__module__,
                Version_result=str(Version(str(torch.__version__))))
            record['runtime']={'torch':torch.__version__,'distribution_version':md.version('torch'),
                'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
                'python':platform.python_version(),'prefix':sys.prefix,
                'gpu_initialized_before':record['gpu_initialized_before']}
            base.validate_runtime_identity(record['runtime'])
            if md.version('triton')!='3.5.1+rocm7.2.1.gita272dfa8':raise RuntimeError('Triton identity changed')
            record['gpu_initialized_after']=torch.cuda.is_initialized()
            record['policy']=policy.snapshot()
            record['classification']='QUALIFIED';qualify(record)
    except BaseException as exc:
        record.update(classification='UNKNOWN',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        sys.setprofile(None);threading.setprofile(None)
        if policy is not None:policy.restore_tensor_guards()
        for owner,name,original in reversed(patches):setattr(owner,name,original)
        gc.collect();record.update(elapsed_seconds=time.monotonic()-began,finished_utc=base.utc_now())
        base.save_json(output/'results.json',record)
    return 0 if record['classification']=='QUALIFIED' else 1

def host(output):
    if output.resolve()!=(E/'dependency-run').resolve():raise RuntimeError('Exclusive CPU evidence path required')
    staging=json.loads((E/'packaging-staging.json').read_text())
    if staging['status']!='PASS' or staging['version']!='25.0':raise RuntimeError('Staging not qualified')
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):base.load_sibling(name)
    spec=importlib.util.spec_from_file_location('cpu_owned_runner',E/'loading.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    # The base helper anchor stays at scripts/compatibility; only guest selection changes.
    runner.SCRIPT=SCRIPT;runner.HOST_DEADLINE=120;runner.NATIVE_DEADLINE=60
    runner.worker_command=base.worker_command;runner.validate_qualified=qualify
    runner.preflight=lambda:None
    def reserve(path):
        base.save_json(E/'CPU-dependency-import.marker',{'started_utc':base.utc_now(),'distro':base.DISTRO,
            'script_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),'output':str(path),'GPU_authorized':False})
        path.mkdir(parents=True,exist_ok=False)
    runner.reserve_attempt=reserve
    def deadlines(now,began,go):
        if now-began>=120 or (go is not None and now-go>=60):raise TimeoutError('CPU import acceptance deadline')
    runner.check_deadlines=deadlines
    return runner.supervise(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker is not None:raise SystemExit(worker(args.worker))
    if args.output is None:parser.error('--output required')
    raise SystemExit(host(args.output))
