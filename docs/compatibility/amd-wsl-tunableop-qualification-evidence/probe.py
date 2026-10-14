"""One CPU-only installed-Torch TunableOp qualification; no decoder changes."""
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

SCRIPT=Path(__file__).resolve();E=SCRIPT.parent;ROOT=SCRIPT.parents[3]
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
SCRATCH=Path('/home/modly/tunableop-cpu-bd8825b6')
SOURCE_SHA='b30632498091a445433ef2d9254c9a1b44ab3e329c6656f0beeae7bd65104a39'
ENV_KEYS=('PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')
GETTERS=(('enabled','is_enabled'),('tuning','tuning_is_enabled'),('record_untuned','record_untuned_is_enabled'))
CONTROLS=('enable','tuning_enable','record_untuned_enable')
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
import amd_wsl_rocdxg_smoke as base

def tunable_environment(environ):
    keys=set(ENV_KEYS)|{key for key in environ if key.startswith('PYTORCH_TUNABLEOP_')}
    return {key:{'set':key in environ,'value':environ.get(key)} for key in sorted(keys)}

def read_flags(tunable):
    flags={}
    for label,name in GETTERS:
        value=getattr(tunable,name)()
        if type(value) is not bool:raise RuntimeError('Expected actual boolean from '+name)
        flags[label]=value
    return flags

def qualify(record):
    if not {'torch.cuda._lazy_init','torch._C._cuda_init'}.issubset(record.get('gpu_entry_functions_instrumented',[])):
        raise RuntimeError('Genuine GPU-entry denial probe not installed')
    if (record.get('classification')!='QUALIFIED' or record.get('gpu_entry_calls')!=[]
        or record.get('gpu_initialized_before') is not False or record.get('gpu_initialized_after') is not False
        or record.get('controls_called')!=list(CONTROLS)
        or record.get('flags_after')!={'enabled':False,'tuning':False,'record_untuned':False}
        or record.get('environment_before')!=record.get('environment_after')
        or record.get('forbidden_imports')!=[] or record.get('restoration_verified') is not True
        or record.get('policy',{}).get('rejected_calls')!=0
        or record.get('policy',{}).get('native_calls')!=0
        or record.get('policy',{}).get('tensor_operations_allowed')!=0
        or 'error' in record):raise RuntimeError('Incomplete CPU TunableOp qualification')
    if set(record.get('flags_before',{}))!={label for label,_ in GETTERS} or any(type(v) is not bool for v in record['flags_before'].values()):
        raise RuntimeError('All initial flags must be independently recorded')
    base.validate_runtime_identity(record['runtime'])

def worker(output):
    output.mkdir(exist_ok=False)
    record={'classification':'UNKNOWN','gpu_entry_calls':[],'forbidden_imports':[],
            'controls_called':[],'started_utc':base.utc_now()}
    torch=policy=None;patches=[];began=time.monotonic()
    def denied(label):
        def callback(*args,**kwargs):
            record['gpu_entry_calls'].append(label)
            raise RuntimeError('CPU qualification reached GPU entry: '+label)
        return callback
    c_entries={'_cuda_init','_cuda_getDeviceCount','_cuda_getDevice','_cuda_setDevice','_cuda_getDeviceProperties','_cuda_synchronize'}
    def profile(frame,event,arg):
        if event=='call' and frame.f_globals.get('__name__')=='torch.cuda' and frame.f_code.co_name in {'_lazy_init','init','device_count','get_device_properties','current_device','set_device','synchronize'}:
            denied('torch.cuda.'+frame.f_code.co_name)()
        if event=='c_call' and getattr(arg,'__module__','')=='torch._C' and getattr(arg,'__name__','') in c_entries:
            denied('torch._C.'+arg.__name__)()
        if policy is not None:policy.profile(frame,event,arg)
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','triton','trellis2','cumesh','o_voxel','nvdiffrast','spconv','torchsparse'}:
            record['forbidden_imports'].append(str(args[0]));raise RuntimeError('Native/model import prohibited')
        if event.startswith('socket.') or event.startswith('subprocess.') or event in ('os.system','os.posix_spawn'):
            raise RuntimeError('Network/process creation prohibited in CPU qualification')
        if event=='open' and isinstance(args[0],(str,bytes)) and '/models/' in str(args[0]).replace('\\','/'):
            raise RuntimeError('No model file access in CPU qualification')
    try:
        base.await_go(sys.stdin.buffer)
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home','tmp','triton-cache','torch-extensions'):(SCRATCH/name).mkdir()
        # TunableOp environment is intentionally inherited and never changed.
        record['environment_before']=tunable_environment(os.environ)
        os.environ.update(HOME=str(SCRATCH/'home'),TMPDIR=str(SCRATCH/'tmp'),
            TRITON_CACHE_DIR=str(SCRATCH/'triton-cache'),TORCH_EXTENSIONS_DIR=str(SCRATCH/'torch-extensions'))
        sys.path.insert(0,str(STAGE))
        from amd_wsl_native_import_policy import ImportPolicy
        policy=ImportPolicy(SCRATCH,writable_roots=(output,));policy.set_phase('native_import')
        policy.install_audit_hook()
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
                    patches.append((owner,name,getattr(owner,name)));setattr(owner,name,denied(owner.__name__+'.'+name))
        record['gpu_entry_functions_instrumented']=[owner.__name__+'.'+name for owner,name,_ in patches]
        policy.install_tensor_guards(torch)
        with policy.tensor_mode(torch):
            from torch.cuda import tunable
            source=Path(tunable.__file__).resolve()
            if source!=Path('/opt/modly-wsl/venv/lib/python3.12/site-packages/torch/cuda/tunable.py') or hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
                raise RuntimeError('Installed TunableOp source identity changed')
            record['source']={'path':str(source),'sha256':SOURCE_SHA}
            record['runtime']={'torch':str(torch.__version__),'distribution_version':md.version('torch'),
                'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
                'python':platform.python_version(),'prefix':sys.prefix,
                'gpu_initialized_before':record['gpu_initialized_before'],'triton_distribution':md.version('triton')}
            base.validate_runtime_identity(record['runtime'])
            if record['runtime']['python']!='3.12.3' or record['runtime']['triton_distribution']!='3.5.1+rocm7.2.1.gita272dfa8':
                raise RuntimeError('Pinned Python/Triton identity changed')
            record['flags_before']=read_flags(tunable)
            base.journal(output,'individual-flags-before',flags=record['flags_before'],environment=record['environment_before'])
            for name in CONTROLS:
                getattr(tunable,name)(False)
                record['controls_called'].append(name)
            record['flags_after']=read_flags(tunable)
            record['environment_after']=tunable_environment(os.environ)
            record['gpu_initialized_after']=torch.cuda.is_initialized()
            record['policy']=policy.snapshot()
            base.journal(output,'individual-flags-after',flags=record['flags_after'])
            if record['flags_after']!={'enabled':False,'tuning':False,'record_untuned':False}:
                raise RuntimeError('Genuine controls did not disable all flags; no environment override or retry')
            record['classification']='QUALIFIED'
    except BaseException as exc:
        record.update(classification='UNKNOWN',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if policy is not None:
            originals=list(policy._tensor_patches)
            policy.restore_tensor_guards()
        else:originals=[]
        for owner,name,original in reversed(patches):setattr(owner,name,original)
        record['restoration_verified']=all(getattr(owner,name) is original for owner,name,original in originals+patches)
        if torch is not None:record['gpu_initialized_after']=torch.cuda.is_initialized()
        sys.setprofile(None);threading.setprofile(None)
        gc.collect();record.update(elapsed_seconds=time.monotonic()-began,finished_utc=base.utc_now())
        if record['classification']=='QUALIFIED':
            try:qualify(record)
            except BaseException as exc:record.update(classification='UNKNOWN',error=str(exc))
        base.save_json(output/'results.json',record)
    return 0 if record['classification']=='QUALIFIED' else 1

def preflight():
    if (E/'CPU-tunableop.marker').exists() or (E/'run').exists():raise RuntimeError('Exclusive CPU attempt already reserved')
    passive=json.loads((E/'linux-preservation-before.json').read_text())
    if passive['status']!='PASS':raise RuntimeError('Linux preservation refused')
    source=E.parent/'amd-wsl-turret-neural-decode-evidence/installed-tunable.py.txt'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Preserved TunableOp source changed')
    helpers={}
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        helper=base.load_sibling(name);path=(ROOT/'scripts/compatibility'/(name+'.py')).resolve()
        if Path(helper.__file__).resolve()!=path:raise RuntimeError('Wrong helper path')
        helpers[name]=str(path)
    command=base.worker_command(base.guest_path(SCRIPT),base.guest_path(E/'run/guest'))
    if command[command.index('--worker')-1]!=base.guest_path(SCRIPT):raise RuntimeError('Wrong guest')
    return {'helpers':helpers,'selected_guest':str(SCRIPT),'command':command,'marker_absent':True,
            'GPU_authorized':False,'source_sha256':SOURCE_SHA}

def host(output):
    if output.resolve()!=(E/'run').resolve():raise RuntimeError('Exclusive CPU evidence path required')
    spec=importlib.util.spec_from_file_location('cpu_owned_runner',E.parent/'amd-wsl-packaging-decoder-loading-evidence/loading.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    # Keep base.SCRIPT anchored at scripts/compatibility. Only the guest changes.
    runner.SCRIPT=SCRIPT;runner.HOST_DEADLINE=120;runner.NATIVE_DEADLINE=60
    runner.worker_command=base.worker_command;runner.validate_qualified=qualify;runner.preflight=preflight
    def reserve(path):
        base.save_json(E/'CPU-tunableop.marker',{'started_utc':base.utc_now(),'distro':base.DISTRO,
            'script_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),'output':str(path),'GPU_authorized':False})
        path.mkdir(parents=True,exist_ok=False)
    runner.reserve_attempt=reserve
    def deadlines(now,began,go):
        if now-began>=120 or (go is not None and now-go>=60):raise TimeoutError('CPU TunableOp qualification deadline')
    runner.check_deadlines=deadlines
    return runner.supervise(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight()));raise SystemExit(0)
    if args.worker is not None:raise SystemExit(worker(args.worker))
    if args.output is None:parser.error('--output required')
    raise SystemExit(host(args.output))
