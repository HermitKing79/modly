"""One monitored metadata-only VRAM check; no weights, native imports or tensors."""
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
SCRATCH=Path('/home/modly/turret-fourth-child-precheck-5a7bf53')
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
sys.path[:0]=[str(E),str(ROOT/'scripts/compatibility')]
from decode import verify_tunable
import amd_wsl_rocdxg_smoke as base

def qualify(record):
    base.validate_runtime_identity(record['runtime'])
    gpu=record['gpu']
    tuning=record.get('tunable_admission',{}).get('after_torch_import',{})
    if (record.get('classification')!='QUALIFIED' or gpu['count']!=1 or 'RX 9070 XT' not in gpu['name']
        or gpu['architecture'].split(':')[0]!='gfx1201' or gpu['free_bytes']<13*1024**3
        or record.get('free_bytes_after_release',0)<13*1024**3
        or record.get('memory_before')!={'allocated':0,'reserved':0}
        or record.get('memory_release')!={'allocated':0,'reserved':0}
        or record.get('policy',{}).get('tensor_operations_allowed')!=0
        or record.get('policy',{}).get('rejected_calls')!=0
        or tuning.get('flags')!={'enabled':False,'tuning':False,'record_untuned':False}
        or tuning.get('environment')!={k:'0' for k in ('PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')}
        or record.get('forbidden_imports')!=[] or 'error' in record or 'cleanup_error' in record):
        raise RuntimeError('Metadata-only 13 GiB VRAM readiness not qualified')

def worker(output):
    output.mkdir(exist_ok=False);began=time.monotonic()
    record={'classification':'UNKNOWN','forbidden_imports':[]};torch=policy=None
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','triton','trellis2','cumesh','o_voxel','nvdiffrast','spconv','torchsparse','safetensors'}:
            record['forbidden_imports'].append(str(args[0]));raise RuntimeError('Readiness forbids model/native imports')
        if event=='open' and isinstance(args[0],(str,bytes)) and '/models/' in str(args[0]).replace('\\','/'):
            raise RuntimeError('Readiness forbids model file access')
    try:
        base.await_go(sys.stdin.buffer)
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home','tmp'):(SCRATCH/name).mkdir()
        os.environ.update(HOME=str(SCRATCH/'home'),TMPDIR=str(SCRATCH/'tmp'),
            PYTORCH_TUNABLEOP_ENABLED='0',PYTORCH_TUNABLEOP_TUNING='0',PYTORCH_TUNABLEOP_RECORD_UNTUNED='0',
            HIP_VISIBLE_DEVICES='0',ROCR_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0')
        sys.path.insert(0,str(STAGE))
        from amd_wsl_native_import_policy import ImportPolicy
        policy=ImportPolicy(SCRATCH,writable_roots=(output,));policy.install_audit_hook();sys.addaudithook(audit)
        import torch
        verify_tunable(torch,record,'after_torch_import')
        record['runtime']={'torch':str(torch.__version__),'distribution_version':md.version('torch'),
            'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
            'python':platform.python_version(),'prefix':sys.prefix,'gpu_initialized_before':torch.cuda.is_initialized()}
        base.validate_runtime_identity(record['runtime'])
        if md.version('triton')!='3.5.1+rocm7.2.1.gita272dfa8':raise RuntimeError('Pinned Triton metadata changed')
        policy.install_tensor_guards(torch);policy.install_profile()
        with policy.tensor_mode(torch):
            count=torch.cuda.device_count();props=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info(0)
            record['gpu']={'count':count,'name':props.name,'architecture':props.gcnArchName,'free_bytes':free,'total_bytes':total}
            record['memory_before']={'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0)}
            if count!=1 or 'RX 9070 XT' not in props.name or props.gcnArchName.split(':')[0]!='gfx1201' or free<13*1024**3:
                raise RuntimeError('Live 13 GiB VRAM admission refused; no neural attempt')
            torch.cuda.set_per_process_memory_fraction(12*1024**3/total,0)
            record['classification']='QUALIFIED'
    except BaseException as exc:
        record.update(classification='UNKNOWN',error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc())
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(0);torch.cuda.empty_cache()
                record['memory_release']={'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0)}
                record['free_bytes_after_release']=torch.cuda.mem_get_info(0)[0]
            except BaseException as exc:record.update(classification='UNKNOWN',cleanup_error=str(exc))
        if policy is not None:
            record['policy']=policy.snapshot();policy.restore_profile();policy.restore_tensor_guards()
        gc.collect();record['elapsed_seconds']=time.monotonic()-began
        if record['classification']=='QUALIFIED':
            try:qualify(record)
            except BaseException as exc:record.update(classification='UNKNOWN',error=str(exc))
        base.save_json(output/'results.json',record)
    return 0 if record['classification']=='QUALIFIED' else 1

def host(output):
    if output.resolve()!=(E/'readiness-run').resolve():raise RuntimeError('Exclusive readiness output required')
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        helper=base.load_sibling(name)
        if Path(helper.__file__).resolve()!=ROOT/'scripts/compatibility'/(name+'.py'):raise RuntimeError('Wrong helper path')
    if (E/'VRAM-readiness.marker').exists() or output.exists() or (E/'neural-attempt.marker').exists():
        raise RuntimeError('Readiness/attempt already reserved')
    spec=importlib.util.spec_from_file_location('ready_owned_runner',E.parent/'amd-wsl-packaging-decoder-loading-evidence/loading.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.SCRIPT=SCRIPT;runner.HOST_DEADLINE=120;runner.NATIVE_DEADLINE=60
    runner.worker_command=base.worker_command;runner.validate_qualified=qualify;runner.preflight=lambda:None
    def reserve(path):
        base.save_json(E/'VRAM-readiness.marker',{'script_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            'GPU_metadata_only':True,'model_loading_authorized':False,'started_utc':base.utc_now()})
        path.mkdir(parents=True,exist_ok=False)
    runner.reserve_attempt=reserve
    def deadlines(now,began,go):
        if now-began>=120 or (go is not None and now-go>=60):raise TimeoutError('VRAM metadata deadline')
    runner.check_deadlines=deadlines
    return runner.supervise(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker is not None:raise SystemExit(worker(a.worker))
    if a.output is None:p.error('--output required')
    raise SystemExit(host(a.output))
