"""One owned installed-WSL CPU acceptance. No native import/GPU/model work."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
smoke=load('existing_smoke',ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py')

def guest(output):
    output.mkdir(exist_ok=False)
    result={'classification':'CPU_UNKNOWN','checks':[],'GPU_entry_attempts':[],
            'native_imports':False,'real_weights_loaded':False,'started_utc':smoke.utc_now()}
    policy=None;torch=None;original_gpu=[];current=[None];callbacks={}
    try:
        smoke.await_go(sys.stdin.buffer)
        result['authorization_accepted']=True
        def deny_import(event,args):
            if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','trellis2','cumesh','o_voxel'}:
                raise RuntimeError('CPU acceptance forbids native/model imports')
        sys.addaudithook(deny_import)
        before=json.loads((E/'installed-source-before.json').read_text())
        for row in before['sources'].values():
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Installed source changed')
        scratch=STAGE/'device-context-correction';scratch.mkdir(exist_ok=True)
        os.environ.update(HOME=str(scratch),TMPDIR=str(scratch),TRITON_CACHE_DIR=str(scratch/'triton-cache'),TORCH_EXTENSIONS_DIR=str(scratch/'extensions'))
        sys.path.insert(0,str(STAGE/'site-packages'))
        import torch
        import importlib.metadata as md
        result['runtime']={'torch':torch.__version__,'distribution':md.version('torch'),'hip':torch.version.hip,'cuda':torch.version.cuda,'python':sys.version.split()[0],'prefix':sys.prefix}
        if result['runtime']!={'torch':'2.9.1+rocm7.2.1.gitff65f5bc','distribution':'2.9.1+rocm7.2.1.lw.gitff65f5bc','hip':'7.2.53211-e1a6bc5663','cuda':None,'python':'3.12.3','prefix':'/opt/modly-wsl/venv'}:raise RuntimeError('Pinned runtime mismatch')
        result['GPU_initialized_before']=torch.cuda.is_initialized()
        if result['GPU_initialized_before']:raise RuntimeError('Premature GPU initialization')
        # Any entry here is a failure, never an accepted frontend rejection.
        def block(owner,name):
            original=getattr(owner,name,None)
            if not callable(original):return
            original_gpu.append((owner,name,original))
            def denial(*a,**k):
                result['GPU_entry_attempts'].append({'function':name,'check':current[0]})
                raise RuntimeError('GPU entry probe reached: '+name)
            setattr(owner,name,denial)
        for name in ('_lazy_init','init','is_available','device_count','current_device','get_device_name','get_device_properties','mem_get_info','synchronize'):block(torch.cuda,name)
        for name in ('_cuda_init','_cuda_getDeviceCount'):block(torch._C,name)
        block(torch,'get_default_device')
        import triton
        import triton.testing
        if triton.__version__!='3.5.1+rocm7.2.1.gita272dfa8':raise RuntimeError('Pinned Triton mismatch')
        result['runtime']['triton']=triton.__version__
        device_module=__import__('torch.utils._device',fromlist=['DeviceContext'])
        originals=frozenset(device_module._device_constructors())
        module=load('corrected_policy',ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py')
        policy=module.ImportPolicy(scratch,writable_roots=(output,));policy.install_tensor_guards(torch);policy.install_runtime_guards(triton);policy.install_audit_hook();policy.install_profile()
        patched=list(policy._tensor_patches)
        factory_originals={o for owner,name,o in patched if owner is torch}
        def profile(frame,event,arg):
            if current[0] and event=='c_call' and (arg in factory_originals or getattr(arg,'__name__',None) in ('to','cuda','new_empty','new_tensor','new_zeros','new_ones','new_full')):
                callbacks[current[0]]=callbacks.get(current[0],0)+1
            policy.profile(frame,event,arg)
        sys.setprofile(profile)
        def check(label,call,expected=None,reject=False):
            current[0]=label;callbacks[label]=0
            smoke.journal(output,'check-entry',check=label)
            try:
                value=call()
            except module.ImportPolicyError:
                if not reject:raise
                if callbacks[label] or result['GPU_entry_attempts']:raise RuntimeError('Rejection occurred after original callback/GPU entry')
                row={'check':label,'outcome':'REJECTED_BEFORE_ORIGINAL','original_callbacks':0}
            else:
                if reject:raise AssertionError('Expected frontend rejection: '+label)
                if value.device.type!=expected:raise AssertionError('Wrong tensor device: '+label)
                row={'check':label,'outcome':'ACCEPTED','device':value.device.type,'shape':list(value.shape),'original_callbacks':callbacks[label]}
            result['checks'].append(row);smoke.journal(output,'check-return',**row);current[0]=None
        with policy.tensor_mode(torch):
            check('native-import-phase-factory-denial',lambda:torch.empty((1,),device='cpu'),reject=True)
            policy.set_phase('cpu_meta')
            check('explicit-CPU',lambda:torch.empty((1,),device='cpu'),'cpu')
            check('explicit-meta',lambda:torch.empty((1,),device='meta'),'meta')
            check('implicit-CPU',lambda:torch.empty((1,)),'cpu')
            with torch.device('cpu'):check('context-implicit-CPU',lambda:torch.empty((1,)),'cpu')
            with torch.device('meta'):check('context-implicit-meta',lambda:torch.empty((1,)),'meta')
            for value in ('cuda','cuda:0',torch.device('cuda')):check('explicit-'+str(value)+'-'+str(type(value)),lambda v=value:torch.empty((1,),device=v),reject=True)
            with torch.device('cuda'):
                check('exact-original-implicit-CUDA',lambda:torch.empty((1,)),reject=True)
                check('explicit-CPU-overrides-CUDA-context',lambda:torch.empty((1,),device='cpu'),'cpu')
                with torch.device('meta'):check('nested-CUDA-then-meta',lambda:torch.empty((1,)),'meta')
            with torch.device('meta'):
                with torch.device('cuda'):check('nested-meta-then-CUDA',lambda:torch.empty((1,)),reject=True)
            cpu=torch.empty((1,),device='cpu');other=torch.empty((1,),device='cpu')
            check('tensor-to-CPU',lambda:cpu.to('cpu'),'cpu')
            check('tensor-to-other-CPU',lambda:cpu.to(other),'cpu')
            check('tensor-to-meta',lambda:cpu.to('meta'),'meta')
            check('tensor-to-dtype-only',lambda:cpu.to(torch.float16),'cpu')
            check('tensor-to-CUDA-positional',lambda:cpu.to('cuda'),reject=True)
            check('tensor-to-CUDA-keyword',lambda:cpu.to(device=torch.device('cuda')),reject=True)
            check('tensor-to-CUDA-index',lambda:cpu.to(0),reject=True)
            check('tensor-cuda',lambda:cpu.cuda(),reject=True)
            check('tensor-new-empty-CUDA',lambda:cpu.new_empty((1,),device='cuda'),reject=True)
            with torch.device('cuda'):check('empty-like-CPU-not-context-constructor',lambda:torch.empty_like(cpu),'cpu')
            for device in ('cpu','meta','cuda'):
                torch.set_default_device(device)
                try:check('set-default-'+device,lambda:torch.empty((1,)),expected=device if device!='cuda' else None,reject=device=='cuda')
                finally:torch.set_default_device(None)
            class ForeignMode(torch.overrides.TorchFunctionMode):
                def __torch_function__(self,*args,**kwargs):raise AssertionError('Foreign mode executed')
            with ForeignMode():check('foreign-mode-before-original',lambda:torch.empty((1,),device='cpu'),reject=True)
            bad=device_module.DeviceContext('cpu');bad.device='cuda'
            with bad:check('malformed-active-device-before-original',lambda:torch.empty((1,)),reject=True)
            cpu=other=None
        gc.collect()
        driver=__import__('triton.runtime.driver',fromlist=['driver']).driver
        if result['GPU_entry_attempts'] or torch.cuda.is_initialized() or driver._active is not None or driver._default is not None:raise RuntimeError('GPU/helper entry or activation occurred')
        if frozenset(device_module._device_constructors())!=originals:raise RuntimeError('Original constructor identity cache changed')
        # Positive c_call controls demonstrate that original-factory observation is live.
        if callbacks['explicit-CPU']<1 or callbacks['tensor-to-CPU']<1:raise RuntimeError('Original callback observation not demonstrated')
        result['policy']=policy.snapshot();policy.restore_profile();policy.restore_runtime_guards();policy.restore_tensor_guards()
        if any(getattr(owner,name) is not original for owner,name,original in patched):raise RuntimeError('Frontend restoration mismatch')
        native_absent=not any(n.startswith(('flex_gemm','hip_utils','__triton_launcher','trellis2')) for n in sys.modules)
        if not native_absent:raise RuntimeError('Excluded native/model module imported')
        result.update(constructor_identity_preserved=True,guards_restored=True,GPU_initialized_after=torch.cuda.is_initialized(),native_module_absent=native_absent,classification='CPU_PASS')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if policy is not None:policy.restore_profile();policy.restore_runtime_guards();policy.restore_tensor_guards()
        for owner,name,original in original_gpu:setattr(owner,name,original)
        if torch is not None:result['GPU_initialized_final']=torch.cuda.is_initialized()
        result['finished_utc']=smoke.utc_now()
        smoke.save_json(output/'results.json',result)
    return 0 if result['classification']=='CPU_PASS' else 1

def helper_preflight():
    # SCRIPT selects the guest; helper resolution stays anchored independently.
    helper_dir=(ROOT/'scripts/compatibility').resolve(strict=True)
    selected=Path(__file__).resolve(strict=True)
    if selected!=(E/'acceptance.py').resolve(strict=True):
        raise RuntimeError('Unexpected acceptance guest script')
    helpers={};rows={}
    expected=json.loads((E/'preservation-before.json').read_text())['unchanged_control_code']
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        path=(helper_dir/(name+'.py')).resolve(strict=True)
        if path.parent!=helper_dir or not path.is_file():raise RuntimeError('Unexpected helper path')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected[name+'.py']:raise RuntimeError('Shared helper changed')
        helpers[name]=load(name,path)
        rows[name]={'path':str(path),'sha256':digest,'imported_from':helpers[name].__file__}
    def fixed_helper(name):
        if name not in helpers:raise RuntimeError('Unexpected supervisor helper: '+name)
        return helpers[name]
    command=smoke.worker_command(smoke.guest_path(selected),smoke.guest_path(E/'run/guest'))
    if command[command.index('--worker')-1]!=smoke.guest_path(selected):raise RuntimeError('Wrong selected guest')
    smoke.save_json(E/'helper-preflight.json',{'helpers':rows,'selected_guest':str(selected),
        'guest_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'worker_command':command,
        'attempt_marker_absent':not(E/'CPU-attempt-started.json').exists(),'GPU_work':False})
    return fixed_helper

def host(output):
    from unittest.mock import patch
    fixed_helper=helper_preflight()
    marker=E/'CPU-attempt-started.json'
    # Reuse production containment unchanged; its GPU PASS contract remains unused.
    with patch.object(smoke,'SCRIPT',Path(__file__).resolve()),patch.object(smoke,'ATTEMPT_MARKER',marker),patch.object(smoke,'load_sibling',side_effect=fixed_helper):
        raw_exit=smoke.supervise(output)
    raw=json.loads((output/'results.json').read_text());worker=raw.get('worker',{})
    passed=(raw_exit==1 and raw.get('classification')=='UNKNOWN' and raw.get('worker_returncode')==0
        and worker.get('classification')=='CPU_PASS' and worker.get('GPU_entry_attempts')==[]
        and worker.get('native_module_absent') is True and worker.get('GPU_initialized_final') is False and worker.get('guards_restored') is True
        and raw.get('observer_healthy') is True and raw.get('windows_job_cleanup',{}).get('owned_tree_exit_verified') is True
        and raw['windows_job_cleanup'].get('forced_job_cleanup') is False
        and raw.get('windows_job_grace',{}).get('empty') is True and raw.get('distro_termination',{}).get('returncode')==0
        and not any('error' in key for key in raw))
    verdict={'classification':'PASS' if passed else 'UNKNOWN','GPU_authorized':False,'acceptance_attempts':1,
        'production_supervisor_exit':raw_exit,'production_supervisor_classification':raw.get('classification'),
        'raw_unknown_reason':'Unmodified GPU supervisor has no vector-add/GPU-cleanup result; CPU verdict separately requires clean real containment.',
        'CPU_checks':len(worker.get('checks',[])),'GPU_entry_attempts':worker.get('GPU_entry_attempts'),
        'policy_sha256':hashlib.sha256((ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py').read_bytes()).hexdigest()}
    smoke.save_json(output/'CPU-verdict.json',verdict);print(json.dumps(verdict));return 0 if passed else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker is not None:raise SystemExit(guest(args.worker))
    if args.output is None:parser.error('--output required')
    raise SystemExit(host(args.output))
