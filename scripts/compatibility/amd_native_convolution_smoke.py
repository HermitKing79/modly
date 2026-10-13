"""Exactly one installed AMD sparse convolution, never an AI model or decoder."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import weakref
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_native_convolution_preflight as preflight
import amd_native_convolution_contract as contract
import amd_native_convolution_policy as policy_module
import amd_sparse_decoder_forward as supervisor
import amd_sparse_decoder_loading as loading
import amd_forward_diagnostics as diagnostics
import amd_blaslt_teardown as teardown
EV=preflight.EV;VENV=preflight.VENV
CPU_PROBE='cpu-probe'
AUTHORIZE=b'EXECUTE_ONE_NATIVE_CONVOLUTION\n'

def identity():
    acceptance=json.loads((ROOT/'docs/compatibility/amd-native-postprocessor-admission-evidence/cpu-probe/results.json').read_text())
    if (acceptance.get('passed') is not True or acceptance['worker'].get('gpu_initialized') is not False
        or acceptance['worker']['constructor_acceptance']['postprocessors'].get('observed',{}).get('python_tail_interfaces') is not True):
        raise ValueError('Full installed Python postprocessor/tail CPU acceptance required; GPU route is NO-GO')
    preflight.verify_historical();teardown.verify_binding_evidence()
    probe=json.loads((EV/CPU_PROBE/'results.json').read_text())
    if probe.get('passed') is not True or probe['worker'].get('gpu_initialized') is not False:raise ValueError('Actual CPU admission required')
    config=probe['worker']
    return dict(config=config,native_contract_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())

def authorize(gate,verified,sha):
    if gate!=dict(status='PASS',gpu_initialized=False,weights_loaded=False,fixture_verified=True) or sha!=verified['native_contract_sha256']:
        raise RuntimeError('Invalid native-convolution CPU gate')

def classify(r,process_ok,healthy,preserved):
    if (r.get('status')=='PASS' and process_ok and healthy and preserved and r.get('convolution_calls')==1
        and all(r.get('native_counts',{}).get(k)==1 for k in contract.Permit().counts)
        and r.get('helper_calls')==1 and r.get('lookup_keys')==[preflight.HIT_KEY] and r.get('forbidden_calls')==[]
        and r.get('elementwise',{}).get('mismatched_elements')==0 and r.get('coordinates_unchanged') is True
        and r.get('release',{}).get('allocated_bytes')==0 and r['release'].get('reserved_bytes')==0
        and r['release'].get('owned_tensors_collected') is True and r.get('workspace_clear_calls')==1):return 'PASS'
    return 'FAIL' if r.get('failure_kind') in ('numerical','native') else 'UNKNOWN'

def worker(scratch):
    r=dict(status='UNKNOWN',stage='startup',convolution_calls=0,weights_loaded=False,sdk_discovery=[])
    torch=native=f=coords=w=b=out=cache=cpu_out=cpu_coords=None
    trace=diagnostics.Trace(lambda line:print(line,flush=True));phase='cpu';failed_native=False
    def stage(name):
        r['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name,phase=phase)
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job required')
        scratch=scratch.resolve()
        if scratch!=EV.resolve()/'run/scratch':raise ValueError('Exclusive owned scratch required')
        teardown.verify_workspace()
        from amd_bounded_gpu_proof import scratch_environment,CACHE_KEYS
        from services.amd_offline_guard import block_external_actions
        env=scratch_environment(scratch);env.update(WORKSPACE_DIR=str(ROOT),TRITON_CACHE_DIR=str(scratch/'t'),FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(scratch/'autotune.json'))
        for k in CACHE_KEYS:
            if k!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[k]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env);block_external_actions(VENV/'Scripts/rocm-sdk.exe',r['sdk_discovery'],scratch)
        allowed_gpu=False
        def deny(*a,**kw):raise RuntimeError('No models, weights, compilation or unapproved GPU access')
        def audit(e,a):
            if e=='import' and str(a[0]).split('.')[0] in ('trellis2','cumesh','o_voxel','transformers','comfy'):deny()
            if e=='import' and str(a[0]).split('.')[0]=='flex_gemm' and not allowed_gpu:deny()
            if e=='open' and isinstance(a[0],(str,bytes)) and Path(os.fsdecode(a[0])).suffix.lower() in ('.safetensors','.pt','.pth','.ckpt','.gguf'):deny()
        sys.addaudithook(audit)
        import torch
        lazy=torch.cuda._lazy_init;init=torch._C._cuda_init
        def gpu_lazy(*a,**kw):
            if not allowed_gpu:deny()
            return lazy(*a,**kw)
        def gpu_init(*a,**kw):
            if not allowed_gpu:deny()
            return init(*a,**kw)
        torch.cuda._lazy_init=gpu_lazy;torch._C._cuda_init=gpu_init
        torch.load=deny;torch.compile=deny;torch.nn.Module.__call__=deny
        stage('cpu-admission')
        meta,source=preflight.inspect_installed(torch);r['runtime']=meta['runtime']
        r['workspace_binding']=teardown.verify_runtime_binding(torch)
        import numpy as np
        import psutil
        r['host_available_bytes']=psutil.virtual_memory().available
        if r['host_available_bytes']<contract.HOST_RAM:raise ValueError('Insufficient host RAM')
        reference,coord_ref,weight,bias=contract.fixture();contract.validate_fixture(reference,coord_ref,weight,bias)
        expected=json.loads((EV/CPU_PROBE/'results.json').read_text())['worker']
        for name,a in zip(('features','coordinates','weights','bias'),(reference,coord_ref,weight,bias)):
            if hashlib.sha256(a.tobytes()).hexdigest()!=expected['fixture'][name]['sha256']:raise ValueError('Fixture differs from CPU review')
        verified=identity()
        private=json.loads((scratch.parent/'private-cache.json').read_text())
        native=policy_module.NativePolicy(torch,private,meta,source,trace)
        r['native_hook_state']=native.hook_state
        r['gate_one']=dict(status='PASS',gpu_initialized=False,weights_loaded=False,fixture_verified=True)
        print(supervisor.EVENT+json.dumps(dict(event='cpu-ready',gate_one=r['gate_one'],native_contract_sha256=verified['native_contract_sha256'])),flush=True)
        if sys.stdin.readline().encode()!=AUTHORIZE:raise RuntimeError('One native convolution authorization required')
        allowed_gpu=True;phase='gpu';os.environ.update(env)
        os.environ.update(FLEX_GEMM_USE_AUTOTUNE_CACHE='1',FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE='0')
        stage('gpu-admission');torch.cuda.init()
        if torch.cuda.device_count()!=1:raise ValueError('Only one visible AMD device required')
        props=torch.cuda.get_device_properties(0)
        r['gpu']=dict(name=props.name,architecture=props.gcnArchName,total_bytes=props.total_memory,index=0)
        trace.emit('state','gpu',gpu=r['gpu'])
        if props.name!='AMD Radeon RX 9070 XT' or props.gcnArchName.split(':')[0]!='gfx1201':raise ValueError('Wrong GPU')
        r['memory_before']=loading.gpu_memory(torch);trace.emit('memory','before',memory=r['memory_before'])
        contract.admit(psutil.virtual_memory().available,r['memory_before']['available_bytes'])
        if r['memory_before']['allocated_bytes'] or r['memory_before']['reserved_bytes']:raise RuntimeError('Unexpected pre-test allocation')
        torch.cuda.set_per_process_memory_fraction(contract.ALLOCATOR_LIMIT/props.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        stage('installed-flex-import')
        import flex_gemm
        r['flex_path']=flex_gemm.__file__;native.attach_flex()
        import importlib
        config=importlib.import_module('flex_gemm.kernels.triton.spconv.config')
        if config.allow_ieee is not True:raise RuntimeError('Unexpected IEEE kernel option')
        native.permit.phase='inputs';stage('input-transfer')
        with policy_module.dispatch_guard(torch,native),torch.inference_mode():
            f=trace.call('features-to-GPU',lambda:torch.from_numpy(reference).to('cuda:0'))
            coords=trace.call('coordinates-to-GPU',lambda:torch.from_numpy(coord_ref).to('cuda:0'))
            w=trace.call('identity-weights-to-GPU',lambda:torch.from_numpy(weight).to('cuda:0'))
            b=trace.call('zero-bias-to-GPU',lambda:torch.from_numpy(bias).to('cuda:0'))
            torch.cuda.synchronize(0);r['memory_after_inputs']=loading.gpu_memory(torch)
            trace.emit('memory','after-inputs',memory=r['memory_after_inputs'])
            native.permit.phase='convolution';stage('single-native-convolution');r['convolution_calls']=1
            began=time.monotonic()
            out,cache=trace.call('genuine-submanifold-convolution',lambda:native.convolve(f,coords,w,b))
            trace.call('convolution-synchronize',lambda:torch.cuda.synchronize(0))
            r['synchronized_convolution_seconds']=time.monotonic()-began;r['memory_after_forward']=loading.gpu_memory(torch)
            trace.emit('memory','after-forward',memory=r['memory_after_forward'])
            native.permit.phase='output';stage('output-copy')
            cpu_out=trace.call('output-to-CPU',out.cpu);cpu_coords=trace.call('coordinates-to-CPU',coords.cpu)
            trace.call('output-copy-synchronize',lambda:torch.cuda.synchronize(0))
        stage('validate-all-elements')
        try:r['elementwise']=contract.compare(cpu_out.numpy(),reference)
        except ValueError:
            r['failure_kind']='numerical';raise
        r['coordinates_unchanged']=bool(np.array_equal(cpu_coords.numpy(),coord_ref))
        if not r['coordinates_unchanged']:r['failure_kind']='numerical';raise ValueError('Coordinate identity changed')
        r['output']=dict(shape=list(out.shape),dtype=str(out.dtype),device=str(out.device),minimum=float(cpu_out.min()),maximum=float(cpu_out.max()),mean=float(cpu_out.float().mean()))
        stage('save-and-readback')
        r['artifacts']={}
        for name,a in (('input-features',reference),('coordinates',coord_ref),('output-features',cpu_out.numpy())):
            path=scratch/(name+'.npy')
            with path.open('xb') as stream:np.save(stream,a,allow_pickle=False)
            read=np.load(path,allow_pickle=False)
            if not np.array_equal(read,a) or read.dtype!=a.dtype:raise RuntimeError('Artifact readback failed')
            r['artifacts'][name]=dict(path=str(path),bytes=path.stat().st_size,sha256=preflight.digest(path),shape=list(a.shape),dtype=str(a.dtype))
        phase='release';stage('release')
        refs=[weakref.ref(t) for t in (f,coords,w,b,out)]
        f=coords=w=b=out=cache=cpu_out=cpu_coords=None;native.release();gc.collect()
        trace.call('release-synchronize',lambda:torch.cuda.synchronize(0))
        r['workspace_clear_calls']=1;trace.call('clear-BLAS-workspaces',torch._C._cuda_clearCublasWorkspaces)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(0)
        r['release']={**loading.gpu_memory(torch),'owned_tensors_collected':all(ref() is None for ref in refs)}
        trace.emit('memory','release',memory=r['release'])
        if r['release']['allocated_bytes'] or r['release']['reserved_bytes'] or not r['release']['owned_tensors_collected']:raise RuntimeError('Zero-allocation release failed')
        r['native_counts']=native.permit.counts.copy()
        if any(v!=1 for v in r['native_counts'].values()):raise RuntimeError('Exact native invocation count failed')
        r['status']='PASS';stage('complete')
    except BaseException as exc:
        r.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if 'failure_kind' not in r:
            r['failure_kind']='native' if ('HIP error' in str(exc) or 'CUDA error' in str(exc)) else 'prerequisite_or_guard'
    finally:
        if native is not None:
            r['post1_scalar_read']=dict(reads=native.post1_scalar.reads,value=native.post1_scalar.value)
            r['post2_scalar_read']=dict(reads=native.post2_scalar.reads,value=native.post2_scalar.value)
            r.update(native_counts=native.permit.counts.copy(),helper_calls=native.helper_calls,lookup_keys=native.lookups,forbidden_calls=native.forbidden_calls)
            # On error issue no additional GPU commands; process containment owns release.
            if r['status']=='PASS':native.close()
        print(supervisor.RESULT+json.dumps(r),flush=True)
    return 0 if r['status']=='PASS' else 1

def profile():
    files=['scripts/compatibility/'+name+'.py' for name in ('amd_native_convolution_smoke','amd_native_convolution_contract','amd_native_convolution_policy','amd_native_post1_scalar','amd_native_postprocessor_admission','test_amd_native_postprocessors','test_amd_native_scalar_dispatch','test_amd_native_convolution_hooks','amd_native_convolution_preflight','amd_native_cache_dispatch_proof','amd_shape_cache_identity','amd_shape_decoder_preflight','amd_blaslt_teardown')]
    files+=supervisor.COMMITTED_FILES
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs/compatibility/amd-native-postprocessor-admission-evidence/cpu-probe/results.json',EV/CPU_PROBE/'results.json',preflight.PRIOR,ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/preflight.json',ROOT/'docs/compatibility/amd-shape-decoder-cache-admission-evidence/native-cache-audit.json',teardown.PINS,teardown.EVIDENCE/'binary-review.json')]
    return SimpleNamespace(kind='native-convolution',evidence=EV,cpu_seconds=45,gpu_seconds=120,overall_seconds=180,
        identity=identity,source_identity=preflight.verify_historical,validate_config=lambda x:None,classify=classify,
        prepare_private_cache=preflight.private_cache,worker_script=Path(__file__).resolve(),authorize=AUTHORIZE,
        authorize_placement=authorize,committed_files=list(dict.fromkeys(files)),maximum_forwards=1)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--worker',action='store_true');a.add_argument('--output',type=Path,required=True);a.add_argument('--harness-commit');args=a.parse_args()
    teardown.verify_workspace()
    if args.worker:raise SystemExit(worker(args.output))
    identity()
    import psutil
    if psutil.virtual_memory().available<contract.HOST_RAM:raise ValueError('Host RAM admission failed before attempt marker')
    raise SystemExit(supervisor.supervise(args.output,args.harness_commit,profile=profile()))
