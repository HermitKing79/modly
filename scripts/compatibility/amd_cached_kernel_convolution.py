"""One genuine cached leaf launch; CPU-built metadata and CPU output reduction."""
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
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_cached_kernel_oracle as oracle
import amd_native_convolution_contract as c
import amd_native_convolution_preflight as p
from amd_cached_kernel_policy import CachedKernelPolicy
import amd_sparse_decoder_forward as supervisor
import amd_sparse_decoder_loading as loading
import amd_forward_diagnostics as diagnostics
import amd_blaslt_teardown as teardown
EV=ROOT/'docs/compatibility/amd-cached-kernel-convolution-evidence'
AUTHORIZE=b'EXECUTE_ONE_CACHED_KERNEL\n'

def arrays():
    f,co,w,b=c.fixture();c.validate_fixture(f,co,w,b);metadata=oracle.build(co);proof=oracle.validate(co,metadata)
    return dict(input=f,weight=w.reshape(1024,27,1024),bias=b,**metadata,output=np.full((2,2049,1024),np.nan,np.float32)),co,proof

def cpu_acceptance(torch,meta,source,scratch):
    from test_amd_native_convolution_hooks import run_constructor_suite
    from test_amd_cached_kernel import installed_policy_tests
    current,helper,leaf=p.inspect_installed(torch,return_leaf=True)
    # The existing worker appends its fixture inventory before this callback.
    if set(meta)-set(current)!={'fixture'} or current!={k:meta[k] for k in current}:raise RuntimeError('Direct leaf admission differs')
    private=p.private_cache(scratch);constructor=run_constructor_suite(torch,private,meta,source)
    policy=installed_policy_tests(torch,private,meta,source,leaf)
    values,coords,proof=arrays()
    artifacts=oracle.save_arrays(scratch,dict(**values,coordinates=coords))
    return dict(passed=constructor['passed'] and policy['passed'],constructor=constructor,policy=policy,metadata=oracle.describe(values),proof=proof,
        artifacts=artifacts,persistence_verified=True,gpu_initialized=torch.cuda.is_initialized(),native_executed=False)

def identity():
    p.verify_historical();teardown.verify_binding_evidence()
    probe=json.loads((EV/'cpu-acceptance/results.json').read_text())
    if not probe.get('passed') or probe['worker'].get('gpu_initialized') is not False:raise RuntimeError('Cached-leaf CPU acceptance required')
    config=probe['worker'];accept=config['constructor_acceptance']
    if not accept['persistence_verified'] or not accept['passed']:raise RuntimeError('Oracle/persistence acceptance required')
    return dict(config=config,native_contract_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())

def authorize(gate,verified,sha):
    if gate!=dict(status='PASS',gpu_initialized=False,weights_loaded=False,fixture_verified=True) or sha!=verified['native_contract_sha256']:raise RuntimeError('Invalid cached-leaf CPU gate')

def counts_valid(counts):
    return set(counts)==set(c.Permit().counts) and all(counts.get(name)==(1 if name in ('convolution','jit','binary','launch') else 0) for name in c.Permit().counts)

def classify(r,process_ok,healthy,preserved):
    if (r.get('status')=='PASS' and process_ok and healthy and preserved and counts_valid(r.get('native_counts',{}))
        and r.get('helper_calls')==1 and r.get('lookup_keys')==[p.HIT_KEY] and r.get('forbidden_calls')==[]
        and r.get('elementwise',{}).get('mismatched_elements')==0 and r.get('elementwise',{}).get('fp32_identity_exact') is True
        and r.get('coordinates_unchanged') is True and r.get('release',{}).get('allocated_bytes')==0
        and r['release'].get('reserved_bytes')==0 and r['release'].get('owned_tensors_collected') is True):return 'PASS'
    return 'FAIL' if r.get('failure_kind') in ('numerical','native') else 'UNKNOWN'

def worker(scratch):
    r=dict(status='UNKNOWN',stage='startup',weights_loaded=False,sdk_discovery=[]);torch=policy=gpu=raw_cpu=None
    trace=diagnostics.Trace(lambda line:print(line,flush=True));phase='cpu';allowed=False
    def stage(name):
        r['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name,phase=phase)
    def memory(name):
        value=loading.gpu_memory(torch);trace.emit('memory',name,memory=value)
        if any(value[key]>c.PEAK_LIMIT for key in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')):raise RuntimeError('1GiB observed smoke memory budget exceeded')
        return value
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job required')
        scratch=scratch.resolve()
        if scratch!=EV.resolve()/'run/scratch':raise RuntimeError('Exclusive scratch required')
        teardown.verify_workspace()
        from amd_bounded_gpu_proof import scratch_environment,CACHE_KEYS
        from services.amd_offline_guard import block_external_actions
        env=scratch_environment(scratch);env.update(WORKSPACE_DIR=str(ROOT),TRITON_CACHE_DIR=str(scratch/'t'),FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(scratch/'autotune.json'))
        for key in CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env);block_external_actions(p.VENV/'Scripts/rocm-sdk.exe',r['sdk_discovery'],scratch)
        def deny(*a,**kw):raise RuntimeError('Cached-leaf proof forbids models/native generators/unapproved GPU')
        def audit(event,args):
            if event=='import' and str(args[0]).split('.')[0] in ('flex_gemm','trellis2','cumesh','o_voxel','transformers','comfy'):deny()
            if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors','.pt','.pth','.ckpt','.gguf'):deny()
        sys.addaudithook(audit)
        import torch
        lazy=torch.cuda._lazy_init;init=torch._C._cuda_init
        def guarded_lazy(*a,**kw):
            if not allowed:deny()
            return lazy(*a,**kw)
        def guarded_init(*a,**kw):
            if not allowed:deny()
            return init(*a,**kw)
        torch.cuda._lazy_init=guarded_lazy;torch._C._cuda_init=guarded_init
        torch.load=deny;torch.compile=deny;torch.nn.Module.__call__=deny
        stage('CPU-oracle-admission')
        meta,source,leaf=p.inspect_installed(torch,return_leaf=True);r['runtime']=meta['runtime']
        teardown.verify_runtime_binding(torch)
        import numpy as np
        import psutil
        if psutil.virtual_memory().available<c.HOST_RAM:raise RuntimeError('Host RAM admission failed')
        values,coordinates,proof=arrays();r['pointer_proof']=proof
        verified=identity();expected=verified['config']['constructor_acceptance']['metadata']
        if oracle.describe(values)!=expected:raise RuntimeError('GPU candidate differs from CPU oracle')
        private=json.loads((scratch.parent/'private-cache.json').read_text())
        policy=CachedKernelPolicy(torch,private,meta,source,trace);policy.select_leaf(leaf)
        r['gate_one']=dict(status='PASS',gpu_initialized=False,weights_loaded=False,fixture_verified=True)
        print(supervisor.EVENT+json.dumps(dict(event='cpu-ready',gate_one=r['gate_one'],native_contract_sha256=verified['native_contract_sha256'])),flush=True)
        if sys.stdin.readline().encode()!=AUTHORIZE:raise RuntimeError('Single cached-kernel authorization required')
        allowed=True;os.environ.update(env);phase='gpu';stage('GPU-admission');torch.cuda.init()
        if torch.cuda.device_count()!=1:raise RuntimeError('Single visible GPU required')
        props=torch.cuda.get_device_properties(0);r['gpu']=dict(name=props.name,architecture=props.gcnArchName,total_bytes=props.total_memory,index=0)
        trace.emit('state','gpu',gpu=r['gpu'])
        if props.name!='AMD Radeon RX 9070 XT' or props.gcnArchName.split(':')[0]!='gfx1201':raise RuntimeError('Wrong GPU')
        before=memory('before');r['memory_before']=before;c.admit(psutil.virtual_memory().available,before['available_bytes'])
        if before['allocated_bytes'] or before['reserved_bytes']:raise RuntimeError('Unexpected pre-test PyTorch allocations')
        torch.cuda.set_per_process_memory_fraction(c.ALLOCATOR_LIMIT/props.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            stage('fixed-buffer-transfers')
            gpu={}
            for name,value in values.items():
                t=trace.call('copy-'+name,lambda value=value:torch.ops.aten._to_copy.default(torch.from_numpy(value),device=torch.device('cuda:0')))
                if tuple(t.shape)!=value.shape or str(t.dtype).removeprefix('torch.')!=str(value.dtype):raise RuntimeError('Transfer changed dtype/shape')
                gpu[name]=t
            t=None;torch.cuda.synchronize(0);r['memory_after_inputs']=memory('after-inputs')
            policy.bind(gpu);policy.permit.phase='convolution';stage('single-cached-kernel-launch');began=time.monotonic()
            trace.call('cached-leaf-host-call',lambda:policy.launch_once(gpu))
            trace.call('cached-leaf-synchronize',lambda:torch.cuda.synchronize(0));r['synchronized_kernel_seconds']=time.monotonic()-began
            r['memory_after_kernel']=memory('after-kernel')
            # Large unexplained native/runtime allocation is a safety failure too.
            if before['available_bytes']-r['memory_after_kernel']['available_bytes']>c.PEAK_LIMIT:raise RuntimeError('Unexpected total GPU memory loss exceeds smoke budget')
            stage('raw-output-copy');raw_cpu=trace.call('raw-splits-to-CPU',lambda:torch.ops.aten._to_copy.default(gpu['output'],device=torch.device('cpu')))
            torch.cuda.synchronize(0)
        r['artifacts']=oracle.save_arrays(scratch,dict(raw_splits=raw_cpu.numpy(),input_features=values['input'],coordinates=coordinates))
        stage('CPU-exact-comparison')
        try:reduced,r['elementwise']=oracle.compare(raw_cpu.numpy(),values['input'])
        except ValueError:r['failure_kind']='numerical';raise
        r['artifacts'].update(oracle.save_arrays(scratch,dict(output_features=reduced)))
        r['coordinates_unchanged']=np.array_equal(coordinates,c.fixture()[1]) and oracle.describe(values)==expected
        if not r['coordinates_unchanged']:raise RuntimeError('CPU fixture or coordinates changed')
        phase='release';stage('release');refs=[weakref.ref(tensor) for tensor in gpu.values()]
        gpu=None;raw_cpu=None;policy.release();gc.collect();torch.cuda.synchronize(0)
        trace.call('clear-BLAS-workspaces',torch._C._cuda_clearCublasWorkspaces)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(0)
        r['release']={**memory('release'),'owned_tensors_collected':all(ref() is None for ref in refs)}
        if r['release']['allocated_bytes'] or r['release']['reserved_bytes'] or not r['release']['owned_tensors_collected']:raise RuntimeError('Zero-allocation release failed')
        if before['available_bytes']-r['release']['available_bytes']>c.PEAK_LIMIT:raise RuntimeError('Unexpected retained native/runtime memory')
        if not counts_valid(policy.permit.counts):raise RuntimeError('Unexpected native kernel counts')
        r['status']='PASS';stage('complete')
    except BaseException as exc:
        r.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        r.setdefault('failure_kind','native' if ('HIP error' in str(exc) or 'CUDA error' in str(exc)) else 'prerequisite_or_guard')
    finally:
        if policy is not None:
            r.update(native_counts=policy.permit.counts.copy(),helper_calls=policy.helper_calls,lookup_keys=policy.lookups,forbidden_calls=policy.forbidden_calls)
            if r['status']=='PASS':policy.close()
        print(supervisor.RESULT+json.dumps(r),flush=True)
    return 0 if r['status']=='PASS' else 1

def profile():
    names=('amd_cached_kernel_convolution','amd_cached_kernel_cpu','amd_cached_kernel_oracle','amd_cached_kernel_policy','test_amd_cached_kernel','test_amd_native_convolution_hooks','amd_native_convolution_preflight','amd_native_convolution_policy','amd_native_post1_scalar','amd_native_convolution_contract','amd_native_cache_dispatch_proof','amd_shape_cache_identity','amd_shape_decoder_preflight','amd_blaslt_teardown')
    files=['scripts/compatibility/'+name+'.py' for name in names]+supervisor.COMMITTED_FILES
    files += [(EV/'cpu-acceptance/results.json').relative_to(ROOT).as_posix(),p.PRIOR.relative_to(ROOT).as_posix(),teardown.PINS.relative_to(ROOT).as_posix(),(teardown.EVIDENCE/'binary-review.json').relative_to(ROOT).as_posix()]
    return SimpleNamespace(kind='native-convolution',evidence=EV,cpu_seconds=45,gpu_seconds=120,overall_seconds=180,identity=identity,
        source_identity=p.verify_historical,validate_config=lambda config:None,classify=classify,prepare_private_cache=p.private_cache,
        worker_script=Path(__file__).resolve(),authorize=AUTHORIZE,authorize_placement=authorize,committed_files=list(dict.fromkeys(files)),maximum_forwards=1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--scratch',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--harness-commit');a=parser.parse_args();teardown.verify_workspace()
    if a.cpu:
        p.EV=EV
        if a.worker:
            if a.scratch.resolve()!=EV.resolve()/'cpu-acceptance/scratch':raise RuntimeError('Owned CPU scratch required')
            raise SystemExit(p.worker(a.scratch,constructor_tests=cpu_acceptance))
        # Existing supervisor passes --worker/--scratch; a tiny CPU launcher is used.
        raise SystemExit(p.supervise(worker_script=ROOT/'scripts/compatibility/amd_cached_kernel_cpu.py',probe_name='cpu-acceptance'))
    if a.worker:raise SystemExit(worker(a.output))
    identity()
    raise SystemExit(supervisor.supervise(a.output,a.harness_commit,profile=profile()))
