"""N2049 metadata-only admission against the actual installed binder and cache."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
from amd_native_cache_dispatch_proof import private_first_group,HIT_KEY,HSACO_SHA
from amd_shape_decoder_preflight import SITE,VENV,digest
import amd_shape_cache_identity as identity
import amd_native_convolution_contract as contract
EV=ROOT/'docs/compatibility/amd-native-convolution-smoke-evidence'
PRIOR=ROOT/'docs/compatibility/amd-shape-decoder-cache-admission-evidence/cpu-probe-5/results.json'
PREFIX='flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk'

def verify_historical():
    audit=json.loads((ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/preflight.json').read_text())
    support=json.loads((ROOT/'docs/compatibility/amd-shape-decoder-cache-admission-evidence/native-cache-audit.json').read_text())
    for path,sha in {**audit['source_sha256'],**audit['native_source_and_binary_sha256']}.items():
        if digest(path)!=sha:raise ValueError('Source/native identity changed: '+path)
    for path,row in support['support_file_identity'].items():
        if digest(path)!=row['sha256']:raise ValueError('Compiler/runtime support changed: '+path)
    tuning=audit['tuning']
    if digest(tuning['path'])!=tuning['sha256']:raise ValueError('Tuning changed')
    table=json.loads(Path(tuning['path']).read_text())['AMD Radeon RX 9070 XT']
    contract.require_tuning(table[PREFIX+'.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk'],table[PREFIX+'.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel'])
    old=json.loads(PRIOR.read_text())['worker']['helper']
    for path,sha in ((old['library'],old['library_sha256']),(old['expected_path'],old['cached_sha256'])):
        if digest(path)!=sha:raise ValueError('Helper/DLL changed')
    return audit,old

def private_cache(scratch):
    name,paths,expected,audit=private_first_group(scratch)
    _,helper=verify_historical()
    dest=scratch/'t'/helper['computed_folder'];dest.mkdir()
    hp=dest/Path(helper['expected_path']).name;shutil.copyfile(helper['expected_path'],hp)
    if digest(hp)!=helper['cached_sha256']:raise ValueError('Helper copy failed')
    tuning=scratch/'autotune.json';shutil.copyfile(audit['tuning']['path'],tuning)
    if digest(tuning)!=audit['tuning']['sha256']:raise ValueError('Tuning copy failed')
    return dict(metadata_name=name,paths=paths,files=expected,helper_path=str(hp),helper=helper,tuning_path=str(tuning),tuning_sha256=audit['tuning']['sha256'])

def inspect_installed(torch,*,return_leaf=False):
    _,expected_helper=verify_historical()
    if torch.cuda.is_initialized():raise RuntimeError('CPU metadata requires uninitialized GPU')
    runtime=dict(python=sys.version.split()[0],torch=str(torch.__version__),hip=torch.version.hip,cuda=torch.version.cuda)
    if runtime!=dict(python='3.12.10',torch='2.13.0+rocm10.0.0',hip='7.15.26333',cuda=None):raise ValueError('Wrong runtime')
    import triton
    from triton import knobs
    from amd_native_convolution_policy import pin_hook_defaults
    pin_hook_defaults(knobs)
    import triton.language as tl
    import triton.compiler.compiler as compiler
    import triton.runtime.build as build
    import triton.backends.amd.driver as driver
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    from triton.runtime.jit import create_function_from_signature
    from triton.runtime.cache import get_cache_key
    srcpath=SITE/'flex_gemm/kernels/triton/spconv/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py'
    fn=next(n for n in ast.parse(srcpath.read_text()).body if isinstance(n,ast.FunctionDef) and n.name.endswith('splitk_kernel'))
    lineno=fn.decorator_list[0].lineno;fn.decorator_list=[]
    ns=dict(__name__=PREFIX,tl=tl)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(srcpath),'exec'),ns)
    raw=ns[fn.name];raw.__code__=raw.__code__.replace(co_firstlineno=lineno);jit=triton.jit(raw)
    class Pointer:
        def __init__(self,dtype):self.dtype=dtype
        def data_ptr(self):return 16
        def ptr_range(self):return 56623104
    args=[Pointer(t) for t in [torch.float16]*3+[torch.uint32,torch.int64,torch.float32]]
    args += [2049,11,1024,1024,27,64,128,32,2,True,Pointer(torch.int32),Pointer(torch.int32)]
    target=GPUTarget('hip','gfx1201',32);backend=HIPBackend(target)
    options=dict(num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3,debug=False,instrumentation_mode='')
    binder=create_function_from_signature(jit.signature,jit.params,backend)
    bound,specialization,_=binder(*args,**options)
    opts,sig,const,attrs=jit._pack_args(backend,options,bound,specialization,None)
    if (sig,const,attrs)!=identity.first_signature():raise ValueError('Actual N2049 binder specialization differs')
    source=compiler.ASTSource(jit,sig,constexprs=const,attrs=attrs)
    env=compiler.get_cache_invalidating_env_vars()
    key=hashlib.sha256(get_cache_key(source,backend,opts,env).encode()).hexdigest()
    if key!=HIT_KEY or env!={}:raise ValueError('Actual kernel cache identity mismatch')
    lib=driver._get_path_to_hip_runtime_dylib()
    text=(SITE/'triton/backends/amd/driver.c').read_text()
    if text.count('/*py_libhip_search_path*/')!=1:raise ValueError('Helper source substitution differs')
    helper_source=text.replace('/*py_libhip_search_path*/',lib.replace('\\','\\\\'),1)
    config=dict(language=['c'],library_dirs=None,include_dirs=driver.include_dirs,libraries=None,ccflags=None)
    hk=identity.helper_key(helper_source.encode(),build.platform_key(),config)
    if hk!=expected_helper['computed_key'] or config!=expected_helper['config'] or lib!=expected_helper['library']:
        raise ValueError('Exact cached helper contract differs')
    meta=dict(runtime=runtime,triton=triton.__version__,N=2049,LOGN=11,grid=[264,2],kernel_key=key,
        jit_source_key=jit.cache_key,cache_environment=env,signature=sig,constants={str(k):v for k,v in const.items()},
        attrs={str(k):v for k,v in attrs.items()},options=opts.__dict__,helper=expected_helper,
        helper_source_sha256=hashlib.sha256(helper_source.encode()).hexdigest(),source_path=str(srcpath),source_sha256=digest(srcpath),
        gpu_initialized=False,weights_loaded=False,native_executed=False)
    return (meta,helper_source,jit) if return_leaf else (meta,helper_source)

def worker(scratch,*,constructor_tests=None):
    r=dict(status='UNKNOWN',gpu_initialized=None)
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job required')
        from amd_bounded_gpu_proof import scratch_environment,CACHE_KEYS
        from services.amd_offline_guard import block_external_actions
        env=scratch_environment(scratch);env['WORKSPACE_DIR']=str(ROOT)
        for k in CACHE_KEYS:
            if k!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[k]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env);block_external_actions(VENV/'Scripts/rocm-sdk.exe',[],scratch)
        def deny(*a,**k):raise RuntimeError('CPU admission forbids GPU/native execution/weights')
        def audit(e,a):
            if e=='import' and str(a[0]).split('.')[0] in ('flex_gemm','cumesh','o_voxel','trellis2','hip_utils'):deny()
            if e=='open' and isinstance(a[0],(str,bytes)) and Path(os.fsdecode(a[0])).suffix in ('.safetensors','.pt','.pth'):deny()
        sys.addaudithook(audit)
        import torch
        torch.cuda._lazy_init=deny;torch._C._cuda_init=deny;torch.cuda.is_available=deny;torch.cuda.get_device_properties=deny
        torch.load=deny;torch.compile=deny;torch.nn.Module.__call__=deny
        import triton.compiler.compiler as compiler
        import triton.runtime.build as build
        import triton.backends.amd.driver as driver
        compiler.compile=deny;build._build=deny;build._load_module_from_path=deny;driver.HIPUtils.__init__=deny
        r,source=inspect_installed(torch)
        f,co,w,b=contract.fixture();contract.validate_fixture(f,co,w,b)
        r['fixture']={k:dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in zip(('features','coordinates','weights','bias'),(f,co,w,b))}
        if constructor_tests is not None:
            r['constructor_acceptance']=constructor_tests(torch,r,source,scratch)
            if r['constructor_acceptance']['passed'] is not True:raise RuntimeError('Complete production constructor regression failed')
        r['status']='PASS';r['gpu_initialized']=torch.cuda.is_initialized()
    except BaseException as exc:r.update(status='UNKNOWN',error=repr(exc),traceback=traceback.format_exc())
    print('NATIVE_PREFLIGHT='+json.dumps(r),flush=True)
    return 0 if r['status']=='PASS' else 1

def supervise(*,worker_script=None,probe_name='cpu-probe'):
    from amd_forward_owned_job import OwnedJob
    from amd_bounded_gpu_proof import scratch_environment
    from amd_sparse_decoder_loading import reap_owned_child
    if probe_name not in ('cpu-probe','cpu-red','cpu-red-isolated','cpu-acceptance','cpu-acceptance-reviewed','cpu-final'):raise ValueError('Exclusive CPU probe name required')
    output=EV/probe_name;output.mkdir();scratch=output/'scratch';scratch.mkdir()
    job=OwnedJob();child=None;r={};start=time.monotonic()
    try:
        with (output/'stdout.txt').open('xb') as out,(output/'stderr.txt').open('xb') as err:
            child=subprocess.Popen([str(VENV/'Scripts/python.exe'),'-B','-I',str(worker_script or Path(__file__).resolve()),'--worker','--scratch',str(scratch)],stdin=subprocess.PIPE,stdout=out,stderr=err,env={**os.environ,**scratch_environment(scratch),'WORKSPACE_DIR':str(ROOT)})
            job.attach(child);child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush();child.stdin.close();child.wait(timeout=120)
            r.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-start)
    except BaseException as exc:r['error']=repr(exc)
    finally:
        r['job_cleanup']=job.finish()
        if child is not None:reap_owned_child(child,r)
    lines=(output/'stdout.txt').read_text().splitlines()
    rows=[json.loads(x.split('=',1)[1]) for x in lines if x.startswith('NATIVE_PREFLIGHT=')]
    if len(rows)==1:r['worker']=rows[0]
    r['passed']=bool(r.get('returncode')==0 and not r.get('error') and r.get('worker',{}).get('status')=='PASS' and r['worker'].get('gpu_initialized') is False and r.get('worker_exit_verified') and r['job_cleanup']['owned_tree_exit_verified'] and not r['job_cleanup']['forced_job_cleanup'])
    (output/'results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k!='worker'}));return 0 if r['passed'] else 1

if __name__=='__main__':
    p=__import__('argparse').ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path);a=p.parse_args()
    raise SystemExit(worker(a.scratch) if a.worker else supervise())
