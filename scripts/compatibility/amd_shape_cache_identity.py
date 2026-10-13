"""CPU-only recomputation of installed Triton and HIP-helper cache identities."""
import argparse
import ast
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_shape_decoder_preflight as previous
VENV,SITE=previous.VENV,previous.SITE
EV=ROOT/'docs/compatibility/amd-shape-decoder-cache-admission-evidence'
OLD=ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence'


def helper_key(source, platform, config):
    h=hashlib.sha256(source);h.update(platform.encode('utf-8'))
    for name, values in config.items():
        if values is None:continue
        h.update(name.encode('utf-8'))
        for value in values:h.update(value.encode('utf-8'));h.update(b':')
    return h.hexdigest()


def first_signature(*,splitk=2):
    # This proof attests only the canonical first dispatch, not other variants.
    if type(splitk) is not int or splitk!=2:raise ValueError('Only verified first-dispatch SPLITK2 is supported')
    names=['input','weight','bias','neighbor','sorted_idx','output','N','LOGN','Ci','Co','V','B1','B2','BK']
    if splitk!=1:names+=['SPLITK']
    names+=['allow_ieee','valid_kernel','valid_kernel_seg']
    values=['*fp16','*fp16','*fp16','*u32','*i64','*fp32' if splitk!=1 else '*fp16',
            'i32','i32','i32','i32']+['constexpr']*(6 if splitk!=1 else 5)+['*i32','*i32']
    sig=dict(zip(names,values));constants={k:v for k,v in zip(names[10:-2],[27,64,128,32]+([splitk] if splitk!=1 else [])+[True])}
    constants={(names.index(k),):v for k,v in constants.items()}
    attrs={(i,):[['tt.divisibility',16],['tt.pointer_range',32]] for i in [0,1,2,3,4,5,len(names)-2,len(names)-1]}
    attrs.update({(8,):[['tt.divisibility',16]],(9,):[['tt.divisibility',16]]})
    # Actual _pack_args constructs attrs in signature argument order.
    attrs={(i,):attrs.get((i,),[]) for i in range(len(names))}
    return sig,constants,attrs


class CacheOnly:
    """CPU contract primitive; does not patch or authorize a GPU execution path."""
    def __init__(self, verified):self.verified=dict(verified)
    def require(self,key):
        if not isinstance(key,str) or key not in self.verified:raise ValueError('Unverified native cache hit')
        return self.verified[key]
    @staticmethod
    def forbidden(action):raise RuntimeError('Cache-only policy forbids '+action)


def worker(scratch):
    r=dict(status='FAIL',gpu_initialized=False,weights_loaded=False,kernel_launches=0,helper_loaded=False,compiled=False,sdk_discovery=[])
    torch=None
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job required')
        scratch=scratch.resolve();scratch.relative_to(EV.resolve())
        from amd_bounded_gpu_proof import scratch_environment,CACHE_KEYS
        from services.amd_offline_guard import block_external_actions
        env=scratch_environment(scratch)
        for key in CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        env['WORKSPACE_DIR']=str(ROOT);os.environ.update(env)
        block_external_actions(VENV/'Scripts/rocm-sdk.exe',r['sdk_discovery'],scratch)
        def deny(*a,**k):raise RuntimeError('CPU cache audit denies GPU, weights, helper execution, compilation, tuning and model calls')
        def audit(event,args):
            if event.startswith('socket.'):return deny()
            if event=='import' and str(args[0]).split('.')[0] in ('flex_gemm','cumesh','o_voxel','trellis2'):return deny()
            if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors','.pt','.pth','.ckpt','.gguf'):return deny()
        sys.addaudithook(audit)
        import torch
        if torch.cuda.is_initialized():return deny()
        if sys.version.split()[0]!='3.12.10' or str(torch.__version__)!='2.13.0+rocm10.0.0' or torch.version.cuda is not None or not torch.version.hip:raise ValueError('Pinned runtime changed')
        torch.cuda._lazy_init=deny;torch.cuda.is_available=deny;torch.cuda.get_device_properties=deny
        torch._C._cuda_init=deny;torch.load=deny;torch.compile=deny;torch.nn.Module.__call__=deny
        import triton
        import triton.runtime.build as build
        import triton.compiler.compiler as compiler
        import triton.backends.amd.driver as driver
        from triton.backends.amd.compiler import HIPBackend
        from triton.backends.compiler import GPUTarget
        from triton.runtime.cache import get_cache_key
        triton.compile=deny;compiler.compile=deny;build._build=deny
        triton.runtime.JITFunction.run=deny;triton.runtime.Autotuner.run=deny;driver.HIPUtils.__init__=deny
        import rocm_sdk
        rocm_sdk.initialize_process=deny;rocm_sdk.preload_libraries=deny
        lib=driver._get_path_to_hip_runtime_dylib()
        source=(SITE/'triton/backends/amd/driver.c').read_text().replace('/*py_libhip_search_path*/',lib.replace('\\','\\\\'),1).encode()
        config=dict(language=['c'],library_dirs=None,include_dirs=driver.include_dirs,libraries=None,ccflags=None)
        key=helper_key(source,build.platform_key(),config)
        # Cross-check our small contract against the exact installed hash function.
        tree=ast.parse((SITE/'triton/runtime/build.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_cache_manager')
        namespace=dict(hashlib=hashlib,platform_key=build.platform_key,get_cache_manager=lambda key:key)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SITE/'triton/runtime/build.py'),'exec'),namespace)
        if key!=namespace['_get_cache_manager'](source,config):raise ValueError('Helper hash formula mismatch')
        folder=base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        original=Path('C:/Users/wattz/.triton/cache')/folder/'hip_utils.cp312-win_amd64.pyd'
        r['helper']=dict(library=lib,library_sha256=previous.digest(lib),platform=build.platform_key(),config=config,
                         computed_key=key,computed_folder=folder,expected_path=str(original),cache_hit=original.is_file(),
                         cached_sha256=previous.digest(original) if original.is_file() else None,installed_formula_matches=True)
        prefix=SITE/'flex_gemm/kernels/triton/spconv/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py'
        tree=ast.parse(prefix.read_text(encoding='utf-8'))
        name='sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel'
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        first_line=fn.decorator_list[0].lineno;fn.decorator_list=[]
        import triton.language as tl
        namespace=dict(__name__='flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk',tl=tl)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(prefix),'exec'),namespace)
        raw=namespace[name]
        # Reconstruct original source-location metadata, not kernel arithmetic.
        raw.__code__=raw.__code__.replace(co_firstlineno=first_line)
        jit=triton.jit(raw)
        signature,constants,attrs=first_signature()
        from triton.runtime.jit import create_function_from_signature
        class MetadataPointer:
            def __init__(self,dtype):self.dtype=dtype
            def data_ptr(self):return 16
            def ptr_range(self):return 1024*1024
        types=[torch.float16,torch.float16,torch.float16,torch.uint32,torch.int64,torch.float32]
        values=[MetadataPointer(t) for t in types]+[3743,11,1024,1024,27,64,128,32,2,True,MetadataPointer(torch.int32),MetadataPointer(torch.int32)]
        backend=HIPBackend(GPUTarget('hip','gfx1201',32))
        binder=create_function_from_signature(jit.signature,jit.params,backend)
        bound,specialization,raw_options=binder(*values,num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3,debug=False,instrumentation_mode='')
        packed_options,packed_sig,packed_constants,packed_attrs=jit._pack_args(backend,dict(num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3,debug=False,instrumentation_mode=''),bound,specialization,None)
        r['binder_crosscheck']=dict(signature=packed_sig,constants={str(k):v for k,v in packed_constants.items()},attrs={str(k):v for k,v in packed_attrs.items()},
                                    source='Installed create_function_from_signature and JITFunction._pack_args; metadata pointers only, no storage/GPU')
        if (packed_sig,packed_constants,packed_attrs)!=(signature,constants,attrs):raise ValueError('Actual installed binder differs from reconstructed signature')
        src=compiler.ASTSource(jit,signature,constexprs=constants,attrs=attrs)
        options=backend.parse_options(dict(num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3,debug=False,instrumentation_mode=''))
        env_vars=compiler.get_cache_invalidating_env_vars()
        key=hashlib.sha256(get_cache_key(src,backend,options,env_vars).encode()).hexdigest()
        folder=base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        expected=Path('C:/Users/wattz/.triton/cache')/folder/(name+'.json')
        r['first_kernel']=dict(source_path=str(prefix),source_sha256=previous.digest(prefix),starting_line_number=jit.starting_line_number,
            jit_source_key=jit.cache_key,ast_source_key=src.hash(),backend_key=backend.hash(),options=options.__dict__,options_key=options.hash(),
            cache_environment=env_vars,computed_hash=key,computed_folder=folder,expected_metadata=str(expected),cache_hit=expected.is_file(),
            historical_hash='d5de3d60ef4c1dfce927f5a59a91fdfa35c7de3400fb71acb214abe5dc4fbea5',
            matches_historical=key=='d5de3d60ef4c1dfce927f5a59a91fdfa35c7de3400fb71acb214abe5dc4fbea5',
            signature=signature,constants={str(k):v for k,v in constants.items()},attrs={str(k):v for k,v in attrs.items()},
            limitation='Original kernel source/location reconstructed for hashing only; actual installed binder cross-checked using metadata pointers. This is not kernel execution or a full runtime admission policy.')
        r.update(status='PASS',gpu_initialized=torch.cuda.is_initialized(),harness_sha256=previous.digest(Path(__file__)),
                 runtime=dict(python=sys.version,torch=str(torch.__version__),hip=torch.version.hip,triton=triton.__version__))
        if r['gpu_initialized']:return deny()
    except BaseException as exc:r.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    print('CACHE_IDENTITY='+json.dumps(r),flush=True)
    return 0 if r['status']=='PASS' else 1


def supervise(name):
    from amd_forward_owned_job import OwnedJob
    from amd_bounded_gpu_proof import scratch_environment
    if name not in ('cpu-probe','cpu-probe-2','cpu-probe-3','cpu-probe-4','cpu-probe-5'):raise ValueError('Bounded exclusive CPU probe name required')
    output=EV/name;output.mkdir();scratch=output/'scratch';scratch.mkdir();job=OwnedJob();child=None;r={};start=time.monotonic()
    try:
        with (output/'stdout.txt').open('xb') as out,(output/'stderr.txt').open('xb') as err:
            env={**os.environ,**scratch_environment(scratch),'WORKSPACE_DIR':str(ROOT)}
            child=subprocess.Popen([str(VENV/'Scripts/python.exe'),'-B','-I',str(Path(__file__).resolve()),'--worker','--scratch',str(scratch)],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child);child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush();child.stdin.close();child.wait(timeout=120)
            r.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-start)
    except BaseException as exc:r.update(error=repr(exc))
    finally:
        r['job_cleanup']=job.finish()
        if child is not None:
            from amd_sparse_decoder_loading import reap_owned_child
            reap_owned_child(child,r)
    rows=[json.loads(s.split('=',1)[1]) for s in (output/'stdout.txt').read_text(encoding='utf-8').splitlines() if s.startswith('CACHE_IDENTITY=')]
    if len(rows)==1:r['worker']=rows[0]
    w=r.get('worker',{})
    passed=(not r.get('error') and r.get('returncode')==0 and w.get('status')=='PASS' and not w.get('error')
            and w.get('gpu_initialized') is False and w.get('weights_loaded') is False and w.get('kernel_launches')==0
            and w.get('helper_loaded') is False and w.get('compiled') is False and r.get('worker_exit_verified') is True
            and r['job_cleanup']['owned_tree_exit_verified'] and not r['job_cleanup']['forced_job_cleanup'])
    r['cpu_probe_passed']=bool(passed)
    with (output/'results.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!='worker'}))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path);p.add_argument('--name',default='cpu-probe')
    a=p.parse_args()
    if a.worker and a.scratch is None:p.error('Owned scratch required')
    raise SystemExit(worker(a.scratch) if a.worker else supervise(a.name))
