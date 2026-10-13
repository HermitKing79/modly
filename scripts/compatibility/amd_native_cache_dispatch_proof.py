"""One CPU-only installed compile/cache entry-point proof, not decoder integration."""
import argparse
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
EV=ROOT/'docs/compatibility/amd-native-cache-dispatch-proof-evidence'
OLD=ROOT/'docs/compatibility/amd-shape-decoder-cache-admission-evidence'
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_shape_cache_identity as identity
from amd_shape_decoder_preflight import SITE,VENV,digest,cache_group_paths
HIT_KEY='d5de3d60ef4c1dfce927f5a59a91fdfa35c7de3400fb71acb214abe5dc4fbea5'
HSACO_SHA='e639008883feef1e37079d2a809a45cdfef85240aaec7ae9f018d8175cf7fe01'
COUNTERS=('compilation_pipeline','native_build','helper_load','gpu_handle_init','jit_run','gpu_init')


def result_passed(record):
    w=record.get('worker',{});calls=w.get('forbidden_calls',{})
    runtime=w.get('runtime',{});hit=w.get('hit',{});miss=w.get('miss',{})
    return (not record.get('error') and record.get('returncode')==0 and record.get('worker_exit_verified') is True
        and record.get('job_cleanup',{}).get('owned_tree_exit_verified') is True
        and record['job_cleanup'].get('forced_job_cleanup') is False
        and w.get('status')=='PASS' and not w.get('error') and w.get('source_identity_verified') is True
        and w.get('gpu_initialized') is False and w.get('weights_loaded') is False
        and all(type(w.get(k)) is int and w[k]==n for k,n in (('installed_compile_calls',2),('installed_constructor_calls',1),('installed_group_calls',1)))
        and set(calls)==set(COUNTERS) and all(type(calls[k]) is int and calls[k]==0 for k in COUNTERS)
        and miss.get('missing_from_historical_and_private_cache') is True and miss.get('rejected') is True
        and isinstance(miss.get('hash'),str) and len(miss['hash'])==64 and miss['hash']!=HIT_KEY
        and hit.get('hash')==HIT_KEY and hit.get('kernel_sha256')==HSACO_SHA
        and all(k in hit and hit[k] is None for k in ('module','function','run_handle'))
        and runtime.get('python')=='3.12.10' and runtime.get('torch')=='2.13.0+rocm10.0.0'
        and 'cuda' in runtime and runtime['cuda'] is None and runtime.get('hip')=='7.15.26333')


class UnverifiedCacheMiss(RuntimeError):
    pass


def normalize_cache_mkdir(path,approved):
    raw=os.fsdecode(path)
    if not raw.startswith('\\\\?\\'):return path
    if len(raw)<7 or not raw[4].isalpha() or raw[5:7]!=':\\':raise ValueError('Foreign extended-path form')
    canonical=Path(raw[4:]).resolve();approved=Path(approved).resolve()
    if canonical!=approved or not approved.is_dir() or not os.path.samefile(raw,approved):
        raise ValueError('Only exact pre-created private cache-folder alias is admitted')
    if len(str(canonical))>=248:raise ValueError('Private cache-folder path too long')
    return str(canonical)


def emit(phase,**values):
    print('DISPATCH_EVENT='+json.dumps(dict(phase=phase,**values)),flush=True)


def private_first_group(scratch):
    """Copy one existing attested container; use the installed cache manager later."""
    audit=json.loads((OLD/'native-cache-audit.json').read_text(encoding='utf-8'))
    previous=json.loads((ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/preflight.json').read_text(encoding='utf-8'))
    for name,h in {**previous['source_sha256'],**previous['native_source_and_binary_sha256']}.items():
        if digest(name)!=h:raise ValueError('Pinned source/native file changed: '+name)
    for name,row in audit['support_file_identity'].items():
        if digest(name)!=row['sha256']:raise ValueError('Pinned compiler/support file changed: '+name)
    if digest(previous['tuning']['path'])!=previous['tuning']['sha256']:raise ValueError('Tuning inventory changed')
    row=next(r for r in previous['cache_metadata'] if r['metadata']['hash']==HIT_KEY)
    metadata=Path(row['path']);group=metadata.parent/('__grp__'+metadata.name)
    if digest(group)!=row['group_sha256']:raise ValueError('Historical group changed')
    children=cache_group_paths(metadata,json.loads(group.read_text())['child_paths'])
    folder=base64.b32encode(bytes.fromhex(HIT_KEY)).decode().rstrip('=')
    destination=scratch/'t'/folder;destination.mkdir(parents=True)
    copied={}
    for name,p in children.items():
        expected=row['files'][name]
        if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:raise ValueError('Historical product changed')
        target=destination/name;shutil.copyfile(p,target)
        if digest(target)!=expected['sha256']:raise ValueError('Private copy differs')
        copied[name]=str(target)
    (destination/group.name).write_text(json.dumps(dict(child_paths=copied)),encoding='utf-8')
    return metadata.name,copied,row['files'],previous


def worker(scratch):
    r=dict(status='UNKNOWN',gpu_initialized=None,weights_loaded=False,source_identity_verified=False,
        forbidden_calls={k:0 for k in COUNTERS},installed_compile_calls=0,installed_constructor_calls=0,installed_group_calls=0)
    torch=None;started=time.monotonic();original_mkdir=os.mkdir
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job attachment required')
        scratch=scratch.resolve();scratch.relative_to(EV.resolve())
        if scratch!=EV.resolve()/'cpu-proof/scratch':raise ValueError('Exclusive scratch required')
        from amd_bounded_gpu_proof import scratch_environment,CACHE_KEYS
        from services.amd_offline_guard import block_external_actions
        env=scratch_environment(scratch)
        # Cache root used below is fixed, fresh and writable; historical roots stay read-only.
        env['TRITON_CACHE_DIR']=str(scratch/'t');env['WORKSPACE_DIR']=str(ROOT)
        for k in CACHE_KEYS:
            if k!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[k]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env)
        discovery=[];block_external_actions(VENV/'Scripts/rocm-sdk.exe',discovery,scratch)
        def forbidden(name):
            def deny(*a,**k):
                r['forbidden_calls'][name]+=1
                raise RuntimeError('CPU-only proof forbids '+name)
            return deny
        def audit(event,args):
            if event.startswith('socket.'):raise RuntimeError('Offline CPU proof')
            if event=='import' and str(args[0]).split('.')[0] in ('flex_gemm','cumesh','o_voxel','trellis2','hip_utils'):
                raise RuntimeError('CPU proof forbids native/model package import: '+str(args[0]))
            if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors','.pt','.pth','.ckpt','.gguf'):
                raise RuntimeError('CPU proof forbids checkpoint access')
        sys.addaudithook(audit)
        metadata_name,private,expected,previous=private_first_group(scratch)
        r['source_identity_verified']=True
        import torch
        r['gpu_initialized']=torch.cuda.is_initialized()
        if r['gpu_initialized']:raise RuntimeError('GPU already initialized')
        r['runtime']=dict(python=sys.version.split()[0],torch=str(torch.__version__),hip=torch.version.hip,cuda=torch.version.cuda)
        if r['runtime']!=dict(python='3.12.10',torch='2.13.0+rocm10.0.0',hip='7.15.26333',cuda=None):raise RuntimeError('Pinned runtime changed')
        torch.cuda._lazy_init=forbidden('gpu_init');torch._C._cuda_init=forbidden('gpu_init')
        torch.cuda.is_available=forbidden('gpu_init');torch.cuda.get_device_properties=forbidden('gpu_init')
        torch.load=forbidden('native_build');torch.compile=forbidden('compilation_pipeline')
        torch.nn.Module.__call__=forbidden('jit_run')
        import triton
        from triton import knobs
        import triton.compiler.compiler as compiler
        import triton.runtime.build as build
        import triton.backends.amd.driver as driver
        from triton.backends.amd.compiler import HIPBackend
        from triton.backends.compiler import GPUTarget
        from triton.runtime.cache import get_cache_key,FileCacheManager
        # These are deny-only backstops, not alternate numerical implementations.
        HIPBackend.add_stages=forbidden('compilation_pipeline')
        compiler.ASTSource.make_ir=forbidden('compilation_pipeline')
        compiler.CompiledKernel._init_handles=forbidden('gpu_handle_init')
        triton.runtime.JITFunction.run=forbidden('jit_run');triton.runtime.JITFunction.warmup=forbidden('jit_run')
        triton.runtime.Autotuner.run=forbidden('jit_run')
        build._build=forbidden('native_build');build._load_module_from_path=forbidden('helper_load')
        build.compile_module_from_src=forbidden('native_build');build.compile_module_from_file=forbidden('native_build')
        driver.compile_module_from_src=forbidden('native_build');driver.HIPUtils.__init__=forbidden('helper_load')
        import rocm_sdk
        rocm_sdk.initialize_process=forbidden('gpu_init');rocm_sdk.preload_libraries=forbidden('helper_load')
        if knobs.cache.manager_class is not None:raise RuntimeError('Foreign cache manager')
        if any(getattr(knobs.compilation,k) for k in ('always_compile','override','dump_ir','store_binary_only')):raise RuntimeError('Foreign compile flags')
        if knobs.compilation.listener is not None or knobs.runtime.add_stages_inspection_hook is not None:raise RuntimeError('Foreign compile hooks')
        # Explicit prior-verified empty env metadata avoids calling libtriton's
        # native environment getter. This proof covers this ASTSource interface.
        compiler.get_cache_invalidating_env_vars=forbidden('compilation_pipeline')
        source=SITE/'flex_gemm/kernels/triton/spconv/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py'
        fn=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel')
        line=fn.decorator_list[0].lineno;fn.decorator_list=[]
        import triton.language as tl
        namespace=dict(__name__='flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk',tl=tl)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        raw=namespace[fn.name];raw.__code__=raw.__code__.replace(co_firstlineno=line)
        jit=triton.jit(raw);signature,constants,attrs=identity.first_signature()
        src=compiler.ASTSource(jit,signature,constexprs=constants,attrs=attrs)
        target=GPUTarget('hip','gfx1201',32);backend=HIPBackend(target)
        options=dict(num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3,debug=False,instrumentation_mode='')
        hit_key=hashlib.sha256(get_cache_key(src,backend,backend.parse_options(options),{}).encode()).hexdigest()
        if hit_key!=HIT_KEY:raise ValueError('Actual current first-dispatch key changed')
        miss_options=dict(options,debug=True)
        miss_key=hashlib.sha256(get_cache_key(src,backend,backend.parse_options(miss_options),{}).encode()).hexdigest()
        missing_folder=base64.b32encode(bytes.fromhex(miss_key)).decode().rstrip('=')
        historical=Path('C:/Users/wattz/.triton/cache')/missing_folder
        private_miss=scratch/'t'/missing_folder
        if miss_key in {x['metadata']['hash'] for x in previous['cache_metadata']} or historical.exists() or private_miss.exists():raise ValueError('Deliberate metadata miss is not missing')
        original_compile=compiler.compile;original_lookup=compiler.get_cache_manager
        original_group=FileCacheManager.get_group
        original_constructor=compiler.CompiledKernel.__init__
        r['entrypoint']=dict(module=original_compile.__module__,file=original_compile.__code__.co_filename,
            source_sha256=digest(original_compile.__code__.co_filename),cache_lookup='triton.compiler.compiler.get_cache_manager',
            unchanged_compile_function=True,env_metadata={},scope='Explicit ASTSource/target/options, not JITFunction.run or full FlexGEMM dispatch')
        r['lookup_keys']=[]
        approved_dir=Path(next(iter(private.values()))).parent
        r['windows_alias_bridge_calls']=0
        def owned_cache_mkdir(path,mode=0o777,*,dir_fd=None):
            if dir_fd is not None:raise ValueError('Directory-descriptor mkdir is not admitted')
            normalized=normalize_cache_mkdir(path,approved_dir)
            if os.fsdecode(path).startswith('\\\\?\\'):r['windows_alias_bridge_calls']+=1
            # Existing audit hooks still authorize this exact physical path.
            return original_mkdir(normalized,mode)
        os.mkdir=owned_cache_mkdir
        def verify_group(group):
            if not isinstance(group,dict) or set(group)!=set(private):raise UnverifiedCacheMiss('Incomplete private group')
            for name,path in group.items():
                p=Path(path).resolve()
                if p!=Path(private[name]).resolve() or p.stat().st_size!=expected[name]['bytes'] or digest(p)!=expected[name]['sha256']:
                    raise UnverifiedCacheMiss('Foreign/changed private product')
        def checked_lookup(key):
            r['lookup_keys'].append(key)
            if key!=HIT_KEY:raise UnverifiedCacheMiss('Unverified cache key: '+key)
            manager=original_lookup(key)
            if type(manager) is not FileCacheManager:raise RuntimeError('Installed local manager required')
            get_group=manager.get_group
            def checked_group(name):
                if name!=metadata_name:raise UnverifiedCacheMiss('Foreign metadata name')
                group=get_group(name);verify_group(group);return group
            manager.get_group=checked_group
            manager.put=forbidden('compilation_pipeline');manager.put_group=forbidden('compilation_pipeline')
            return manager
        compiler.get_cache_manager=checked_lookup
        watched={original_compile.__code__:'installed_compile_calls',original_group.__code__:'installed_group_calls',original_constructor.__code__:'installed_constructor_calls'}
        def profile(frame,event,arg):
            if event=='call' and frame.f_code in watched:r[watched[frame.f_code]]+=1
        sys.setprofile(profile)
        emit('installed-miss-entry',key=miss_key)
        try:
            original_compile(src,target=target,options=miss_options,_env_vars={})
        except UnverifiedCacheMiss as exc:
            r['miss']=dict(hash=miss_key,rejected=True,exception=repr(exc),missing_from_historical_and_private_cache=True,
                missing_directory_created=private_miss.exists(),only_changed_option='debug=True')
        else:raise RuntimeError('Installed miss unexpectedly returned')
        if private_miss.exists():raise RuntimeError('Miss created a cache directory')
        emit('installed-miss-rejected',**r['miss'])
        emit('installed-hit-entry',key=HIT_KEY)
        result=original_compile(src,target=target,options=options,_env_vars={})
        sys.setprofile(None)
        r['hit']=dict(hash=result.hash,kernel_sha256=hashlib.sha256(result.kernel).hexdigest(),kernel_bytes=len(result.kernel),
            module=result.module,function=result.function,run_handle=result._run,
            result_class=type(result).__module__+'.'+type(result).__name__,private_products_verified=len(private))
        if any(x is not None for x in (result.module,result.function,result._run)):raise RuntimeError('GPU handles unexpectedly loaded')
        emit('installed-hit-return',**r['hit'])
        result=None
        r.update(status='PASS',gpu_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
            harness_sha256=digest(Path(__file__)),sdk_discovery=discovery,
            protected_deny_only=['HIPBackend.add_stages','ASTSource.make_ir','CompiledKernel._init_handles','JITFunction.run/warmup',
                'Autotuner.run','build._build/_load_module_from_path/compile_module_from_src/compile_module_from_file',
                'amd.driver.compile_module_from_src/HIPUtils.__init__','PyTorch GPU initialization and model calls'],
            not_exercised=['FlexGEMM PersistentCacheAutoTuner.__call__/_benchmark','FlexGEMM TritonPersistentCacheAutotuner.run/_bench',
                'HIP-helper build/loadability','JITFunction._do_compile/run and native neighbor processing'],
            flex_policy='FlexGEMM package import prohibited; custom tuner entry points located/read, not patched or exercised. No benchmark capability admitted.',
            native_code_policy='Host PyTorch/Triton extension modules imported for metadata; no compiler passes, cached helper/HSACO execution or GPU handles.')
    except BaseException as exc:r.update(status='UNKNOWN',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
    finally:
        sys.setprofile(None)
        os.mkdir=original_mkdir
    print('NATIVE_DISPATCH='+json.dumps(r),flush=True)
    return 0 if r['status']=='PASS' else 1


def supervise():
    from amd_forward_owned_job import OwnedJob
    from amd_bounded_gpu_proof import scratch_environment
    output=EV/'cpu-proof';output.mkdir();scratch=output/'scratch';scratch.mkdir()
    job=OwnedJob();child=None;r={};started=time.monotonic()
    with (output/'worker-started.json').open('x',encoding='utf-8') as f:json.dump(dict(cpu_only=True,maximum_seconds=120,harness_sha256=digest(Path(__file__))),f)
    try:
        with (output/'stdout.txt').open('xb') as out,(output/'stderr.txt').open('xb') as err:
            env={**os.environ,**scratch_environment(scratch),'WORKSPACE_DIR':str(ROOT)}
            child=subprocess.Popen([str(VENV/'Scripts/python.exe'),'-B','-I',str(Path(__file__).resolve()),'--worker','--scratch',str(scratch)],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child);child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush();child.stdin.close()
            remaining=120-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('CPU worker deadline elapsed')
            child.wait(timeout=remaining);r.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:r.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        r['job_cleanup']=job.finish()
        if child is not None:
            from amd_sparse_decoder_loading import reap_owned_child
            reap_owned_child(child,r)
    rows=[json.loads(s.split('=',1)[1]) for s in (output/'stdout.txt').read_text(encoding='utf-8').splitlines() if s.startswith('NATIVE_DISPATCH=')]
    if len(rows)==1:r['worker']=rows[0]
    passed=result_passed(r) and r['worker'].get('harness_sha256')==digest(Path(__file__))
    r.update(narrow_entrypoint_proof_passed=bool(passed),classification='PASS' if passed else 'UNKNOWN',full_decoder_recommendation='NO-GO')
    with (output/'results.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!='worker'}))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path)
    a=p.parse_args()
    if a.worker and a.scratch is None:p.error('Owned scratch required')
    raise SystemExit(worker(a.scratch) if a.worker else supervise())
