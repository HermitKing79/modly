"""Deny misses at genuine installed dispatch seams for exactly one convolution."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import numpy as np
import amd_native_convolution_contract as c
import amd_native_convolution_preflight as p
from amd_native_cache_dispatch_proof import normalize_cache_mkdir

HOOKS_SHA='7468fb1fb4740bb9237c57216b04499400825f62a6177da4214153900feda824'
CHAIN_DEFAULTS=dict(launch_enter_hook=False,launch_exit_hook=True,kernel_load_start_hook=False,kernel_load_end_hook=True,kernel_unload_hook=False)
_INSTALLED_DEFAULTS=None

def pin_hook_defaults(knobs):
    global _INSTALLED_DEFAULTS
    path=Path(knobs.__file__).resolve()
    if path!=(p.SITE/'triton/knobs.py').resolve() or p.digest(path)!=HOOKS_SHA:
        raise ValueError('Installed Triton hook source identity changed')
    defaults={name:knobs.runtime_knobs.__dict__[name] for name in CHAIN_DEFAULTS}
    if _INSTALLED_DEFAULTS is None:_INSTALLED_DEFAULTS=defaults
    if any(defaults[name] is not _INSTALLED_DEFAULTS[name] for name in CHAIN_DEFAULTS):
        raise ValueError('Installed native hook defaults replaced')

def validate_hook_state(knobs):
    if _INSTALLED_DEFAULTS is None:raise ValueError('Source-verified installed hook defaults must be pinned first')
    pin_hook_defaults(knobs)
    path=Path(knobs.__file__).resolve()
    states={}
    for name,reverse in CHAIN_DEFAULTS.items():
        default=_INSTALLED_DEFAULTS[name];value=getattr(knobs.runtime,name)
        if (type(default) is not knobs.HookChain or set(vars(default))!={'calls','reversed'}
            or type(default.calls) is not list or len(default.calls)!=0
            or type(default.reversed) is not bool or default.reversed is not reverse):
            raise ValueError('Foreign/nonempty default native hook: '+name)
        # None is the already-supported disabled state. Otherwise require the
        # genuine installed class-default instance, never an arbitrary chain.
        if value is not None and value is not default:raise ValueError('Foreign native hook: '+name)
        states[name]=dict(state='None' if value is None else 'installed-empty-HookChain',callbacks=0,reversed=reverse)
    for obj,names in ((knobs.runtime,('add_stages_inspection_hook','jit_cache_hook','jit_post_compile_hook')),(knobs.compilation,('listener',))):
        for name in names:
            if getattr(obj,name) is not None:raise ValueError('Foreign native callback: '+name)
            states[name]=dict(state='None')
    return dict(source_path=str(path),source_sha256=HOOKS_SHA,hooks=states)

class NativePolicy:
    def __init__(self,torch,private,meta,source,trace):
        self.torch=torch;self.private=private;self.meta=meta;self.trace=trace
        self.permit=c.Permit();self.leaf=self.outer=self.inner=None;self.restores=[]
        self.helper_calls=0;self.lookups=[];self.forbidden_calls=[]
        import triton
        import triton.compiler.compiler as compiler
        import triton.runtime.build as build
        import triton.backends.amd.driver as driver
        from triton.backends.amd.compiler import HIPBackend
        from triton.runtime.cache import FileCacheManager
        from triton import knobs
        if knobs.cache.manager_class is not None or any(getattr(knobs.compilation,k) for k in ('always_compile','override','dump_ir','store_binary_only')):
            raise ValueError('Foreign compilation/cache policy')
        self.hook_state=validate_hook_state(knobs)
        self.patch(HIPBackend,'add_stages',self.deny('lowering'))
        self.patch(compiler.ASTSource,'make_ir',self.deny('IR construction'))
        original_loader=build._load_module_from_path
        checked_helper=c.helper_loader(source,driver.include_dirs,Path(private['helper_path']),private['helper']['cached_sha256'],p.digest,original_loader)
        def helper(*a,**kw):
            if self.permit.phase!='convolution':raise RuntimeError('Helper outside selected operation')
            self.helper_calls+=1
            return trace.call('cached-HIP-helper-load',lambda:checked_helper(*a,**kw))
        self.patch(driver,'compile_module_from_src',helper)
        for name in ('_build','_load_module_from_path','compile_module_from_src','compile_module_from_file'):
            self.patch(build,name,self.deny(name))
        original_lookup=compiler.get_cache_manager
        def lookup(key):
            if self.permit.phase!='convolution' or key!=p.HIT_KEY or self.lookups:raise RuntimeError('Unverified/duplicate kernel cache request: '+str(key))
            self.lookups.append(key);manager=original_lookup(key)
            if type(manager) is not FileCacheManager:raise RuntimeError('Foreign cache manager')
            original=manager.get_group
            def group(name):
                if name!=private['metadata_name']:raise RuntimeError('Unexpected cache metadata')
                rows=original(name)
                if not isinstance(rows,dict) or set(rows)!=set(private['paths']):raise RuntimeError('Incomplete cached container')
                for n,path in rows.items():
                    expected=private['files'][n]
                    if Path(path).resolve()!=Path(private['paths'][n]).resolve() or Path(path).stat().st_size!=expected['bytes'] or p.digest(path)!=expected['sha256']:
                        raise RuntimeError('Cached product integrity failure')
                return rows
            manager.get_group=group;manager.put=self.deny('cache put');manager.put_group=self.deny('cache put_group')
            return manager
        self.patch(compiler,'get_cache_manager',lookup)
        original_compile=compiler.compile
        def compile_checked(src,*a,**kw):
            if self.permit.phase!='convolution' or src.fn is not self.leaf:raise RuntimeError('Unexpected compile source')
            sig,const,attrs=__import__('amd_shape_cache_identity').first_signature()
            if src.signature!=sig or src.constants!=const or src.attrs!=attrs:raise RuntimeError('Actual GPU specialization differs from CPU admission')
            return original_compile(src,*a,**kw)
        self.patch(compiler,'compile',compile_checked)
        self.patch(triton,'compile',compile_checked)
        self.patch(importlib.import_module('triton.compiler'),'compile',compile_checked)
        original_init=compiler.CompiledKernel._init_handles
        def handles(kernel):
            self.permit.take('binary')
            if kernel.hash!=p.HIT_KEY or hashlib.sha256(kernel.kernel).hexdigest()!=p.HSACO_SHA or kernel.metadata.target.arch!='gfx1201':raise RuntimeError('Unverified native kernel')
            if kernel.metadata.global_scratch_size or kernel.metadata.profile_scratch_size or kernel.metadata.launch_cooperative_grid:raise RuntimeError('Unexpected native scratch/launch mode')
            return trace.call('cached-HSACO-load',lambda:original_init(kernel))
        self.patch(compiler.CompiledKernel,'_init_handles',handles)
        original_launch=driver.HIPLauncher.__call__
        def launch(obj,*args,**kw):
            self.permit.take('launch')
            if args[:3]!=(264,2,1) or kw:raise RuntimeError('Unexpected kernel launch grid')
            return trace.call('selected-HIP-launch',lambda:original_launch(obj,*args,**kw))
        self.patch(driver.HIPLauncher,'__call__',launch)
        original_run=triton.runtime.JITFunction.run
        def jit_run(obj,*args,grid,warmup,**kw):
            self.permit.take('jit')
            if obj is not self.leaf or obj.cache_key!=meta['jit_source_key'] or warmup is not False or obj.pre_run_hooks:
                raise RuntimeError('Unexpected JIT function/warmup/hooks')
            scalar=dict(N=2049,LOGN=11,Ci=1024,Co=1024,V=27,B1=64,B2=128,BK=32,SPLITK=2,allow_ieee=True)
            bound=dict(zip(obj.arg_names,args));bound.update(kw)
            if any(type(bound.get(k)) is not type(v) or bound[k]!=v for k,v in scalar.items()):raise RuntimeError('Unexpected JIT scalar specialization')
            opts=dict(num_warps=4,num_ctas=1,num_stages=2,waves_per_eu=3)
            if set(kw)-set(obj.arg_names)-set(opts) or any(kw.get(k)!=v for k,v in opts.items()):raise RuntimeError('Unexpected JIT options')
            expected={'input':((2049,1024),torch.float16),'weight':((1024,27,1024),torch.float16),'bias':((1024,),torch.float16),
                      'neighbor':((2049,27),torch.uint32),'sorted_idx':((2049,),torch.int64),'output':((2,2049,1024),torch.float32)}
            for name,(shape,dtype) in expected.items():self.tensor(bound[name],shape,dtype)
            for name in ('valid_kernel','valid_kernel_seg'):
                tensor=bound[name]
                self.tensor(tensor,tuple(tensor.shape),torch.int32)
                if tensor.ndim!=1 or tensor.numel()>891:raise RuntimeError('Unexpected kernel index allocation')
            actual_grid=grid(bound) if callable(grid) else grid
            if actual_grid!=(264,2):raise RuntimeError('Unexpected selected grid')
            return original_run(obj,*args,grid=grid,warmup=warmup,**kw)
        self.patch(triton.runtime.JITFunction,'run',jit_run)
        self.patch(triton.runtime.JITFunction,'warmup',self.deny('JIT warmup'))
        self.patch(triton.runtime.Autotuner,'run',self.deny('unselected autotuner'))
        original_mkdir=os.mkdir;approved=Path(next(iter(private['paths'].values()))).parent
        def mkdir(path,mode=0o777,*,dir_fd=None):
            if dir_fd is not None:raise RuntimeError('Unexpected mkdir dir_fd')
            return original_mkdir(normalize_cache_mkdir(path,approved),mode)
        self.patch(os,'mkdir',mkdir)

    def patch(self,obj,name,replacement):
        self.restores.append((obj,name,getattr(obj,name)));setattr(obj,name,replacement)
    def deny(self,name):
        def reject(*a,**kw):
            self.forbidden_calls.append(name);raise RuntimeError('Native smoke forbids '+name)
        return reject
    def tensor(self,t,shape,dtype):
        if tuple(t.shape)!=shape or t.dtype!=dtype or t.device!=self.torch.device('cuda:0') or not t.is_contiguous() or t.requires_grad:
            raise RuntimeError('Unexpected native tensor contract')

    def attach_flex(self):
        torch=self.torch
        tuner=importlib.import_module('flex_gemm.utils.autotuner')
        mod=importlib.import_module(p.PREFIX)
        self.outer=mod.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk
        self.inner=mod.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel
        leaf=self.inner
        while not isinstance(leaf,__import__('triton').runtime.JITFunction):leaf=leaf.fn
        self.leaf=leaf
        if leaf.device_caches:raise RuntimeError('Kernel already has device cache')
        c.require_tuning(self.outer.cache,self.inner.cache)
        original_outer=tuner.PersistentCacheAutoTuner.__call__
        def outer(obj,*a,**kw):
            self.permit.take('outer')
            if obj is not self.outer or kw or obj.key_fn(*a,**kw)!=c.OUTER_KEY:raise RuntimeError('Foreign outer tuning call')
            c.require_tuning(obj.cache,self.inner.cache)
            return original_outer(obj,*a,**kw)
        self.patch(tuner.PersistentCacheAutoTuner,'__call__',outer)
        original_inner=tuner.TritonPersistentCacheAutotuner.run
        def inner(obj,*a,**kw):
            self.permit.take('inner')
            values={**dict(zip(obj.arg_names,a)),**kw};named={k:v for k,v in values.items() if k in obj.arg_names}
            key=[named[k] for k in obj.keys if k in named]
            key += [str(v.dtype) for v in named.values() if hasattr(v,'dtype')]
            if obj is not self.inner or str(tuple(key))!=c.INNER_KEY:raise RuntimeError('Foreign inner tuning key')
            c.require_tuning(self.outer.cache,obj.cache)
            try:return original_inner(obj,*a,**kw)
            finally:obj.nargs=None
        self.patch(tuner.TritonPersistentCacheAutotuner,'run',inner)
        for obj,names in ((tuner.PersistentCacheAutoTuner,('_benchmark',)),(tuner.TritonPersistentCacheAutotuner,('_bench','warmup','prune_configs')),(tuner,('save_autotune_cache',))):
            for name in names:self.patch(obj,name,self.deny(name))
        native=importlib.import_module('flex_gemm.kernels.cuda')
        names={'hashmap_build_submanifold_conv_neighbour_map_cuda':'neighbor','neighbor_map_post_process_for_masked_implicit_gemm_1':'post1','neighbor_map_post_process_for_masked_implicit_gemm_2':'post2'}
        for name in dir(native):
            if name.startswith('_') or not callable(getattr(native,name)):continue
            if name not in names:self.patch(native,name,self.deny('unselected native '+name));continue
            original=getattr(native,name);tag=names[name]
            def call(*a,_fn=original,_tag=tag,**kw):
                self.permit.take(_tag)
                if kw:raise RuntimeError('Unexpected native keyword arguments')
                if _tag=='neighbor':
                    if len(a)!=12 or a[3:]!=(3,32,32,3,3,3,1,1,1):raise RuntimeError('Neighbor scalar contract')
                    for t in a[:2]:self.tensor(t,(4098,),torch.uint32)
                    self.tensor(a[2],(2049,4),torch.int32)
                elif _tag=='post1':
                    if len(a)!=1:raise RuntimeError('post1 arity')
                    self.tensor(a[0],(2049,27),torch.uint32)
                else:
                    if len(a)!=3 or type(a[2]) is not int or a[2]!=64:raise RuntimeError('post2 block contract')
                    self.tensor(a[0],(2049,),torch.int32);self.tensor(a[1],(2049,),torch.int64)
                return self.trace.call('native:'+_tag,lambda:_fn(*a))
            self.patch(native,name,call)
        self.spconv=importlib.import_module('flex_gemm.ops.spconv')
        if self.spconv.ALGORITHM!='masked_implicit_gemm_splitk' or self.spconv.HASHMAP_RATIO!=2.0:
            raise RuntimeError('Unexpected sparse algorithm/hashmap ratio')

    def convolve(self,f,co,w,b):
        self.permit.take('convolution')
        for t,shape,dtype in ((f,(2049,1024),self.torch.float16),(co,(2049,4),self.torch.int32),(w,(1024,3,3,3,1024),self.torch.float16),(b,(1024,),self.torch.float16)):
            self.tensor(t,shape,dtype)
        return self.spconv.sparse_submanifold_conv3d(f,co,c.SHAPE,w,b,neighbor_cache=None,dilation=(1,1,1))

    def release(self):
        if self.inner is not None:self.inner.nargs=None
        if self.leaf is not None:self.leaf.device_caches.clear()
        self.permit.phase='release'

    def close(self):
        for obj,name,original in reversed(self.restores):setattr(obj,name,original)
        self.restores.clear()

def dispatch_guard(torch,policy):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Bound(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};phase=policy.permit.phase
            def tensors(value):
                if isinstance(value,torch.Tensor):yield value
                elif isinstance(value,(list,tuple)):
                    for v in value:yield from tensors(v)
                elif isinstance(value,dict):
                    for v in value.values():yield from tensors(v)
            gpu=any(t.is_cuda for t in tensors((args,kwargs))) or str(kwargs.get('device','')).startswith('cuda')
            if gpu:
                if phase not in ('inputs','convolution','output'):raise RuntimeError('GPU tensor operation outside admitted phase')
                if phase=='convolution':
                    admitted={'aten.empty.memory_format','aten.empty_strided.default','aten.full.default','aten.fill_.Scalar','aten.view.default',
                        'aten.sort.default','aten.sort.stable','aten.argsort.default','aten.cumsum.default','aten.select.int',
                        'aten._local_scalar_dense.default','aten.sum.dim_IntList','aten._to_copy.default','aten.to.dtype','aten.to.dtype_layout'}
                    if str(func) not in admitted:raise RuntimeError('Unexpected convolution Torch operation: '+str(func))
                    if str(func) in ('aten.empty.memory_format','aten.empty_strided.default','aten.full.default'):
                        dtype=kwargs.get('dtype',torch.float32);c.allocation(tuple(args[0]),torch.empty((),dtype=dtype).element_size())
            result=func(*args,**kwargs)
            if gpu:
                if torch.cuda.max_memory_allocated(0)>c.PEAK_LIMIT or torch.cuda.memory_reserved(0)>c.PEAK_LIMIT:
                    raise RuntimeError('Native smoke exceeds 1GiB observed PyTorch budget')
                if phase=='convolution':
                    for t in tensors(result):
                        alias=(str(func)=='aten.view.default' and t.untyped_storage().data_ptr()==args[0].untyped_storage().data_ptr())
                        c.result_allocation(str(func),tuple(t.shape),t.element_size(),alias)
            return result
    return Bound()
