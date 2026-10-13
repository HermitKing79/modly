"""CPU test replay of pinned installed Python bodies; only native result simulated."""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import typing
import numpy as np
import amd_native_convolution_contract as contract
from amd_native_convolution_preflight import SITE,digest

SOURCES={
 'flex_gemm/ops/spconv/submanifold_conv3d.py':'51ac1ac963e46228e7d0f928c6237af5e0677705278ebef73bffdb7b43e69aab',
 'flex_gemm/kernels/triton/spconv/sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.py':'21d5efc8f59f312c62fa45241fc13a9e446cd4c13f11e360cf92216d151301bb'}

def metadata(t):
    return dict(tensor_impl=t._cdata,storage_impl=t.untyped_storage()._cdata,storage_bytes=t.untyped_storage().nbytes(),
        shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),offset=t.storage_offset(),requires_grad=t.requires_grad)

def replay(torch,boundary):
    """Same source host bodies and Function.apply, precomputed CPU neighbor metadata."""
    import triton
    source={}
    for relative,sha in SOURCES.items():
        path=SITE/relative
        if digest(path)!=sha:raise RuntimeError('Python tail source identity changed')
        source[relative]=ast.parse(path.read_text())
    f,co,w,b=contract.fixture();contract.validate_fixture(f,co,w,b)
    tensors=[torch.from_numpy(a) for a in (f,co,w,b)]
    features,coordinates,weight,bias=tensors
    calls=[]
    class KernelBoundary:
        def __getitem__(self,grid):
            def simulate(*args,**kwargs):
                if calls or args[6:]!=(2049,11,1024,1024,27) or grid(dict(B1=64,B2=128))!=(264,2):raise RuntimeError('Foreign/repeated simulated native boundary')
                output=args[5]
                if tuple(output.shape)!=(2,2049,1024) or output.dtype!=torch.float32 or output.device.type!='cpu':raise RuntimeError('Invalid simulated split result')
                calls.append(dict(simulated=True,native_execution=False,output=metadata(output),weight=metadata(args[1]),grid=[264,2],splitk=kwargs['SPLITK']))
                output.numpy()[0]=f.astype(np.float32);output.numpy()[1]=0
                boundary(args[1],output)
            return simulate
    ns={**vars(typing),'torch':torch,'math':math,'triton':triton,'config':SimpleNamespace(allow_ieee=True),
        'sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel':KernelBoundary()}
    split=next(node for node in source[list(SOURCES)[1]].body if isinstance(node,ast.FunctionDef) and node.name=='sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk')
    split.decorator_list=[]
    exec(compile(ast.Module(body=[split],type_ignores=[]),str(SITE/list(SOURCES)[1]),'exec'),ns)
    selected=ns[split.name]
    def configured(*args,**kw):
        if kw:raise RuntimeError('Unexpected source host arguments')
        return selected(*args,SPLITK=2)
    algo=SimpleNamespace(**{key:key.lower() for key in ('EXPLICIT_GEMM','IMPLICIT_GEMM','IMPLICIT_GEMM_SPLITK','MASKED_IMPLICIT_GEMM','MASKED_IMPLICIT_GEMM_SPLITK')})
    ns.update(Function=torch.autograd.Function,Algorithm=algo,spconv=SimpleNamespace(ALGORITHM=algo.MASKED_IMPLICIT_GEMM_SPLITK),
        kernels=SimpleNamespace(triton=SimpleNamespace(sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk=configured)))
    body=[node for node in source[list(SOURCES)[0]].body if getattr(node,'name',None) in ('SubMConv3dNeighborCache','SubMConv3dFunction','sparse_submanifold_conv3d')]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(SITE/list(SOURCES)[0]),'exec'),ns)
    cache=ns['SubMConv3dNeighborCache'](neighbor_map=torch.zeros((2049,27),dtype=torch.uint32),sorted_idx=torch.arange(2049,dtype=torch.int64),
        valid_kernel_64=torch.zeros(599,dtype=torch.int32),valid_kernel_seg_64=torch.zeros(34,dtype=torch.int32))
    def run():
        result=ns['sparse_submanifold_conv3d'](features,coordinates,contract.SHAPE,weight,bias,neighbor_cache=cache)
        if len(calls)!=1 or result[1] is not cache:raise RuntimeError('Unexpected source final return/cache')
        if not np.array_equal(result[0].numpy(),f):raise RuntimeError('CPU simulated-boundary reference mismatch')
        return result
    return SimpleNamespace(run=run,weight=weight,features=features,cache=cache,calls=calls,source_sha256=SOURCES.copy(),
        fixture_sha256={key:hashlib.sha256(a.tobytes()).hexdigest() for key,a in zip(('features','coordinates','weights','bias'),(f,co,w,b))})

def observe(torch,case):
    from torch.utils._python_dispatch import TorchDispatchMode
    rows=[];held=[]
    def encode(value):
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu':raise RuntimeError('Observational capture forbids GPU tensors')
            held.extend((value,value.untyped_storage()));return metadata(value)
        if isinstance(value,(tuple,list)):return [encode(v) for v in value]
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        return value if value is None or type(value) in (bool,int,float,str) else str(value)
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            if 'device' in kwargs and torch.device(kwargs['device']).type!='cpu':raise RuntimeError('Observation forbids non-CPU allocation')
            before=encode((args,kwargs));old={t.untyped_storage()._cdata for t in held if isinstance(t,torch.Tensor)}
            result=func(*args,**kwargs);after=encode(result)
            outputs=[]
            def collect(v):
                if isinstance(v,torch.Tensor):outputs.append(v)
                elif isinstance(v,(list,tuple)):
                    for child in v:collect(child)
            collect(result)
            rows.append(dict(operator=str(func),schema=str(func._schema),arguments=before,result=after,
                fresh_storage_bytes=sum({t.untyped_storage()._cdata:t.untyped_storage().nbytes() for t in outputs if t.untyped_storage()._cdata not in old}.values())))
            return result
    with Observe():result=case.run()
    return dict(operators=rows,final_output=metadata(result[0]),cache_identity_preserved=result[1] is case.cache,
        native_result_boundary=case.calls,simulated_reference_match=True,source_sha256=case.source_sha256,fixture_sha256=case.fixture_sha256)
