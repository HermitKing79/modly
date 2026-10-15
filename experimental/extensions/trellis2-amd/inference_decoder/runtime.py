"""One source-pinned parent decode's BF16 configuration and native-call boundary."""
import sys
import threading
from run_contract import MAX_INITIAL_ROWS

ALLOWED_NATIVE = 'hashmap_build_submanifold_conv_neighbour_map'

def configure(model, torch, mix_types, conv_config, triton_config):
    if conv_config.FLEX_GEMM_HASHMAP_RATIO != 2.0:
        raise RuntimeError('Pinned hashmap ratio changed')
    before={k:(str(p.dtype),list(p.shape)) for k,p in model.named_parameters()}
    # Reference amd_fixes.py:223-247: float the blocks, selectively BF16 MIX modules.
    # Do not touch from_latent/output_layer or normalization arithmetic.
    model.blocks.float()
    def convert(layer):
        if isinstance(layer,mix_types):
            for p in layer.parameters():p.data=p.data.to(dtype=torch.bfloat16)
    model.blocks.apply(convert)
    model.dtype=torch.bfloat16
    changes={}
    for k,p in model.named_parameters():
        dtype,shape=before[k]
        if list(p.shape)!=shape or p.device.type!='cpu' or p.requires_grad:
            raise RuntimeError('Runtime conversion changed shape/device/gradient contract: '+k)
        if str(p.dtype)!=dtype:
            if dtype!='torch.float16' or p.dtype!=torch.bfloat16 or not k.startswith('blocks.'):
                raise RuntimeError('Unexpected precision conversion: '+k)
            changes[k]={'from':dtype,'via':'torch.float32','to':str(p.dtype),'rounding':'BF16 rounding; checkpoint unchanged'}
        elif p.dtype!=torch.float32:
            raise RuntimeError('Unconverted decoder parameter: '+k)
    conv_config.FLEX_GEMM_ALGO='explicit_gemm'
    triton_config.allow_tf32=False
    return changes

def validate_tensor_contract(x,c,initial_count):
    if (type(initial_count) is not int or not 0<initial_count<=MAX_INITIAL_ROWS or tuple(x.shape)!=(initial_count,32) or str(x.dtype)!='torch.float32'
        or tuple(c.shape)!=(initial_count,4) or str(c.dtype)!='torch.int32'
        or not x.is_contiguous() or not c.is_contiguous()):
        raise RuntimeError('Require entire run-verified F32[N,32]/I32[N,4] input within qualified bounds')

class NativeBoundary:
    """Retain original neighbour-map binding; deny all other native/compile paths."""
    def __init__(self,cuda,names,hooks,torch,conv_config,spconv,emit):
        self.cuda,self.hooks,self.torch=cuda,hooks,torch
        self.conv_config,self.spconv,self.emit=conv_config,spconv,emit
        self.names=tuple(names);self.originals={n:getattr(cuda,n) for n in names}
        self.calls=0;self.inside=False;self.phase='closed'
        self.sites_called=set()
        self.old_profile=self.old_thread_profile=None
        if len(names)!=18 or ALLOWED_NATIVE not in names or not all(callable(f) for f in self.originals.values()):
            raise RuntimeError('Qualified native closure differs')

    def install(self):
        for name in self.names:
            def wrapper(*args,_name=name,**kwargs):return self.call(_name,args,kwargs)
            setattr(self.cuda,name,wrapper)
        self.old_profile,self.old_thread_profile=sys.getprofile(),threading.getprofile()
        sys.setprofile(self.profile);threading.setprofile(self.profile)
        return self

    def call(self,name,args,kwargs):
        if self.phase!='neural' or name!=ALLOWED_NATIVE or self.inside or self.hooks.active_conv is None:
            raise RuntimeError('Unadmitted native binding/lifecycle: '+name)
        site,module,x=self.hooks.active_conv
        if kwargs or len(args)!=12 or self.calls>=40 or site in self.sites_called:
            raise RuntimeError('Unexpected native signature/count')
        keys,vals,coords,*scalars=args
        n=x.coords.shape[0];spatial=tuple(x.spatial_shape)
        volume=spatial[0]*spatial[1]*spatial[2]
        key_dtype=self.torch.uint32 if volume<2**32 else self.torch.uint64
        if (coords is not x.coords or coords.dtype!=self.torch.int32 or tuple(coords.shape)!=(n,4)
            or coords.device.type!='cuda' or coords.device.index not in (None,0)
            or tuple(keys.shape)!=(2*n,) or tuple(vals.shape)!=(2*n,)
            or keys.dtype!=key_dtype or vals.dtype!=self.torch.uint32
            or any(t.device!=x.feats.device or not t.is_contiguous() for t in (keys,vals,coords))
            or any(type(v) is not int for v in scalars)
            or tuple(scalars)!=(*spatial,3,3,3,1,1,1)
            or self.conv_config.FLEX_GEMM_ALGO!='explicit_gemm'
            or self.conv_config.FLEX_GEMM_HASHMAP_RATIO!=2.0 or self.spconv.ALGORITHM!='explicit_gemm'
            or self.spconv.HASHMAP_RATIO!=2.0
            or self.torch.cuda.current_stream(0).cuda_stream!=0):
            raise RuntimeError('Native neighbour-map provenance/shape/device/algorithm/stream mismatch')
        self.emit('native-entry',site=site,rows=n,call=self.calls+1,host_return_is_not_sync=True)
        self.sites_called.add(site)
        self.inside=True
        try:result=self.originals[name](*args)
        finally:self.inside=False
        if tuple(result.shape)!=(n,27) or result.dtype!=self.torch.uint32 or result.device!=coords.device:
            raise RuntimeError('Native map output contract failed')
        if site=='blocks.3.4.conv2':self.hooks.final_neighbor_map=result
        if self.hooks.tile_large_convolutions and site=='blocks.2.8.conv2':
            if self.hooks.large_neighbor_map is not None:raise RuntimeError('Duplicate large native map')
            self.hooks.large_neighbor_map=result;self.hooks.large_neighbor_coords=coords
        self.calls+=1
        self.emit('native-host-return',site=site,rows=n,call=self.calls,host_return_is_not_sync=True)
        return result

    def profile(self,frame,event,arg):
        if event=='c_call' and (getattr(arg,'__module__','') or '')=='flex_gemm.kernels.cuda':
            if not self.inside or arg is not self.originals[ALLOWED_NATIVE]:
                raise RuntimeError('Foreign native entry rejected')
        if event=='call':
            mod,name=frame.f_globals.get('__name__',''),frame.f_code.co_name
            if mod=='flex_gemm.utils.autotuner' and name in ('__call__','run','warmup','_benchmark','_bench','load_autotune_cache','save_autotune_cache','get_autotune_cache'):
                raise RuntimeError('Autotuning/cache execution rejected')
            if mod.startswith('trellis2.') and name in ('forward','decode','decode_latent'):
                allowed=('trellis2.models.sc_vaes.sparse_unet_vae','trellis2.modules.norm',
                    'trellis2.modules.sparse.basic','trellis2.modules.sparse.linear',
                    'trellis2.modules.sparse.conv.conv','trellis2.modules.sparse.spatial.spatial2channel')
                if self.phase!='neural' or mod not in allowed:raise RuntimeError('Foreign model/geometry forward rejected: '+mod)

    def remove(self):
        self.phase='closed'
        sys.setprofile(self.old_profile);threading.setprofile(self.old_thread_profile)
        for name,original in self.originals.items():setattr(self.cuda,name,original)
        self.hooks=None
