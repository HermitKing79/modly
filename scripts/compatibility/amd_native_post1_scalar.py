"""One fixed native-post1 allocation count, never a general scalar permission."""
from contextlib import contextmanager
from pathlib import Path

N=2049; V=27; LENGTH=N*V; EXPECTED=35353
SOURCE_SHA='f5de4b1b1f8ddcf54ccbdb76c7a7ff8759801c3cc312fb0ab8f869d3b78abb92'

class Post1Scalar:
    def __init__(self,policy,torch):
        self.policy=policy;self.torch=torch;self.active=False;self.entries=0;self.reads=0
        self.device=None;self.mask=self.prefix=self.last=None;self.value=None
        self.mask_storage=self.prefix_storage=None

    def clear(self):
        self.mask=self.prefix=self.last=None
        self.mask_storage=self.prefix_storage=None

    @contextmanager
    def scope(self,device):
        from amd_native_convolution_preflight import SITE,digest
        if (self.active or self.entries or self.policy.permit.phase!='convolution'
            or self.policy.permit.counts['post1']!=1):raise RuntimeError('Unapproved/repeated post1 scalar scope')
        if digest(SITE/'flex_gemm/kernels/cuda/spconv/neighbor_map.hip')!=SOURCE_SHA:raise RuntimeError('Native post1 source identity changed')
        self.device=self.torch.device(device)
        if self.device not in (self.torch.device('cpu'),self.torch.device('cuda:0')):raise RuntimeError('Unsupported post1 device')
        # CPU supports operator acceptance tests only: the native wrapper still
        # requires CUDA:0 and validates the genuine neighbor input before entry.
        self.entries=1;self.active=True
        try:
            yield
            if self.reads!=1:raise RuntimeError('Exactly one post1 scalar read required')
        finally:self.active=False;self.clear()

    def tensor(self,t,shape):
        if (type(t) is not self.torch.Tensor or t.dtype!=self.torch.int32 or tuple(t.shape)!=shape
            or t.device!=self.device or not t.is_contiguous() or t.requires_grad):
            raise RuntimeError('Unexpected post1 tensor metadata')

    def same(self,a,b):return type(a) is self.torch.Tensor and b is not None and a._cdata==b._cdata

    def storage(self,t,storage,offset):
        if storage is None or t.untyped_storage()._cdata!=storage._cdata or t.storage_offset()!=offset:
            raise RuntimeError('Post1 tensor storage provenance changed')

    def before(self,func,args,kw):
        ops=self.torch.ops.aten
        if func in (ops.item.default,ops._local_scalar_dense.default):
            if (not self.active or self.policy.permit.phase!='convolution' or len(args)!=1 or kw
                or self.reads or not self.same(args[0],self.last)):
                raise RuntimeError('Foreign/repeated/out-of-scope scalar read')
            self.tensor(args[0],())
            self.tensor(self.prefix,(LENGTH,))
            self.storage(self.prefix,self.prefix_storage,0)
            self.storage(args[0],self.prefix_storage,LENGTH-1)
            self.reads=1
            return 'read'
        if not self.active:return None
        if self.policy.permit.phase!='convolution':raise RuntimeError('Post1 scalar lifecycle phase changed')
        if func is ops.empty.memory_format and tuple(args[0])==(LENGTH,) and kw.get('dtype')==self.torch.int32:
            if (self.mask is not None or len(args)!=1 or set(kw)-{'dtype','layout','device','pin_memory','memory_format'}
                or self.torch.device(kw.get('device','cpu'))!=self.device
                or kw.get('layout') not in (None,self.torch.strided) or kw.get('pin_memory') not in (None,False)
                or kw.get('memory_format') not in (None,self.torch.contiguous_format)):
                raise RuntimeError('Unexpected/repeated post1 mask allocation')
            return 'mask'
        if func is ops.cumsum.default:
            if (len(args)!=2 or not self.same(args[0],self.mask) or type(args[1]) is not int or args[1]!=0
                or kw!={'dtype':self.torch.int32} or self.prefix is not None):raise RuntimeError('Foreign post1 cumsum')
            self.tensor(args[0],(LENGTH,));self.storage(args[0],self.mask_storage,0);return 'prefix'
        if func is ops.select.int:
            if (len(args)!=3 or not self.same(args[0],self.prefix) or type(args[1]) is not int or args[1]!=0
                or type(args[2]) is not int or args[2] not in (-1,LENGTH-1) or kw or self.last is not None):
                raise RuntimeError('Foreign post1 last-element selection')
            self.tensor(args[0],(LENGTH,));self.storage(args[0],self.prefix_storage,0)
            return 'last'
        return None

    def after(self,kind,result):
        if kind in ('mask','prefix'):
            self.tensor(result,(LENGTH,))
            if result.storage_offset()!=0:raise RuntimeError('Unexpected post1 allocation offset')
            setattr(self,kind,result)
            # Retain the original StorageImpl too: .data can replace storage
            # while preserving TensorImpl identity and shape/dtype metadata.
            setattr(self,kind+'_storage',result.untyped_storage())
        elif kind=='last':
            self.tensor(result,())
            self.storage(result,self.prefix_storage,LENGTH-1)
            self.last=result
        elif kind=='read':
            if type(result) is not int or not N<=result<=LENGTH or result!=EXPECTED:
                raise RuntimeError('Post1 count outside proven bounds or deterministic fixture identity')
            self.value=result
