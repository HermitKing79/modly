"""Two fixed native postprocessor contracts, never general scalar permission."""
from contextlib import contextmanager
from pathlib import Path

N=2049; V=27; LENGTH=N*V; EXPECTED=35353
SOURCE_SHA='f5de4b1b1f8ddcf54ccbdb76c7a7ff8759801c3cc312fb0ab8f869d3b78abb92'

class Post1Scalar:
    tag='post1';length=LENGTH;minimum=N;maximum=LENGTH;expected=EXPECTED
    def __init__(self,policy,torch):
        self.policy=policy;self.torch=torch;self.active=False;self.entries=0;self.reads=0
        self.device=None;self.mask=self.prefix=self.last=None;self.value=None
        self.mask_storage=self.prefix_storage=None
        self.factories={};self.created=[];self.sorted=None;self.sorted_storage=None

    def clear(self):
        self.mask=self.prefix=self.last=None
        self.mask_storage=self.prefix_storage=None
        self.created.clear();self.sorted=self.sorted_storage=None

    @contextmanager
    def scope(self,device):
        from amd_native_convolution_preflight import SITE,digest
        if (self.active or self.entries or self.policy.permit.phase!='convolution'
            or self.policy.permit.counts[self.tag]!=1
            or any(getattr(self.policy,tag+'_scalar',None).active for tag in ('post1','post2')
                   if getattr(self.policy,tag+'_scalar',None) is not None and tag!=self.tag)):
            raise RuntimeError('Unapproved/repeated/nested postprocessor scope')
        if digest(SITE/'flex_gemm/kernels/cuda/spconv/neighbor_map.hip')!=SOURCE_SHA:raise RuntimeError('Native post1 source identity changed')
        self.device=self.torch.device(device)
        if self.device not in (self.torch.device('cpu'),self.torch.device('cuda:0')):raise RuntimeError('Unsupported post1 device')
        # CPU supports operator acceptance tests only: the native wrapper still
        # requires CUDA:0 and validates the genuine neighbor input before entry.
        self.entries=1;self.active=True
        try:
            yield
            if self.reads!=1 or self.policy.permit.phase!='convolution':raise RuntimeError('Exactly one postprocessor scalar read in its phase required')
        finally:self.active=False;self.clear()

    def tensor(self,t,shape):
        self.metadata(t,shape,self.torch.int32)

    def metadata(self,t,shape,dtype):
        if (type(t) is not self.torch.Tensor or t.dtype!=dtype or tuple(t.shape)!=shape
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
            self.tensor(self.prefix,(self.length,))
            self.storage(self.prefix,self.prefix_storage,0)
            self.storage(args[0],self.prefix_storage,self.length-1)
            self.reads=1
            return 'read'
        if not self.active:return None
        if self.policy.permit.phase!='convolution':raise RuntimeError('Post1 scalar lifecycle phase changed')
        if func is ops.empty.memory_format:
            if (len(args)!=1 or set(kw)-{'dtype','layout','device','pin_memory','memory_format'}
                or self.torch.device(kw.get('device','cpu'))!=self.device
                or any(type(x) is not int for x in args[0])
                or kw.get('layout') not in (None,self.torch.strided)
                or (kw.get('pin_memory') is not None and kw.get('pin_memory') is not False)
                or kw.get('memory_format') not in (None,self.torch.contiguous_format)):
                raise RuntimeError('Unexpected postprocessor allocation arguments')
            key=(tuple(args[0]),kw.get('dtype'))
            limits=self.factory_limits()
            if key not in limits or self.factories.get(key,0)>=limits[key]:raise RuntimeError('Unexpected/repeated postprocessor allocation shape/dtype')
            scalar_output=(self.tag=='post1' and key==((35353,),self.torch.uint32)) or (self.tag=='post2' and key==((599,),self.torch.int32))
            if scalar_output and self.value!=self.expected:raise RuntimeError('Scalar-dependent allocation before verified count')
            self.factories[key]=self.factories.get(key,0)+1
            return ('factory',key)
        if func is ops.cumsum.default:
            if (len(args)!=2 or not self.same(args[0],self.mask) or type(args[1]) is not int or args[1]!=0
                or kw!={'dtype':self.torch.int32} or self.prefix is not None):raise RuntimeError('Foreign post1 cumsum')
            self.tensor(args[0],(self.length,));self.storage(args[0],self.mask_storage,0);return 'prefix'
        if func is ops.select.int:
            if (len(args)!=3 or not self.same(args[0],self.prefix) or type(args[1]) is not int or args[1]!=0
                or type(args[2]) is not int or args[2] not in (-1,self.length-1) or kw or self.last is not None):
                raise RuntimeError('Foreign post1 last-element selection')
            self.tensor(args[0],(self.length,));self.storage(args[0],self.prefix_storage,0)
            return 'last'
        if func in (ops.argsort.default,ops.sort.default,ops.sort.stable):
            binary=[row for row in self.created if row[2]==(2049,) and row[3]==self.torch.int32]
            if (self.tag!='post1' or len(binary)!=2 or self.sorted is not None or not args
                or not self.same(args[0],binary[1][0]) or len(args)>3
                or (len(args)>1 and (type(args[1]) is not int or args[1]!=-1))
                or (len(args)>2 and args[2] is not False) or set(kw)-{'dim','descending','stable'}
                or type(kw.get('dim',-1)) is not int or kw.get('dim',-1)!=-1 or kw.get('descending',False) is not False
                or (kw.get('stable') is not None and kw.get('stable') is not False)):
                raise RuntimeError('Foreign/repeated post1 sort')
            self.metadata(args[0],(2049,),self.torch.int32);self.storage(args[0],binary[1][1],0)
            return 'sorted'
        raise RuntimeError('Unexpected postprocessor Torch operation: '+str(func))

    def factory_limits(self):
        t=self.torch
        if self.tag=='post1':return {((2049,),t.int32):2,((55323,),t.int32):1,((55323,),t.uint32):1,((35353,),t.uint32):2,((28,),t.uint32):1}
        return {((33,),t.int32):1,((34,),t.int32):1,((599,),t.int32):1}

    def finish(self,outputs):
        if not self.active or self.policy.permit.phase!='convolution' or self.factories!=self.factory_limits() or self.reads!=1 or self.value!=self.expected:
            raise RuntimeError('Incomplete postprocessor allocation/read contract')
        if type(outputs) is not tuple or len(outputs)!=(5 if self.tag=='post1' else 2):raise RuntimeError('Malformed postprocessor outputs')
        def created(shape,dtype):return [row for row in self.created if row[2]==shape and row[3]==dtype]
        if self.tag=='post1':
            rows=[created((2049,),self.torch.int32)[0],(self.sorted,self.sorted_storage,(2049,),self.torch.int64),
                  *created((35353,),self.torch.uint32),created((28,),self.torch.uint32)[0]]
        else:rows=[created((599,),self.torch.int32)[0],(self.prefix,self.prefix_storage,(34,),self.torch.int32)]
        for output,(original,storage,shape,dtype) in zip(outputs,rows):
            if not self.same(output,original):raise RuntimeError('Foreign postprocessor returned tensor')
            self.metadata(output,shape,dtype);self.storage(output,storage,0)

    def after(self,kind,result):
        if isinstance(kind,tuple):
            _,(shape,dtype)=kind;self.metadata(result,shape,dtype)
            if result.storage_offset()!=0:raise RuntimeError('Unexpected postprocessor allocation offset')
            self.created.append((result,result.untyped_storage(),shape,dtype))
            kind='mask' if shape==(self.length,) and dtype==self.torch.int32 else None
        if kind=='sorted':
            self.sorted=result[-1] if isinstance(result,tuple) else result
            self.metadata(self.sorted,(2049,),self.torch.int64)
            if self.sorted.storage_offset()!=0:raise RuntimeError('Unexpected sort allocation offset')
            self.sorted_storage=self.sorted.untyped_storage()
        if kind in ('mask','prefix'):
            self.tensor(result,(self.length,))
            if result.storage_offset()!=0:raise RuntimeError('Unexpected post1 allocation offset')
            setattr(self,kind,result)
            # Retain the original StorageImpl too: .data can replace storage
            # while preserving TensorImpl identity and shape/dtype metadata.
            setattr(self,kind+'_storage',result.untyped_storage())
        elif kind=='last':
            self.tensor(result,())
            self.storage(result,self.prefix_storage,self.length-1)
            self.last=result
        elif kind=='read':
            if type(result) is not int or not self.minimum<=result<=self.maximum or result!=self.expected:
                raise RuntimeError('Post1 count outside proven bounds or deterministic fixture identity')
            self.value=result

class Post2Scalar(Post1Scalar):
    tag='post2';length=34;minimum=33;maximum=891;expected=599
