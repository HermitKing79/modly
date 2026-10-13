"""Fixed CPU contracts for N2049/C1024 centre-identity native convolution."""
import math
import numpy as np

N=2049; C=1024; SHAPE=(1,C,3,32,32)
ALLOCATOR_LIMIT=12*1024**3; FREE_VRAM=13*1024**3
HOST_RAM=1024**3; PEAK_LIMIT=1024**3; FACTORY_LIMIT=32*1024**2
OUTER_KEY='(2^11, 1024, 1024, 27)'
INNER_KEY="(11, 1024, 1024, 27, 2, True, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.uint32', 'torch.int64', 'torch.float32')"
INNER_CONFIG=dict(kwargs=dict(B1=64,B2=128,BK=32,waves_per_eu=3),num_warps=4,num_ctas=1,num_stages=2,maxnreg=None,pre_hook=None,ir_override=None)

def fixture():
    i=np.arange(N,dtype=np.int32);j=np.arange(C,dtype=np.int32)
    f=(((i[:,None]*17+j[None,:]*31)%2049-1024)/1024).astype(np.float16)
    coords=np.stack([np.zeros_like(i),i//1024,(i//32)%32,i%32],1)
    w=np.zeros((C,3,3,3,C),dtype=np.float16);w[j,1,1,1,j]=1
    return f,coords,w,np.zeros(C,dtype=np.float16)

def validate_fixture(f,coords,w,b):
    expected=fixture()
    for actual,ref in zip((f,coords,w,b),expected):
        if actual.dtype!=ref.dtype or actual.shape!=ref.shape or not actual.flags.c_contiguous or not np.array_equal(actual,ref):
            raise ValueError('Not the exact deterministic native-convolution fixture')
    if len(np.unique(coords,axis=0))!=N:raise ValueError('Duplicate coordinates')

def compare(out,reference):
    if out.shape!=(N,C) or out.dtype!=np.float16 or not np.isfinite(out).all():raise ValueError('Invalid native output')
    errors=np.abs(out.astype(np.float32)-reference.astype(np.float32))
    row=dict(elements=int(out.size),mismatched_elements=int(np.count_nonzero(out!=reference)),atol=0.,rtol=0.,max_absolute_error=float(errors.max()),finite=True)
    if row['mismatched_elements']:raise ValueError('Elementwise identity mismatch: '+str(row))
    return row

def outer_key(n,ci,co,v):return f'(2^{int(math.log2(n))}, {ci}, {co}, {v})'

def require_tuning(outer,inner):
    config=inner.get(INNER_KEY)
    if hasattr(config,'__dict__'):config=config.__dict__
    if outer.get(OUTER_KEY)!={'SPLITK':2} or config!=INNER_CONFIG:raise ValueError('Exact verified tuning hit required')

def admit(ram,free):
    if ram<HOST_RAM or free<FREE_VRAM:raise ValueError('Host RAM or available VRAM admission failed')

def allocation(shape,itemsize):
    size=1
    for x in shape:
        if type(x) is not int or x<0:raise ValueError('Invalid allocation shape')
        size*=x
        if size*itemsize>FACTORY_LIMIT:raise ValueError('Unexpected convolution allocation exceeds 32MiB')
    return size*itemsize

class Permit:
    def __init__(self):
        self.phase='cpu';self.counts={k:0 for k in ('convolution','neighbor','post1','post2','outer','inner','jit','binary','launch')}
    def take(self,name):
        if self.phase!='convolution' or name not in self.counts or self.counts[name]!=0:raise RuntimeError('Unapproved/duplicate native path: '+name)
        self.counts[name]+=1

def result_allocation(operation,shape,itemsize,aliases_input):
    if operation=='aten.view.default':
        if not aliases_input or math.prod(shape)*itemsize>56623104:raise ValueError('Unexpected view storage')
        return 0
    return allocation(shape,itemsize)

def helper_loader(source,include_dirs,path,sha,digest,loader):
    calls=0
    def cached_only(*args,**kw):
        nonlocal calls
        if args or kw!=dict(src=source,name='hip_utils',include_dirs=include_dirs) or digest(path)!=sha:
            raise ValueError('Unverified cached HIP helper request')
        if calls:raise RuntimeError('Repeated helper load')
        calls+=1
        # Original installed extension loader; propagate failure, never rebuild.
        return loader('hip_utils',str(path))
    return cached_only
