"""Fixed CPU metadata/oracle for one genuine cached SplitK launch, not a kernel."""
import hashlib
from pathlib import Path
import numpy as np
import amd_native_convolution_contract as c

SENTINEL=np.uint32(0xffffffff)

def build(coords):
    if coords.shape!=(2049,4) or coords.dtype!=np.int32 or len(np.unique(coords,axis=0))!=2049:raise ValueError('Invalid fixed coordinates')
    offsets=np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)],dtype=np.int32)
    points=coords[:,None,1:]+offsets[None]
    keys=points[:,:,0]*1024+points[:,:,1]*32+points[:,:,2]
    inside=((points>=0)&(points<np.array([3,32,32]))).all(axis=2)
    # The unchanged fixture enumerates these flat keys in source-index order.
    if not np.array_equal(coords[:,1]*1024+coords[:,2]*32+coords[:,3],np.arange(2049)):raise ValueError('Foreign coordinate fixture/order')
    neighbor=np.full(keys.shape,SENTINEL,dtype=np.uint32)
    valid=inside&(keys>=0)&(keys<2049)
    neighbor[valid]=keys[valid].astype(np.uint32)
    flags=neighbor!=SENTINEL
    gray=(flags.astype(np.int32)*(1<<np.arange(27))).sum(axis=1).astype(np.int32)
    binary=gray.copy()
    for shift in range(1,27):binary^=gray>>shift
    order=np.argsort(binary,kind='stable').astype(np.int64)
    indices=[];segments=[0]
    for start in range(0,2049,64):
        active=np.flatnonzero(flags[order[start:start+64]].any(axis=0))
        indices.extend(active.tolist());segments.append(len(indices))
    return dict(neighbor=neighbor,sorted_idx=order,valid_kernel=np.array(indices,dtype=np.int32),valid_kernel_seg=np.array(segments,dtype=np.int32))

def validate(coords,metadata):
    expected={'neighbor':((2049,27),np.uint32),'sorted_idx':((2049,),np.int64),'valid_kernel':((599,),np.int32),'valid_kernel_seg':((34,),np.int32)}
    if set(metadata)!=set(expected):raise ValueError('Unexpected kernel metadata fields')
    for name,(shape,dtype) in expected.items():
        a=metadata[name]
        if a.shape!=shape or a.dtype!=dtype or not a.flags.c_contiguous:raise ValueError('Kernel metadata layout: '+name)
    if coords.shape!=(2049,4) or coords.dtype!=np.int32 or not np.all(coords[:,0]==0):raise ValueError('Invalid coordinate metadata')
    lookup={tuple(map(int,p)):i for i,p in enumerate(coords)}
    if len(lookup)!=2049:raise ValueError('Duplicate coordinates')
    gray=[];signals=0
    # Independent tuple-based membership; does not call the vectorized builder.
    for i,point in enumerate(coords):
        bits=0
        for v,(dx,dy,dz) in enumerate(( (x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1))):
            target=(0,int(point[1])+dx,int(point[2])+dy,int(point[3])+dz)
            expected_index=lookup.get(target,0xffffffff)
            if int(metadata['neighbor'][i,v])!=expected_index:raise ValueError('Neighbor membership/order/sentinel mismatch')
            if expected_index!=0xffffffff:bits|=1<<v;signals+=1
        if int(metadata['neighbor'][i,13])!=i:raise ValueError('Missing identity centre')
        gray.append(bits)
    if signals!=35353:raise ValueError('Valid signal count mismatch')
    binary=gray.copy()
    for i,g in enumerate(gray):
        for shift in range(1,27):binary[i]^=g>>shift
    order=metadata['sorted_idx'].tolist()
    if sorted(order)!=list(range(2049)) or order!=sorted(range(2049),key=lambda i:(binary[i],i)):raise ValueError('Noncanonical sorted permutation')
    seg=metadata['valid_kernel_seg'];kernels=metadata['valid_kernel']
    if seg[0]!=0 or seg[-1]!=599 or np.any(np.diff(seg)<=0):raise ValueError('Invalid segment offsets')
    counts=[]
    for block in range(33):
        bits=0
        for index in order[block*64:(block+1)*64]:bits|=gray[index]
        allowed=[v for v in range(27) if bits&(1<<v)]
        actual=kernels[int(seg[block]):int(seg[block+1])].tolist()
        if actual!=allowed or 13 not in actual:raise ValueError('Invalid per-block kernel membership/order')
        counts.append(len(actual))
    valid=metadata['neighbor'][metadata['neighbor']!=SENTINEL]
    if valid.min()<0 or valid.max()>=2049:raise ValueError('Input pointer index exceeds feature allocation')
    # Source masked sorted load uses other=0 for padded lanes. Thus even the
    # following unconditional neighbor load uses an in-bounds row; sentinel
    # feature addresses are masked, and padded output stores are masked.
    return dict(signals=signals,kernel_entries=599,segments=34,block_counts=counts,grid=[264,2],
        feature_last_element=2049*1024-1,weight_last_element=1024*27*1024-1,output_last_element=2*2049*1024-1,
        padded_sorted_rows_use_zero=True,sentinel_input_reads_masked=True,padded_output_stores_masked=True,
        source_sort_ties='Stable CPU order; equal binary codes have identical Gray masks, so tie order cannot alter block unions.')

def describe(arrays):
    return {k:dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in arrays.items()}

def save_arrays(scratch,arrays):
    scratch=Path(scratch).resolve();rows={}
    for name,a in arrays.items():
        if not name.replace('-','').replace('_','').isalnum() or a.nbytes>64*1024**2:raise ValueError('Bounded artifact name/size required')
        path=scratch/(name+'.npy')
        with path.open('xb') as f:np.save(f,a,allow_pickle=False);f.flush();__import__('os').fsync(f.fileno())
        read=np.load(path,allow_pickle=False)
        if read.dtype!=a.dtype or read.shape!=a.shape or not np.array_equal(read,a,equal_nan=True):raise ValueError('Artifact readback failed')
        rows[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(a.shape),dtype=str(a.dtype))
    return rows

def compare(raw,reference):
    if raw.shape!=(2,2049,1024) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Nonfinite/unwritten/invalid raw SplitK result')
    reduced=raw.sum(axis=0,dtype=np.float32)
    # Check before casting too, preventing FP16 rounding from hiding an error.
    if not np.array_equal(reduced,reference.astype(np.float32)):raise ValueError('Exact FP32 identity mismatch')
    result=reduced.astype(np.float16);stats=c.compare(result,reference)
    return result,dict(**stats,raw_elements=int(raw.size),raw_finite=True,fp32_identity_exact=True)
