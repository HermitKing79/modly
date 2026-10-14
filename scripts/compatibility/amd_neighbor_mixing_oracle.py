"""Independent tuple/integer CPU reference for one fixed three-tap convolution."""
import numpy as np
import amd_native_convolution_contract as c
import amd_cached_kernel_oracle as metadata_oracle

CASE='center1-axis3plus_half-axis1minus_quarter'
# Stored coordinate order is [batch,axis1,axis2,axis3]; names avoid conflating
# the native generator's W/H/D labels with pipeline D/H/W terminology.
TAPS=(((0,0,0),1.,13),((0,0,1),.5,14),((-1,0,0),-.25,4))

def fixture():
    f,co,w,b=c.fixture();j=np.arange(1024)
    w[j,1,1,2,j]=.5;w[j,0,1,1,j]=-.25
    return f,co,w,b

def validate_fixture(f,co,w,b):
    for actual,expected in zip((f,co,w,b),fixture()):
        if actual.dtype!=expected.dtype or actual.shape!=expected.shape or not actual.flags.c_contiguous or not np.array_equal(actual,expected):
            raise ValueError('Not the exact three-tap fixture')

def _target(point,offset):
    return (int(point[0]),*(int(point[i+1])+offset[i] for i in range(3)))

def _coefficients(w,offset):
    j=np.arange(1024)
    # Read actual weight values in [Co,k1,k2,k3,Ci], with Co=Ci diagonal.
    return (w[j,offset[0]+1,offset[1]+1,offset[2]+1,j].astype(np.float32)*4).astype(np.int32)

def reference(f,co,w,b):
    validate_fixture(f,co,w,b)
    lookup={tuple(map(int,point)):i for i,point in enumerate(co)}
    units=(f.astype(np.float32)*1024).astype(np.int32)
    numerator=np.zeros((2049,1024),np.int32);valid=[];outside=[];sparse=[]
    for offset,_,_ in TAPS:
        coefficient=_coefficients(w,offset);nv=no=ns=0
        for row,point in enumerate(co):
            target=_target(point,offset);source=lookup.get(target)
            if source is None:
                if any(x<0 or x>=limit for x,limit in zip(target[1:],(3,32,32))):no+=1
                else:ns+=1
            else:numerator[row]+=units[source]*coefficient;nv+=1
        valid.append(nv);outside.append(no);sparse.append(ns)
    # |numerator|<=7168<2**24, so this power-of-two scaling is exact F32.
    if np.max(np.abs(numerator))>7168:raise ValueError('Exact-arithmetic bound exceeded')
    output=numerator.astype(np.float32)/np.float32(4096)
    coverage=dict(case=CASE,offsets=[list(t[0]) for t in TAPS],kernel_indices=[t[2] for t in TAPS],weights=[t[1] for t in TAPS],
        valid_rows=valid,out_of_bounds_missing=outside,in_bounds_sparse_missing=sparse,valid_noncentral_terms=sum(valid[1:])*1024,
        nonidentity_elements=int(np.count_nonzero(output!=f.astype(np.float32))),reference_numerator_min=int(numerator.min()),reference_numerator_max=int(numerator.max()),
        exact_f32=True,atol=0.,rtol=0.,coordinate_order='batch,axis1,axis2,axis3',reference_uses_precomputed_neighbor_map=False)
    return output,coverage

def split_reference(f,co,w,metadata,full_reference):
    # Full reference above never uses these arrays. Here the already-validated
    # source split assignment is checked separately against that full result.
    metadata_oracle.validate(co,metadata)
    lookup={tuple(map(int,point)):i for i,point in enumerate(co)}
    units=(f.astype(np.float32)*1024).astype(np.int32)
    numerator=np.zeros((2,2049,1024),np.int32);channel_blocks=np.arange(1024)//32
    for block in range(33):
        a,z=map(int,metadata['valid_kernel_seg'][block:block+2]);allowed=metadata['valid_kernel'][a:z].tolist()
        threshold=16*len(allowed) # ceil(32*segment_length/2), genuine SPLITK2.
        for offset,_,v in TAPS:
            if v not in allowed:continue
            k=32*allowed.index(v)+channel_blocks;in_first=k<threshold
            coefficient=_coefficients(w,offset)
            for row in metadata['sorted_idx'][block*64:(block+1)*64]:
                source=lookup.get(_target(co[row],offset))
                if source is None:continue
                contribution=units[source]*coefficient
                numerator[0,row,in_first]+=contribution[in_first]
                numerator[1,row,~in_first]+=contribution[~in_first]
    result=numerator.astype(np.float32)/np.float32(4096)
    if not np.array_equal(result[0]+result[1],full_reference):raise ValueError('CPU SplitK routing disagrees with independent tuple reference')
    return result

def compare(raw,reference,partials):
    if raw.shape!=(2,2049,1024) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Invalid/nonfinite/unwritten SplitK output')
    if reference.shape!=(2049,1024) or reference.dtype!=np.float32 or not np.isfinite(reference).all():raise ValueError('Invalid independent F32 reference')
    if partials.shape!=raw.shape or partials.dtype!=np.float32 or not np.isfinite(partials).all():raise ValueError('Invalid CPU split reference')
    if not np.array_equal(raw,partials):raise ValueError('Exact per-split reference mismatch')
    reduced=raw.sum(axis=0,dtype=np.float32)
    if not np.array_equal(reduced,reference):raise ValueError('Exact F32 neighbour-mixing mismatch')
    return reduced,dict(elements=int(reduced.size),raw_elements=int(raw.size),mismatched_elements=0,split_mismatched_elements=0,
        max_absolute_error=0.,atol=0.,rtol=0.,finite=True,raw_finite=True,fp32_reference_exact=True,per_split_reference_exact=True)
