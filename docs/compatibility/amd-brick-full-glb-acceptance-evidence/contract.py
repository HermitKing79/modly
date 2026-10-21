"""Pure NumPy original-index compaction and chunked exact surface checks."""
import sys
sys.dont_write_bytecode=True
def compaction(faces,n):
    import numpy as np
    if (type(n)is not int or not 0<n<=2000000 or not isinstance(faces,np.ndarray) or faces.dtype!=np.dtype('<i4') or faces.ndim!=2 or faces.shape[1]!=3 or not 0<len(faces)<=4173584):raise ValueError('Bounded original triangle table')
    used=np.zeros(n,dtype=np.bool_)
    for start in range(0,len(faces),32768):
        f=faces[start:start+32768]
        if np.any(f<0) or np.any(f>=n):raise ValueError('Foreign index')
        used[f.reshape(-1)]=True
    ids=np.flatnonzero(used);inverse=np.full(n,-1,dtype='<i4');inverse[ids]=np.arange(len(ids),dtype='<i4')
    return ids,inverse
def verify_geometry(vertices,faces,actual_v,actual_f):
    import numpy as np
    ids,inverse=compaction(faces,len(vertices))
    if vertices.dtype!=np.dtype('<f4') or actual_v.dtype!=np.dtype('<f4') or actual_v.shape!=(len(ids),3) or actual_f.dtype!=np.dtype('<i4') or actual_f.shape!=faces.shape:raise ValueError('Exact geometry dtype and shape')
    for start in range(0,len(ids),32768):
        a=actual_v[start:start+32768];b=vertices[ids[start:start+32768]]
        if not np.isfinite(a).all() or not np.array_equal(a.view('<u4'),b.view('<u4')):raise ValueError('Original vertex position bits changed')
    for start in range(0,len(faces),32768):
        if not np.array_equal(actual_f[start:start+32768],inverse[faces[start:start+32768]]):raise ValueError('Original triangle order/corners/winding/multiplicity changed')
    return ids,inverse
