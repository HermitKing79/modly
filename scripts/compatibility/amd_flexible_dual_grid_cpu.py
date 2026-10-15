"""Inference-only CPU translation of Microsoft's flexible dual-grid converter.

No model or native GPU imports. Global sorted integer keys replace ONLY the
native hashmap. Original row/axis ordering, vertex indices and winding survive.
Reference: microsoft/TRELLIS.2 de38fdd, o-voxel/convert/flexible_dual_grid.py.
"""
from pathlib import Path
import numpy as np

GRID=512
MAX_ROWS=2_000_000
OFFSETS=np.array([[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
                  [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
                  [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]],np.int32)
SPLIT_1=np.array([0,1,2,0,2,3])
SPLIT_2=np.array([0,1,3,3,1,2])


def transform_features(features):
    if (features.dtype!=np.float32 or features.ndim!=2 or features.shape[1]!=7
        or not np.isfinite(features).all()):raise ValueError('Finite F32[N,7] features required')
    x=features[:,:3]
    z=np.exp(-np.abs(x))
    sigmoid=np.where(x>=0,np.float32(1)/(np.float32(1)+z),z/(np.float32(1)+z))
    dual=np.float32(2)*sigmoid-np.float32(.5)
    q=features[:,6]
    # PyTorch F.softplus's default beta=1, threshold=20; no precision promotion.
    weight=np.where(q>20,q,np.log1p(np.exp(-np.abs(q)))+np.maximum(q,np.float32(0)))
    return dual,features[:,3:6]>0,weight


def vertices(coords,dual):
    return (coords.astype(np.float32)+dual)*np.float32(1/512)-np.float32(.5)


def integer_keys(coords):
    return (coords[...,0].astype(np.int64)*GRID+coords[...,1])*GRID+coords[...,2]


def faces(coords,flags,weight,batch_rows=32768,progress=None):
    n=len(coords)
    if (coords.dtype!=np.int32 or coords.shape!=(n,3) or not 0<n<=MAX_ROWS
        or np.any(coords<0) or np.any(coords>=GRID)):raise ValueError('Invalid original grid coordinates')
    if flags.dtype!=np.bool_ or flags.shape!=(n,3):raise ValueError('Boolean[N,3] flags required')
    if weight.dtype!=np.float32 or weight.shape!=(n,) or not np.isfinite(weight).all() or np.any(weight<0):
        raise ValueError('Finite nonnegative F32[N] softplus weights required')
    if not 1<=batch_rows<=65536:raise ValueError('Bounded row batch required')
    key=integer_keys(coords);order=np.argsort(key);sorted_key=key[order]
    if np.any(sorted_key[1:]==sorted_key[:-1]):raise ValueError('Duplicate coordinate')
    result=[]
    stats={'flagged_quads':0,'valid_quads':0,'missing_neighbour_quads':0,'out_of_grid_quads':0,'split_1':0,'split_2':0,
           'global_original_indices':True,'batch_rows':batch_rows}
    for start in range(0,n,batch_rows):
        rows,axes=np.nonzero(flags[start:start+batch_rows])
        neighbour=coords[start+rows,None,:]+OFFSETS[axes]
        bounds=((neighbour>=0)&(neighbour<GRID)).all(axis=(1,2))
        query=integer_keys(neighbour)
        pos=np.searchsorted(sorted_key,query)
        # Do not dereference the end sentinel or allow coordinate-key wraparound.
        within=pos<n;found=np.zeros(pos.shape,bool)
        found[within]=sorted_key[pos[within]]==query[within]
        valid=bounds&found.all(axis=1)
        quad=order[pos[valid]].astype(np.int32)
        q=weight[quad]
        choose=q[:,0]*q[:,2]>q[:,1]*q[:,3]
        triangles=quad[:,SPLIT_2].copy()
        triangles[choose]=quad[choose][:,SPLIT_1]
        result.append(triangles.reshape(-1,3))
        stats['flagged_quads']+=len(rows);stats['valid_quads']+=len(quad)
        stats['missing_neighbour_quads']+=int((~valid).sum())
        stats['out_of_grid_quads']+=int((~bounds).sum())
        stats['split_1']+=int(choose.sum());stats['split_2']+=int((~choose).sum())
        if progress is not None:progress(start+min(batch_rows,n-start),stats)
    return np.concatenate(result,axis=0),stats


def write_ply(path,verts,triangles):
    path=Path(path)
    if (verts.dtype!=np.float32 or verts.ndim!=2 or verts.shape[1]!=3 or not np.isfinite(verts).all()
        or triangles.dtype!=np.int32 or triangles.ndim!=2 or triangles.shape[1]!=3
        or (len(triangles) and (triangles.min()<0 or triangles.max()>=len(verts)))):
        raise ValueError('Invalid mesh arrays')
    header=('ply\nformat binary_little_endian 1.0\ncomment original global vertex order; no repairs\n'
            f'element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n'
            f'element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n').encode('ascii')
    with path.open('xb') as f:
        f.write(header);f.write(verts.astype('<f4',copy=False).tobytes())
        for start in range(0,len(triangles),32768):
            chunk=triangles[start:start+32768]
            record=np.empty(len(chunk),dtype=[('count','u1'),('indices','<i4',(3,))])
            record['count']=3;record['indices']=chunk;f.write(record.tobytes())


def read_ply(path):
    with Path(path).open('rb') as f:
        lines=[]
        while len(lines)<20:
            line=f.readline(512).decode('ascii').strip();lines.append(line)
            if line=='end_header':break
        if lines[-1]!='end_header' or lines[:2]!=['ply','format binary_little_endian 1.0']:
            raise ValueError('Unexpected PLY header')
        nv=int(next(x.split()[2] for x in lines if x.startswith('element vertex ')))
        nf=int(next(x.split()[2] for x in lines if x.startswith('element face ')))
        offset=f.tell()
    if Path(path).stat().st_size!=offset+nv*12+nf*13:raise ValueError('PLY payload size differs')
    v=np.memmap(path,mode='r',offset=offset,dtype='<f4',shape=(nv,3))
    row=np.memmap(path,mode='r',offset=offset+nv*12,dtype=[('count','u1'),('indices','<i4',(3,))],shape=(nf,))
    if np.any(row['count']!=3):raise ValueError('Nontriangle PLY')
    return v,row['indices']
