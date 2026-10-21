"""One fixed index-only candidate; exact source-ID transfer and frozen gates."""
import sys
sys.dont_write_bytecode=True
import numpy as np

TARGET_TRIANGLES=2_086_792
ERROR=.005
FLAGS=['LockBorder']
GATES=dict(minimum_triangle_reduction=.20,silhouette_IoU=.98,
           visible_foliage_recall=.99,geometry_diagonal=[.005,.01,.03],
           RGB=[.02,.05,.15],scalar_RMSE=.02,boundary_length_growth=.05,
           component_losses=0,new_nonmanifold_edges=0,new_components=0,
           zero_area_triangles=0)

def need(ok,message):
    if not ok:raise ValueError(message)

def scan(vertices,faces):
    need(vertices.dtype==np.float32 and vertices.ndim==2 and vertices.shape[1]==3 and 0<len(vertices)<=1_957_891,'Bounded original F32 positions')
    need(faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3 and 0<len(faces)<=4_173_584,'Bounded I32 triangles')
    low=np.full(3,np.inf);high=-low
    for start in range(0,len(vertices),32768):
        v=vertices[start:start+32768];need(np.isfinite(v).all(),'Nonfinite position');low=np.minimum(low,v.min(0));high=np.maximum(high,v.max(0))
    area=0.;zero=0
    for start in range(0,len(faces),32768):
        f=faces[start:start+32768];need((f>=0).all() and (f<len(vertices)).all(),'Index outside original vertex table')
        p=vertices[f].astype(np.float64);c=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.linalg.norm(c,axis=1)
        zero+=int((length==0).sum());area+=float(length.sum()*.5)
    need(zero==0,'Zero-area or repeated-index triangles')
    return dict(vertices=len(vertices),triangles=len(faces),bounds=[low.tolist(),high.tolist()],area=area,zero_area=zero)

def compact(faces,n):
    need(type(n)is int and 0<n<=1_957_891,'Bounded original vertex count')
    need(faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3,'I32 triangle table')
    used=np.zeros(n,np.bool_)
    for s in range(0,len(faces),32768):
        a=faces[s:s+32768];need((a>=0).all() and (a<n).all(),'Index outside source');used[a.ravel()]=True
    ids=np.flatnonzero(used).astype(np.int32);inverse=np.full(n,-1,np.int32);inverse[ids]=np.arange(len(ids),dtype=np.int32)
    return ids,inverse[faces]

def component_lineage(faces,labels,expected):
    need(labels.dtype==np.int32 and labels.ndim==1,'Original source component labels')
    counts=np.zeros(int(labels.max())+1,np.int64)
    for s in range(0,len(faces),32768):
        f=faces[s:s+32768];need((f>=0).all() and (f<len(labels)).all(),'Foreign component index')
        x=labels[f];need((x==x[:,:1]).all(),'New triangle bridges disconnected original surfaces')
        np.add.at(counts,x[:,0],1)
    actual=np.flatnonzero(counts).tolist();expected=list(map(int,expected))
    return dict(all_components_retained=actual==sorted(expected),missing=sorted(set(expected)-set(actual)),
                foreign=sorted(set(actual)-set(expected)),faces_by_original_component={str(i):int(counts[i]) for i in actual})

def stats(error):
    a=np.asarray(error,np.float64);need(a.size>0 and np.isfinite(a).all(),'Finite nonempty error population')
    return dict(rmse=float(np.sqrt(np.mean(a*a))),p95=float(np.percentile(np.abs(a),95)),maximum=float(np.abs(a).max()))

def gate(row,limits):return all(row[k]<=x for k,x in zip(('rmse','p95','maximum'),limits))

def linear_to_encoded(a):
    a=np.clip(np.asarray(a,np.float64),0,1)
    return np.where(a<=.0031308,12.92*a,1.055*np.power(a,1/2.4)-.055)

def oriented_keys(faces,n):
    need(type(n)is int and 0<n<=1_957_891,'Bounded key radix')
    result=np.empty(len(faces),np.uint64)
    for s in range(0,len(faces),32768):
        f=faces[s:s+32768];need((f>=0).all() and (f<n).all(),'Foreign oriented face index')
        j=np.argmin(f,axis=1);r=np.arange(len(f));a=f[r,j].astype(np.uint64);b=f[r,(j+1)%3].astype(np.uint64);c=f[r,(j+2)%3].astype(np.uint64)
        result[s:s+len(f)]=(a*np.uint64(n)+b)*np.uint64(n)+c
    return result

def upper_bounds(points,referenced_vertices):
    """Exact nearest-vertex distances upper-bound sampled surface distance."""
    from scipy.spatial import cKDTree
    need(points.ndim==referenced_vertices.ndim==2 and points.shape[1]==referenced_vertices.shape[1]==3,'Point/vertex coordinates')
    need(np.isfinite(points).all() and np.isfinite(referenced_vertices).all() and len(referenced_vertices)>0,'Finite referenced positions')
    tree=cKDTree(referenced_vertices.astype(np.float64),compact_nodes=False,balanced_tree=False)
    distance,ids=tree.query(points.astype(np.float64),k=1,eps=0,workers=4)
    return distance,ids

def reverse_samples(vertices,faces,areas,n=2048):
    need(type(n)is int and 0<n<=2048 and areas.shape==(len(faces),) and np.isfinite(areas).all() and (areas>0).all(),'Positive complete candidate areas and bounded reverse count')
    chosen=np.searchsorted(np.cumsum(areas),(np.arange(n)+.5)*areas.sum()/n).astype(np.int32)
    uv=np.random.default_rng(20261022).random((n,2));root=np.sqrt(uv[:,0])
    bary=np.column_stack((1-root,root*(1-uv[:,1]),root*uv[:,1]))
    points=np.einsum('ni,nij->nj',bary,vertices[faces[chosen]].astype(np.float64)).astype(np.float32)
    need((bary>0).all() and (bary<1).all() and np.isfinite(points).all(),'Strict finite reverse interiors')
    return chosen,bary,points
