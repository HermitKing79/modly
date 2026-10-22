"""Read-only global incidence, oriented fan and full simplicial-link census."""
import sys
sys.dont_write_bytecode=True
import gc,time
import numpy as np
BATCH=32768
MAX_FREE_DEGREE=128
MAX_LINK_COMPARISONS=4_000_000_000

def need(ok,message):
    if not ok:raise ValueError(message)

def keys_for(f,n):
    a=f[:,[0,1,2]];b=f[:,[1,2,0]]
    return np.minimum(a,b).astype(np.uint64)*np.uint64(n)+np.maximum(a,b).astype(np.uint64)

def member(q,sorted_values):
    if len(sorted_values)==0:return np.zeros(q.shape,np.bool_)
    p=np.searchsorted(sorted_values,q);return (p<len(sorted_values))&(sorted_values[np.minimum(p,len(sorted_values)-1)]==q)

def unordered(f,n):
    t=np.sort(f,axis=1).astype(np.uint64)
    return (t[:,0]*np.uint64(n)+t[:,1])*np.uint64(n)+t[:,2]

def occurrences(f,keys,inc,n,check):
    need(len(f)*3<2**31 and len(keys)<2**31 and int(inc.sum(dtype=np.int64))==len(f)*3,'int32 occurrence capacity and cache total')
    slots=np.empty(len(f)*3,np.int32);direction=np.empty(len(f)*3,np.bool_)
    used=np.zeros(n,np.bool_);degree=np.zeros(n,np.int32)
    for s in range(0,len(f),BATCH):
        check();b=f[s:s+BATCH];need((b>=0).all() and (b<n).all(),'Original face range')
        used[b.ravel()]=True;np.add.at(degree,b.ravel(),1)
        k=keys_for(b,n);at=np.searchsorted(keys,k)
        need((at<len(keys)).all() and np.array_equal(keys[at],k),'Exact cached endpoint key correspondence')
        slots[3*s:3*(s+len(b))]=at.ravel();direction[3*s:3*(s+len(b))]=(b[:,[0,1,2]]<b[:,[1,2,0]]).ravel()
    actual=np.bincount(slots,minlength=len(keys));need(np.array_equal(actual,inc),'Complete original cached incidence equality');del actual
    order=np.argsort(slots,kind='stable').astype(np.int32);del slots
    offsets=np.empty(len(keys)+1,np.int32);offsets[0]=0;np.cumsum(inc,dtype=np.int32,out=offsets[1:])
    two=np.flatnonzero(inc==2).astype(np.int32);oa=order[offsets[two]];ob=order[offsets[two]+1]
    consistent=np.zeros(len(keys),np.bool_);consistent[two]=direction[oa]!=direction[ob]
    conflict=two[direction[oa]==direction[ob]]
    del oa,ob,two,direction;gc.collect()
    return order,offsets,used,degree,consistent,conflict

def component_labels(nodes,left,right):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    need(len(left)==len(right) and nodes<2**31,'Bounded sparse adjacency')
    graph=coo_matrix((np.ones(len(left),np.uint8),(left.astype(np.int32),right.astype(np.int32))),shape=(nodes,nodes)).tocsr()
    count,labels=connected_components(graph,directed=False,return_labels=True)
    del graph;gc.collect();return int(count),labels.astype(np.int32,copy=False)

def fan_and_sheets(f,order,offsets,consistent,used,check):
    good=np.flatnonzero(consistent).astype(np.int32);oa=order[offsets[good]];ob=order[offsets[good]+1]
    fa=oa//3;fb=ob//3
    check();sheet_count,sheets=component_labels(len(f),np.minimum(fa,fb),np.maximum(fa,fb));check()
    # Each edge occurrence starts at its own corner. Opposite winding pairs
    # must be matched by actual original vertex ID, never by corner number.
    ka=oa%3;kb=ob%3;va=f[fa,ka];vb=f[fb,kb]
    start_same=va==vb
    lb=fb*3+np.where(start_same,kb,(kb+1)%3)
    hb=fb*3+np.where(start_same,(kb+1)%3,kb)
    la=fa*3+ka;ha=fa*3+(ka+1)%3
    need(np.array_equal(f.ravel()[la],f.ravel()[lb]) and np.array_equal(f.ravel()[ha],f.ravel()[hb]),'Both fan links preserve exact original vertex ID')
    left=np.concatenate((np.minimum(la,lb),np.minimum(ha,hb)));right=np.concatenate((np.maximum(la,lb),np.maximum(ha,hb)))
    del good,oa,ob,fa,fb,ka,kb,va,vb,start_same,la,lb,ha,hb;gc.collect()
    fan_count,fan_labels=component_labels(len(f)*3,left,right);del left,right;check()
    fan_min=np.full(len(used),len(f)*3,np.int32);fan_max=np.full(len(used),-1,np.int32)
    sheet_min=np.full(len(used),len(f),np.int32);sheet_max=np.full(len(used),-1,np.int32)
    for s in range(0,len(f),BATCH):
        check();ids=f[s:s+BATCH].ravel();a=fan_labels[3*s:3*(s+len(f[s:s+BATCH]))]
        np.minimum.at(fan_min,ids,a);np.maximum.at(fan_max,ids,a)
        x=np.repeat(sheets[s:s+BATCH],3);np.minimum.at(sheet_min,ids,x);np.maximum.at(sheet_max,ids,x)
    multi_fan=used&(fan_min!=fan_max);multi_sheet=used&(sheet_min!=sheet_max)
    del fan_labels,fan_min,fan_max,sheet_min,sheet_max;gc.collect()
    return sheets,multi_fan,multi_sheet,dict(oriented_sheet_components=sheet_count,corner_fan_components=fan_count,multiple_fan_vertices=int(multi_fan.sum()),multiple_sheet_vertices=int(multi_sheet.sum()))

def touched(f,vertices,check):
    a=np.empty(len(f),np.bool_)
    for s in range(0,len(f),BATCH):check();a[s:s+BATCH]=vertices[f[s:s+BATCH]].any(1)
    return a

def from_faces(f,mask,n,check):
    a=np.zeros(n,np.bool_)
    for s in range(0,len(f),BATCH):check();a[f[s:s+BATCH][mask[s:s+BATCH]].ravel()]=True
    return a

def protection(v,f,keys,inc,order,offsets,used,multi_fan,multi_sheet,conflict,check):
    n=len(v);nm=np.zeros(n,np.bool_);boundary=np.zeros(n,np.bool_)
    for selector,target in ((inc>2,nm),(inc==1,boundary)):
        k=keys[selector];target[k//np.uint64(n)]=True;target[k%np.uint64(n)]=True
    junction_faces=touched(f,nm,check);boundary_faces=touched(f,boundary,check)
    boundary_edge_faces=np.zeros(len(f),np.bool_);boundary_edge_faces[order[offsets[np.flatnonzero(inc==1)]]//3]=True
    # Referenced equal-position classes only. Unused source rows are preserved
    # but are not phantom interfaces in a mesh which never indexes them.
    ids=np.flatnonzero(used);xyz=v[ids].copy();xyz[xyz==0]=0
    _,inverse,counts=np.unique(xyz,axis=0,return_inverse=True,return_counts=True);del xyz
    alias=np.zeros(n,np.bool_);alias[ids]=counts[inverse]>1
    protected=junction_faces|boundary_faces|touched(f,multi_fan|multi_sheet|alias,check)
    protected[order[offsets[conflict]]//3]=True;protected[order[offsets[conflict]+1]//3]=True
    protected_v=from_faces(f,protected,n,check);guard_faces=touched(f,protected_v,check)
    locks=from_faces(f,guard_faces,n,check)
    locked_classes=np.zeros(len(counts),np.bool_);locked_classes[inverse[locks[ids]]]=True;locks[ids]=locked_classes[inverse]
    unprotected=~protected;free_triangles=unprotected&~touched(f,locks,check)
    guard_affected=unprotected&~free_triangles
    # A guarded corner does not make the opposite edge unavailable. These are
    # face-cohort counts; only full_links certifies actual endpoint eligibility.
    return dict(nonmanifold_vertices=nm,boundary_vertices=boundary,nonmanifold_touch_faces=junction_faces,boundary_touch_faces=boundary_faces,boundary_edge_touch_faces=boundary_edge_faces,alias_vertices=alias,protected_faces=protected,protected_vertices=protected_v,locked_vertices=locks,pre_guard_interior_faces=unprotected,fully_unlocked_interior_faces=free_triangles,guard_affected_interior_faces=guard_affected)

def full_links(f,keys,order,offsets,consistent,protected,locks,n,check):
    from scipy.sparse import coo_matrix
    a=(keys//np.uint64(n)).astype(np.int32);b=(keys%np.uint64(n)).astype(np.int32)
    possible=consistent&~locks[a]&~locks[b]
    ee=np.flatnonzero(possible).astype(np.int32)
    if len(ee):
        rows=order[offsets[ee]]//3;other=order[offsets[ee]+1]//3
        ee=ee[~protected[rows]&~protected[other]]
    tested=len(ee);eligible=[];rejected_neighbour=0;rejected_link_edge=0
    if not tested:return np.empty(0,np.int32),dict(pre_link_candidate_edges=0,link_tested_edges=0,certified_edges=0,link_comparisons=0,maximum_free_degree=0),None
    graph=coo_matrix((np.ones(len(a)*2,np.uint8),(np.concatenate((a,b)),np.concatenate((b,a)))),shape=(n,n)).tocsr()
    degree=np.diff(graph.indptr);D=int(max(degree[a[ee]].max(),degree[b[ee]].max()))
    need(D<=MAX_FREE_DEGREE,'Incomplete: free-neighbour degree exceeds128')
    comparisons=int(tested)*D*D;need(comparisons<=MAX_LINK_COMPARISONS,'Incomplete: padded link comparison work exceeds4e9')
    triangle_keys=np.empty(len(f),np.uint64)
    for s in range(0,len(f),BATCH):check();triangle_keys[s:s+BATCH]=unordered(f[s:s+BATCH],n)
    triangle_keys.sort();batch=min(8192,max(1,8*1024**2//max(1,D*D)));r=np.arange(D,dtype=np.int32)
    for s in range(0,tested,batch):
        check();edge=ee[s:s+batch];x,y=a[edge],b[edge]
        fx=order[offsets[edge]]//3;fy=order[offsets[edge]+1]//3
        tx=f[fx];ty=f[fy]
        c=tx[np.arange(len(edge)),np.argmax((tx!=x[:,None])&(tx!=y[:,None]),axis=1)]
        d=ty[np.arange(len(edge)),np.argmax((ty!=x[:,None])&(ty!=y[:,None]),axis=1)]
        ix=graph.indptr[x,None]+r;iy=graph.indptr[y,None]+r
        nx=graph.indices[np.minimum(ix,len(graph.indices)-1)];ny=graph.indices[np.minimum(iy,len(graph.indices)-1)]
        nx=np.where(r<degree[x,None],nx,-1);ny=np.where(r<degree[y,None],ny,-2)
        # Distinct -1/-2 padding cannot match itself or any valid neighbour.
        count=(nx[:,:,None]==ny[:,None,:]).sum(axis=(1,2))
        good=(c!=d)&(count==2)&(nx==c[:,None]).any(1)&(nx==d[:,None]).any(1)&(ny==c[:,None]).any(1)&(ny==d[:,None]).any(1)
        rejected_neighbour+=int((~good).sum())
        shared_edge=member(unordered(np.column_stack((x,c,d)),n),triangle_keys)&member(unordered(np.column_stack((y,c,d)),n),triangle_keys)
        rejected_link_edge+=int((good&shared_edge).sum());eligible.append(edge[good&~shared_edge])
    result=np.concatenate(eligible) if eligible else np.empty(0,np.int32)
    del graph,triangle_keys;gc.collect()
    return result,dict(pre_link_candidate_edges=tested,link_tested_edges=tested,certified_edges=len(result),rejected_common_vertex_link=rejected_neighbour,rejected_common_link_edge=rejected_link_edge,link_comparisons=comparisons,maximum_free_degree=D,batch_edge_limit=batch),None

def octants(points,mid):return ((points[:,0]>=mid[0]).astype(np.uint8)+2*(points[:,1]>=mid[1]).astype(np.uint8)+4*(points[:,2]>=mid[2]).astype(np.uint8))

def distributions(v,f,labels,masks,keys,eligible,check):
    mid=(v.min(0).astype(np.float64)+v.max(0))/2
    regions=np.empty(len(f),np.uint8);components=np.empty(len(f),np.int32)
    for s in range(0,len(f),BATCH):
        check();b=f[s:s+BATCH];ll=labels[b];need((ll==ll[:,:1]).all(),'Original component identity for every face')
        components[s:s+len(b)]=ll[:,0];regions[s:s+len(b)]=octants(v[b].mean(1,dtype=np.float64),mid)
    face_components=np.unique(components);by_component={};by_region={}
    stage={name:a for name,a in masks.items() if len(a)==len(f)}
    for name,a in dict(original_faces=np.ones(len(f),np.bool_),**stage).items():
        co=np.bincount(components[a],minlength=int(labels.max())+1);sp=np.bincount(regions[a],minlength=8)
        for c in face_components:by_component.setdefault(str(int(c)),{})[name]=int(co[c])
        for i in range(8):by_region.setdefault(str(i),{})[name]=int(sp[i])
    for name,a in masks.items():
        if len(a)!=len(v):continue
        co=np.bincount(labels[a],minlength=int(labels.max())+1);sp=np.bincount(octants(v[a].astype(np.float64),mid),minlength=8)
        for c in face_components:by_component[str(int(c))][name]=int(co[c])
        for i in range(8):by_region[str(i)][name]=int(sp[i])
    k=keys[eligible];x=(k//np.uint64(len(v))).astype(np.int32);y=(k%np.uint64(len(v))).astype(np.int32)
    need(np.array_equal(labels[x],labels[y]),'Eligible original source component identity')
    co=np.bincount(labels[x],minlength=int(labels.max())+1);sp=np.bincount(octants((v[x].astype(np.float64)+v[y])/2,mid),minlength=8)
    for c in face_components:by_component[str(int(c))]['certified_edges']=int(co[c])
    for i in range(8):by_region[str(i)]['certified_edges']=int(sp[i])
    return dict(original_face_bearing_vertex_components=len(face_components),by_original_component=by_component,by_octant=by_region,octant_definition='bit0 X>=AABBmidX; bit1 Y>=midY; bit2 Z>=midZ. Faces use centroids; vertices use positions; edges use endpoint midpoints. Separate denominators.',unused_source_vertices=int((np.bincount(f.ravel(),minlength=len(v))==0).sum())),regions
