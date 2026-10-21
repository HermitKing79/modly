"""Bounded original-ID selection/protection; contains no simplifier call."""
import sys
sys.dont_write_bytecode=True
import numpy as np
MAX_FACES=6000
SEED=(1456609,1456613)
RINGS=3

def need(ok,message):
    if not ok:raise ValueError(message)

def edge_keys(f,n):
    a=f[:,[0,1,2]];b=f[:,[1,2,0]]
    return np.minimum(a,b).astype(np.uint64)*np.uint64(n)+np.maximum(a,b).astype(np.uint64)

def member(values,sorted_keys):
    if len(sorted_keys)==0:return np.zeros(values.shape,np.bool_)
    at=np.searchsorted(sorted_keys,values);return (at<len(sorted_keys))&(sorted_keys[np.minimum(at,len(sorted_keys)-1)]==values)

def select(f,n):
    initial=[];edge_faces=[]
    for s in range(0,len(f),32768):
        b=f[s:s+32768];initial.extend((s+np.flatnonzero(np.isin(b,SEED).any(1))).tolist())
        edge_faces.extend((s+np.flatnonzero((b==SEED[0]).any(1)&(b==SEED[1]).any(1))).tolist())
    need(len(edge_faces)==4 and 0<len(initial)<=MAX_FACES,'Complete original four-face edge and bounded endpoint stars')
    chosen=np.array(sorted(initial),np.int32);rings=[]
    for ring in range(RINGS):
        wanted=np.unique(edge_keys(f[chosen],n));found=[]
        for s in range(0,len(f),32768):
            hit=member(edge_keys(f[s:s+32768],n),wanted).any(1)
            found.extend((s+np.flatnonzero(hit)).tolist())
            need(len(found)<=MAX_FACES,'Fixed original patch exceeds6000; no truncation or alternate rings')
        expanded=np.array(sorted(found),np.int32);addition=np.setdiff1d(expanded,chosen,assume_unique=True)
        rings.append(addition);chosen=expanded
    return chosen,np.array(initial,np.int32),np.array(edge_faces,np.int32),rings

def local_edges(f,n):
    k=edge_keys(f,n);flat=k.ravel();order=np.argsort(flat,kind='stable')
    unique,starts,counts=np.unique(flat[order],return_index=True,return_counts=True)
    rows=order//3;directions=(f[:,[0,1,2]]<f[:,[1,2,0]]).ravel()[order]
    return unique,starts,counts,rows,directions

def same_position_classes(v):
    # Canonical +0 also conservatively closes -0 aliases. No epsilon welding.
    z=v.copy();z[z==0]=0
    _,inverse,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    return inverse,counts

def classify(v,f,source_ids,source_faces,outer_rows,global_keys,global_inc,N):
    need(v.dtype==np.float32 and v.ndim==2 and v.shape[1]==3 and np.isfinite(v).all(),'Finite original F32 patch positions')
    need(f.dtype==source_faces.dtype==np.int32 and f.shape==source_faces.shape and 0<len(f)<=MAX_FACES and (f>=0).all() and (f<len(v)).all(),'Bounded original I32 patch faces')
    need(np.array_equal(source_ids[f],source_faces),'Exact original vertex and oriented corner correspondence')
    n=len(v);k,starts,counts,face_rows,direction=local_edges(source_faces,N)
    at=np.searchsorted(global_keys,k);need((at<len(global_keys)).all() and np.array_equal(global_keys[at],k),'All patch edges map to actual original graph')
    global_count=global_inc[at];need((counts<=global_count).all(),'Patch incidence cannot exceed original')
    opposite=np.zeros(len(k),np.bool_)
    for j in np.flatnonzero(counts==2):opposite[j]=bool(direction[starts[j]]!=direction[starts[j]+1])
    consistent=(global_count==2)&(counts==2)&opposite
    crop=counts<global_count
    real_boundary=global_count==1
    # Edge-connected consistently oriented sheet labels, without joining sheets
    # through a shared vertex alone or crossing non-manifold/winding interfaces.
    parent=np.arange(len(f),dtype=np.int32)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    for j in np.flatnonzero(consistent):
        a,b=map(int,face_rows[starts[j]:starts[j]+2]);a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    sheets=np.array([find(i) for i in range(len(f))],np.int32)
    # Mark global junction/boundary vertices, even when the crop hides faces.
    global_nm=global_keys[global_inc>2];global_boundary=global_keys[global_inc==1]
    nm_vertices=np.unique(np.concatenate((global_nm//np.uint64(N),global_nm%np.uint64(N))))
    boundary_vertices=np.unique(np.concatenate((global_boundary//np.uint64(N),global_boundary%np.uint64(N))))
    bad_vertex=np.isin(source_ids,nm_vertices);boundary_vertex=np.isin(source_ids,boundary_vertices)
    first_sheet=np.full(n,len(f),np.int32);last_sheet=np.full(n,-1,np.int32)
    np.minimum.at(first_sheet,f.ravel(),np.repeat(sheets,3));np.maximum.at(last_sheet,f.ravel(),np.repeat(sheets,3))
    multi=first_sheet!=last_sheet
    # A global sheet can reconnect far away from a pinched vertex. Inspect the
    # local incident-face fan too; ambiguous crop-split fans are locked rather
    # than claimed to be genuine global pinches.
    vertex_faces=[[] for _ in range(n)];vertex_edges=[[] for _ in range(n)]
    for row,tri in enumerate(f):
        for x in tri:vertex_faces[int(x)].append(row)
    for j in np.flatnonzero(consistent):
        a_global,b_global=divmod(int(k[j]),N)
        for x in (int(np.searchsorted(source_ids,a_global)),int(np.searchsorted(source_ids,b_global))):vertex_edges[x].append(j)
    local_multi=np.zeros(n,np.bool_)
    for x,star in enumerate(vertex_faces):
        fan={row:row for row in star}
        def fan_root(row):
            while fan[row]!=row:fan[row]=fan[fan[row]];row=fan[row]
            return row
        for j in vertex_edges[x]:
            a,b=map(int,face_rows[starts[j]:starts[j]+2]);a,b=fan_root(a),fan_root(b)
            if a!=b:fan[max(a,b)]=min(a,b)
        local_multi[x]=len({fan_root(row) for row in star})>1
    aliases,alias_counts=same_position_classes(v);alias_vertex=alias_counts[aliases]>1
    protected=(bad_vertex[f]|boundary_vertex[f]|multi[f]|local_multi[f]|alias_vertex[f]).any(1)
    protected[np.asarray(outer_rows,np.int32)]=True
    interface=crop|real_boundary|((global_count==2)&~opposite)
    for j in np.flatnonzero(interface):protected[face_rows[starts[j]:starts[j]+counts[j]]]=True
    protected_vertices=np.zeros(n,np.bool_);protected_vertices[f[protected].ravel()]=True
    # Guard all neighbours of immutable protected vertices; close position
    # classes to prevent native wedge/position alias bypasses.
    guard_faces=protected_vertices[f].any(1)
    locked=np.zeros(n,np.bool_);locked[f[guard_faces].ravel()]=True
    locked=np.isin(aliases,np.unique(aliases[locked]))
    eligible=[];link_rejected=[]
    for j in np.flatnonzero(consistent):
        a_global,b_global=divmod(int(k[j]),N);a=int(np.searchsorted(source_ids,a_global));b=int(np.searchsorted(source_ids,b_global))
        rows=face_rows[starts[j]:starts[j]+2]
        if locked[a] or locked[b] or protected[rows].any() or sheets[rows[0]]!=sheets[rows[1]]:continue
        star_a=f[(f==a).any(1)];star_b=f[(f==b).any(1)]
        na=np.unique(star_a);nb=np.unique(star_b)
        shared=np.setdiff1d(np.intersect1d(na,nb),np.array([a,b],np.int32))
        opposite_vertices=np.setdiff1d(np.unique(f[rows]),np.array([a,b],np.int32))
        if len(opposite_vertices)!=2 or not np.array_equal(shared,opposite_vertices):link_rejected.append(j);continue
        c,d=map(int,opposite_vertices)
        # A shared link EDGE also violates the full simplicial link condition;
        # shared-neighbour equality alone would incorrectly admit tetrahedra.
        shared_link_edge=(((star_a==c).any(1)&(star_a==d).any(1)).any() and ((star_b==c).any(1)&(star_b==d).any(1)).any())
        if shared_link_edge:link_rejected.append(j);continue
        eligible.append(j)
    # A disjoint closed-neighbourhood certificate is only an estimate, not a
    # simplification. It freezes an upper target without mutating any faces.
    selected=[];busy=np.zeros(n,np.bool_)
    for j in eligible:
        a_global,b_global=divmod(int(k[j]),N);a=int(np.searchsorted(source_ids,a_global));b=int(np.searchsorted(source_ids,b_global))
        neighbourhood=np.unique(f[((f==a)|(f==b)).any(1)])
        if not busy[neighbourhood].any():selected.append(j);busy[neighbourhood]=True
    eligible_faces=np.unique(np.concatenate([face_rows[starts[j]:starts[j]+2] for j in eligible])) if eligible else np.empty(0,np.int32)
    needed=int(np.ceil(.2*len(eligible_faces)/2))
    ready=bool(len(eligible_faces)>0 and len(selected)>=max(1,needed))
    summary=dict(original_patch_triangles=len(f),original_patch_vertices=n,sheet_components=int(len(np.unique(sheets))),global_nonmanifold_patch_edges=int((global_count>2).sum()),real_boundary_patch_edges=int(real_boundary.sum()),artificial_crop_edges=int(crop.sum()),orientation_inconsistent_two_face_edges=int(((global_count==2)&(counts==2)&~opposite).sum()),consistent_two_face_interior_edges=int(consistent.sum()),global_junction_vertices_in_patch=int(bad_vertex.sum()),global_boundary_vertices_in_patch=int(boundary_vertex.sum()),multi_sheet_vertices=int(multi.sum()),local_multiple_fan_vertices=int(local_multi.sum()),position_alias_vertices=int(alias_vertex.sum()),protected_faces=int(protected.sum()),protected_vertices=int(protected_vertices.sum()),guard_locked_vertices=int(locked.sum()),unlocked_vertices=int((~locked).sum()),unprotected_faces=int((~protected).sum()),strict_eligible_edges=len(eligible),eligible_face_population=len(eligible_faces),link_condition_rejected=len(link_rejected),non_touching_certified_edges=len(selected),collapses_needed_for_20pct_eligible_faces=needed,eligible_interior_reduction_admission=ready)
    return summary,dict(protected_faces=protected,protected_vertices=protected_vertices,vertex_locks=locked,sheet_labels=sheets,edge_keys=k,patch_edge_incidence=counts,global_edge_incidence=global_count,eligible_edge_rows=np.array(eligible,np.int32),certified_edge_rows=np.array(selected,np.int32))
