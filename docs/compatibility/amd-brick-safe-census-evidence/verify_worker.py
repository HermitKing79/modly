"""Independent bounded durable readback; no census engine or graph rebuild."""
import sys
sys.dont_write_bytecode=True
import json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E))
BATCH=32768
VERTEX_MASKS=('nonmanifold_vertices','boundary_vertices','alias_vertices','protected_vertices','locked_vertices','multiple_fan_vertices','multiple_sheet_vertices')
FACE_MASKS=('nonmanifold_touch_faces','boundary_touch_faces','boundary_edge_touch_faces','protected_faces','pre_guard_interior_faces','fully_unlocked_interior_faces','guard_affected_interior_faces','certified_edge_touch_faces')

def need(ok,message):
    if not ok:raise ValueError(message)

def verify_occurrences(f,keys,inc,order,offsets,consistent,n,check,rows=BATCH,sheets=None):
    import numpy as np
    m=len(f)*3;e=len(keys)
    need(f.dtype==np.int32 and f.ndim==2 and f.shape[1]==3 and 0<m<2**31,'Source triangle schema')
    need(order.dtype==offsets.dtype==np.int32 and order.shape==(m,) and offsets.shape==(e+1,),'Occurrence/offset schema')
    need(consistent.dtype==np.bool_ and consistent.shape==(e,) and keys.dtype==np.uint64,'Cache/winding schema')
    need(offsets[0]==0 and offsets[-1]==m and np.array_equal(np.diff(offsets),inc) and (inc>0).all(),'Exact cached group multiplicities')
    need((keys[1:]>keys[:-1]).all(),'Sorted unique cached keys')
    seen=np.zeros(m,np.bool_);degree=np.zeros(n,np.int32);conflict_faces=np.zeros(len(f),np.bool_)
    previous=-1;previous_group=-1
    for first in range(0,m,rows):
        check();stop=min(m,first+rows);o=order[first:stop]
        need((o>=0).all() and (o<m).all() and not seen[o].any(),'Occurrence permutation range/repetition')
        group=np.searchsorted(offsets[1:],np.arange(first,stop),side='right')
        need((o[1:][group[1:]==group[:-1]]>o[:-1][group[1:]==group[:-1]]).all(),'Stable within-edge occurrence order')
        need(previous_group!=int(group[0]) or previous<int(o[0]),'Stable group across readback batches')
        face=o//3;corner=o%3;a=f[face,corner];b=f[face,(corner+1)%3]
        need((a>=0).all() and (a<n).all() and (b>=0).all() and (b<n).all(),'Source index bounds')
        actual=np.minimum(a,b).astype(np.uint64)*np.uint64(n)+np.maximum(a,b).astype(np.uint64)
        need(np.array_equal(actual,keys[group]),'Every occurrence maps to its exact original endpoint key')
        seen[o]=True;np.add.at(degree,a,1);previous=int(o[-1]);previous_group=int(group[-1])
    need(seen.all(),'Complete occurrence permutation')
    conflicts=0
    for first in range(0,e,BATCH):
        check();ids=np.arange(first,min(e,first+BATCH));two=ids[inc[ids]==2];expected=np.zeros(len(ids),np.bool_)
        x=order[offsets[two]];y=order[offsets[two]+1]
        dx=f[x//3,x%3]<f[x//3,(x%3+1)%3];dy=f[y//3,y%3]<f[y//3,(y%3+1)%3]
        expected[two-first]=dx!=dy
        need(np.array_equal(consistent[ids],expected),'Every saved consistent edge has opposite original winding')
        bad=two[dx==dy];conflicts+=len(bad);conflict_faces[order[offsets[bad]]//3]=True;conflict_faces[order[offsets[bad]+1]//3]=True
        if sheets is not None:need(np.array_equal(sheets[x[dx!=dy]//3],sheets[y[dx!=dy]//3]),'Consistent face neighbours share their saved sheet label')
    return degree,conflict_faces,dict(orientation_conflict_edges=conflicts,consistent_two_face_edges=int(consistent.sum()))

def verify_masks(f,keys,inc,order,offsets,consistent,sheets,masks,degree,conflict_faces,check,eligible=None):
    import numpy as np
    n=len(degree);used=degree>0
    for name in VERTEX_MASKS:need(masks[name].dtype==np.bool_ and masks[name].shape==(n,) and not (masks[name]&~used).any(),'Referenced vertex mask: '+name)
    for name in FACE_MASKS:need(masks[name].dtype==np.bool_ and masks[name].shape==(len(f),),'Face mask: '+name)
    need(sheets.dtype==np.int32 and sheets.shape==(len(f),) and (sheets>=0).all() and (sheets<len(f)).all(),'Saved sheet-label schema')
    nm=np.zeros(n,np.bool_);boundary=np.zeros(n,np.bool_);edge_faces=np.zeros(len(f),np.bool_)
    for first in range(0,len(keys),BATCH):
        check();row=np.arange(first,min(first+BATCH,len(keys)));k=keys[row];a=(k//np.uint64(n)).astype(np.int32);b=(k%np.uint64(n)).astype(np.int32)
        need((a>=0).all() and (a<n).all() and (b>=0).all() and (b<n).all(),'Cached endpoint range')
        nm[a[inc[row]>2]]=True;nm[b[inc[row]>2]]=True;boundary[a[inc[row]==1]]=True;boundary[b[inc[row]==1]]=True
        edge_faces[order[offsets[row[inc[row]==1]]]//3]=True
    need(np.array_equal(nm,masks['nonmanifold_vertices']) and np.array_equal(boundary,masks['boundary_vertices']),'Cache-derived boundary/junction vertices')
    need(np.array_equal(edge_faces,masks['boundary_edge_touch_faces']),'Boundary-edge face mask')
    expected_protected=np.empty(len(f),np.bool_);pv=np.zeros(n,np.bool_);lo=np.full(n,len(f),np.int32);hi=np.full(n,-1,np.int32)
    danger=nm|boundary|masks['multiple_fan_vertices']|masks['multiple_sheet_vertices']|masks['alias_vertices']
    for first in range(0,len(f),BATCH):
        check();b=f[first:first+BATCH];part=slice(first,first+len(b))
        need(np.array_equal(nm[b].any(1),masks['nonmanifold_touch_faces'][part]) and np.array_equal(boundary[b].any(1),masks['boundary_touch_faces'][part]),'Incident source-face mask identity')
        expected_protected[part]=danger[b].any(1)|conflict_faces[part];pv[b[expected_protected[part]].ravel()]=True
        label=np.repeat(sheets[part],3);np.minimum.at(lo,b.ravel(),label);np.maximum.at(hi,b.ravel(),label)
    need(np.array_equal(used&(lo!=hi),masks['multiple_sheet_vertices']),'Per-vertex saved sheet-label reduction')
    need(np.array_equal(expected_protected,masks['protected_faces']) and np.array_equal(pv,masks['protected_vertices']),'Exact protected-face/vertex boolean closure')
    locks=np.zeros(n,np.bool_)
    for first in range(0,len(f),BATCH):check();b=f[first:first+BATCH];locks[b[pv[b].any(1)].ravel()]=True
    # Every referenced alias vertex is already incident to a protected face;
    # native position-class closure therefore cannot add another guard vertex.
    need(np.array_equal(locks,masks['locked_vertices']),'Exact complete one-ring guard locks')
    for first in range(0,len(f),BATCH):
        check();b=f[first:first+BATCH];part=slice(first,first+len(b));interior=~expected_protected[part];affected=locks[b].any(1)
        for name,value in [('pre_guard_interior_faces',interior),('fully_unlocked_interior_faces',interior&~affected),('guard_affected_interior_faces',interior&affected)]:need(np.array_equal(value,masks[name][part]),'Stage mask arithmetic: '+name)
    quasi=[]
    for first in range(0,len(keys),BATCH):
        check();row=np.arange(first,min(first+BATCH,len(keys)));k=keys[row];a=(k//np.uint64(n)).astype(np.int32);b=(k%np.uint64(n)).astype(np.int32)
        candidate=row[consistent[row]&~locks[a]&~locks[b]]
        if len(candidate):candidate=candidate[~expected_protected[order[offsets[candidate]]//3]&~expected_protected[order[offsets[candidate]+1]//3]]
        quasi.append(candidate.astype(np.int32))
    touch=np.zeros(len(f),np.bool_)
    if eligible is not None and len(eligible):touch[order[offsets[eligible]]//3]=True;touch[order[offsets[eligible]+1]//3]=True
    need(np.array_equal(touch,masks['certified_edge_touch_faces']),'Certified edge/face union')
    return np.concatenate(quasi)

def link_from_stars(a,b,star_a,star_b,c,d):
    if a==b or c==d:return False
    common=(set(map(int,star_a.ravel()))&set(map(int,star_b.ravel())))-{a,b}
    shared_edge=any(c in t and d in t for t in star_a) and any(c in t and d in t for t in star_b)
    return common=={c,d} and not shared_edge

def link_witnesses(f,keys,order,offsets,edge_rows,n,check):
    import numpy as np
    endpoints=sorted({int(x) for k in keys[edge_rows] for x in (k//np.uint64(n),k%np.uint64(n))});stars={x:[] for x in endpoints};total=0
    for first in range(0,len(f),BATCH):
        check();block=f[first:first+BATCH]
        for t in block[np.isin(block,endpoints).any(1)]:
            for x in t:
                if int(x) in stars:stars[int(x)].append(tuple(map(int,t)));total+=1
        need(total<=32768,'Bounded independent witness stars; no truncation')
    answer=[]
    for edge in edge_rows:
        a,b=divmod(int(keys[edge]),n);rows=order[offsets[edge]:offsets[edge]+2]//3
        opposite=[set(map(int,f[row]))-{a,b} for row in rows];need(all(len(x)==1 for x in opposite),'Witness opposite source corners')
        c,d=[next(iter(x)) for x in opposite];answer.append(dict(edge_id=int(edge),pass_link=bool(link_from_stars(a,b,np.asarray(stars[a],np.int32),np.asarray(stars[b],np.int32),c,d))))
    return answer

def verify_distributions(v,f,labels,masks,eligible,keys,face_octants,record,check):
    import numpy as np
    used=np.zeros(len(v),np.bool_)
    for first in range(0,len(f),BATCH):check();used[f[first:first+BATCH].ravel()]=True
    components=np.unique(labels[used]);need(set(record['by_original_component'])==set(map(str,map(int,components))) and set(record['by_octant'])==set(map(str,range(8))),'Distribution component/octant keys')
    low=np.full(3,np.inf);high=-low
    for first in range(0,len(v),BATCH):check();p=v[first:first+BATCH].astype(np.float64);need(np.isfinite(p).all(),'Finite source positions');low=np.minimum(low,p.min(0));high=np.maximum(high,p.max(0))
    mid=(low+high)/2
    def bins(p):return np.sum((p>=mid).astype(np.uint8)*np.array([1,2,4],np.uint8),axis=1,dtype=np.uint8)
    names=('original_faces',)+FACE_MASKS+VERTEX_MASKS+('certified_edges',);co={k:np.zeros(len(components),np.int64) for k in names};sp={k:np.zeros(8,np.int64) for k in names}
    def add(name,component,region):co[name][:]+=np.bincount(np.searchsorted(components,component),minlength=len(components));sp[name][:]+=np.bincount(region,minlength=8)
    for first in range(0,len(f),BATCH):
        check();b=f[first:first+BATCH];part=slice(first,first+len(b));cl=labels[b];need((cl==cl[:,:1]).all(),'Every original face has one cached source component')
        region=bins(v[b].astype(np.float64).sum(1)/3);need(np.array_equal(region,face_octants[part]),'Every saved face-centroid octant')
        add('original_faces',cl[:,0],region)
        for name in FACE_MASKS:m=masks[name][part];add(name,cl[m,0],region[m])
    for first in range(0,len(v),BATCH):
        check();part=slice(first,first+BATCH);region=bins(v[part].astype(np.float64))
        for name in VERTEX_MASKS:m=masks[name][part];add(name,labels[part][m],region[m])
    for first in range(0,len(eligible),BATCH):
        check();k=keys[eligible[first:first+BATCH]];a=(k//np.uint64(len(v))).astype(np.int32);b=(k%np.uint64(len(v))).astype(np.int32)
        need(np.array_equal(labels[a],labels[b]),'Every eligible source component');add('certified_edges',labels[a],bins((v[a].astype(np.float64)+v[b])/2))
    for j,c in enumerate(components):need(record['by_original_component'][str(int(c))]=={name:int(co[name][j]) for name in names},'All component distribution rows')
    for j in range(8):need(record['by_octant'][str(j)]=={name:int(sp[name][j]) for name in names},'All separate-denominator octant rows')
    need(record['original_face_bearing_vertex_components']==len(components) and record['unused_source_vertices']==int((~used).sum()),'Distribution population totals')

def main():
    import common as c
    c.begin();started=time.monotonic();out=c.P/'verify'
    def check():
        if time.monotonic()-started>150:raise TimeoutError('Incomplete verification:150-second work budget')
    try:
        import numpy as np
        c.authenticate();directory=c.P/'diagnose';receipt=c.plain(directory/'result.json');parent=json.loads(receipt.read_text());host=json.loads(c.plain(c.P/'diagnose-host.json').read_text())
        need(host['classification']=='PASS' and host['worker_receipt_sha256']==c.digest(receipt) and parent.get('census_completed') is True,'Completed exact owned census predecessor')
        need(parent['classification'] in {'BLOCKED','CENSUS_COMPLETE'} and parent['actual_simplification_calls']==0,'Read-only census status')
        for name,row in parent['files'].items():
            check();need(Path(name).name==name,'Manifest basename');p=c.plain(directory/name);need(p.stat().st_size==row['bytes'] and c.digest(p)==row['sha256'],'Every durable census artifact hash: '+name)
        v,f,rgba,keys,inc=c.load(np);n=len(v);m=3*len(f);e=len(keys);summary=parent['summary']
        labels=np.load(c.plain(c.original.CACHE/'vertex_components.npy'),mmap_mode='r',allow_pickle=False)
        need(labels.dtype==np.int32 and labels.shape==(n,) and (labels>=0).all() and (labels<n).all(),'Cached source component schema')
        specs=dict(edge_occurrence_order=(np.int32,(m,)),edge_offsets=(np.int32,(e+1,)),consistent_edges=(np.bool_,(e,)),global_sheet_labels=(np.int32,(len(f),)),eligible_edge_ids=(np.int32,(summary['certified_edges'],)),face_degree=(np.int32,(n,)),face_octants=(np.uint8,(len(f),)))
        specs.update({name:(np.bool_,(n,)) for name in VERTEX_MASKS});specs.update({name:(np.bool_,(len(f),)) for name in FACE_MASKS})
        need({name for name in parent['files'] if name.endswith('.npy')}=={name+'.npy' for name in specs},'Exact census array set')
        arrays={name:np.load(c.plain(directory/(name+'.npy')),mmap_mode='r',allow_pickle=False) for name in specs}
        for name,(dtype,shape) in specs.items():need(arrays[name].dtype==dtype and arrays[name].shape==shape,'Durable array schema: '+name)
        order=arrays['edge_occurrence_order'];off=arrays['edge_offsets'];con=arrays['consistent_edges'];sheets=arrays['global_sheet_labels'];eligible=arrays['eligible_edge_ids']
        need(not len(eligible) or ((eligible>=0).all() and (eligible<e).all() and (eligible[1:]>eligible[:-1]).all()),'Sorted unique eligible edge IDs')
        degree,conflict,stats=verify_occurrences(f,keys,inc,order,off,con,n,check,sheets=sheets);need(np.array_equal(degree,arrays['face_degree']),'Every source face-degree count')
        masks={name:arrays[name] for name in FACE_MASKS+VERTEX_MASKS};quasi=verify_masks(f,keys,inc,order,off,con,sheets,masks,degree,conflict,check,eligible=eligible)
        need(len(quasi)==summary['pre_link_candidate_edges'] and summary['link_tested_edges']==len(quasi),'All pre-link endpoint/protection counts')
        if len(eligible):at=np.searchsorted(quasi,eligible);need((at<len(quasi)).all() and np.array_equal(quasi[at],eligible),'Every saved eligible endpoint and protected-face condition')
        for name,a in masks.items():need(int(a.sum())==summary[name],'Every staged mask count: '+name)
        for name,value in dict(original_vertices=n,original_triangles=len(f),original_edges=e,referenced_vertices=int((degree>0).sum()),unused_vertices=int((degree==0).sum()),original_nonmanifold_edges=int((inc>2).sum()),original_boundary_edges=int((inc==1).sum()),certified_edge_face_population=int(masks['certified_edge_touch_faces'].sum()),**stats).items():need(summary[name]==value,'Source census arithmetic: '+name)
        need(int(np.unique(sheets).size)==summary['oriented_sheet_components'],'Saved sheet-label population')
        population=eligible if len(eligible) else quasi
        need(len(eligible)>0 or len(quasi)<=32,'Zero population cannot be independently certified by a bounded link replay; endpoint certificate is nonzero')
        witness_ids=population[np.unique(np.linspace(0,len(population)-1,min(32,len(population)),dtype=np.int64))] if len(population) else np.empty(0,np.int32)
        witnesses=link_witnesses(f,keys,order,off,witness_ids,n,check)
        need(all(x['pass_link']==bool(len(eligible)) for x in witnesses),'Independent full-link witness verdicts')
        if len(quasi):need(summary['rejected_common_vertex_link']+summary['rejected_common_link_edge']+len(eligible)==len(quasi),'Link result population arithmetic')
        record=json.loads(c.plain(directory/'distributions.json').read_text());verify_distributions(v,f,labels,masks,eligible,keys,arrays['face_octants'],record,check)
        c.save(out/'result.json',dict(classification='PASS',census_result_SHA256=c.digest(receipt),census_host_SHA256=c.digest(c.P/'diagnose-host.json'),all_manifest_files_verified=len(parent['files']),occurrence_permutation_keys_multiplicities_winding_exact=True,all_stage_boolean_relations_and_counts_exact=True,all_component_and_octant_counts_exact=True,eligible_edges=len(eligible),zero_eligible_endpoint_certificate=len(quasi)==0,full_link_witnesses=witnesses,scope='No census graph recomputed. Saved fan/alias classifications remain authenticated outputs of the separately source-reviewed census; boolean protection closure and distributions are independently replayed. Positive link population receives all-endpoint checks and bounded full-link witnesses only.',no_new_graph_simplification_field_query_or_render=True,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:c.save(out/'result.json',dict(classification='BLOCKED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
