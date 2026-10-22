"""Independent bounded saved-context verification; no producer qualification calls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path,PurePosixPath
import json,time,traceback,math
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E))

def need(ok,message):
    if not ok:raise ValueError(message)

def same(actual,expected,where):
    need(type(actual)is type(expected),'Exact JSON type: '+where)
    if type(expected)is dict:
        need(set(actual)==set(expected),'Exact JSON keys: '+where)
        for key in expected:same(actual[key],expected[key],where+'.'+key)
    elif type(expected)is list:
        need(len(actual)==len(expected),'Exact JSON length: '+where)
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,where+str(i))
    else:need(actual==expected and (type(actual)is not float or math.isfinite(actual)),'Exact JSON value: '+where)

def strict_json(path):
    need(path.stat().st_size<=2*1024**2,'Bounded JSON receipt')
    def unique(pairs):
        result={}
        for key,value in pairs:need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def link_ok(f,a,b,star_a,star_b):
    left=[tuple(map(int,f[row])) for row in star_a];right=[tuple(map(int,f[row])) for row in star_b]
    need(all(a in t and len(set(t))==3 for t in left) and all(b in t and len(set(t))==3 for t in right),'Full original star corners')
    edge_faces=set(map(int,star_a))&set(map(int,star_b))
    if len(edge_faces)!=2:return False
    opposite={int(x) for row in edge_faces for x in f[row]}-{a,b}
    if len(opposite)!=2:return False
    va={x for t in left for x in t}-{a};vb={x for t in right for x in t}-{b}
    ea={tuple(sorted(x for x in t if x!=a)) for t in left}
    eb={tuple(sorted(x for x in t if x!=b)) for t in right}
    return va&vb==opposite and not ea&eb

def guard_closure(v,f,local_faces,inherited_faces,inherited_vertices,inherited_locks):
    import numpy as np
    protected=local_faces|inherited_faces;vertices=inherited_vertices.copy();vertices[f[protected].ravel()]=True
    locks=inherited_locks.copy();locks[f[vertices[f].any(1)].ravel()]=True
    positions=v.copy();positions[positions==0]=0
    _,classes=np.unique(positions,axis=0,return_inverse=True)
    locks=np.isin(classes,np.unique(classes[locks]))
    return protected,vertices,locks

def screen(s):
    k=s['initial_non_touching_opportunities'];e=s['eligible_face_population'];f=s['original_patch_faces']
    need(all(type(x)is int and x>=0 for x in (k,e,f)) and f>0 and type(s['qualified'])is bool,'Integer screen inputs')
    count=k>=10;interior=e>0 and 10*k>=e;whole=40*k>=f
    return dict(meaningful_pass=bool(s['qualified'] and count and interior and whole),passes_10_disjoint=count,
                passes_20pct_eligible=interior,passes_5pct_patch=whole,estimated_removed_faces=2*k,
                eligible_face_fraction=2*k/e if e else None,whole_patch_fraction=2*k/f)

def replay_progression(f,keys,order,offsets,n,baseline,baseline_outer,check,targets=(128,512,2048),cap=6000):
    import numpy as np
    current=set(map(int,baseline));outer=set(map(int,baseline_outer));depth=3;pending=list(targets)
    records=[dict(depth=3,face_count=len(current),added_count=len(outer))];snapshots=[];base_labels=['baseline']
    def retain(labels):
        if depth==3:base_labels.extend(x for x in labels if x not in base_labels)
        elif snapshots and snapshots[-1]['depth']==depth:snapshots[-1]['labels'].extend(x for x in labels if x not in snapshots[-1]['labels'])
        else:snapshots.append(dict(depth=depth,labels=list(labels),source_face_ids=np.array(sorted(current),np.int32),outer_face_ids=np.array(sorted(outer),np.int32)))
    for _ in range(cap-len(current)+1):
        check()
        if len(current)==cap:retain(['max']);terminal=dict(reason='exact_cap',largest_depth=depth,largest_face_count=len(current));break
        pairs=set()
        for row in current:
            t=f[row]
            for j in range(3):a,b=int(t[j]),int(t[(j+1)%3]);pairs.add(min(a,b)*n+max(a,b))
        wanted=np.array(sorted(pairs),np.uint64);groups=np.searchsorted(keys,wanted)
        need((groups<len(keys)).all() and np.array_equal(keys[groups],wanted),'Independent ring endpoint/cache equality')
        expanded=set(current)
        for i,g in enumerate(groups):
            if i%1024==0:check()
            start,stop=int(offsets[g]),int(offsets[g+1]);need(0<stop-start<=4,'Bounded original ring incidence')
            expanded.update(int(x)//3 for x in order[start:stop])
        if len(expanded)>cap:
            retain(['max']);terminal=dict(reason='next_complete_ring_over_cap',largest_depth=depth,largest_face_count=len(current),prospective_depth=depth+1,prospective_face_count=len(expanded));break
        if expanded==current:retain(['max']);terminal=dict(reason='fixed_point',largest_depth=depth,largest_face_count=len(current));break
        outer=expanded-current;current=expanded;depth+=1;records.append(dict(depth=depth,face_count=len(current),added_count=len(outer)))
        labels=[]
        while pending and len(current)>=pending[0]:labels.append(str(pending.pop(0)))
        if labels:retain(labels)
    else:raise ValueError('Incomplete independent ring progression')
    return dict(baseline_depth=3,baseline_labels=base_labels,ring_records=records,terminal=terminal,snapshots=snapshots)

def recompute(v,f,rgba,keys,inc,labels,c,face_ids,outer,choice,classify,check):
    import numpy as np
    check();need(face_ids.dtype==np.int32 and face_ids.ndim==1 and 0<len(face_ids)<=6000 and (face_ids>=0).all() and (face_ids<len(f)).all() and (face_ids[1:]>face_ids[:-1]).all(),'Bounded exact sorted source faces')
    need(outer.dtype==np.int32 and outer.ndim==1 and (outer[1:]>outer[:-1]).all() and np.isin(outer,face_ids).all(),'Exact outer-ring IDs')
    sf=np.asarray(f[face_ids]);need(sf.dtype==np.int32 and ((sf[:,0]!=sf[:,1])&(sf[:,1]!=sf[:,2])&(sf[:,0]!=sf[:,2])).all(),'Original distinct I32 triangle corners')
    ids,degree=np.unique(sf,return_counts=True);ids=ids.astype(np.int32);cf=np.searchsorted(ids,sf).astype(np.int32);cv=np.asarray(v[ids]);col=np.asarray(rgba[ids])
    need(cv.dtype==np.float32 and np.isfinite(cv).all() and col.dtype==np.uint8 and col.shape==(len(ids),4),'Original F32/native-RGBA schemas')
    need((labels[sf]==labels[sf[:,0],None]).all() and len(np.unique(labels[ids]))==1,'Original component correspondence')
    local_summary,local=classify(cv,cf,ids,sf,np.searchsorted(face_ids,outer).astype(np.int32),keys,inc,len(v));check()
    inherited_f=np.asarray(c['protected_faces'][face_ids]);inherited_v=np.asarray(c['protected_vertices'][ids]);inherited_l=np.asarray(c['locked_vertices'][ids])
    protected,pv,locks=guard_closure(cv,cf,local['protected_faces'],inherited_f,inherited_v,inherited_l)
    complete_ids=set(map(int,ids[degree==c['face_degree'][ids]]));stars={x:[] for x in complete_ids};edges={};parents=list(range(len(sf)))
    def root(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    for row,(original,t) in enumerate(zip(face_ids,sf)):
        for x in t:
            if int(x) in stars:stars[int(x)].append(int(original))
        for j in range(3):
            a,b=int(t[j]),int(t[(j+1)%3]);key=min(a,b)*len(v)+max(a,b)
            if key in edges:
                left,right=root(row),root(edges[key][0]);parents[max(left,right)]=min(left,right)
            edges.setdefault(key,[]).append(row)
    components=len({root(i) for i in range(len(sf))});need(components==1,'Raw original edge-connected context')
    patch_keys=np.array(sorted(edges),np.uint64);at=np.searchsorted(keys,patch_keys)
    need((at<len(keys)).all() and np.array_equal(keys[at],patch_keys),'Every local source edge/cache identity')
    globally=c['eligible_edge_ids'];p=np.searchsorted(globally,at);present=np.zeros(len(at),bool)
    if len(globally):present=(p<len(globally))&(globally[np.minimum(p,len(globally)-1)]==at)
    touch=at[present];both=complete=guarded=rejected=0;eligible=[];eligible_faces=np.zeros(len(sf),bool)
    for edge in touch:
        check();edge=int(edge);occ=c['order'][c['offsets'][edge]:c['offsets'][edge+1]];rows=occ//3
        if not np.isin(rows,face_ids).all():continue
        both+=1;a,b=divmod(int(keys[edge]),len(v))
        if a not in stars or b not in stars:continue
        complete+=1;ia,ib=np.searchsorted(ids,[a,b]);local_rows=np.searchsorted(face_ids,rows)
        if locks[ia] or locks[ib] or protected[local_rows].any():continue
        guarded+=1;need(len(occ)==2 and inc[edge]==2 and c['consistent'][edge] and len(set(map(int,c['global_sheet_labels'][rows])))==1,'Original manifold/winding/sheet prerequisites')
        directions=[bool(f[int(o)//3,int(o)%3]<f[int(o)//3,(int(o)%3+1)%3]) for o in occ]
        need(directions[0]!=directions[1],'Actual opposite original winding')
        if link_ok(f,a,b,stars[a],stars[b]):eligible.append(edge);eligible_faces[local_rows]=True
        else:rejected+=1
    need(np.isin(keys[np.array(eligible,np.int32)],local['edge_keys'][local['eligible_edge_rows']]).all(),'Global qualification only narrows the pinned local classifier')
    busy=set();capacity=[]
    for edge in sorted(eligible,key=lambda x:int(keys[x])):
        check();a,b=divmod(int(keys[edge]),len(v));neighbours={int(x) for row in set(stars[a])|set(stars[b]) for x in f[row]}
        if not busy&neighbours:capacity.append(edge);busy.update(neighbours)
    k=len(capacity);ef=int(eligible_faces.sum());mutable=int((~protected).sum());need(2*k<=ef<=mutable,'Disjoint opportunity and mutable-face bounds')
    s=dict(original_patch_faces=len(sf),original_patch_vertices=len(ids),source_component=int(labels[ids[0]]),raw_original_edge_connected_components=components,eligible_subset_of_independent_local_classifier=True,
           selected_component_faces=choice['selected_component_faces'],selected_component_edges=choice['selected_component_edges'],eligible_face_components=choice['eligible_face_components'],seed_edge_id=choice['seed_edge_id'],
           globally_certified_edges_touching_patch=len(touch),both_edge_faces_represented=both,complete_global_endpoint_star_edges=complete,post_crop_inherited_guard_edges=guarded,
           lost_incomplete_edge_face_context=len(touch)-both,lost_incomplete_endpoint_star_context=both-complete,lost_to_actual_crop_or_inherited_guards=complete-guarded,full_link_rejected=rejected,
           strict_eligible_edges=len(eligible),eligible_face_population=ef,immutable_face_floor=int(protected.sum()),optimistic_removable_face_ceiling=mutable,guard_locked_vertices=int(locks.sum()),
           initial_non_touching_opportunities=k,initial_topological_removal_estimate_faces=2*k,estimate_fraction_of_eligible_faces=2*k/ef if ef else None,initial_estimate_fraction_of_entire_patch=2*k/len(sf),
           optimistic_removable_ceiling_fraction_of_entire_patch=mutable/len(sf),twenty_percent_is_informational_not_admission=True,qualified=bool(ef and k>=1))
    sv=np.array(sorted(stars),np.int32);so=np.r_[0,np.cumsum([len(stars[int(x)]) for x in sv])].astype(np.int32)
    arrays=dict(vertices=cv,faces=cf,source_faces=sf,source_face_ids=face_ids,source_vertex_ids=ids,original_native_RGBA=col,source_component_labels=np.asarray(labels[ids],np.int32),
                inherited_protected_faces=inherited_f,inherited_protected_vertices=inherited_v,inherited_vertex_locks=inherited_l,local_protected_faces=local['protected_faces'],local_protected_vertices=local['protected_vertices'],local_vertex_locks=local['vertex_locks'],
                protected_faces=protected,protected_vertices=pv,vertex_locks=locks,eligible_faces=eligible_faces,eligible_edge_ids=np.array(eligible,np.int32),non_touching_edge_ids=np.array(capacity,np.int32),global_sheet_labels=np.asarray(c['global_sheet_labels'][face_ids],np.int32),
                complete_star_vertex_ids=sv,complete_star_offsets=so,complete_star_source_face_ids=np.array([row for x in sv for row in stars[int(x)]],np.int32),outer_face_ids=outer)
    return s,arrays,local_summary

def verify_snapshot(v,f,rgba,keys,inc,labels,c,arrays,summary,local_summary,choice,classify,check):
    import numpy as np
    s,expected,local=recompute(v,f,rgba,keys,inc,labels,c,arrays['source_face_ids'],arrays['outer_face_ids'],choice,classify,check)
    need(set(arrays)==set(expected),'Exact24 snapshot array schemas')
    for name,want in expected.items():
        actual=arrays[name];need(actual.dtype==want.dtype and actual.shape==want.shape,'Exact array shape/dtype: '+name)
        need(np.array_equal(actual.view(np.uint32),want.view(np.uint32)) if want.dtype==np.float32 else np.array_equal(actual,want),'Every saved array value/bit: '+name)
    same(summary,s,'summary');same(local_summary,local,'local classifier summary')
    return screen(s)

def main():
    import common as c
    c.begin();started=time.monotonic();out=c.P/'verify'
    def check():
        if time.monotonic()-started>150:raise TimeoutError('150-second independent context verification budget')
    try:
        import numpy as np
        c.authenticate();directory=c.P/'context';path=c.plain(directory/'result.json');top=strict_json(path);host=strict_json(c.plain(c.P/'context-host.json'))
        need(host['returncode']==0 and host['worker_receipt_sha256']==c.digest(path) and host['sources_unchanged'] and host['job_cleanup']['owned_tree_exit_verified'] and not host['job_cleanup']['forced_job_cleanup'] and not any(k in host for k in ('error','measurement_error')),'Exact naturally settled owned context predecessor')
        need(top['classification'] in {'PASS','BLOCKED'} and top['observations_completed'] is True and not any(k in top for k in ('error','traceback')) and top['actual_simplification_calls']==0 and top['census_reruns']==0 and top['baseline_requalified'] is False,'Completed contexts only; no baseline/census replay')
        choice=c.frozen_choice();same(top['frozen_choice'],choice,'frozen choice')
        need(len(top['observations'])<=4,'At most4 distinct new snapshots')
        for name,row in top['files'].items():
            check();rel=PurePosixPath(name);need(name==rel.as_posix() and not rel.is_absolute() and all(x not in ('..','.') for x in rel.parts) and '\\' not in name,'Confined context artifact')
            artifact=c.plain(directory.joinpath(*rel.parts));need(artifact.stat().st_size==row['bytes'] and artifact.stat().st_size<=4*1024**2 and c.digest(artifact)==row['sha256'],'Every context artifact size/hash')
        old=strict_json(c.plain(c.RECOVERY/'patch/result.json'));old_selection=strict_json(c.plain(c.RECOVERY/'patch/selection.json'))
        baseline=np.load(c.plain(c.RECOVERY/'patch/source_face_ids.npy'),mmap_mode='r',allow_pickle=False);outer=np.array(old_selection['ring_added_face_ids'][-1],np.int32)
        need(len(baseline)==44 and len(outer)==16 and old['classification']=='BLOCKED','Authenticated baseline history only')
        v,f,rgba,keys,inc=c.load(np);labels=np.load(c.plain(c.original.CACHE/'vertex_components.npy'),mmap_mode='r',allow_pickle=False)
        names=dict(order='edge_occurrence_order',offsets='edge_offsets',consistent='consistent_edges',protected_faces='protected_faces',protected_vertices='protected_vertices',locked_vertices='locked_vertices',global_sheet_labels='global_sheet_labels',eligible_edge_ids='eligible_edge_ids',face_degree='face_degree')
        ctx={name:np.load(c.plain(c.CENSUS/'diagnose'/(value+'.npy')),mmap_mode='r',allow_pickle=False) for name,value in names.items()}
        built=replay_progression(f,keys,ctx['order'],ctx['offsets'],len(v),baseline,outer,check)
        progression={name:built[name] for name in ('baseline_depth','baseline_labels','ring_records','terminal')}
        same(top['progression'],progression,'complete progression');same(strict_json(c.plain(directory/'progression.json')),progression,'saved progression')
        expected_baseline=dict(depth=3,labels=built['baseline_labels'],result_SHA256=c.digest(c.RECOVERY/'patch/result.json'),summary=old['summary']);same(top['baseline'],expected_baseline,'historical baseline')
        same(strict_json(c.plain(directory/'observations.json')),dict(baseline=expected_baseline,observations=top['observations']),'saved observations')
        need(len(top['observations'])==len(built['snapshots']),'Every fixed checkpoint represented once')
        prior=c.pin('independent_guard_local_classifier',c.ROOT/'docs/compatibility/amd-brick-junction-protected-evidence/patch_contract.py','336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        verified=[];expected_files={'progression.json','observations.json'}
        for row,snapshot in zip(top['observations'],built['snapshots']):
            check();need(set(row)=={'depth','labels','relative_directory','summary','local_classifier_summary','screening'},'Exact snapshot metadata')
            same(row['depth'],snapshot['depth'],'snapshot depth');same(row['labels'],snapshot['labels'],'checkpoint labels');need(row['relative_directory']=='depth-%04d'%row['depth'],'Fixed depth output directory')
            sub=directory/row['relative_directory'];per=strict_json(c.plain(sub/'result.json'));need(set(per)==set(row)|{'files'},'Exact saved per-depth receipt')
            for name in row:same(per[name],row[name],'per-depth '+name)
            arrays={}
            for name,item in per['files'].items():
                need(Path(name).name==name and name.endswith('.npy'),'Exact private NPY basename');a=c.plain(sub/name);need(a.stat().st_size<=1024**2 and item==top['files'][row['relative_directory']+'/'+name],'Bounded NPY manifest correspondence')
                arrays[name[:-4]]=np.load(a,mmap_mode='r',allow_pickle=False);expected_files.add(row['relative_directory']+'/'+name)
            expected_files.add(row['relative_directory']+'/result.json')
            need(np.array_equal(arrays['source_face_ids'],snapshot['source_face_ids']) and np.array_equal(arrays['outer_face_ids'],snapshot['outer_face_ids']),'Independent complete-ring snapshot and actual outer-ring identity')
            result=verify_snapshot(v,f,rgba,keys,inc,labels,ctx,arrays,row['summary'],row['local_classifier_summary'],choice,prior.classify,check)
            same(row['screening'],result,'integer screening');verified.append(dict(depth=row['depth'],faces=len(snapshot['source_face_ids']),meaningful_pass=result['meaningful_pass']));del arrays
        selected=next((x['depth'] for x in verified if x['meaningful_pass']),None)
        same(top['selected_depth'],selected,'earliest qualifying depth');need(top['classification']==('PASS' if selected is not None else 'BLOCKED'),'Qualification verdict matches all screens')
        same(top['diagram'],'patch-protection.png' if selected is not None else None,'Selected-only diagnostic')
        if selected is not None:expected_files.add('patch-protection.png')
        need(set(top['files'])==expected_files,'Exact intended worker-owned artifact set')
        c.save(out/'result.json',dict(classification='PASS',context_qualification=top['classification'],context_result_SHA256=c.digest(path),context_host_SHA256=c.digest(c.P/'context-host.json'),baseline_requalified=False,census_reruns=0,verified_new_contexts=verified,selected_depth=selected,all_context_arrays_and_summary_values_exact=True,all_protected_masks_and_complete_stars_replayed=True,all_strict_global_links_and_disjoint_certificates_replayed=True,all_checkpoint_and_earliest_screening_exact=True,scope='All new fixed snapshots independently replayed against authenticated original data using the pinned local classifier and separate ring/protection/link/greedy logic. Global census fan/alias outputs remain authenticated source-reviewed inputs. Old44-face baseline authenticated only, never requalified.',actual_simplification_calls=0,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:c.save(out/'result.json',dict(classification='BLOCKED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
