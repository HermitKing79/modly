"""Reproduce the saved brick rejection; no constructor or MeshLib import."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,time,traceback
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from verify_saved import verify_counter,relative_artifact_name,_topology,_labels

def require(ok,message):
    if not ok:raise ValueError(message)

def verify_failure_arrays(np,original,saved,counter):
    """Independently check partial original-FaceId readback, not compact IDs."""
    v,f=original['vertices'],original['faces'];N,F=len(v),len(f)
    require(0<N<=4000 and 0<F<=5000 and v.dtype==np.float32 and v.shape==(N,3) and np.isfinite(v).all(),'Bounded original F32 positions')
    require(f.dtype==np.int32 and f.shape==(F,3) and (f>=0).all() and (f<N).all(),'Original bounded triangle rows')
    for name,size in [('source_vertex_ids',N),('source_face_ids',F)]:
        source,actual=original[name],saved[name]
        require(source.dtype.kind in 'iu' and source.shape==(size,) and len(np.unique(source))==size and (source>=0).all(),'Original unique source IDs')
        require(actual.dtype==source.dtype and actual.shape==source.shape and np.array_equal(actual,source),'All original source-ID copies unchanged')
    for name,source in [('vertices_after_native',v),('faces_after_native',f)]:
        actual=saved[name]
        require(actual.dtype==source.dtype and actual.shape==source.shape and actual.tobytes()==source.tobytes(),'Every original heap input bit unchanged')
    rv=saved['vertices_readback'];require(rv.dtype in (np.dtype('float32'),np.dtype('float64')) and rv.shape==(N,3) and np.isfinite(rv).all(),'All allocated original position slots retained')
    bits=np.uint32 if rv.dtype==np.float32 else np.uint64
    require(np.array_equal(rv.view(bits),v.astype(rv.dtype).view(bits)),'Every allocated position is exact original F32 promotion, including unused/invalid slots')
    vf=saved['valid_face_ids'];failed=saved['failed_face_ids'];rf=saved['faces_readback'];vv=saved['valid_vertex_ids']
    require(vf.dtype==np.int32 and vf.ndim==1 and 0<len(vf)<F and np.array_equal(vf,np.unique(vf)) and (vf>=0).all() and (vf<F).all(),'Sorted noncontiguous original valid FaceIds')
    M=len(vf);missing=np.setdiff1d(np.arange(F,dtype=np.int32),vf)
    require(failed.dtype==np.int32 and failed.shape==missing.shape and np.array_equal(failed,missing),'Failed region equals every missing original FaceId')
    require(counter['failed_ids']==counter['settings_failed_ids']==failed.tolist(),'Both actual failed regions equal the missing FaceIds')
    require(counter['region_before']==counter['settings_region_before']==list(range(F)),'Every original face was requested')
    verify_counter(counter['before'],counter['settings_before'],counter['after'],counter['settings_after'],counter['failed_ids'],F)
    require(counter['after']['values']==[F-M] and F-M>0,'Positive observed native skipped count exactly explains missing faces')
    require(rf.dtype==np.int32 and rf.shape==(M,3) and (rf>=0).all() and (rf<N).all(),'Retained native triangles use original vertex IDs')
    target=f[vf];rotations=np.full(M,-1,np.int8)
    for shift in range(3):rotations[(rf==np.roll(target,-shift,axis=1)).all(axis=1)]=shift
    require((rotations>=0).all(),'Every retained row matches its actual original FaceId with original cyclic winding')
    require(vv.dtype==np.int32 and vv.ndim==1 and np.array_equal(vv,np.unique(rf)),'Native valid vertices equal exactly the retained triangle references')
    source_stats,source_edges,source_el,_=_topology(f)
    retained_stats,retained_edges,_,_=_topology(rf)
    names=['native_undirected_edge_ids','native_halfedge_ids','native_left_faces','native_right_faces']
    eid,hid,left,right=[saved[name] for name in names]
    for name,array in zip(names,[eid,hid,left,right]):require(array.dtype==np.int32 and array.shape==(M,3),'Aligned native metadata: '+name)
    valid_set=set(int(x) for x in vf)
    require((eid>=0).all() and (hid>=0).all() and len(np.unique(hid))==3*M,'Actual distinct native face-left halfedges')
    require(np.array_equal(left,np.repeat(vf[:,None],3,axis=1)),'Native left faces retain noncontiguous original FaceIds')
    require(all(int(x)==-1 or int(x)in valid_set for x in right.flat),'Right faces are actual surviving FaceIds or raw invalid -1')
    native={};owners={};pair_ids={};face_to_row={int(face):i for i,face in enumerate(vf)}
    for row,tri in enumerate(rf):
        face=int(vf[row])
        for corner in range(3):
            a,b=int(tri[corner]),int(tri[(corner+1)%3]);pair=tuple(sorted((a,b)));edge=int(eid[row,corner])
            require(edge not in owners or owners[edge]==pair,'Native edge owns one original endpoint pair')
            owners[edge]=pair;pair_ids.setdefault(pair,set()).add(edge)
            native.setdefault(edge,[]).append((face,row,a,b,int(right[row,corner]),int(hid[row,corner])))
    for group in native.values():
        require(len(group)in (1,2),'Native edge has at most its two observed sides')
        if len(group)==1:require(group[0][4]==-1,'A native boundary cannot reference an unseen face')
        else:
            a,b=group
            require(a[0]!=b[0] and a[2]==b[3] and a[3]==b[2] and a[4]==b[0] and b[4]==a[0],
                    'Native actual-FaceId adjacency is reciprocal and oppositely oriented')
    native_labels=_labels(M,[[x[1] for x in group] for group in native.values()]);membership={}
    for row,label in enumerate(native_labels):membership.setdefault(label,set()).add(source_el[int(vf[row])])
    require(all(len(x)==1 for x in membership.values()),'No merge across original source edge components')
    missing_set=set(int(x) for x in missing);junctions=[];touching=set()
    for pair,group in sorted(source_edges.items()):
        if len(group)<=2:continue
        incident=[x[0] for x in group];deleted=[x for x in incident if x in missing_set];touching.update(deleted)
        native_groups=[]
        for edge in sorted(pair_ids.get(pair,())):
            native_groups.append(dict(native_edge_id=edge,native_face_ids=[x[0] for x in native[edge]],native_halfedge_ids=[x[5] for x in native[edge]],native_right_face_ids=[x[4] for x in native[edge]]))
        junctions.append(dict(source_vertex_ids=[int(original['source_vertex_ids'][x]) for x in pair],
                              original_source_face_ids=[int(original['source_face_ids'][x]) for x in incident],
                              rejected_source_face_ids=[int(original['source_face_ids'][x]) for x in deleted],native_groups=native_groups))
    original_referenced=np.unique(f);lost_vertices=np.setdiff1d(original_referenced,vv)
    witness=dict(missing_native_face_ids=missing.tolist(),missing_source_face_ids=original['source_face_ids'][missing].tolist(),
                 missing_source_vertex_ids=original['source_vertex_ids'][lost_vertices].tolist(),
                 rejected_source_triangles=[dict(source_face_id=int(original['source_face_ids'][face]),source_vertex_ids=original['source_vertex_ids'][f[face]].tolist()) for face in missing],
                 original_nonmanifold_junctions=junctions)
    summary=dict(brick_input_transport='NO-GO',foliage_input_transport='NOT RUN',sheet_correspondence='FAILED',
                 semantic_sheet_pairing='UNRESOLVED',remeshing_admission='NOT AUTHORISED',
                 original_vertices=N,original_triangles=F,allocated_position_rows=len(rv),native_valid_vertices=len(vv),native_valid_faces=M,
                 missing_original_faces=F-M,missing_valid_original_vertices=len(lost_vertices),observed_skipped_count=counter['after']['values'][0],
                 both_failed_regions_equal_missing_original_faces=True,all_allocated_original_position_bits_exact=True,all_original_heap_and_source_ID_copies_exact=True,
                 retained_original_FaceId_orientation_and_multiplicity_exact=True,
                 cyclic_rotation_counts={str(i):int((rotations==i).sum()) for i in range(3)},original_topology=source_stats,retained_endpoint_topology=retained_stats,
                 native_topology=dict(undirected_edges=len(native),boundary_edges=sum(len(x)==1 for x in native.values()),paired_edges=sum(len(x)==2 for x in native.values()),
                                      source_endpoint_pairs_split=sum(len(x)>1 for x in pair_ids.values()),edge_connected_face_components=len(set(native_labels)),cross_original_component_merges=0),
                 rejection_incidence=dict(rejected_faces_touching_original_nonmanifold_edges=len(touching),original_nonmanifold_edges_touching_rejected_faces=sum(bool(x['rejected_source_face_ids']) for x in junctions),
                                          rejected_faces_not_touching_original_nonmanifold_edges=len(missing)-len(touching)),
                 interpretation='The zero-loss transport contract fails on actual face/valid-vertex deletion. Junction incidence overlap is descriptive, not proof of why MeshLib rejected a face or of an original semantic sheet pairing.')
    return summary,witness

def main():
    import common as c
    c.begin('verify');began=time.monotonic();out=c.P/'verify'
    result=dict(classification='BLOCKED',verification_completed=False,brick_input_transport='NO-GO',foliage_input_transport='NOT RUN',sheet_correspondence='UNRESOLVED',remeshing_admission='NOT AUTHORISED',verification_constructor_calls=0)
    def check():require(time.monotonic()-began<150,'150-second failure-readback budget')
    def manifest(directory,entries):
        for rel,item in entries.items():
            relative_artifact_name(rel);path=c.plain(directory/rel)
            require(path.is_relative_to(directory) and path.is_file() and path.stat().st_size==item['bytes'] and item['bytes']<=2*1024**2 and c.digest(path)==item['sha256'],'Exact bounded saved artifact');check()
    try:
        c.authenticate();plan=c.load_plan();check()
        record=c.read_json(c.P/'import/result.json');host=c.read_json(c.P/'import-host.json')
        require(c.digest(c.P/'import/result.json')=='6a47a4df80fc32610bbb2ff2072d15c38f635c621de92d8dab684d766e55ed1f','The sole preserved failed native attempt')
        require(record['classification']==host['classification']==host['worker_classification']=='BLOCKED' and host['worker_receipt_sha256']==c.digest(c.P/'import/result.json'),'Exact failed predecessor binding')
        require(host['returncode']==0 and host['sources_unchanged']is True and host['job_cleanup']['owned_tree_exit_verified']is True and host['job_cleanup']['forced_job_cleanup']is False,'Naturally settled failed constructor')
        require(record['constructor_calls']==1 and record['remesh_calls']==record['target_generation_calls']==0 and record['constructors_completed']is False,'One brick call only')
        require([x['name'] for x in record['patches']]==['brick'] and not (c.P/'import/foliage').exists(),'Foliage was not attempted')
        require(record['input_manifest_SHA256']==c.digest(c.INPUT_MANIFEST) and record['library_pins_SHA256']==c.digest(c.SDK),'Original input and SDK identity')
        manifest(c.P/'import',record['files']);patch=plan['patches'][0];directory=c.P/'import/brick';row=c.read_json(directory/'result.json')
        require(row==record['patches'][0] and row['constructor_calls']==1 and row['classification']=='BLOCKED' and row['error']=='No skipped or failed original faces','Specific preserved loss gate')
        manifest(directory,row['files'])
        jsons={rel:c.read_json(directory/rel) for rel in row['files'] if rel.endswith('.json')}
        counter=row['counter'];require(jsons['counter-after.json']==counter,'Final counter journal')
        require(jsons['counter-before.json']=={k:counter[k] for k in ('before','settings_before','region_before','settings_region_before')}
                and jsons['counter-holder-before.json']==counter['before'] and jsons['counter-after-holder.json']==counter['after'],'Every pre/post scalar snapshot')
        journal=jsons['before-constructor.json']
        require(journal['constructor_call_ordinal']==1 and journal['name']=='brick' and journal['counter_before']==counter['before'] and journal['settings_before']==counter['settings_before']
                and journal['inputfiles']==patch['inputfiles'] and journal['region_before']==journal['settings_region_before']==list(range(781))
                and journal['allowNonManifoldEdge']is True and journal['duplicateNonManifoldVertices']is False,'Exact single original-brick constructor journal')
        import numpy as np
        original={}
        for name,item in patch['inputfiles'].items():
            array=np.load(c.plain(item['path']),mmap_mode='r',allow_pickle=False);schema=patch['array_schemas'][name]
            require(array.dtype==np.dtype(schema['dtype']) and list(array.shape)==schema['shape'],'Original NPY schema');original[name]=array
        saved={Path(rel).stem:np.load(c.plain(directory/rel),mmap_mode='r',allow_pickle=False) for rel in row['files'] if rel.endswith('.npy')}
        require(len(saved)==13 and 'cyclic_rotations' not in saved,'Exactly the failure-path arrays, no invented success rotation output')
        summary,witness=verify_failure_arrays(np,original,saved,counter)
        require(summary['original_vertices']==457 and summary['original_triangles']==781 and summary['native_valid_faces']==777 and summary['native_valid_vertices']==456
                and summary['missing_original_faces']==4 and summary['missing_valid_original_vertices']==1 and saved['vertices_readback'].dtype==np.float64,'Exact observed brick-loss population')
        require(summary['original_topology']==row['original_topology'] and summary['original_topology']['nonmanifold_source_edges']==4
                and row['skipped_face_count']==row['failed_face_count']==4 and row['heap_inputs_unchanged']is True,'Independent original topology and native loss counts')
        c.save(out/'brick-rejected-source-memberships.json',witness);check()
        result.update(summary,classification='PASS',verification_completed=True,constructor_result_SHA256=c.digest(c.P/'import/result.json'),constructor_host_SHA256=c.digest(c.P/'import-host.json'),
                      original_input_manifest_SHA256=c.digest(c.INPUT_MANIFEST),library_pins_SHA256=c.digest(c.SDK),observed_constructor_calls=1,
                      verification_scope='PASS means the saved native rejection was independently reproduced, not that transport passed. No new constructor, repair, method, sample or appearance operation ran.',elapsed_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(classification='BLOCKED',verification_completed=False,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
    result['files']={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.is_file() and p.suffix=='.json' and p.name!='result.json'}
    c.save(out/'result.json',result)

if __name__=='__main__':main()
