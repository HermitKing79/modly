"""Independent bounded saved-data verification; never imports MeshLib."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path,PurePosixPath
import os,time,traceback
E=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def relative_artifact_name(name):
    require(type(name)is str and bool(name) and '\\' not in name and ':' not in name,'Canonical POSIX artifact name')
    path=PurePosixPath(name)
    require(not path.is_absolute() and bool(path.parts) and all(x not in {'.','..'} for x in path.parts)
            and path.as_posix()==name,'Confined canonical relative artifact path')
    return name

def verify_counter(before,settings_before,after,settings_after,failed_ids,face_count):
    def scalar(row):
        require(type(row)is dict and set(row)=={'length','values','getter'},'Exact saved counter state')
        require(type(row['length'])is int and row['length']==1 and type(row['values'])is list and len(row['values'])==1,'One counter slot')
        value=row['values'][0];getter=row['getter']
        require(type(value)is int and type(getter)is dict and set(getter)=={'ok','value'} and getter['ok']is True
                and type(getter['value'])is int and getter['value']==value,'Coherent actual scalar getter')
        return value
    require(scalar(before)==-23456 and type(settings_before)is int and settings_before==-23456,'Distinctive actual bound pre-call sentinel')
    value=scalar(after)
    require(type(settings_after)is int and settings_after==value and 0<=value<=face_count,'Actual overwritten bound counter')
    require(type(failed_ids)is list and all(type(x)is int and 0<=x<face_count for x in failed_ids)
            and failed_ids==sorted(set(failed_ids)) and value==len(failed_ids),'Counter equals independently enumerated failed region')
    return True

def _labels(count,groups):
    parent=list(range(count))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for group in groups:
        if group:
            first=group[0]
            for other in group[1:]:
                a,b=find(first),find(other)
                if a!=b:parent[max(a,b)]=min(a,b)
    return [find(i) for i in range(count)]

def _incidence(faces):
    edges={};vertices={}
    for face,tri in enumerate(faces):
        for corner in range(3):
            a,b=int(tri[corner]),int(tri[(corner+1)%3])
            edges.setdefault(tuple(sorted((a,b))),[]).append((face,corner,a,b))
            vertices.setdefault(a,[]).append(face)
    return edges,vertices

def _topology(faces):
    edges,vertices=_incidence(faces);counts=[len(x) for x in edges.values()]
    el=_labels(len(faces),[[x[0] for x in group] for group in edges.values()])
    vl=_labels(len(faces),list(vertices.values()))
    return dict(unique_source_edges=len(edges),boundary_source_edges=sum(x==1 for x in counts),
                nonmanifold_source_edges=sum(x>2 for x in counts),maximum_source_edge_incidence=max(counts,default=0),
                edge_connected_face_components=len(set(el)),vertex_connected_face_components=len(set(vl))),edges,el,vl

def verify_arrays(np,original,saved):
    """Return scalar statistics and private NM memberships from saved arrays."""
    v,f=original['vertices'],original['faces'];n,F=len(v),len(f)
    require(0<n<=4000 and 0<F<=5000 and v.dtype==np.float32 and v.shape==(n,3) and np.isfinite(v).all(),'Bounded finite original F32 positions')
    require(f.dtype==np.int32 and f.shape==(F,3) and (f>=0).all() and (f<n).all(),'Bounded original I32 triangles')
    for name,size in [('source_vertex_ids',n),('source_face_ids',F)]:
        source=original[name];actual=saved[name]
        require(source.dtype.kind in 'iu' and source.shape==(size,) and (source>=0).all() and len(np.unique(source))==size,'Unique original source IDs')
        require(actual.dtype==source.dtype and actual.shape==source.shape and np.array_equal(actual,source),'Every saved source ID and its row association')
    for name,source in [('vertices_after_native',v),('faces_after_native',f)]:
        actual=saved[name]
        require(actual.dtype==source.dtype and actual.shape==source.shape and actual.tobytes()==source.tobytes(),'Exact unchanged original heap copies')
    for name,size in [('valid_vertex_ids',n),('valid_face_ids',F)]:
        actual=saved[name]
        require(actual.dtype==np.int32 and actual.shape==(size,) and np.array_equal(actual,np.arange(size,dtype=np.int32)),'Dense unchanged actual native IDs')
    failed=saved['failed_face_ids'];require(failed.dtype==np.int32 and failed.shape==(0,),'Zero failed source faces')
    rv,rf=saved['vertices_readback'],saved['faces_readback']
    require(rv.dtype in (np.dtype('float32'),np.dtype('float64')) and rv.shape==v.shape and np.isfinite(rv).all(),'Native vertex schema')
    target=v.astype(rv.dtype);kind=np.uint32 if rv.dtype==np.float32 else np.uint64
    require(np.array_equal(rv.view(kind),target.view(kind)),'Exact native promoted F32 position bits including signed zero')
    require(rf.dtype==np.int32 and rf.shape==f.shape and (rf>=0).all() and (rf<n).all(),'Native original FaceId schema')
    rotations=saved['cyclic_rotations']
    require(rotations.dtype==np.int8 and rotations.shape==(F,) and (rotations>=0).all() and (rotations<=2).all(),'Recorded cyclic rotations')
    columns=(np.arange(3)[None,:]+rotations[:,None])%3
    require(np.array_equal(rf,np.take_along_axis(f,columns,axis=1)),'Every original FaceId, oriented corner and triangle multiplicity')
    original_stats,source_edges,source_el,source_vl=_topology(f)
    actual_stats,actual_edges,actual_el,actual_vl=_topology(rf)
    require(original_stats==actual_stats and source_el==actual_el and source_vl==actual_vl,'Complete original source incidence and component relationships')
    names=['native_undirected_edge_ids','native_halfedge_ids','native_left_faces','native_right_faces']
    eid,hid,left,right=[saved[name] for name in names]
    for name,array in zip(names,[eid,hid,left,right]):require(array.dtype==np.int32 and array.shape==(F,3),'Exact aligned native metadata: '+name)
    require((eid>=0).all() and (hid>=0).all() and (right>=-1).all() and (right<F).all(),'Bounded actual native edge/FaceId values')
    require(np.array_equal(left,np.repeat(np.arange(F,dtype=np.int32)[:,None],3,axis=1)),'Every actual halfedge has the current face on its left')
    require(len(np.unique(hid))==3*F,'One distinct face-left halfedge per actual corner')
    native={};owners={};source_to_native={}
    for face,tri in enumerate(rf):
        for corner in range(3):
            a,b=int(tri[corner]),int(tri[(corner+1)%3]);pair=tuple(sorted((a,b)));edge=int(eid[face,corner])
            require(edge not in owners or owners[edge]==pair,'Native edge cannot connect different original endpoint pairs')
            owners[edge]=pair;source_to_native.setdefault(pair,set()).add(edge)
            native.setdefault(edge,[]).append((face,corner,a,b,int(right[face,corner]),int(hid[face,corner])))
    for group in native.values():
        require(len(group)in(1,2),'Native left/right edge incidence must be representable')
        if len(group)==1:require(group[0][4]==-1,'Native boundary edge cannot reference an unseen face')
        else:
            a,b=group
            require(a[0]!=b[0] and a[2]==b[3] and a[3]==b[2] and a[4]==b[0] and b[4]==a[0],
                    'Actual native paired faces must be reciprocal with opposite edge directions')
    native_el=_labels(F,[[x[0] for x in group] for group in native.values()])
    native_components={}
    for face,label in enumerate(native_el):native_components.setdefault(label,set()).add(source_el[face])
    require(all(len(group)==1 for group in native_components.values()),'Native adjacency cannot merge distinct original edge components')
    details=[]
    for pair,occurrences in sorted(source_edges.items()):
        if len(occurrences)<=2:continue
        groups=[]
        for edge in sorted(source_to_native[pair]):
            groups.append(dict(native_undirected_edge_id=edge,source_face_ids=[int(original['source_face_ids'][x[0]]) for x in native[edge]],
                               native_halfedge_ids=[x[5] for x in native[edge]],native_right_face_ids=[x[4] for x in native[edge]]))
        details.append(dict(source_vertex_ids=[int(original['source_vertex_ids'][x]) for x in pair],
                            original_source_face_ids=[int(original['source_face_ids'][x[0]]) for x in occurrences],native_edge_groups=groups))
    native_stats=dict(undirected_edges=len(native),boundary_edges=sum(len(x)==1 for x in native.values()),paired_two_face_edges=sum(len(x)==2 for x in native.values()),
                      edge_connected_face_components=len(set(native_el)),vertex_connected_face_components=actual_stats['vertex_connected_face_components'],
                      source_pairs_split_into_multiple_native_edges=sum(len(x)>1 for x in source_to_native.values()),
                      nonmanifold_source_pairs_split=sum(len(source_to_native[pair])>1 for pair,x in source_edges.items() if len(x)>2),
                      cross_original_edge_component_merges=0)
    summary=dict(classification='PASS',vertices=n,triangles=F,position_bits_exact=True,oriented_FaceId_multiplicity_exact=True,
                 source_ID_colour_row_association_exact=True,colour_scope='Unchanged authenticated external colour rows remain associated through exact original source IDs; no native colour attribute or fit was created.',
                 cyclic_rotation_counts={str(i):int((rotations==i).sum()) for i in range(3)},original_topology=original_stats,native_topology=native_stats,
                 sheet_correspondence='UNRESOLVED',sheet_reason='Original incidence and observed native pairing are known, but no authenticated original junction sheet partition was supplied.',remeshing_admission='NOT AUTHORISED')
    return summary,details

def main():
    sys.path.insert(0,str(E));import common as c
    c.begin('verify');began=time.monotonic();out=c.P/'verify'
    result=dict(classification='BLOCKED',brick_input_transport='BLOCKED',foliage_input_transport='NOT RUN',sheet_correspondence='UNRESOLVED',remeshing_admission='NOT AUTHORISED',patches=[])
    def check():require(time.monotonic()-began<150,'150-second saved-data verification budget')
    def manifest(directory,entries):
        require(type(entries)is dict,'Artifact manifest')
        for rel,row in entries.items():
            relative_artifact_name(rel)
            path=c.plain(directory/rel)
            require(path.is_relative_to(directory) and path.is_file() and path.stat().st_size==row['bytes'] and row['bytes']<=2*1024**2 and c.digest(path)==row['sha256'],'Exact saved artifact identity')
            check()
    def write_array(name,array,np):
        with (out/name).open('xb') as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    try:
        c.authenticate();plan=c.load_plan();lineage=c.load_lineage();check()
        record=c.read_json(c.P/'import/result.json');host=c.read_json(c.P/'import-host.json')
        require(record['classification']==host['classification']=='PASS' and host['worker_receipt_sha256']==c.digest(c.P/'import/result.json'),'Qualified saved constructor predecessor')
        require(record['constructor_calls']==2 and record['remesh_calls']==record['target_generation_calls']==0 and record['constructors_completed']is True,'Only the two conditional original constructors')
        require(record['input_manifest_SHA256']==c.digest(c.INPUT_MANIFEST) and record['library_pins_SHA256']==c.digest(c.SDK),'Exact original inputs and SDK')
        require([row['name'] for row in record['patches']]==['brick','foliage'],'Brick then conditional foliage')
        manifest(c.P/'import',record['files'])
        result['constructor_result_SHA256']=c.digest(c.P/'import/result.json');result['constructor_host_SHA256']=c.digest(c.P/'import-host.json')
        import numpy as np
        for ordinal,(patch,producer) in enumerate(zip(plan['patches'],record['patches']),1):
            check();name=patch['name'];directory=c.P/'import'/name;row=c.read_json(directory/'result.json')
            require(row==producer and row['classification']=='PASS' and row['constructor_calls']==1 and row['remesh_calls']==0,'Exact complete per-patch producer receipt')
            manifest(directory,row['files'])
            original={}
            for key,item in patch['inputfiles'].items():
                array=np.load(c.plain(item['path']),mmap_mode='r',allow_pickle=False);schema=patch['array_schemas'][key]
                require(array.dtype==np.dtype(schema['dtype']) and list(array.shape)==schema['shape'],'Exact original NPY schema');original[key]=array
            saved={Path(rel).stem:np.load(c.plain(directory/rel),mmap_mode='r',allow_pickle=False) for rel in row['files'] if rel.endswith('.npy')}
            summary,private=verify_arrays(np,original,saved)
            require(summary['vertices']==patch['vertex_count'] and summary['triangles']==patch['triangle_count'],'Every original patch row')
            require(summary['original_topology']==row['original_topology'],'Independent original topology agrees with observed source statistics')
            counter=row['counter'];before=c.read_json(directory/'counter-before.json');after=c.read_json(directory/'counter-after.json');journal=c.read_json(directory/'before-constructor.json')
            require(before=={key:counter[key] for key in ['before','settings_before','region_before','settings_region_before']} and after==counter,'Durable before/after counter observations')
            require(c.read_json(directory/'counter-holder-before.json')==counter['before'] and c.read_json(directory/'counter-after-holder.json')==counter['after'],'Durable actual scalar storage observations')
            F=patch['triangle_count'];require(counter['region_before']==counter['settings_region_before']==list(range(F)),'Complete original pre-call regions')
            require(counter['failed_ids']==counter['settings_failed_ids']==saved['failed_face_ids'].tolist(),'All independent failed-face enumerations agree')
            verify_counter(counter['before'],counter['settings_before'],counter['after'],counter['settings_after'],counter['failed_ids'],F)
            require(counter['after']['values']==[0] and row['skipped_face_count']==row['failed_face_count']==0,'Zero skipped or failed original faces')
            require(journal['name']==name and journal['constructor_call_ordinal']==ordinal and journal['counter_before']==counter['before'] and journal['settings_before']==counter['settings_before']
                    and journal['region_before']==journal['settings_region_before']==list(range(F)) and journal['inputfiles']==patch['inputfiles']
                    and journal['allowNonManifoldEdge']is True and journal['duplicateNonManifoldVertices']is False,'Exact one-call journal and retained settings')
            # Only authenticated external original rows are inspected; no sampling.
            direct_item=lineage['patches'][name]['files']['direct.npy'];direct=np.load(c.plain(direct_item['path']),mmap_mode='r',allow_pickle=False)
            require(direct.dtype==np.float32 and direct.shape==(patch['vertex_count'],6) and np.isfinite(direct).all(),'Authenticated finite external colour/material source rows')
            if name=='foliage':
                linear=np.load(c.plain(lineage['foliage_linear_RGB']['path']),mmap_mode='r',allow_pickle=False)
                require(linear.dtype==np.float32 and linear.shape==(patch['vertex_count'],3) and np.isfinite(linear).all(),'Authenticated external foliage linear colour rows')
            summary['counter_observed_write_and_zero_failed_faces']=True
            c.save(out/(name+'-junction-memberships.json'),private)
            write_array(name+'-source-vertex-ids.npy',saved['source_vertex_ids'],np);write_array(name+'-source-face-ids.npy',saved['source_face_ids'],np)
            result['patches'].append(dict(name=name,**summary));result[name+'_input_transport']='PASS'
        require(len(result['patches'])==2,'Both independent patch checks completed')
        result.update(classification='PASS',verification_completed=True,elapsed_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(classification='BLOCKED',verification_completed=False,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
    result['files']={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in {'.json','.npy'} and p.name!='result.json'}
    c.save(out/'result.json',result)

if __name__=='__main__':main()
