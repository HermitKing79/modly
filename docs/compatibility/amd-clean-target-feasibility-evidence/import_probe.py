"""One no-duplication constructor transport probe per patch; no target operation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import time,traceback,os

class AdmissionError(ValueError):
    """A measured contract rejection, distinct from an unexpected API fault."""

def need(ok,message):
    if not ok:raise AdmissionError(message)

def compare_roundtrip(np,vertices,faces,source_vertex_ids,source_face_ids,readback_vertices,readback_faces,valid_face_ids,valid_vertex_ids):
    n=len(vertices);count=len(faces)
    need(vertices.dtype==np.float32 and vertices.shape==(n,3) and n>0 and np.isfinite(vertices).all(),'Finite original F32 vertex matrix')
    need(faces.dtype==np.int32 and faces.shape==(count,3) and count>0 and (faces>=0).all() and (faces<n).all(),'Original I32 triangle matrix')
    for ids,size in ((source_vertex_ids,n),(source_face_ids,count)):
        need(ids.dtype.kind in 'iu' and ids.shape==(size,) and (ids>=0).all() and len(np.unique(ids))==size,'Unique exact original source IDs')
    for actual,size in ((valid_face_ids,count),(valid_vertex_ids,n)):
        need(actual.dtype.kind in 'iu' and actual.shape==(size,) and np.array_equal(actual,np.arange(size)),'Dense unchanged native valid IDs')
    need(readback_vertices.dtype in (np.dtype('float32'),np.dtype('float64')) and readback_vertices.shape==vertices.shape
         and np.isfinite(readback_vertices).all(),'Finite exact native position schema')
    promoted=vertices.astype(readback_vertices.dtype);bits=np.uint32 if readback_vertices.dtype==np.float32 else np.uint64
    need(np.array_equal(readback_vertices.view(bits),promoted.view(bits)),'Every original position bit and exact F64 promotion, including signed zero')
    need(readback_faces.dtype.kind in 'iu' and readback_faces.shape==faces.shape and (readback_faces>=0).all()
         and (readback_faces<n).all(),'Every original native FaceId triangle')
    rotations=np.full(count,-1,np.int8)
    for shift in range(3):rotations[(readback_faces==np.roll(faces,-shift,axis=1)).all(1)]=shift
    need((rotations>=0).all(),'Original per-FaceId orientation/corners, allowing only cyclic rotations')
    # Dense original IDs and exact per-FaceId oriented triples also prove the
    # complete source triangle multiset/multiplicity and ID correspondence.
    summary=dict(original_vertices=n,original_triangles=count,dense_valid_vertex_IDs_exact=True,dense_valid_face_IDs_exact=True,
                 original_F32_position_bits_exact=True,native_position_dtype=str(readback_vertices.dtype),
                 source_vertex_ID_dtype=str(source_vertex_ids.dtype),source_face_ID_dtype=str(source_face_ids.dtype),
                 source_ID_correspondence_exact=True,oriented_triangle_multiplicity_exact=True,reversed_or_reordered_faces=0,
                 cyclic_rotation_counts={str(i):int((rotations==i).sum()) for i in range(3)})
    return summary,rotations

def topology_statistics(np,faces,edge_ids=None):
    """Bounded original edge/vertex connectivity, plus observed native pairing."""
    groups={};vertex_owner={};edge_parent=list(range(len(faces)));vertex_parent=list(range(len(faces)))
    def root(parent,x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def unite(parent,a,b):
        a,b=root(parent,a),root(parent,b);parent[max(a,b)]=min(a,b)
    native={}
    for row,tri in enumerate(faces):
        for vertex in tri:
            vertex=int(vertex)
            if vertex in vertex_owner:unite(vertex_parent,row,vertex_owner[vertex])
            else:vertex_owner[vertex]=row
        for corner in range(3):
            pair=tuple(sorted((int(tri[corner]),int(tri[(corner+1)%3]))));old=groups.setdefault(pair,[])
            if old:unite(edge_parent,row,old[0])
            old.append(row)
            if edge_ids is not None:native.setdefault(pair,set()).add(int(edge_ids[row,corner]))
    counts=[len(rows) for rows in groups.values()]
    result=dict(unique_source_edges=len(groups),boundary_source_edges=sum(x==1 for x in counts),
                nonmanifold_source_edges=sum(x>2 for x in counts),maximum_source_edge_incidence=max(counts,default=0),
                edge_connected_face_components=len({root(edge_parent,i) for i in range(len(faces))}),
                vertex_connected_face_components=len({root(vertex_parent,i) for i in range(len(faces))}))
    if edge_ids is not None:
        owner={}
        for pair,values in native.items():
            for value in values:
                need(value not in owner or owner[value]==pair,'Native edge ID cannot alias different source endpoint pairs');owner[value]=pair
        result.update(observed_native_undirected_edges=len(owner),source_pairs_with_multiple_native_edges=sum(len(x)>1 for x in native.values()),
                      original_source_sheet_pairing_proven=False)
    return result

def main():
    c.begin();began=time.monotonic();out=c.P/'import'
    record=dict(classification='BLOCKED',constructor_calls=0,remesh_calls=0,target_generation_calls=0,patches=[],
                interpretation='Constructor transport only; no original junction/sheet pairing or remesh admission is established.')
    def check():
        if time.monotonic()-began>150:raise TimeoutError('150-second constructor probe budget')
    def write(directory,name,array):
        check();path=directory/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        saved=np.load(path,mmap_mode='r',allow_pickle=False)
        need(saved.dtype==array.dtype and saved.shape==array.shape,'Private readback schema')
        need(np.array_equal(saved.view(np.uint32 if array.dtype==np.float32 else np.uint64),array.view(np.uint32 if array.dtype==np.float32 else np.uint64))
             if array.dtype.kind=='f' else np.array_equal(saved,array),'Private exact readback bits')
    def files(directory):
        return {p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(directory.rglob('*'))
                if p.is_file() and p.suffix in {'.json','.npy'} and p!=directory/'result.json'}
    try:
        c.authenticate();plan=c.load_plan();check()
        import numpy as np
        from meshlib import mrmeshpy as mr,mrmeshnumpy as mn
        def cpu_modules():
            forbidden=[name for name in sys.modules if name.split('.')[0] in {'torch','bpy','cupy','pycuda'}
                       or name.startswith(('meshlib.mrcudapy','meshlib.mrviewerpy'))]
            need(not forbidden,'CPU constructor module closure only')
        cpu_modules();record['input_manifest_SHA256']=c.digest(c.P/'input-manifest.json')
        record['library_pins_SHA256']=c.digest(E/'constructor-library-pins.json')
        for patch in plan['patches']:
            check();directory=out/patch['name'];directory.mkdir()
            row=dict(name=patch['name'],classification='BLOCKED',constructor_calls=0,remesh_calls=0)
            record['patches'].append(row)
            try:
                arrays={}
                for name,item in patch['inputfiles'].items():
                    value=np.load(c.plain(item['path']),allow_pickle=False);schema=patch['array_schemas'][name]
                    need(value.dtype==np.dtype(schema['dtype']) and list(value.shape)==schema['shape'],'Exact authenticated original patch schema')
                    arrays[name]=value.copy()
                v,f,sv,sf=(arrays[name] for name in ('vertices','faces','source_vertex_ids','source_face_ids'))
                need(np.isfinite(v).all() and (f>=0).all() and (f<len(v)).all(),'Finite in-bounds original constructor inputs')
                need(len(np.unique(sv))==len(sv) and len(np.unique(sf))==len(sf),'Unique original source correspondence before native call')
                row['original_topology']=topology_statistics(np,f)
                native_v=v.copy();native_f=f.copy();region=mr.FaceBitSet(len(f),True);skipped=mr.int_output(0)
                settings=mr.MeshBuilder.BuildSettings();settings.allowNonManifoldEdge=True;settings.shiftFaceId=0
                settings.region=region;settings.skippedFaceCount=skipped
                journal=dict(name=patch['name'],constructor_call_ordinal=record['constructor_calls']+1,duplicateNonManifoldVertices=False,
                             allowNonManifoldEdge=True,original_vertices=len(v),original_triangles=len(f),inputfiles=patch['inputfiles'])
                c.save(directory/'before-constructor.json',journal)
                row['constructor_calls']=1;record['constructor_calls']+=1;started=time.monotonic()
                mesh=mn.meshFromFacesVerts(native_f,native_v,settings,duplicateNonManifoldVertices=False)
                row['constructor_elapsed_seconds']=time.monotonic()-started;check();cpu_modules()
                failed=np.flatnonzero(mn.getNumpyBitSet(region)).astype(np.int32)
                need(np.array_equal(mn.getNumpyBitSet(region),mn.getNumpyBitSet(settings.region)),'Actual output failed-face region agrees')
                skipped_count=int(skipped.value);row.update(skipped_face_count=skipped_count,failed_face_count=len(failed))
                valid_f=np.flatnonzero(mn.getNumpyBitSet(mesh.topology.getValidFaces())).astype(np.int32)
                valid_v=np.flatnonzero(mn.getNumpyBitSet(mesh.topology.getValidVerts())).astype(np.int32)
                rv=np.asarray(mn.getNumpyVerts(mesh));rf=np.empty((len(valid_f),3),np.int32);re=np.empty_like(rf)
                for index,face in enumerate(valid_f):
                    if index%256==0:check()
                    tri=mesh.topology.getTriVerts(mr.FaceId(int(face)));values=[int(tri[j].get()) for j in range(3)]
                    need(all(0<=x<2**31 for x in values),'Bounded native source vertex IDs');rf[index]=values
                    edge=[mr.EdgeId(),mr.EdgeId(),mr.EdgeId()];mesh.topology.getTriEdges(mr.FaceId(int(face)),*edge)
                    found={}
                    for item in edge:
                        pair=(int(mesh.topology.org(item).get()),int(mesh.topology.dest(item).get()));eid=int(item.undirected().get())
                        need(0<=eid<2**31 and pair not in found,'Distinct bounded native oriented edge IDs');found[pair]=eid
                    for corner in range(3):
                        pair=(values[corner],values[(corner+1)%3]);need(pair in found,'Native edge readback aligned to actual oriented face corners');re[index,corner]=found[pair]
                saved=dict(vertices_readback=rv,faces_readback=rf,native_undirected_edge_ids=re,valid_face_ids=valid_f,valid_vertex_ids=valid_v,
                           failed_face_ids=failed,source_vertex_ids=sv,source_face_ids=sf,vertices_after_native=native_v,faces_after_native=native_f)
                for name,array in saved.items():write(directory,name,array)
                row['native_topology']=topology_statistics(np,rf,re)
                row['heap_inputs_unchanged']=bool(np.array_equal(native_v.view(np.uint32),v.view(np.uint32)) and np.array_equal(native_f,f))
                need(row['heap_inputs_unchanged'],'Native constructor cannot mutate source heap copies')
                need(skipped_count==0 and len(failed)==0,'No skipped or failed original faces')
                summary,rotations=compare_roundtrip(np,v,f,sv,sf,rv,rf,valid_f,valid_v)
                write(directory,'cyclic_rotations',rotations)
                row.update(classification='PASS',correspondence=summary,source_sheet_pairing_admitted=False)
                del mesh
            except BaseException as error:
                row.update(classification='BLOCKED',error=str(error),traceback=traceback.format_exc())
                if not isinstance(error,AdmissionError):raise
            finally:
                row['files']=files(directory);c.save(directory/'result.json',row)
        cpu_modules();record.update(classification='PASS' if all(row['classification']=='PASS' for row in record['patches']) else 'BLOCKED',
                                     constructors_completed=True,CPU_only_modules_observed=True,elapsed_seconds=time.monotonic()-began,files=files(out))
        check();c.save(out/'result.json',record)
    except BaseException as error:
        record.update(classification='BLOCKED',constructors_completed=False,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began,files=files(out))
        c.save(out/'result.json',record);print('CONSTRUCTOR_PROBE_BLOCKED '+str(error),flush=True)

if __name__=='__main__':main()
