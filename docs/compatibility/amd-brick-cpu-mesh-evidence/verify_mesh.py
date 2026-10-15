"""Independent CPU witnesses for the existing brick mesh; never extracts faces."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback

E=Path(__file__).resolve().parent;ROOT=E.parents[2]
BASE=ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579'
OUTPUT=BASE/'mesh'
SOURCE=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors'
SOURCE_SHA='d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861'
ROWS=1957891
CHILDREN=[21391,97313,438684,1957891]
FILES=('mesh.ply','vertices.npy','faces.npy','mesh.json')
BATCH=32768
RESULT=E/'independent-mesh-analysis.json'
HOST_RESULT=E/'independent-mesh-analysis-host.json'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def plain(path):
    path=Path(path).absolute()
    if str(path).startswith('\\\\') or path!=path.resolve():raise ValueError('Canonical local path required')
    for ancestor in (path,*path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            if ancestor.is_symlink() or getattr(ancestor.lstat(),'st_file_attributes',0)&0x400:
                raise ValueError('Reparse/junction/symlink prohibited: '+str(ancestor))
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Plain single-link file required')
    return path


def snapshot(path):
    path=plain(path);before=path.stat();sha=digest(path);after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed while hashing')
    return dict(bytes=after.st_size,sha256=sha,mtime_ns=after.st_mtime_ns,file_id=after.st_ino,links=after.st_nlink)


def unique(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ValueError('Duplicate JSON key')
        value[key]=item
    return value


def read_json(path):
    plain(path)
    if Path(path).stat().st_size>256*1024:raise ValueError('Metadata exceeds bounded size')
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique)


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def incidence(np,sorted_edges,need,batch=BATCH):
    """Count sorted edge runs with bounded temporaries and cross-batch carry."""
    total=boundary=two=nonmanifold=maximum=0;last=None;pending=0
    def count(value):
        nonlocal total,boundary,two,nonmanifold,maximum
        total+=1;boundary+=value==1;two+=value==2;nonmanifold+=value>2;maximum=max(maximum,value)
    for start in range(0,len(sorted_edges),batch):
        block=sorted_edges[start:start+batch]
        changes=np.r_[0,np.flatnonzero(block[1:]!=block[:-1])+1,len(block)]
        keys=block[changes[:-1]];lengths=np.diff(changes)
        for key,length in zip(keys.tolist(),lengths.tolist()):
            if last is not None and key!=last:count(pending);pending=0
            last=key;pending+=length
        need(True,'Edge-incidence deadline')
    if last is not None:count(pending)
    return dict(unique_edges=total,boundary_edges=boundary,two_face_edges=two,
                nonmanifold_edges=nonmanifold,maximum_incidence=maximum)


def geometry(np,c,f,v,t,need,pv=None,pf=None):
    """Original checker's witnesses, with bounded omitted-quad query batches."""
    n,nf=len(v),len(t)
    need(0<n<=ROWS and 0<nf<=6*n and nf%2==0,'Bounded nonempty paired triangles required')
    need(v.shape==(n,3) and v.dtype==np.float32 and t.shape==(nf,3) and t.dtype==np.int32,'Mesh array schema')
    maximum_error=0.;zero_area=repeated_indices=0
    for start in range(0,n,BATCH):
        x=f[start:start+BATCH,:3];z=np.exp(-np.abs(x));sig=np.empty_like(x);positive=x>=0
        sig[positive]=1/(1+z[positive]);sig[~positive]=z[~positive]/(1+z[~positive])
        expected=(c[start:start+BATCH].astype(np.float32)+(2*sig-.5))*np.float32(1/512)-.5
        maximum_error=max(maximum_error,float(np.max(np.abs(v[start:start+BATCH]-expected))))
        need(np.isfinite(expected).all() and np.isfinite(v[start:start+BATCH]).all(),'Finite vertices')
        if pv is not None:need(np.array_equal(v[start:start+BATCH],pv[start:start+BATCH]),'PLY vertex readback')
    need(maximum_error==0,'Source vertex transform/order')
    face_keys=np.empty(nf,np.int64)
    for start in range(0,nf,BATCH):
        tt=t[start:start+BATCH]
        need(tt.min()>=0 and tt.max()<n,'Face indices')
        if pf is not None:need(np.array_equal(tt,pf['idx'][start:start+BATCH]) and np.all(pf['count'][start:start+BATCH]==3),'PLY face readback')
        repeated_indices+=int(np.count_nonzero((tt[:,0]==tt[:,1])|(tt[:,1]==tt[:,2])|(tt[:,2]==tt[:,0])))
        tri=v[tt].astype(np.float64)
        zero_area+=int(np.count_nonzero(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0))
        sorted_tri=np.sort(tt,axis=1).astype(np.int64)
        face_keys[start:start+len(tt)]=(sorted_tri[:,0]*n+sorted_tri[:,1])*n+sorted_tri[:,2]
    need(n**3-1<2**63 and repeated_indices==0 and zero_area==0,'Nondegenerate valid faces')
    face_keys.sort();duplicates=int(np.count_nonzero(face_keys[1:]==face_keys[:-1]));del face_keys
    need(duplicates==0,'Duplicate triangle')
    offsets=np.array([[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
                      [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
                      [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]],np.int32)
    represented=np.zeros((n,3),bool);last=-1;split1=0
    for start in range(0,nf//2,BATCH):
        pairs=t[2*start:2*min(start+BATCH,nf//2)].reshape(-1,2,3);a,b=pairs[:,0],pairs[:,1]
        one=(b[:,0]==a[:,0])&(b[:,1]==a[:,2]);two=(b[:,0]==a[:,2])&(b[:,1]==a[:,1])
        need(np.all(one^two),'Consecutive source split/winding pattern')
        q=np.column_stack((a[:,0],a[:,1],np.where(one,a[:,2],b[:,2]),np.where(one,b[:,2],a[:,2])))
        delta=c[q[:,1]]-c[q[:,0]];matches=np.all(delta[:,None,:]==offsets[:,1,:],axis=2)
        need(np.all(matches.sum(axis=1)==1),'Quad axis')
        axes=matches.argmax(axis=1)
        need(np.all(c[q]==c[q[:,0],None,:]+offsets[axes]),'Original coordinate quad corners/winding')
        need(np.all(f[q[:,0],3+axes]>0),'Original positive source flag')
        raw=f[q,6];w=np.maximum(raw,np.float32(0))+np.log1p(np.exp(-np.abs(raw)));w=np.where(raw>20,raw,w)
        need(np.array_equal(one,w[:,0]*w[:,2]>w[:,1]*w[:,3]),'Float32 split weight rule')
        keys=q[:,0].astype(np.int64)*3+axes
        need(keys[0]>last and np.all(keys[1:]>keys[:-1]),'Quad order or duplicate anchor')
        last=int(keys[-1]);represented[q[:,0],axes]=True;split1+=int(one.sum())
    sorted_keys=np.sort((c[:,0].astype(np.int64)*512+c[:,1])*512+c[:,2])
    flagged=omitted=outside=0
    for start in range(0,n,BATCH):
        flags=f[start:start+BATCH,3:6]>0;flagged+=int(flags.sum())
        rows,axes=np.nonzero(flags&~represented[start:start+BATCH]);rows+=start;omitted+=len(rows)
        query_coords=c[rows,None,:]+offsets[axes]
        query=(query_coords[:,:,0].astype(np.int64)*512+query_coords[:,:,1])*512+query_coords[:,:,2]
        positions=np.searchsorted(sorted_keys,query);inside=positions<n;found=np.zeros(positions.shape,bool)
        found[inside]=sorted_keys[positions[inside]]==query[inside]
        in_grid=np.all((query_coords>=0)&(query_coords<512),axis=(1,2));outside+=int((~in_grid).sum())
        need(not np.any(found.all(axis=1)&in_grid),'Omitted valid source quad')
    del sorted_keys,represented,query_coords,query,positions,found
    edges=np.empty(nf*3,np.int64)
    for start in range(0,nf,BATCH):
        tt=t[start:start+BATCH].astype(np.int64)
        for j,(a,b) in enumerate(((0,1),(1,2),(2,0))):
            edges[3*start+j:3*(start+len(tt)):3]=np.minimum(tt[:,a],tt[:,b])*n+np.maximum(tt[:,a],tt[:,b])
        need(True,'Edge-key deadline')
    edges.sort();edge_incidence=incidence(np,edges,need);del edges
    return dict(index_and_PLY_readback=True,original_vertices_exact=True,max_vertex_transform_error=maximum_error,
        every_quad_source_verified=True,original_winding_verified=True,no_omitted_valid_quads=True,
        valid_quads=nf//2,split_1=split1,split_2=nf//2-split1,flagged_quads=flagged,
        omitted_missing_corner_quads=omitted,omitted_out_of_grid_quads=outside,
        zero_area_triangles=zero_area,repeated_index_triangles=repeated_indices,duplicate_triangles=duplicates,
        edge_incidence=edge_incidence,components_measured=False,self_intersections_measured=False,
        topology_repair=False,edge_incidence_is_diagnostic=True)


def ply_arrays(np,path,n,nf,opened,need):
    with Path(path).open('rb') as stream:
        lines=[]
        for _ in range(20):
            line=stream.readline(512).decode('ascii').strip();lines.append(line)
            if line=='end_header':break
        offset=stream.tell()
    expected=['ply','format binary_little_endian 1.0',f'element vertex {n}',
              'property float x','property float y','property float z',f'element face {nf}',
              'property list uchar int vertex_indices','end_header']
    need([line for line in lines if not line.startswith('comment ')]==expected,'Exact binary PLY header/schema')
    need(Path(path).stat().st_size==offset+12*n+13*nf,'PLY lengths')
    v=np.memmap(path,mode='r',dtype='<f4',offset=offset,shape=(n,3));opened.append(v)
    f=np.memmap(path,mode='r',dtype=[('count','u1'),('idx','<i4',(3,))],offset=offset+12*n,shape=(nf,));opened.append(f)
    return v,f,dict(bytes=offset,lines=lines)


def worker():
    if sys.stdin.buffer.read()!=b'GO\n':raise RuntimeError('Missing owned analysis gate')
    began=time.monotonic();record=dict(classification='UNKNOWN',CPU_only=True,forbidden_imports=[],mesh_extraction_executed=False)
    opened=[]
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0].lower() in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy','cumesh','nvdiffrast'}:
            record['forbidden_imports'].append(str(args[0]));raise RuntimeError('Forbidden model/GPU import')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('Forbidden child/network')
    sys.addaudithook(audit)
    def need(ok,why):
        if time.monotonic()-began>175:raise TimeoutError('175-second worker analysis bound')
        if not ok:raise ValueError(why)
    try:
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts/compatibility'))
        import amd_generation_mesh as mesh
        api=mesh._contract_api();contract=api.load(SOURCE.parent/'run-contract.json')
        metadata=read_json(OUTPUT/'mesh.json');sidecar=read_json(SOURCE.parent/'provenance.json')
        need(metadata['input_sha256']==SOURCE_SHA and metadata['vertices']==ROWS
             and metadata['learned_subdivisions']==CHILDREN and metadata['PLY_readback_exact'] is True
             and metadata['zero_area_triangles']==0 and metadata['topology_repair'] is False
             and metadata['native_extraction'] is False and metadata['material_decoded'] is False,'Extraction metadata')
        need(set(metadata['artifacts'])==set(FILES)-{'mesh.json'} and sorted(p.name for p in OUTPUT.iterdir())==sorted(FILES),'Exact four-file mesh closure')
        input_guest=str(contract.input_paths['coords.npy'])
        need(input_guest.startswith('/mnt/c/'),'Authenticated original coordinate drive')
        input_coords=Path('C:/'+input_guest[len('/mnt/c/'):]);input_coords.resolve().relative_to(ROOT.resolve())
        watched=[SOURCE,SOURCE.parent/'run-contract.json',SOURCE.parent/'provenance.json',input_coords,
                 *[OUTPUT/name for name in FILES],Path(__file__),Path(mesh.__file__),Path(mesh.converter.__file__)]
        before={str(path):snapshot(path) for path in watched}
        need(before[str(SOURCE)]['sha256']==SOURCE_SHA,'Genuine source SHA256')
        for name,expected in metadata['artifacts'].items():
            actual=before[str(OUTPUT/name)];need(actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes'],'Mesh artifact hash/size: '+name)
        api.artifact_binding(sidecar['provenance'],contract)
        need(sidecar['run_contract']==contract.document() and sidecar['output']['sha256']==SOURCE_SHA,'Source provenance/contract')
        arrays,children=mesh.read_neural(SOURCE,SOURCE_SHA,contract=contract);opened.extend(arrays.values())
        need(children==CHILDREN and len(arrays)==10,'Genuine ten-tensor support')
        tensor_hashes={name:hashlib.sha256(memoryview(array)).hexdigest() for name,array in arrays.items()}
        need(tensor_hashes==sidecar['output']['tensor_payload_sha256'],'All ten source tensor payload hashes')
        original=api.read_npy(input_coords,contract.input_sha256['coords.npy'],(4430,4),'int32')
        need(np.array_equal(arrays['subdivision_0_coordinates'],original),'Exact original input coordinates')
        c=arrays['coordinates'][:,1:];f=arrays['features']
        v=np.load(OUTPUT/'vertices.npy',mmap_mode='r',allow_pickle=False,max_header_size=4096);opened.append(v)
        t=np.load(OUTPUT/'faces.npy',mmap_mode='r',allow_pickle=False,max_header_size=4096);opened.append(t)
        nf=metadata['triangles']
        need(type(nf) is int and 0<nf<=6*ROWS and v.shape==(ROWS,3) and t.shape==(nf,3),'Exact output sizes')
        need(v.flags.c_contiguous and t.flags.c_contiguous and v.offset+v.nbytes==(OUTPUT/'vertices.npy').stat().st_size
             and t.offset+t.nbytes==(OUTPUT/'faces.npy').stat().st_size,'NPY layout/trailing bytes')
        pv,pf,ply_header=ply_arrays(np,OUTPUT/'mesh.ply',ROWS,nf,opened,need)
        analysis=geometry(np,c,f,v,t,need,pv,pf)
        conversion=metadata['conversion']
        expected=dict(flagged_quads=analysis['flagged_quads'],valid_quads=analysis['valid_quads'],
            missing_neighbour_quads=analysis['omitted_missing_corner_quads'],out_of_grid_quads=analysis['omitted_out_of_grid_quads'],
            split_1=analysis['split_1'],split_2=analysis['split_2'],global_original_indices=True,batch_rows=32768)
        need(conversion==expected,'Independent source witness/conversion metadata agreement')
        need(metadata['bounds']==dict(min=v.min(axis=0).tolist(),max=v.max(axis=0).tolist()),'Mesh bounds metadata')
        record.update(analysis,source_path=str(SOURCE),source_sha256=SOURCE_SHA,run_contract_sha256=contract.sha256,
            source_job_sha256=contract.source_job_sha256,final_support_profile=contract.final_support_profile,
            ten_tensor_payload_sha256=tensor_hashes,vertices=ROWS,triangles=nf,ply_header=ply_header,
            metadata=metadata,originals_before=before,numpy=np.__version__,
            memory_accounting=dict(maximum_triangles=6*ROWS,face_key_bytes=nf*8,edge_key_bytes=nf*24,
                                  omitted_query_row_batch=BATCH,edge_run_batch=BATCH,
                                  duplicate_face_keys_freed_before_edges=True,whole_verifier_bound_claimed=False))
        after={str(path):snapshot(path) for path in watched}
        need(before==after,'Source/inputs/output changed during verification')
        record.update(originals_after=after,immutability_verified=True,classification='PASS_GEOMETRY_READBACK')
    except BaseException as exc:record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        closed=True
        for array in opened:
            try:array._mmap.close();closed=closed and array._mmap.closed
            except BaseException:closed=False
        record.update(all_memmaps_closed=closed,mappings_opened=len(opened),elapsed_seconds=time.monotonic()-began)
        if not closed:record['classification']='UNKNOWN'
    save(RESULT,record)
    print(json.dumps({key:record.get(key) for key in ('classification','vertices','triangles','valid_quads','zero_area_triangles','edge_incidence','all_memmaps_closed','elapsed_seconds','error')}),flush=True)
    return 0 if record['classification']=='PASS_GEOMETRY_READBACK' else 1


def host():
    import psutil
    if RESULT.exists() or HOST_RESULT.exists():raise FileExistsError('Independent evidence already exists')
    ram=psutil.virtual_memory().available;disk=psutil.disk_usage(str(BASE)).free
    if ram<12*1024**3 or disk<2*1024**3:raise RuntimeError('12GiB available RAM and 2GiB free disk required')
    sys.path.insert(0,str(ROOT/'scripts/compatibility'))
    from amd_forward_owned_job import OwnedJob,ExtendedLimits
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=6*1024**3
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
    record=dict(required_RAM_bytes=12*1024**3,available_RAM_bytes=ram,required_disk_bytes=2*1024**3,
                free_disk_bytes=disk,process_memory_limit_bytes=6*1024**3,timeout_seconds=180)
    process=None
    try:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (E/'independent-mesh-analysis-worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
            job.attach(process);process.stdin.write(b'GO\n');process.stdin.close();process.wait(timeout=180)
        record.update(returncode=process.returncode,worker_exit_verified=True,timed_out=False)
        memory=ExtendedLimits()
        if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(memory),ctypes.sizeof(memory),None):raise ctypes.WinError()
        record['peak_process_committed_bytes']=memory.PeakProcessMemory
    except BaseException as exc:record.update(error=str(exc),timed_out=isinstance(exc,subprocess.TimeoutExpired))
    finally:
        if process is not None and not process.stdin.closed:process.stdin.close()
        try:record['job_cleanup']=job.finish()
        except BaseException as exc:record['job_cleanup_error']=str(exc)
        cleanup=record.get('job_cleanup',{})
        passed=(record.get('returncode')==0 and record.get('worker_exit_verified') is True
                and record.get('timed_out') is False and not record.get('error')
                and cleanup.get('owned_tree_exit_verified') is True and cleanup.get('forced_job_cleanup') is False
                and not record.get('job_cleanup_error'))
        record['classification']='PASS' if passed else 'UNKNOWN';save(HOST_RESULT,record)
    return 0 if passed else 1


def fixture_only():
    import numpy as np
    def need(ok,why):
        if not ok:raise ValueError(why)
    c=np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]],np.int32)
    f=np.zeros((4,7),np.float32);f[0,5]=1
    v=(c.astype(np.float32)+np.float32(.5))*np.float32(1/512)-np.float32(.5)
    split2=np.array([[0,1,3],[3,1,2]],np.int32)
    result=geometry(np,c,f,v,split2,need)
    need(result['valid_quads']==1 and result['split_2']==1 and result['edge_incidence']==dict(unique_edges=5,boundary_edges=4,two_face_edges=1,nonmanifold_edges=0,maximum_incidence=2),'Small positive source witness')
    weighted=f.copy();weighted[[0,2],6]=2
    split1=np.array([[0,1,2],[0,2,3]],np.int32)
    need(geometry(np,c,weighted,v,split1,need)['split_1']==1,'Nonzero source split weights')
    collapsed=f.copy();collapsed[:,:3]=np.where(c>0,np.float32(-100),np.float32(0))
    collapsed_vertices=np.full_like(v,np.float32(.5/512-.5))
    failures=0
    for features,vertices,faces in ((weighted,v,split2),(f,v,split2[:,::-1].copy()),(f,v+np.float32(.01),split2),
                                     (collapsed,collapsed_vertices,split2),(f,v,np.array([[0,1,1],[1,1,2]],np.int32))):
        try:geometry(np,c,features,vertices,faces,need)
        except ValueError:failures+=1
        else:raise AssertionError('Corrupt geometry fixture accepted')
    # Omitted valid second quad, preserving valid first-quad source order.
    cc=np.concatenate((c,c+np.array([3,0,0],np.int32)));ff=np.concatenate((f,f));vv=np.concatenate((v,v+np.array([3/512,0,0],np.float32)))
    try:geometry(np,cc,ff,vv,split2,need)
    except ValueError as exc:need('Omitted valid' in str(exc),'Omission rejection reason')
    else:raise AssertionError('Omitted valid source quad accepted')
    runs=incidence(np,np.array([1,1,1,2,3,3,4,4,4,4],np.int64),need,batch=2)
    need(runs==dict(unique_edges=4,boundary_edges=1,two_face_edges=1,nonmanifold_edges=2,maximum_incidence=4),'Cross-batch incidence carry')
    forbidden={'torch','triton','flex_gemm','o_voxel','trellis2','cupy','cumesh','nvdiffrast'}
    need(not any(name.split('.')[0].lower() in forbidden for name in sys.modules),'Unexpected model/GPU import')
    print(json.dumps(dict(classification='PASS_CPU_FIXTURES',positive_source_splits=2,negative_geometry_cases=failures+1,
                          edge_run_boundary_case=True,model_or_GPU_imported=False,mesh_extraction_executed=False)))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    raise SystemExit(fixture_only() if args.fixture_only else worker() if args.worker else host())
