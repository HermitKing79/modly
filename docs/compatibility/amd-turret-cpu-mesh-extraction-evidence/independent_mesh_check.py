"""Read existing arrays only; independently check quad witnesses and edge incidence."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

E=Path(__file__).resolve().parent;ROOT=E.parents[2]
A=ROOT.parent/'artifacts/amd-turret-cpu-mesh-ad07f96'
S=ROOT.parent/'artifacts/amd-wsl-turret-writer-ready-4870856/turret-neural-output.safetensors'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2)
def worker():
    if sys.stdin.buffer.read()!=b'GO\n':raise RuntimeError('Missing analysis gate')
    began=time.monotonic();r={'classification':'UNKNOWN','CPU_only':True,'forbidden_imports':[]}
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy'}:
            r['forbidden_imports'].append(str(args[0]));raise RuntimeError('Forbidden import')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('Forbidden child/network')
    sys.addaudithook(audit)
    def need(ok,why):
        if time.monotonic()-began>85:raise TimeoutError('Analysis bound')
        if not ok:raise ValueError(why)
    try:
        import numpy as np
        original=json.loads((E/'extraction-worker.json').read_text())
        r['file_hashes']={n:digest(A/n) for n in original['artifacts']}
        need(all(h==original['artifacts'][n]['sha256'] for n,h in r['file_hashes'].items()),'Artifact hash')
        need(digest(S)=='dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22','Source hash')
        with S.open('rb') as f:
            hs=int.from_bytes(f.read(8),'little');need(0<hs<=131072,'Header bound');header=json.loads(f.read(hs))
        for name,row in header.items():
            if name=='__metadata__':continue
            dt={'F32':'<f4','I32':'<i4','BF16':'<u2'}[row['dtype']]
            arr=np.memmap(S,mode='r',offset=8+hs+row['data_offsets'][0],dtype=dt,shape=tuple(row['shape']))
            need(hashlib.sha256(memoryview(arr)).hexdigest()==original['source']['ten_tensor_payload_sha256'][name],'Raw source '+name)
        def source(name):
            row=header[name]
            return np.memmap(S,mode='r',offset=8+hs+row['data_offsets'][0],dtype={'F32':'<f4','I32':'<i4'}[row['dtype']],shape=tuple(row['shape']))
        c=source('coordinates')[:,1:];f=source('features')
        v=np.load(A/'vertices.npy',mmap_mode='r',allow_pickle=False);t=np.load(A/'faces.npy',mmap_mode='r',allow_pickle=False)
        n=len(v);nf=len(t);need(v.shape==(1842968,3) and v.dtype==np.float32 and t.shape==(4012360,3) and t.dtype==np.int32,'Array schema')
        with (A/'turret-full.ply').open('rb') as stream:
            lines=[]
            for _ in range(20):
                line=stream.readline(512).decode('ascii').strip();lines.append(line)
                if line=='end_header':break
            off=stream.tell()
        need(lines[:2]==['ply','format binary_little_endian 1.0'] and lines[-1]=='end_header','PLY header')
        need(f'element vertex {n}' in lines and f'element face {nf}' in lines and (A/'turret-full.ply').stat().st_size==off+12*n+13*nf,'PLY lengths')
        pv=np.memmap(A/'turret-full.ply',mode='r',dtype='<f4',offset=off,shape=(n,3))
        pf=np.memmap(A/'turret-full.ply',mode='r',dtype=[('count','u1'),('idx','<i4',(3,))],offset=off+12*n,shape=(nf,))
        max_vertex_error=0.
        for start in range(0,n,32768):
            x=f[start:start+32768,:3];z=np.exp(-np.abs(x));sig=np.empty_like(x);positive=x>=0
            sig[positive]=1/(1+z[positive]);sig[~positive]=z[~positive]/(1+z[~positive])
            expected=(c[start:start+32768].astype(np.float32)+(2*sig-.5))*np.float32(1/512)-.5
            max_vertex_error=max(max_vertex_error,float(np.max(np.abs(v[start:start+32768]-expected))))
            need(np.array_equal(v[start:start+32768],pv[start:start+32768]) and np.isfinite(expected).all(),'Vertex readback/finite')
        need(max_vertex_error==0,'Source vertex transform/order')
        for start in range(0,nf,32768):
            tt=t[start:start+32768];need(tt.min()>=0 and tt.max()<n and np.array_equal(tt,pf['idx'][start:start+32768]) and np.all(pf['count'][start:start+32768]==3),'Face readback/index')
        offsets=np.array([[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[[0,0,0],[1,0,0],[1,0,1],[0,0,1]],[[0,0,0],[0,1,0],[1,1,0],[1,0,0]]],np.int32)
        represented=np.zeros((n,3),bool);last=-1;split1=0
        for start in range(0,nf//2,32768):
            pairs=t[2*start:2*min(start+32768,nf//2)].reshape(-1,2,3);a,b=pairs[:,0],pairs[:,1]
            one=(b[:,0]==a[:,0])&(b[:,1]==a[:,2]);two=(b[:,0]==a[:,2])&(b[:,1]==a[:,1])
            need(np.all(one^two),'Consecutive source split pattern')
            q=np.column_stack((a[:,0],a[:,1],np.where(one,a[:,2],b[:,2]),np.where(one,b[:,2],a[:,2])))
            delta=c[q[:,1]]-c[q[:,0]];matches=np.all(delta[:,None,:]==offsets[:,1,:],axis=2)
            need(np.all(matches.sum(axis=1)==1),'Quad axis')
            axes=matches.argmax(axis=1)
            need(np.all(c[q]==c[q[:,0],None,:]+offsets[axes]),'Original coordinate quad corners')
            need(np.all(f[q[:,0],3+axes]>0),'Original positive source flag')
            raw=f[q,6];w=np.maximum(raw,np.float32(0))+np.log1p(np.exp(-np.abs(raw)));w=np.where(raw>20,raw,w)
            need(np.array_equal(one,w[:,0]*w[:,2]>w[:,1]*w[:,3]),'Float32 split weight rule')
            keys=q[:,0].astype(np.int64)*3+axes;need(keys[0]>last and np.all(keys[1:]>keys[:-1]),'Quad order or duplicate anchor')
            last=int(keys[-1]);represented[q[:,0],axes]=True;split1+=int(one.sum())
        missing_row,missing_axis=np.nonzero((f[:,3:6]>0)&~represented)
        # Check only omitted flag witnesses, not a second mesh construction.
        sorted_keys=np.sort((c[:,0].astype(np.int64)*512+c[:,1])*512+c[:,2])
        query_coords=c[missing_row,None,:]+offsets[missing_axis]
        query=(query_coords[:,:,0].astype(np.int64)*512+query_coords[:,:,1])*512+query_coords[:,:,2]
        positions=np.searchsorted(sorted_keys,query);inside=positions<n;found=np.zeros(positions.shape,bool)
        found[inside]=sorted_keys[positions[inside]]==query[inside]
        in_grid=np.all((query_coords>=0)&(query_coords<512),axis=(1,2))
        need(not np.any(found.all(axis=1)&in_grid),'Omitted valid source quad')
        r.update(index_and_PLY_readback=True,original_vertices_exact=True,max_vertex_transform_error=max_vertex_error,every_quad_source_verified=True,valid_quads=nf//2,split_1=split1,split_2=nf//2-split1,omitted_missing_corner_quads=len(missing_row),omitted_out_of_grid_quads=int((~in_grid).sum()))
        edges=np.empty(nf*3,np.int64)
        for start in range(0,nf,32768):
            tt=t[start:start+32768].astype(np.int64)
            for j,(a,b) in enumerate(((0,1),(1,2),(2,0))):
                edges[3*start+j:3*(start+len(tt)):3]=np.minimum(tt[:,a],tt[:,b])*n+np.maximum(tt[:,a],tt[:,b])
            need(True,'Deadline')
        edges.sort();starts=np.r_[0,np.flatnonzero(edges[1:]!=edges[:-1])+1];counts=np.diff(np.r_[starts,len(edges)])
        r['edge_incidence']={'unique_edges':len(counts),'boundary_edges':int((counts==1).sum()),'two_face_edges':int((counts==2).sum()),'nonmanifold_edges':int((counts>2).sum()),'maximum_incidence':int(counts.max()),'no_orientation_or_self_intersection_test':True}
        r['preview_safe']=True;r['classification']='PASS_GEOMETRY_READBACK';r['numpy']=np.__version__
    except BaseException as exc:r.update(error=str(exc),traceback=traceback.format_exc())
    r['elapsed_seconds']=time.monotonic()-began;save(E/'independent-mesh-analysis.json',r);print(json.dumps(r),flush=True)
    return 0 if r.get('preview_safe') else 1
def host():
    sys.path.insert(0,str(ROOT/'scripts/compatibility'))
    from amd_forward_owned_job import OwnedJob,ExtendedLimits
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=2*1024**3
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
    r={};p=None
    try:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],stdin=subprocess.PIPE,env=env);job.attach(p);p.stdin.write(b'GO\n');p.stdin.close();p.wait(timeout=90);r['returncode']=p.returncode
        m=ExtendedLimits()
        if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(m),ctypes.sizeof(m),None):raise ctypes.WinError()
        r['peak_process_committed_bytes']=m.PeakProcessMemory
    except BaseException as exc:r['error']=str(exc)
    finally:
        if p is not None and not p.stdin.closed:p.stdin.close()
        r['job_cleanup']=job.finish();save(E/'independent-mesh-analysis-host.json',r)
    return r.get('returncode',1)
if __name__=='__main__':raise SystemExit(worker() if '--worker' in sys.argv else host())
