"""One CPU-only conversion of the immutable saved neural turret; no models."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

E=Path(__file__).resolve().parent
ROOT=E.parents[2]
SOURCE=ROOT.parent/'artifacts/amd-wsl-turret-writer-ready-4870856/turret-neural-output.safetensors'
OUTPUT=ROOT.parent/'artifacts/amd-turret-cpu-mesh-ad07f96'
SOURCE_SHA='dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22'
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
GIB=1024**3


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,record):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)


def validate_source(np):
    start=SOURCE.stat()
    if start.st_size!=96433760 or digest(SOURCE)!=SOURCE_SHA:raise ValueError('Immutable neural input hash/size differs')
    with SOURCE.open('rb') as f:
        size=int.from_bytes(f.read(8),'little')
        if not 0<size<=128*1024:raise ValueError('Unbounded safetensors header')
        header=json.loads(f.read(size))
    meta=header.pop('__metadata__');provenance=json.loads(meta['provenance']);summary=json.loads(meta['summary'])
    receipt=json.loads((E/'source-artifact-before.json').read_text())
    sidecar=json.loads(SOURCE.with_name('provenance.json').read_text())
    if (provenance!=sidecar['provenance'] or summary['rows']!=1842968 or not summary['finite']
        or sidecar['output']['sha256']!=SOURCE_SHA or receipt['sha256']!=SOURCE_SHA
        or provenance['resolution']!=512 or provenance['forward_calls']!=1 or provenance['mesh_extraction'] is not False):
        raise ValueError('Neural source provenance differs')
    schema={'features':('F32',[1842968,7]),'coordinates':('I32',[1842968,4])}
    for i,n in enumerate((3743,17015,79169,378891)):
        schema[f'subdivision_{i}_logits']=('BF16',[n,8]);schema[f'subdivision_{i}_coordinates']=('I32',[n,4])
    if set(header)!=set(schema):raise ValueError('Ten tensor closure differs')
    tensors={};ranges=[];hashes={}
    for name,(dtype,shape) in schema.items():
        row=header[name];a,b=row['data_offsets'];item={'F32':4,'I32':4,'BF16':2}[dtype]
        if (row['dtype']!=dtype or row['shape']!=shape or type(a)is not int or type(b)is not int
            or not 0<=a<b<=start.st_size-8-size or b-a!=item*int(np.prod(shape))):raise ValueError('Tensor layout differs: '+name)
        arr=np.memmap(SOURCE,mode='r',dtype={'F32':'<f4','I32':'<i4','BF16':'<u2'}[dtype],offset=8+size+a,shape=tuple(shape))
        raw=hashlib.sha256(memoryview(arr)).hexdigest();hashes[name]=raw
        if raw!=receipt['tensor_payload_sha256'][name]:raise ValueError('Tensor bytes differ: '+name)
        if dtype=='F32' and not np.isfinite(arr).all():raise ValueError('Nonfinite features')
        if dtype=='BF16' and np.any((arr&0x7f80)==0x7f80):raise ValueError('Nonfinite subdivision logits')
        tensors[name]=arr;ranges.append((a,b))
    ranges.sort()
    if ranges[0][0] or ranges[-1][1]!=start.st_size-8-size or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:])):
        raise ValueError('Incomplete/overlapping payload')
    for name,arr in tensors.items():
        if 'coordinates' not in name:continue
        res=512 if name=='coordinates' else 32*2**int(name.split('_')[1])
        if np.any(arr[:,0]!=0) or np.any(arr[:,1:]<0) or np.any(arr[:,1:]>=res):raise ValueError('Coordinate bounds: '+name)
        keys=(arr[:,1].astype(np.int64)*res+arr[:,2])*res+arr[:,3]
        if len(np.unique(keys))!=len(arr):raise ValueError('Coordinate uniqueness: '+name)
    for i,children in enumerate((17015,79169,378891,1842968)):
        u=tensors[f'subdivision_{i}_logits']
        if int((((u&0x8000)==0)&((u&0x7fff)!=0)).sum())!=children:raise ValueError('Subdivision count linkage')
    if (SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)!=(start.st_size,start.st_mtime_ns):raise ValueError('Source changed')
    return tensors,{'sha256':SOURCE_SHA,'bytes':start.st_size,'ten_tensor_payload_sha256':hashes,
                    'provenance':provenance,'all_tensors_validated':True,'read_only_memory_maps':True}


def worker():
    if sys.stdin.buffer.read()!=b'GO\n':raise RuntimeError('CPU worker authorization missing')
    began=time.monotonic();record={'classification':'UNKNOWN','CPU_only':True,'GPU_imports':[]}
    def no_ai(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy'}:
            record['GPU_imports'].append(str(args[0]));raise RuntimeError('CPU mesh worker forbids AI/GPU imports')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('CPU worker forbids child/network execution')
    sys.addaudithook(no_ai)
    try:
        import numpy as np
        import amd_flexible_dual_grid_cpu as fdg
        arrays,source=validate_source(np)
        coords=arrays['coordinates'][:,1:];features=arrays['features']
        if OUTPUT.exists() or OUTPUT.is_symlink():raise RuntimeError('Fresh absent mesh leaf required')
        OUTPUT.mkdir(exist_ok=False)
        def progress(done,stats):
            if done==len(coords) or done%262144==0:print(json.dumps({'rows_processed':done,'elapsed_seconds':time.monotonic()-began,**stats}),flush=True)
        transform_started=time.monotonic()
        dual,flags,weight=fdg.transform_features(features);verts=fdg.vertices(coords,dual)
        tris,stats=fdg.faces(coords,flags,weight,batch_rows=32768,progress=progress)
        extraction_seconds=time.monotonic()-transform_started
        if not len(tris) or not np.isfinite(verts).all() or tris.min()<0 or tris.max()>=len(verts):raise ValueError('Invalid full mesh')
        # The genuine margin permits half a voxel beyond the nominal AABB.
        if verts.min()<-.5-.5/512 or verts.max()>.5+.5/512:raise ValueError('Geometric scale differs')
        repeated=zero_area=0
        for start in range(0,len(tris),32768):
            t=tris[start:start+32768]
            repeated+=int(((t[:,0]==t[:,1])|(t[:,1]==t[:,2])|(t[:,0]==t[:,2])).sum())
            p=verts[t].astype(np.float64)
            cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            zero_area+=int((np.einsum('ij,ij->i',cross,cross)==0).sum())
        if repeated or zero_area:raise ValueError(f'Incorrect/degenerate triangles: repeated={repeated}, zero_area={zero_area}')
        # Three sorted original indices fit one signed 64-bit key at N<=2M.
        face_key=np.empty(len(tris),np.int64);n=len(verts)
        for start in range(0,len(tris),32768):
            t=np.sort(tris[start:start+32768].astype(np.int64),axis=1)
            face_key[start:start+len(t)]=(t[:,0]*n+t[:,1])*n+t[:,2]
        face_key.sort();duplicates=int((face_key[1:]==face_key[:-1]).sum());del face_key
        if duplicates:raise ValueError('Duplicate source triangles: '+str(duplicates))
        # Do not alter support, vertices, winding or triangles after conversion.
        np.save(OUTPUT/'vertices.npy',verts,allow_pickle=False);np.save(OUTPUT/'faces.npy',tris,allow_pickle=False)
        fdg.write_ply(OUTPUT/'turret-full.ply',verts,tris)
        pv,pf=fdg.read_ply(OUTPUT/'turret-full.ply')
        if not np.array_equal(pv,verts) or not np.array_equal(pf,tris):raise ValueError('PLY readback differs')
        for name,arr in [('vertices.npy',verts),('faces.npy',tris)]:
            if not np.array_equal(np.load(OUTPUT/name,mmap_mode='r',allow_pickle=False),arr):raise ValueError('Array readback differs')
        if digest(SOURCE)!=SOURCE_SHA:raise ValueError('Neural source changed during extraction')
        record.update(classification='PASS',numpy=np.__version__,source=source,vertex_count=len(verts),face_count=len(tris),
            coordinate_order='original [D,H,W] as XYZ, no permutation',grid_size=512,aabb=[[-.5]*3,[.5]*3],
            bbox=[verts.min(axis=0).tolist(),verts.max(axis=0).tolist()],connectivity=stats,
            repeated_index_triangles=repeated,zero_area_triangles=zero_area,duplicate_triangles=duplicates,unused_vertices_retained=True,
            extraction_seconds=extraction_seconds,mesh_repaired=False,source_unchanged=True,
            artifacts={n:{'path':str(OUTPUT/n),'bytes':(OUTPUT/n).stat().st_size,'sha256':digest(OUTPUT/n)} for n in ('turret-full.ply','vertices.npy','faces.npy')})
        save(OUTPUT/'provenance.json',record)
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    record['elapsed_seconds']=time.monotonic()-began
    save(E/'extraction-worker.json',record)
    print(record['classification'],flush=True)
    return 0 if record['classification']=='PASS' else 1


def supervise():
    from amd_forward_owned_job import OwnedJob,ExtendedLimits
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    if m.available<6*GIB or shutil.disk_usage(OUTPUT.parent).free<2*GIB:raise RuntimeError('6GiB RAM/2GiB disk admission refused')
    if OUTPUT.exists() or (E/'extraction-attempt.marker').exists():raise RuntimeError('Exclusive extraction already reserved')
    # Reuse established Windows ownership, adding only a hard CPU process cap.
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2000|0x100;limits.ProcessMemory=2*GIB
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
    save(E/'extraction-attempt.marker',{'CPU_only':True,'source_sha256':SOURCE_SHA,'available_ram_bytes':m.available,
        'disk_free_bytes':shutil.disk_usage(OUTPUT.parent).free,'process_memory_cap_bytes':2*GIB,'deadline_seconds':180,
        'source_code_sha256':digest(ROOT/'scripts/compatibility/amd_flexible_dual_grid_cpu.py'),'worker_sha256':digest(__file__)})
    record={'CPU_only':True};child=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        with (E/'extraction.stdout.txt').open('xb') as out,(E/'extraction.stderr.txt').open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child)
            child.stdin.write(b'GO\n');child.stdin.close()
            child.wait(timeout=180);record['worker_returncode']=child.returncode
            measured=ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(measured),ctypes.sizeof(measured),None):raise ctypes.WinError()
            record['peak_process_committed_bytes']=measured.PeakProcessMemory
            record['peak_job_committed_bytes']=measured.PeakJobMemory
    except BaseException as exc:record.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        save(E/'extraction-host.json',record)
    return 0 if record.get('worker_returncode')==0 and not record['job_cleanup']['forced_job_cleanup'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    raise SystemExit(worker() if args.worker else supervise())
