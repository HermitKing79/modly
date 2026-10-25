"""Owned CPU-only inspection of the first successful standalone zombie decode."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
DECODE=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-standalone-d77c7ef/decode'
SOURCE=DECODE/'run/guest/shape-decoded.safetensors'
SOURCE_SHA='e589fec9f5ea58ec7bdb07c00c3c40911b6191d08055611fac17ddee65aa7918'
HARNESS='2c6990b9224e9e4a0c5cd383dadcc1a3808c3e3a'
ROWS=326232
CHILDREN=[4277,18312,77797,326232]
CONTRACT_SHA='13a140490d933d0c9f21af701af36caf54384b0fe0e5dc9c4714d0328dea6343'
CHECKPOINT_SHA='e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581'
SOURCE_IMAGE=ROOT.parent/'artifacts/amd-preview-dinov3-7d11b36/cpu-input/source-image.png'
SOURCE_IMAGE_SHA='db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e'
GIB=1024**3
CAP=6*GIB
DEADLINE=180
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def plain(path):
    path=Path(path).absolute()
    if path!=path.resolve() or str(path).startswith('\\\\'):raise ValueError('Canonical local path required')
    for parent in (path,*path.parents):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):raise ValueError('Redirected path denied')
    if path.is_file() and path.stat().st_nlink!=1:raise ValueError('Single-link file required')
    return path


def read(path):
    path=plain(path)
    if path.stat().st_size>8*1024**2:raise ValueError('Bounded metadata required')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def receipts():
    return read(DECODE/'final-result.json'),read(DECODE/'run/results.json'),read(SOURCE.parent/'results.json'),read(SOURCE.parent/'provenance.json')


def validate_receipts(final,host,guest,sidecar):
    check=(final['classification']=='PASS' and final['supervisor']=='QUALIFIED' and final['copy_error'] is None
        and final['harness_commit']==HARNESS and final['run_contract_sha256']==CONTRACT_SHA
        and final['independent_verification']['ten_tensors_verified'] is True
        and final['independent_verification']['rows']==ROWS and final['independent_verification']['sha256']==SOURCE_SHA
        and host['classification']=='QUALIFIED' and host['worker_returncode']==0 and host['observer_healthy'] is True
        and host['minimum_host_ram_bytes']>=12*GIB and host['elapsed_seconds']<480
        and host['windows_job_cleanup']['owned_tree_exit_verified'] is True
        and host['windows_job_cleanup']['forced_job_cleanup'] is False and host['distro_termination']['returncode']==0
        and guest['classification']=='QUALIFIED' and guest['forward_calls']==1 and guest['convolutions_completed']==40
        and guest['memory_release']==dict(allocated=0,reserved=0)
        and sidecar['output']==guest['output'] and sidecar['provenance']['inputs']==guest['saved_inputs']
        and sidecar['output']['rows']==ROWS and sidecar['output']['sha256']==SOURCE_SHA
        and sidecar['provenance']['checkpoint_sha256']==CHECKPOINT_SHA
        and final['independent_verification']['tensor_payload_sha256']==guest['output']['tensor_payload_sha256'])
    if not check:raise ValueError('Complete successful decode, provenance and clean owned settlement required')


def authenticate():
    import numpy as np
    import amd_generation_mesh as mesh
    final,host,guest,sidecar=receipts();validate_receipts(final,host,guest,sidecar)
    for name,row in final['allowlisted_copy'].items():
        if name not in ('run-contract.json','phases.jsonl','results.json','provenance.json','shape-decoded.safetensors'):
            raise ValueError('Unexpected copied evidence')
        p=plain(SOURCE.parent/name)
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Copied evidence hash differs: '+name)
    if set(final['allowlisted_copy'])!={'run-contract.json','phases.jsonl','results.json','provenance.json','shape-decoded.safetensors'}:
        raise ValueError('Incomplete copied evidence closure')
    api=mesh._contract_api();contract=api.load(DECODE/'run-contract.json')
    if contract.sha256!=CONTRACT_SHA or api.load(SOURCE.parent/'run-contract.json').document()!=contract.document():
        raise ValueError('Original host/guest contract differs')
    if sidecar['run_contract']!=contract.document() or final['independent_verification']['source_job_sha256']!=contract.source_job_sha256:
        raise ValueError('Output contract/source job differs')
    api.artifact_binding(sidecar['provenance'],contract)
    if digest(SOURCE_IMAGE)!=SOURCE_IMAGE_SHA:raise ValueError('Original source image changed')
    arrays,children=mesh.read_neural(plain(SOURCE),SOURCE_SHA,contract=contract)
    try:
        hashes={name:hashlib.sha256(memoryview(array)).hexdigest() for name,array in arrays.items()}
        if len(arrays)!=10 or children!=CHILDREN or hashes!=sidecar['output']['tensor_payload_sha256']:
            raise ValueError('Ten-tensor bytes or learned linkage differ')
        for name,guest_path in contract.input_paths.items():
            if not str(guest_path).startswith('/mnt/c/'):raise ValueError('Original Windows input drive required')
            path=plain(Path('C:/'+str(guest_path)[7:]));path.relative_to(ROOT)
            original=api.read_npy(path,contract.input_sha256[name],(1000,4 if name=='coords.npy' else 32),'int32' if name=='coords.npy' else 'float32')
            if name=='coords.npy':
                if not np.array_equal(arrays['subdivision_0_coordinates'],original):raise ValueError('Original coordinates/order differ')
                if digest(path.parents[1]/'results.json')!=contract.source_job_sha256:raise ValueError('Original shape job changed')
        with SOURCE.open('rb') as stream:
            import struct
            length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
        if json.loads(header['__metadata__']['provenance'])!=sidecar['provenance'] or json.loads(header['__metadata__']['summary'])['subdivisions']!=guest['output']['subdivisions']:
            raise ValueError('Raw tensor metadata differs from complete sidecar')
    finally:
        for array in arrays.values():array._mmap.close()
    return contract,dict(classification='PASS_TEN_TENSOR_PROVENANCE',source_sha256=SOURCE_SHA,
        bytes=SOURCE.stat().st_size,rows=ROWS,learned_subdivisions=children,tensor_payload_sha256=hashes,
        run_contract_sha256=contract.sha256,source_job_sha256=contract.source_job_sha256,
        checkpoint_sha256=CHECKPOINT_SHA,source_image_sha256=SOURCE_IMAGE_SHA,
        complete_sidecar_and_header_agree=True,host_minimum_RAM_bytes=host['minimum_host_ram_bytes'],model_or_GPU_imported=False)


def components(np,faces,n):
    parents=list(range(n));sizes=[1]*n
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    for a,b,c in faces.tolist():
        for x,y in ((a,b),(a,c)):
            x,y=find(x),find(y)
            if x!=y:
                if sizes[x]<sizes[y]:x,y=y,x
                parents[y]=x;sizes[x]+=sizes[y]
    used=np.unique(faces);roots=np.fromiter((find(int(i)) for i in used),dtype=np.int32,count=len(used))
    keys,counts=np.unique(roots,return_counts=True)
    face_roots=np.fromiter((find(int(i)) for i in faces[:,0]),dtype=np.int32,count=len(faces))
    fk,fc=np.unique(face_roots,return_counts=True);order=np.argsort(fc)[::-1]
    return dict(components_measured=True,surface_components=len(keys),referenced_vertices=len(used),unused_vertices=n-len(used),
        largest_component_vertices=int(counts.max()),largest_component_triangles=int(fc.max()),
        largest_ten_component_triangle_counts=fc[order[:10]].tolist())


def inspect_mesh():
    import numpy as np
    import amd_generation_mesh as mesh
    contract,authentication=authenticate();save(BASE/'authentication-inspect.json',authentication)
    checker=load('_original_mesh_witnesses',ROOT/'docs/compatibility/amd-brick-cpu-mesh-evidence/verify_mesh.py')
    opened=[];began=time.monotonic()
    def need(ok,why):
        if time.monotonic()-began>175:raise TimeoutError('Existing 175-second analysis worker bound')
        if not ok:raise ValueError(why)
    try:
        arrays,children=mesh.read_neural(SOURCE,SOURCE_SHA,contract=contract);opened.extend(arrays.values())
        output=BASE/'mesh';metadata=read(output/'mesh.json')
        need(metadata['input_sha256']==SOURCE_SHA and metadata['vertices']==ROWS and metadata['learned_subdivisions']==CHILDREN,'Mesh binding')
        for name,row in metadata['artifacts'].items():
            p=plain(output/name);need(digest(p)==row['sha256'] and p.stat().st_size==row['bytes'],'Mesh artifact hash/length')
        v=np.load(output/'vertices.npy',mmap_mode='r',allow_pickle=False);t=np.load(output/'faces.npy',mmap_mode='r',allow_pickle=False)
        opened.extend((v,t));pv,pf,header=checker.ply_arrays(np,output/'mesh.ply',ROWS,len(t),opened,need)
        result=checker.geometry(np,arrays['coordinates'][:,1:],arrays['features'],v,t,need,pv,pf)
        result.update(components(np,t,len(v)))
        result.update(classification='PASS_GEOMETRY_READBACK',vertices=len(v),triangles=len(t),
            exact_duplicate_vertex_positions=len(v)-len(np.unique(v,axis=0)),
            bounds=dict(min=v.min(axis=0).tolist(),max=v.max(axis=0).tolist()),
            source_sha256=SOURCE_SHA,PLY_sha256=digest(output/'mesh.ply'),run_contract_sha256=contract.sha256,
            self_intersections_measured=False,geometry_repaired=False,geometry_smoothed=False)
        need(result['bounds']==metadata['bounds'],'Independent bounds')
        result['elapsed_seconds']=time.monotonic()-began
        save(BASE/'mesh-analysis.json',result);print(json.dumps(result),flush=True)
    finally:
        for array in opened:array._mmap.close()


def admission(ram,disk):
    return dict(available_RAM_bytes=ram,free_disk_bytes=disk,required_RAM_bytes=12*GIB,required_disk_bytes=2*GIB,
        admitted=ram>=12*GIB and disk>=2*GIB)


def review_previews():
    from PIL import Image,ImageDraw
    folder=BASE/'previews';metadata=read(folder/'render-metadata.json')
    if metadata['classification']!='PASS_CPU_PREVIEWS' or metadata['device']!='CPU' or not metadata['imported_vertex_and_face_payloads_exact']:
        raise ValueError('Complete CPU geometry preview receipt required')
    if metadata['source_PLY_sha256']!=digest(BASE/'mesh/mesh.ply'):raise ValueError('Rendered mesh binding differs')
    if len(metadata['views'])!=8:raise ValueError('Eight distinct views required')
    sheet=Image.new('RGB',(1536,824),'#30353d');draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(metadata['views']):
        path=folder/row['name']
        if digest(path)!=row['sha256']:raise ValueError('Preview changed')
        with Image.open(path) as image:
            if image.size!=(768,768):raise ValueError('Preview dimensions')
            thumb=image.resize((384,384),Image.Resampling.LANCZOS)
            sheet.paste(thumb,((index%4)*384,(index//4)*412))
        draw.text(((index%4)*384+10,(index//4)*412+388),f"Azimuth {row['azimuth_degrees']} degrees",fill='white')
    sheet.save(folder/'eight-view-contact-sheet.png')
    if digest(folder/'silhouette.png')!=metadata['silhouette_sha256']:raise ValueError('Silhouette changed')
    with Image.open(folder/'view-000.png') as image:
        image.crop((310,40,460,230)).resize((450,570),Image.Resampling.NEAREST).save(folder/'front-head-detail.png')
    # Preserve the first analysis receipt, including the inherited checker's
    # default flag. Its computed component counts are valid; append clarification.
    save(BASE/'mesh-analysis-clarification.json',dict(original_analysis_sha256=digest(BASE/'mesh-analysis.json'),
        components_measured=True,reason='Inherited witness default flag was not overwritten; component calculation/counts ran and were recorded',
        geometry_or_analysis_reexecuted=False,original_evidence_modified=False))
    save(BASE/'preview-review.json',dict(classification='PASS_EIGHT_VIEWS_AND_SILHOUETTE',
        source_PLY_sha256=metadata['source_PLY_sha256'],original_preview_hashes_verified=True,
        composite_sha256=digest(folder/'eight-view-contact-sheet.png'),head_detail_sha256=digest(folder/'front-head-detail.png'),
        head_detail_is_nearest_neighbour_crop_of_original_render=True,geometry_unchanged=True))


def resource():
    # Reuse the qualified native reader, without its diagnostic entrypoint.
    module=load('_cpu_mesh_memory',Path(__file__).with_name('amd_zombie_ram_diagnostic.py'))
    return admission(module.windows_memory()['available_bytes'],shutil.disk_usage(BASE).free)


def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO plus EOF required')
    forbidden=[]
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0].lower() in {'torch','triton','trellis2','flex_gemm','o_voxel','cupy','cumesh','nvdiffrast'}:
            forbidden.append(str(args[0]));raise RuntimeError('Model/GPU import denied')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):
            forbidden.append(event);raise RuntimeError('Child/network denied')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and (any(c in str(args[1]) for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            plain(os.fsdecode(args[0])).relative_to(BASE)
        if event=='os.mkdir':plain(args[0]).relative_to(BASE)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):raise RuntimeError('Destructive mutation denied')
    sys.addaudithook(audit)
    sys.path[:0]=[str(ROOT/'scripts/compatibility'),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
    if stage=='fixture':
        import socket
        for call in (lambda:__import__('torch'),lambda:subprocess.Popen(['wsl.exe']),lambda:socket.create_connection(('127.0.0.1',9))):
            try:call()
            except RuntimeError:pass
            else:raise AssertionError('CPU policy bypass')
        save(BASE/'fixture.json',dict(classification='PASS',blocked=forbidden));return
    if stage=='extract':
        contract,result=authenticate();save(BASE/'authentication-extract.json',result)
        import amd_generation_mesh as mesh
        print(json.dumps(mesh.extract(SOURCE,SOURCE_SHA,BASE/'mesh',contract=contract)),flush=True)
    elif stage=='inspect':inspect_mesh()
    elif stage=='review':review_previews()
    else:raise ValueError('Unknown CPU stage')
    if forbidden:raise RuntimeError('Forbidden operation attempted')


def supervise(stage):
    sys.path.insert(0,str(ROOT/'scripts/compatibility'))
    from amd_forward_owned_job import OwnedJob,ExtendedLimits
    plain(BASE);initial=resource()
    if not initial['admitted']:raise RuntimeError('CPU admission failed: '+json.dumps(initial))
    ev=BASE/(stage+'-run');ev.mkdir(exist_ok=False)
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=CAP
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
    record=dict(stage=stage,initial_resource=initial,process_cap_bytes=CAP,deadline_seconds=DEADLINE,samples=[])
    child=None;began=time.monotonic()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TEMP=str(BASE),TMP=str(BASE))
    if stage=='render':
        command=[str(BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1',
            '--python',str(Path(__file__).with_name('amd_zombie_mesh_render.py'))]
    else:
        # Direct base interpreter avoids the Windows venv redirector PID trap.
        command=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',stage]
    try:
        with (ev/'stdout.txt').open('xb') as out,(ev/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child);record['owned_job_attached']=True
            child.stdin.write(b'GO\n')
            child.stdin.close()
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=DEADLINE:raise TimeoutError('Original 180-second CPU stage deadline')
                row=resource();row['elapsed_seconds']=elapsed;record['samples'].append(row)
                if not row['admitted']:raise RuntimeError('Live CPU resource floor failed')
                time.sleep(.2)
            record['worker_returncode']=child.returncode
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5:time.sleep(.025)
    except BaseException as exc:record.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        peak=ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):
            record['peak_process_committed_bytes']=peak.PeakProcessMemory;record['peak_job_committed_bytes']=peak.PeakJobMemory
        else:record['measurement_error']='Job peak query failed'
        cleanup_errors=[]
        try:
            if child is not None and not child.stdin.closed:child.stdin.close()
        except BaseException as exc:cleanup_errors.append('stdin: '+str(exc))
        # An attachment failure leaves the process outside the Job. Never rely
        # on an empty Job to settle that exact owned Popen handle.
        if child is not None and not record.get('owned_job_attached'):
            try:
                if child.poll() is None:
                    child.terminate();record['unattached_owned_child_terminated']=True
            except BaseException as exc:cleanup_errors.append('owned terminate: '+str(exc))
        try:record['job_cleanup']=job.finish()
        except BaseException as exc:cleanup_errors.append('Job: '+str(exc))
        if child is not None:
            try:child.wait(timeout=5);record['worker_exit_verified']=True
            except BaseException as exc:
                cleanup_errors.append('owned wait: '+str(exc))
                try:
                    if child.poll() is None:child.kill()
                    child.wait(timeout=5);record['worker_exit_verified']=True
                except BaseException as final:cleanup_errors.append('owned kill/reap: '+str(final))
        record['cleanup_errors']=cleanup_errors
        record['elapsed_seconds']=time.monotonic()-began
        cleanup=record.get('job_cleanup',{})
        record['classification']='PASS' if (record.get('worker_returncode')==0 and record.get('worker_exit_verified') is True
            and not record.get('error') and not record.get('measurement_error') and not cleanup_errors
            and cleanup.get('owned_tree_exit_verified') is True and cleanup.get('forced_job_cleanup') is False) else 'FAIL'
        save(ev/'host-result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    return 0 if record['classification']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('fixture','extract','inspect','render','review'))
    parser.add_argument('--worker',choices=('fixture','extract','inspect','review'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:raise SystemExit(supervise(args.stage))
    else:print('DRY_RUN: explicit CPU stage required; no model, WSL or generation entrypoint')
