"""One contained CPU invocation of the existing mesh CLI; no numerical changes."""
import argparse,ctypes,hashlib,json,os,runpy,shutil,stat,subprocess,sys,time,traceback
from pathlib import Path

E=Path(__file__).resolve().parent;ROOT=E.parents[2]
PRIVATE=ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579'
SOURCE=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors'
SOURCE_SHA='d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861'
OUTPUT=PRIVATE/'mesh';CLI=ROOT/'scripts/compatibility/amd_generation_mesh.py'
GIB=1024**3;PROCESS_CAP=6*GIB;RAM_FLOOR=12*GIB;DISK_FLOOR=2*GIB;DEADLINE=180
sys.path.insert(0,str(ROOT/'scripts/compatibility'))

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,r):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def plain(p,exists=True):
    p=Path(p).absolute()
    for q in (p,*p.parents):
        if not q.exists():
            if q==p and not exists:continue
            raise ValueError('Missing path ancestor: '+str(q))
        s=q.lstat()
        if q.is_symlink() or getattr(s,'st_file_attributes',0)&0x400:raise ValueError('Reparse/symlink refused: '+str(q))
    if p.exists() and p.is_file() and p.stat().st_nlink!=1:raise ValueError('Single-link file required')
    return p
def extraction_argv():
    return [str(CLI),'--source',str(SOURCE),'--sha256',SOURCE_SHA,'--output',str(OUTPUT),
            '--run-contract',str(SOURCE.parent/'run-contract.json')]
def admission(ram,disk):
    return dict(available_ram_bytes=ram,disk_free_bytes=disk,required_ram_bytes=RAM_FLOOR,
                required_disk_bytes=DISK_FLOOR,admitted=ram>=RAM_FLOOR and disk>=DISK_FLOOR)
def resource():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return admission(m.available,shutil.disk_usage(PRIVATE).free)
def process_memory(child):
    class C(ctypes.Structure):
        _fields_=[('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)]+[(n,ctypes.c_size_t) for n in
            ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    value=C();value.cb=ctypes.sizeof(value)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(C),ctypes.c_uint32]
    api.GetProcessMemoryInfo.restype=ctypes.c_int
    if not api.GetProcessMemoryInfo(int(child._handle),ctypes.byref(value),ctypes.sizeof(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(peak_working_set_bytes=value.peak_working_set,working_set_bytes=value.working_set,
                peak_pagefile_bytes=value.peak_pagefile,private_bytes=value.private)
def worker(ev,fixture=False,verify=False):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact GO plus EOF required')
    forbidden=[]
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy','nvdiffrast','cumesh'}:
            forbidden.append(str(args[0]));raise RuntimeError('CPU-only import denied: '+str(args[0]))
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):
            forbidden.append(event);raise RuntimeError('CPU-only child/network denied: '+event)
    sys.addaudithook(audit)
    if fixture:
        denied=[]
        for label,call in [('torch',lambda:__import__('torch')),
                           ('subprocess',lambda:subprocess.Popen([sys.executable,'-c','pass']))]:
            try:call()
            except RuntimeError:denied.append(label)
            else:raise AssertionError('CPU guard bypass: '+label)
        import socket
        with socket.socket() as sock:
            try:sock.connect(('127.0.0.1',9))
            except RuntimeError:denied.append('network')
            else:raise AssertionError('CPU guard network bypass')
        save(ev/'fixture-result.json',dict(CPU_only=True,denied=denied));time.sleep(.25);return 0
    target=E/'verify_mesh.py' if verify else CLI
    sys.argv=[str(target)] if verify else extraction_argv()
    try:
        runpy.run_path(str(target),run_name='__main__')
        if forbidden:raise RuntimeError('Forbidden operation was attempted')
        return 0
    finally:
        save(ev/'worker-policy.json',dict(CPU_only=True,forbidden=forbidden,target=str(target)))

def supervise(ev,fixture=False,verify=False):
    from amd_forward_owned_job import OwnedJob,ExtendedLimits
    ev=Path(ev);initial=resource()
    if not initial['admitted']:raise RuntimeError('CPU pre-marker admission: '+json.dumps(initial))
    if not fixture:
        plain(PRIVATE);plain(SOURCE)
        if SOURCE.stat().st_size!=104148612 or digest(SOURCE)!=SOURCE_SHA:raise ValueError('Neural source identity')
        if not verify:
            plain(OUTPUT,exists=False)
            if OUTPUT.exists():raise FileExistsError('No extraction retry: '+str(OUTPUT))
        ev.mkdir(exist_ok=False)
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2000|0x100;limits.ProcessMemory=PROCESS_CAP
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
    save(ev/'attempt-marker.json',dict(CPU_only=True,fixture=fixture,verify=verify,source_sha256=SOURCE_SHA,
        process_memory_cap_bytes=PROCESS_CAP,deadline_seconds=DEADLINE,initial_resource=initial,
        mesh_host_sha256=digest(__file__),CLI_sha256=digest(CLI),converter_sha256=digest(ROOT/'scripts/compatibility/amd_flexible_dual_grid_cpu.py')))
    record=dict(CPU_only=True,fixture=fixture,verify=verify,initial_resource=initial,resource_samples=[],timed_out=False)
    child=None;started=time.monotonic();last_resource=0
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TEMP=str(PRIVATE),TMP=str(PRIVATE))
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(ev)]
    if fixture:command+=['--fixture']
    if verify:command+=['--verify']
    record['command']=command
    try:
        with (ev/'stdout.txt').open('xb') as out,(ev/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child);record['job_attached_before_GO']=True
            child.stdin.write(b'GO\n');child.stdin.close()
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=DEADLINE:record['timed_out']=True;raise TimeoutError('180-second CPU deadline')
                if elapsed-last_resource>=1 or not record['resource_samples']:
                    row=resource();row.update(elapsed_seconds=elapsed)
                    record['resource_samples'].append(row);last_resource=elapsed
                    if not row['admitted']:raise RuntimeError('Live CPU admission failed: '+json.dumps(row))
                time.sleep(.05)
            record['worker_returncode']=child.returncode
            record['process_memory']=process_memory(child)
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5:time.sleep(.025)
            record['natural_settlement_seconds']=time.monotonic()-grace
    except BaseException as exc:record.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        try:
            peak=ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
            record.update(peak_process_committed_bytes=peak.PeakProcessMemory,peak_job_committed_bytes=peak.PeakJobMemory)
        except BaseException as exc:record['measurement_error']=str(exc)
        if child is not None and child.stdin is not None and not child.stdin.closed:child.stdin.close()
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        record['elapsed_seconds']=time.monotonic()-started
        record['classification']='PASS' if (record.get('worker_returncode')==0 and 'error' not in record
            and 'measurement_error' not in record and not record['job_cleanup']['forced_job_cleanup']
            and record['job_cleanup']['owned_tree_exit_verified']) else 'PARTIAL'
        save(ev/'host-result.json',record)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--fixture',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.worker,args.fixture,args.verify))
    result=supervise(E/('verification-run' if args.verify else 'extraction-run'),verify=args.verify)
    print(json.dumps(result),flush=True);raise SystemExit(0 if result['classification']=='PASS' else 1)
