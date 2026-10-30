"""Isolated, bounded CPU CGAL proof. Default inert; no neural/WSL route."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'artifacts/amd-cgal-global-proof-20261030-v1'
ENV = BASE / 'isolated-native'
CGAL = ENV / 'CGAL-6.2.1'
SCIP = ENV / 'scipoptsuite-10.1.0-win-x64'
CMAKE = ENV / 'cmake-3.31.10-windows-x86_64/bin/cmake.exe'
BUILD = BASE / 'build-shared'
CPU_SITE = BASE / 'cpu-python-qualified/Lib/site-packages'
COMPILER_HELPER = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/vctip.exe')
COMPILER_HELPER_SHA = 'e175c981ca866689bf8f8d9bd5ba1e71e2eb1c9d389de555744b0cb439250996'
GIB = 1024**3
CAP = 6*GIB
DEADLINE = 180
STAGES = ('authenticate','verify-sample','provision-cpu-qualified','configure-eigen-final','configure-shared','build-runtime','build-planes','sample-official','prepare-ai','extract-planes','assess-planes')


def save(path, data):
    with path.open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
            or any(s.endswith((' ', '.')) for s in p.parts)
            or str(p) != name.rstrip('/')):
        raise ValueError('Unsafe archive path')
    return Path(*p.parts)


def admission(ram, disk):
    return ram >= 12*GIB and disk >= 2*GIB


def selected_boost(path):
    return path.parts[:2] == ('boost_1_87_0', 'boost') or path == Path('boost_1_87_0/LICENSE_1_0.txt')


def spawn_allowed(stage, event_args):
    if stage not in ('configure-eigen-final','configure-shared','build-runtime','build-planes','sample-official','extract-planes'):
        return False
    expected=command(stage)
    cwd=BASE/'official-sample' if stage=='sample-official' else BASE
    executable,args,directory,_env=event_args
    exact=args==expected or args==tuple(expected) or args==subprocess.list2cmdline(expected)
    return executable==expected[0] and exact and directory==str(cwd)


def worker(stage):
    if sys.stdin.buffer.read(4) != b'GO\n': raise RuntimeError('Exact owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    ordinary = audit_policy(BASE)
    def guard(event, args):
        # Native build commands are fixed below; their entire tree is Job-owned.
        if event == 'subprocess.Popen':
            if not spawn_allowed(stage, args): raise RuntimeError('Exact CPU command/cwd required')
            return
        ordinary(event, args)
    sys.addaudithook(guard)
    if stage == 'authenticate':
        import amd_ai_hard_surface_assessment as old
        old.BASE = BASE
        sys.path.insert(0, 'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        old.authenticate(); return
    if stage == 'verify-sample':
        from amd_cgal_official_check import verify
        save(BASE/'official-sample-verification.json',verify(BASE));return
    if stage == 'provision-cpu-qualified':
        import venv,base64,csv,io
        row=json.loads((BASE/'numpy-provenance.json').read_text());archive=BASE/'downloads'/row['name']
        if sha(archive)!=row['sha256']:raise ValueError('NumPy archive identity')
        root=CPU_SITE.parents[1]
        if root.exists():raise ValueError('CPU environment consumed')
        venv.EnvBuilder(with_pip=False,symlinks=False).create(root)
        site=root/'Lib/site-packages'
        with zipfile.ZipFile(archive) as z:
            record=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
            allrows=list(csv.reader(io.StringIO(z.read(record).decode())))
            rows={a:(b,c) for a,b,c in allrows}
            if len(rows)!=len(allrows) or set(rows)!={e.filename for e in z.infolist() if not e.is_dir()}:raise ValueError('NumPy wheel closure')
            for entry in z.infolist():
                if entry.is_dir():continue
                if stat.S_ISLNK(entry.external_attr >> 16):raise ValueError('NumPy wheel link denied')
                name=entry.filename;target=site/safe_member(name);data=z.read(entry)
                if name!=record:
                    h,n=rows[name];actual='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                    if h!=actual or n!=str(len(data)):raise ValueError('NumPy payload RECORD differs')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as out:out.write(data)
        sys.path.insert(0,str(site));import numpy as np
        if np.__version__!='2.3.5' or int(np.arange(4)@np.arange(4))!=14:raise ValueError('Isolated CPU NumPy smoke check')
        save(BASE/'cpu-python-install.json',dict(classification='PASS',numpy=np.__version__,site=str(site),archive_sha256=row['sha256'],existing_site_packages_shared=False))
        return
    if stage in ('prepare-ai','assess-planes'):
        sys.path.insert(0,str(CPU_SITE))
        import amd_cgal_ai_input as ai
        (ai.prepare if stage=='prepare-ai' else ai.assess)();return
    if not json.loads((BASE/'private-use-licence-review.json').read_text())['native_private_use_qualified']:
        raise RuntimeError('Dependency licence gate')
    cwd = BASE/'official-sample' if stage == 'sample-official' else BASE
    if stage == 'sample-official': cwd.mkdir(exist_ok=False)
    if stage == 'extract-planes':
        if json.loads((BASE/'ai-input/receipt.json').read_text())['classification']!='PASS_SOURCE_ONLY_TRAIN_INPUT':raise ValueError('Qualified training input required')
        (BASE/'ai-planes').mkdir(exist_ok=False)
    env = dict(os.environ)
    env['PATH'] = str(SCIP/'bin') + os.pathsep + os.environ.get('PATH', '')
    result = subprocess.run(command(stage), executable=command(stage)[0], cwd=str(cwd), env=env, check=False, shell=False)
    if result.returncode: raise RuntimeError('Native command exit '+str(result.returncode))


def command(stage):
    return [x.replace('\\','/') if x.startswith('-D') else x for x in _command(stage)]


def _command(stage):
    if stage == 'configure-eigen-final':
        return [str(CMAKE),'-S',str(ENV/'eigen-3.4.0'),'-B',str(BASE/'eigen-build-final'),
                '-G','Visual Studio 17 2022','-A','x64','-DBUILD_TESTING=OFF',
                '-DEIGEN_BUILD_DOC=OFF','-DEIGEN_BUILD_PKGCONFIG=OFF',
                '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON','-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF',
                '-DCMAKE_INSTALL_PREFIX='+str(BASE/'eigen-install')]
    if stage == 'configure-shared':
        return [str(CMAKE), '-S', str(BASE/'project-shared'), '-B', str(BUILD),
                '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143',
                '-DCGAL_DIR='+str(CGAL), '-DBOOST_ROOT='+str(ENV/'boost_1_87_0'),
                '-DBoost_NO_SYSTEM_PATHS=ON', '-DEIGEN3_INCLUDE_DIR='+str(ENV/'eigen-3.4.0'),
                '-DBoost_DIR='+str(BASE/'boost-header-config'),
                '-DEigen3_DIR='+str(BASE/'eigen-build-final'),
                '-DSCIP_DIR='+str(SCIP/'lib/cmake/scip'), '-DCGAL_DISABLE_GMP=ON',
                '-DMODLY_SCIP_ROOT='+str(SCIP),
                '-DGMP_DIR='+str(ENV/'auxiliary/gmp'), '-DMPFR_DIR='+str(ENV/'auxiliary/gmp'),
                '-DGMP_INCLUDE_DIR='+str(ENV/'auxiliary/gmp/include'),
                '-DGMP_INCLUDE_DIRS='+str(ENV/'auxiliary/gmp/include'),
                '-DGMP_LIBRARIES='+str(ENV/'auxiliary/gmp/lib/gmp.lib'),
                '-DMPFR_INCLUDE_DIR='+str(ENV/'auxiliary/gmp/include'),
                '-DMPFR_LIBRARIES='+str(ENV/'auxiliary/gmp/lib/mpfr.lib'),
                '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']
    if stage == 'build-runtime':
        return [str(CMAKE), '--build', str(BUILD), '--config', 'Release',
                '--target', 'official_sample', '--parallel', '1']
    if stage=='build-planes':
        return [str(CMAKE),'--build',str(BUILD),'--config','Release','--target','ai_planes','--parallel','1']
    if stage == 'sample-official':
        return [str(BUILD/'Release/official_sample.exe'), str(CGAL/'data/points_3/ball.ply')]
    if stage=='extract-planes':
        return [str(BUILD/'Release/ai_planes.exe'),str(BASE/'ai-input/train-native.ply'),'100',str(BASE/'ai-planes/assigned.ply'),str(BASE/'ai-planes/planes.csv')]
    raise ValueError('No native command for this stage')


def compiler_cleanup_stage(stage):
    return stage in ('configure-eigen-final','configure-shared','build-runtime','build-planes')


def compiler_outputs_complete(stage):
    if stage == 'configure-eigen-final':
        return all((BASE/'eigen-build-final'/p).is_file() for p in ('Eigen3Config.cmake','Eigen3Targets.cmake'))
    if stage == 'configure-shared':
        p=BUILD/'official_sample.vcxproj'
        return p.is_file() and 'CGAL_USE_SCIP' in p.read_text()
    if stage == 'build-runtime':
        p=BUILD/'Release/official_sample.exe'
        return p.is_file() and p.stat().st_size>100000 and p.open('rb').read(2)==b'MZ'
    if stage=='build-planes':
        p=BUILD/'Release/ai_planes.exe'
        return p.is_file() and p.stat().st_size>100000 and p.open('rb').read(2)==b'MZ'
    return False


def finish_exact_compiler_helper(job, stage, stage_start_filetime, deadline, resource):
    """Only the evidenced optional VC helper; never solver/application cleanup."""
    if not compiler_cleanup_stage(stage) or not compiler_outputs_complete(stage):
        raise RuntimeError('Compiler helper stage/output qualification failed')
    class Pids(ctypes.Structure):
        _fields_=[('assigned',ctypes.c_uint32),('count',ctypes.c_uint32),('ids',ctypes.c_size_t*128)]
    ids=Pids()
    if not job.api.QueryInformationJobObject(job.handle,3,ctypes.byref(ids),ctypes.sizeof(ids),None):raise ctypes.WinError()
    if ids.assigned!=1 or ids.count!=1:raise RuntimeError('Exactly one owned helper required')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
    k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    k.QueryFullProcessImageNameW.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)]
    k.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_uint64)]*4
    k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=k.OpenProcess(0x1000|0x100000|1,False,ids.ids[0])
    if not handle:raise ctypes.WinError()
    try:
        member=ctypes.c_int();buf=ctypes.create_unicode_buffer(32768);n=ctypes.c_uint32(len(buf))
        born=ctypes.c_uint64();end=ctypes.c_uint64();kernel=ctypes.c_uint64();user=ctypes.c_uint64()
        if not k.IsProcessInJob(handle,job.handle,ctypes.byref(member)) or not member.value:raise RuntimeError('Helper membership differs')
        if not k.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(n)):raise ctypes.WinError()
        if not k.GetProcessTimes(handle,ctypes.byref(born),ctypes.byref(end),ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError()
        if born.value<stage_start_filetime or born.value>int((time.time()+11644473600)*10000000):raise RuntimeError('Helper start identity differs')
        pin=json.loads((BASE/'compiler-helper-pin.json').read_text());path=Path(buf.value)
        from amd_zombie_cpu_mesh import plain
        if plain(path)!=plain(COMPILER_HELPER) or plain(Path(pin['path']))!=plain(COMPILER_HELPER) or pin['sha256']!=COMPILER_HELPER_SHA:
            raise RuntimeError('Unexpected owned compiler process/pin')
        s=path.stat()
        if (s.st_size,s.st_ino,s.st_mtime_ns)!=(pin['bytes'],pin['file_id'],pin['mtime_ns']) or sha(path)!=COMPILER_HELPER_SHA:
            raise RuntimeError('Pinned compiler helper differs')
        expected_birth=born.value
        if not resource()['admitted'] or time.monotonic()>=deadline:raise RuntimeError('Cleanup resource/deadline gate')
        if not k.GetProcessTimes(handle,ctypes.byref(born),ctypes.byref(end),ctypes.byref(kernel),ctypes.byref(user)) or born.value!=expected_birth:
            raise RuntimeError('Retained helper identity differs')
        if not k.IsProcessInJob(handle,job.handle,ctypes.byref(member)) or not member.value:raise RuntimeError('Helper escaped owned Job')
        if not k.TerminateProcess(handle,0):raise ctypes.WinError()
        if k.WaitForSingleObject(handle,1000)!=0 or job.active()!=0:raise RuntimeError('Helper/owned Job not empty')
        if not resource()['admitted'] or time.monotonic()>=deadline:raise RuntimeError('Post-helper resource/deadline gate')
        return dict(pid=int(ids.ids[0]),creation_filetime=expected_birth,image=str(path),sha256=pin['sha256'],
                    retained_handle_termination=True,job_membership_reverified=True,helper_exit_verified=True,
                    owned_job_empty=True,natural_helper_exit=False)
    finally:k.CloseHandle(handle)


def supervise(stage):
    from amd_forward_owned_job import OwnedJob, ExtendedLimits
    from amd_zombie_cpu_mesh import plain
    from amd_zombie_ram_diagnostic import windows_memory
    def resource():
        ram=windows_memory()['available_bytes']; disk=shutil.disk_usage(BASE).free
        return dict(available_RAM_bytes=ram, free_disk_bytes=disk, admitted=admission(ram,disk))
    plain(BASE); initial=resource()
    if not initial['admitted']: raise RuntimeError('CPU resource admission refused')
    ev=BASE/(stage+'-run'); ev.mkdir(exist_ok=False)
    job=OwnedJob(); caps=ExtendedLimits(); caps.Basic.Flags=0x2300
    caps.ProcessMemory=CAP; caps.JobMemory=CAP
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(caps),ctypes.sizeof(caps)):
        raise ctypes.WinError()
    record=dict(stage=stage,initial_resource=initial,process_and_job_cap_bytes=CAP,deadline_seconds=DEADLINE,samples=[],entry_sha256=sha(Path(__file__)))
    (ev/'entry-source.py.txt').write_bytes(Path(__file__).read_bytes())
    child=None; began=time.monotonic(); stage_start_filetime=int((time.time()+11644473600)*10000000)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MSBUILDDISABLENODEREUSE='1',VSCMD_SKIP_SENDTELEMETRY='1',TEMP=str(BASE),TMP=str(BASE))
    try:
        with (ev/'stdout.txt').open('xb') as out,(ev/'stderr.txt').open('xb') as err:
            child=subprocess.Popen([sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',stage],stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
            job.attach(child); record['owned_job_attached']=True
            # Limit the owned CPU worker and its descendants to at most 4 cores.
            k=ctypes.WinDLL('kernel32',use_last_error=True)
            k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
            k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
            mask=ctypes.c_size_t(); system=ctypes.c_size_t()
            if not k.GetProcessAffinityMask(int(child._handle),ctypes.byref(mask),ctypes.byref(system)): raise ctypes.WinError()
            chosen=sum(1<<i for i in [i for i in range(64) if mask.value&(1<<i)][:4])
            if not chosen or not k.SetProcessAffinityMask(int(child._handle),chosen): raise ctypes.WinError()
            record['cpu_affinity_mask']=chosen
            child.stdin.write(b'GO\n'); child.stdin.close()
            while child.poll() is None:
                if time.monotonic()-began >= DEADLINE: raise TimeoutError('180-second CPU stage deadline')
                r=resource();r['elapsed_seconds']=time.monotonic()-began;record['samples'].append(r)
                if not r['admitted']: raise RuntimeError('Live CPU RAM/disk floor')
                time.sleep(.2)
            record['worker_returncode']=child.returncode
            class Pids(ctypes.Structure):
                _fields_=[('assigned',ctypes.c_uint32),('count',ctypes.c_uint32),('ids',ctypes.c_size_t*128)]
            ids=Pids()
            if job.api.QueryInformationJobObject(job.handle,3,ctypes.byref(ids),ctypes.sizeof(ids),None):
                tail=[]
                k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
                k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
                k.QueryFullProcessImageNameW.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)]
                k.CloseHandle.argtypes=[ctypes.c_void_p]
                for pid in ids.ids[:min(ids.count,128)]:
                    h=k.OpenProcess(0x1000,False,pid)
                    if not h:continue
                    try:
                        member=ctypes.c_int();buf=ctypes.create_unicode_buffer(32768);n=ctypes.c_uint32(len(buf))
                        if k.IsProcessInJob(h,job.handle,ctypes.byref(member)) and member.value and k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(n)):
                            tail.append(dict(pid=pid,image=buf.value,job_membership_reverified=True))
                    finally:k.CloseHandle(h)
                record['owned_processes_after_worker_exit']=tail
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5:
                if time.monotonic()-began>=DEADLINE:raise TimeoutError('CPU cleanup deadline')
                r=resource();r['phase']='owned-tail-grace';record['samples'].append(r)
                if not r['admitted']:raise RuntimeError('Owned-tail resource floor')
                time.sleep(.2)
            if job.active() and child.returncode==0 and compiler_cleanup_stage(stage):
                record['compiler_helper_cleanup']=finish_exact_compiler_helper(job,stage,stage_start_filetime,began+DEADLINE,resource)
    except BaseException as exc:
        record.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        peaks=ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peaks),ctypes.sizeof(peaks),None):
            record.update(peak_process_committed_bytes=peaks.PeakProcessMemory,peak_job_committed_bytes=peaks.PeakJobMemory)
        else: record['peak_measurement_error']=True
        errors=[]
        if child is not None and not child.stdin.closed: child.stdin.close()
        if child is not None and not record.get('owned_job_attached') and child.poll() is None: child.terminate()
        try: record['cleanup']=job.finish()
        except BaseException as exc: errors.append(repr(exc))
        if child is not None:
            try: child.wait(timeout=5);record['worker_exit_verified']=True
            except BaseException as exc: errors.append(repr(exc))
        record['cleanup_errors']=errors;record['elapsed_seconds']=time.monotonic()-began
        success=(record.get('worker_returncode')==0 and record.get('worker_exit_verified') and not record.get('error') and not record.get('peak_measurement_error') and not errors and record.get('cleanup',{}).get('owned_tree_exit_verified') and not record['cleanup']['forced_job_cleanup'])
        record['classification']=('PASS_WITH_OWNED_COMPILER_HELPER_CLEANUP' if record.get('compiler_helper_cleanup') else 'PASS') if success else 'FAIL'
        save(ev/'host-result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    return int(record['classification']=='FAIL')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
    if a.worker: worker(a.worker);return 0
    if a.stage:return supervise(a.stage)
    print('INERT: explicit bounded CPU stage required');return 0


if __name__=='__main__': raise SystemExit(main())
