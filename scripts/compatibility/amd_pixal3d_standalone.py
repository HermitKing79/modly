"""Standalone Windows owner. Default does no work; live requires deliberate --run."""
import argparse,ctypes,json,os,re,shutil,subprocess,sys,time,urllib.parse,urllib.request,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common as c

FILES=['scripts/compatibility/'+name for name in ('Invoke-Pixal3DHipProof.ps1','amd_pixal3d_standalone.py',
    'amd_pixal3d_proof_common.py','amd_pixal3d_guest.py','amd_pixal3d_hip_ops.py','test_amd_pixal3d_standalone.py',
    'amd_zombie_d_handoff.py','amd_zombie_ram_diagnostic.py','amd_forward_owned_job.py')]+[
    'docs/compatibility/amd-pixal3d-standalone-hip-contract.json',
    'docs/compatibility/amd-pixal3d-native-single-image-proof-contract.json']

def sequence(invoke):
    for stage in ('provision','operators'):invoke(stage)

def linux_path(path):
    value=c.plain(path).as_posix()
    if not re.fullmatch(r'[CD]:/[A-Za-z0-9_./ ()+\-]+',value):raise ValueError('Canonical C/D path required')
    return '/mnt/'+value[0].lower()+'/'+value[3:]

def wsl_command(stage,case,token,cleanup=False):
    if stage not in ('provision','operators') or not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('Exact stage/token required')
    if not re.fullmatch(r'/mnt/c/[A-Za-z0-9_./ ()+\-]+',case) or '..' in case.split('/'):raise ValueError('Canonical /mnt/c evidence required')
    cfg=c.lock();entry=linux_path(Path(__file__).with_name('amd_pixal3d_guest.py'))
    return [str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe'),'-d',cfg['distro'],'-u',cfg['guest_user'],
        '--exec','/usr/bin/env','MODLY_PIXAL_PROOF_TOKEN='+token,'PYTHONDONTWRITEBYTECODE=1',
        cfg['guest_python'],'-B','-I','-S',entry,'--cleanup' if cleanup else '--controller',stage,'--case',case,'--token',token]

def accept_operators(result,cleanup):
    if (result.get('classification')!='PASS' or set(result.get('operators',{}))!=set(c.OPERATORS)
        or any(row.get('status')!='PASS' for row in result['operators'].values())
        or result.get('memory_release')!=dict(allocated=0,reserved=0)
        or cleanup.get('natural_exit') is not True or cleanup.get('group_empty') is not True):
        raise RuntimeError('Incomplete operator or natural cleanup evidence')

def verify_publication(commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('Exact commit required')
    def git(*args):return subprocess.check_output(['git','-C',str(c.ROOT),*args],text=True,encoding='utf-8',timeout=15).strip()
    if git('rev-parse','HEAD')!=commit:raise ValueError('Current checkout differs from published handoff')
    for name in FILES:
        actual=(c.ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n')
        committed=subprocess.check_output(['git','-C',str(c.ROOT),'show',commit+':'+name],text=True,encoding='utf-8',timeout=15)
        if committed.replace('\r\n','\n')!=actual:raise ValueError('Uncommitted/changed proof source: '+name)
    cfg=c.lock();old=c.ROOT/cfg['approved_contract']
    import hashlib
    if hashlib.sha256(old.read_text(encoding='utf-8').encode()).hexdigest()!=cfg['approved_contract_canonical_sha256']:raise ValueError('Original approved contract changed')
    for ref in cfg['historical_commits']:
        if git('rev-parse',ref)!=ref:raise ValueError('Historical reference unavailable')

def desktop_absent():
    # Read-only process inventory; never close or terminate a desktop application.
    from ctypes import wintypes as w
    class Entry(ctypes.Structure):
        _fields_=[('size',w.DWORD),('usage',w.DWORD),('pid',w.DWORD),('heap',ctypes.c_size_t),('module',w.DWORD),('threads',w.DWORD),('parent',w.DWORD),('priority',w.LONG),('flags',w.DWORD),('exe',w.WCHAR*260)]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];api.CreateToolhelp32Snapshot.restype=w.HANDLE
    api.Process32FirstW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)];api.Process32NextW.argtypes=api.Process32FirstW.argtypes
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.CreateToolhelp32Snapshot(2,0)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError()
    try:
        entry=Entry();entry.size=ctypes.sizeof(entry);found=[];ok=api.Process32FirstW(handle,ctypes.byref(entry))
        if not ok:raise ctypes.WinError()
        while ok:
            if entry.exe.lower() in ('codex.exe','chatgpt.exe'):found.append(entry.exe)
            ok=api.Process32NextW(handle,ctypes.byref(entry))
        if found:raise RuntimeError('Close desktop applications before live proof: '+', '.join(found))
    finally:api.CloseHandle(handle)

def allocated_size(path):
    high=ctypes.c_uint32();api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCompressedFileSizeW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)]
    api.GetCompressedFileSizeW.restype=ctypes.c_uint32;ctypes.set_last_error(0)
    low=api.GetCompressedFileSizeW(str(c.plain(path)),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError()
    return high.value*2**32+low

class Runner:
    def __init__(self,case,token,cfg):
        self.case=case;self.token=token;self.cfg=cfg;self.windows=c.plain(cfg['windows_root']);self.vhd=Path('D:/AI/Modly-WSL/Ubuntu24.04/ext4.vhdx')
        self.memory_log=case/('payload-host-memory.jsonl' if '--payload-child' in sys.argv else 'host-memory.jsonl')
        if not self.vhd.is_file():raise RuntimeError('Qualified distro VHD location unavailable; growth guard cannot be disabled')
        self.vhd_before=allocated_size(self.vhd)
    def memory(self):
        from amd_zombie_d_handoff import windows_memory
        return windows_memory()
    def live(self):
        row=self.memory();D=shutil.disk_usage('D:/').free;output=shutil.disk_usage(self.case).free
        allocated=allocated_size(self.vhd)
        c.journal(self.memory_log,'sample',observer_pid=os.getpid(),**row,D_free_bytes=D,output_free_bytes=output,VHD_allocated_bytes=allocated)
        c.require_resources(row['available_bytes'],D,output)
        if allocated-self.vhd_before>self.cfg['guest_growth_limit_bytes']:raise RuntimeError('Original guest physical growth cap exceeded')
        return row
    def final_gate(self,stage):
        row=self.live();c.journal(self.case/'phases.jsonl','final-22GiB',stage=stage,**row);c.require_ram(row,22*c.GIB)
    def obtain(self,row,destination):
        if destination.exists():raise FileExistsError('No retry/overwrite: '+str(destination))
        cache=row.get('cache');cache=(c.ROOT.parent/cache if cache and not Path(cache).is_absolute() else Path(cache)) if cache else None
        if cache and cache.exists():
            c.verify(cache,row);source=cache.open('rb');origin='verified-cache'
        else:
            parsed=urllib.parse.urlsplit(row['url'])
            if parsed.scheme!='https' or parsed.hostname not in ('files.pythonhosted.org','raw.githubusercontent.com','repo.radeon.com'):raise ValueError('Unapproved download origin')
            source=urllib.request.urlopen(row['url'],timeout=30);origin='download'
            if urllib.parse.urlsplit(source.geturl()).hostname not in ('files.pythonhosted.org','raw.githubusercontent.com','repo.radeon.com'):source.close();raise ValueError('Foreign download redirect')
        began=time.monotonic();total=0
        c.journal(self.case/'phases.jsonl','payload-start',filename=destination.name,origin=origin,url=row['url'])
        with source,destination.open('xb') as target:
            while True:
                self.live()
                if time.monotonic()-began>self.cfg['provision_seconds']:raise TimeoutError('180-second payload deadline; no retry')
                block=source.read(1024**2)
                if not block:break
                total+=len(block)
                if total>row['bytes']:raise RuntimeError('Payload exceeds pinned bytes')
                target.write(block)
        c.verify(destination,row)
        c.journal(self.case/'phases.jsonl','payload-verified',filename=destination.name,bytes=total,sha256=row['sha256'],origin=origin)
    def prepare_payload(self):
        return self.payload_owned()
    def payload_owned(self):
        """Independent live monitoring even while HTTPS reads or hashing block."""
        from amd_forward_owned_job import OwnedJob,ExtendedLimits
        leaf=self.case/'payload-stage';leaf.mkdir(exist_ok=False);job=OwnedJob();limits=ExtendedLimits()
        limits.Basic.Flags=0x2100;limits.ProcessMemory=6*c.GIB
        if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            job.close();raise ctypes.WinError()
        command=[sys.executable,'-B','-I','-S',str(Path(__file__).resolve()),'--payload-child','--case',str(self.case),'--token',self.token]
        receipt=dict(classification='FAIL');child=None;began=time.monotonic()
        try:
            with (leaf/'stdout.txt').open('xb') as out,(leaf/'stderr.txt').open('xb') as err:
                child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=str(self.case))
                job.attach(child);receipt['job_attached']=True
                self.live();child.stdin.write(b'GO\n');child.stdin.close()
                while child.poll() is None:
                    self.live()
                    if time.monotonic()-began>self.cfg['provision_seconds']:raise TimeoutError('180-second payload CPU stage deadline')
                    time.sleep(.5)
                receipt['exit_code']=child.returncode
                if child.returncode:raise RuntimeError('Payload failed; no WSL or retry')
                receipt['classification']='PASS'
        except BaseException as exc:receipt['error']=type(exc).__name__+': '+str(exc)
        finally:
            if child and child.stdin and not child.stdin.closed:child.stdin.close()
            peak=ExtendedLimits()
            if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):receipt['peak_process_commit_bytes']=peak.PeakProcessMemory
            else:receipt['measurement_error']='Windows Job peak query failed'
            try:receipt['cleanup']=job.finish()
            except BaseException as exc:receipt['cleanup_error']=str(exc)
            if child:
                if not receipt.get('job_attached') and child.poll() is None:child.kill()
                try:child.wait(timeout=5)
                except Exception as exc:receipt['reap_error']=str(exc)
            receipt['seconds']=time.monotonic()-began
            if ('error' in receipt or 'cleanup_error' in receipt or 'reap_error' in receipt or 'measurement_error' in receipt
                or receipt.get('cleanup',{}).get('forced_job_cleanup') is not False):receipt['classification']='FAIL'
            c.save(leaf/'result.json',receipt)
        if receipt['classification']!='PASS':raise RuntimeError('Payload stage failed: '+str(receipt.get('error','cleanup failure')))
    def prepare_payload_body(self):
        if self.windows.exists():raise FileExistsError('One-shot Windows root already used; stop for review')
        self.windows.mkdir(exist_ok=False);downloads=self.windows/'downloads';downloads.mkdir()
        # Check Kitchen first; do not provision a guest if this wheel lacks HIP.
        kitchen=next(p for p in self.cfg['packages'] if p['name']=='comfy-kitchen')
        self.obtain(kitchen,downloads/kitchen['filename']);payload=c.hip_payload(downloads/kitchen['filename'])
        c.save(self.case/'HIP-wheel.json',payload)
        for row in self.cfg['packages']:
            if row!=kitchen:self.obtain(row,downloads/row['filename'])
        paths=[downloads/p['filename'] for p in self.cfg['packages']];expansion=c.bounded_expansion(paths)
        if sum(p['bytes'] for p in self.cfg['packages'])>self.cfg['download_limit_bytes']:raise RuntimeError('Original download budget exceeded')
        if shutil.disk_usage('D:/').free<50*c.GIB+expansion:raise RuntimeError('D: cannot retain original floor after isolated install')
        sources=self.windows/'sources';sources.mkdir()
        for row in self.cfg['sources']:
            target=c.inside(sources,row['path']);target.parent.mkdir(parents=True,exist_ok=True);self.obtain(row,target)
        c.save(self.case/'payload.json',dict(windows_root=str(self.windows),guest_downloads=linux_path(downloads),guest_sources=linux_path(sources),
             packages=self.cfg['packages'],sources=self.cfg['sources'],expansion_bytes=expansion,hip_payload=payload))
    def cleanup_guest(self,stage):
        command=wsl_command(stage,linux_path(self.case),self.token,cleanup=True)
        with (self.case/stage/'cleanup-stdout.txt').open('xb') as out,(self.case/stage/'cleanup-stderr.txt').open('xb') as err:
            completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=self.cfg['cleanup_seconds'],creationflags=0x08000000)
        if completed.returncode:raise RuntimeError('Owned guest cleanup verification failed')
        return json.loads((self.case/stage/'cleanup.json').read_text())
    def invoke(self,stage):
        from amd_forward_owned_job import OwnedJob,ExtendedLimits
        leaf=self.case/stage;leaf.mkdir(exist_ok=False);command=wsl_command(stage,linux_path(self.case),self.token)
        job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=6*c.GIB
        if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            job.close();raise ctypes.WinError()
        child=None;receipt=dict(stage=stage,classification='FAIL');began=time.monotonic()
        try:
            with (leaf/'stdout.txt').open('xb') as out,(leaf/'stderr.txt').open('xb') as err:
                child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=str(self.case),creationflags=0x08000000)
                job.attach(child);receipt['job_attached']=True
                # Guest owns a nonce-bound group and waits before all setup/GPU work.
                ready=leaf/'ownership.json'
                owner=None
                while owner is None:
                    self.live()
                    if child.poll() is not None:raise RuntimeError('Guest exited before ready/GO')
                    if time.monotonic()-began>120:raise TimeoutError('Original CPU ready deadline')
                    if ready.exists():
                        try:owner=json.loads(ready.read_text())
                        except json.JSONDecodeError:pass  # Concurrent first write is not a new attempt.
                    if owner is None:time.sleep(.2)
                if owner['token']!=self.token or owner['stage']!=stage or owner['pgid']!=owner['pid']:raise RuntimeError('Guest ownership handshake differs')
                self.final_gate(stage)
                child.stdin.write(b'GO\n');child.stdin.close();receipt['GO_sent']=True
                deadline=self.cfg['provision_seconds'] if stage=='provision' else self.cfg['operator_seconds']
                while child.poll() is None:
                    self.live()
                    if time.monotonic()-began>deadline:raise TimeoutError('Host stage deadline')
                    time.sleep(.5)
                receipt['exit_code']=child.returncode
                if child.returncode:raise RuntimeError(stage+' failed; no later stage or retry')
                result=json.loads((leaf/'result.json').read_text())
                if result['classification']!='PASS':raise RuntimeError('Guest result rejected')
                receipt['classification']='PASS'
        except BaseException as exc:receipt.update(error=type(exc).__name__+': '+str(exc))
        finally:
            try:
                if child and child.stdin and not child.stdin.closed:child.stdin.close()
                if (leaf/'ownership.json').exists():receipt['guest_cleanup']=self.cleanup_guest(stage)
                elif child is not None:receipt['cleanup_unverified']=True
            except BaseException as exc:receipt['cleanup_error']=str(exc)
            peak=ExtendedLimits()
            if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):receipt['Windows_client_peak_commit_bytes']=peak.PeakProcessMemory
            else:receipt['measurement_error']='Windows Job peak query failed'
            try:receipt['Windows_job_cleanup']=job.finish()
            except BaseException as exc:receipt['job_cleanup_error']=str(exc)
            if child:
                try:child.wait(timeout=5)
                except BaseException:
                    if not receipt.get('job_attached') and child.poll() is None:child.kill();child.wait(timeout=5)
                    receipt['client_reap_error']=True
            receipt['seconds']=time.monotonic()-began
            if (receipt.get('guest_cleanup',{}).get('natural_exit') is not True or receipt.get('cleanup_unverified')
                or any(k in receipt for k in ('error','cleanup_error','job_cleanup_error','client_reap_error','measurement_error'))
                or receipt.get('Windows_job_cleanup',{}).get('forced_job_cleanup') is not False):receipt['classification']='FAIL'
            c.save(leaf/'host-result.json',receipt)
        if receipt['classification']!='PASS':raise RuntimeError(stage+': '+str(receipt.get('error','cleanup failed')))
        if stage=='operators':accept_operators(result,receipt['guest_cleanup'])

def run(commit):
    token=uuid.uuid4().hex
    case=c.plain(c.ROOT.parent/'artifacts')/('amd-pixal3d-standalone-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+token)
    case.mkdir(exist_ok=False);runner=None
    summary=dict(classification='FAIL',commit=commit,case=str(case),operators={n:dict(status='NOT_RUN') for n in c.OPERATORS},
        model_weights=False,generation=False,automatic_retry=False,execution_started=False,next='STOP_FOR_REVIEW_NO_GENERATION')
    try:
        if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-admin Windows PowerShell required')
        verify_publication(commit);desktop_absent();cfg=c.lock()
        if c.digest(sys.executable)!=cfg['windows_python_sha256']:raise ValueError('Standalone base Python changed')
        runner=Runner(case,token,cfg)
        def record(row):c.journal(case/'admission.jsonl','admission',**row)
        def admitted(_):summary['execution_started']=True;runner.prepare_payload()
        c.admit_once(runner.memory,time.sleep,record,admitted)
        sequence(runner.invoke)
        result=json.loads((case/'operators/result.json').read_text());summary.update(classification='PASS_MODEL_FREE_HIP',operators=result['operators'],
            memory_peak=result['memory_peak'],memory_release=result['memory_release'],Linux_worker_peak_RSS_bytes=result.get('Linux_worker_peak_RSS_bytes'),
            operator_stage_seconds=result.get('seconds'),next='GO_FOR_PREPARING_SEPARATE_SINGLE_IMAGE_HANDOFF_ONLY')
    except BaseException as exc:summary.update(error=type(exc).__name__+': '+str(exc),next='STOP_FOR_REVIEW_NO_GENERATION')
    finally:
        partial=case/'operators/result.json'
        if partial.exists():
            try:
                if partial.stat().st_size>8*1024**2:raise ValueError('Bounded result required')
                result=json.loads(partial.read_text())
                summary.update(operators=result.get('operators',summary['operators']))
                for key in ('memory_peak','memory_release','error','cleanup_error','Linux_worker_peak_RSS_bytes','seconds'):
                    if key in result:summary['operator_'+key]=result[key]
            except Exception as exc:summary.update(classification='FAIL',evidence_read_error=str(exc),next='STOP_FOR_REVIEW_NO_GENERATION')
        if summary['execution_started'] and runner is not None:
            try:summary['post_exit_host_memory']=runner.live()
            except Exception as exc:summary.update(classification='FAIL',post_exit_error=str(exc),next='STOP_FOR_REVIEW_NO_GENERATION')
        c.save(case/'summary.json',summary)
        with (case/'REPORT.txt').open('x',encoding='utf-8') as stream:
            reason=summary.get('error') or summary.get('post_exit_error') or summary.get('operator_cleanup_error') or summary.get('evidence_read_error') or 'All required model-free operators passed.'
            stream.write(summary['classification']+'\n'+reason+'\n')
            for name,row in summary['operators'].items():stream.write(name+': '+row['status']+'\n')
            for key in ('operator_stage_seconds','Linux_worker_peak_RSS_bytes','memory_peak','memory_release'):
                if key in summary:stream.write(key+': '+str(summary[key])+'\n')
            stream.write('No model weights or generation. '+summary['next']+'\nEvidence: '+str(case)+'\n')
        print(str(case/'REPORT.txt'));print(summary['classification'],flush=True)
    return 0 if summary['classification']=='PASS_MODEL_FREE_HIP' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--commit')
    parser.add_argument('--payload-child',action='store_true');parser.add_argument('--case');parser.add_argument('--token');args=parser.parse_args(argv)
    if args.payload_child:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned payload GO/EOF required')
        if os.name!='nt' or not args.case or not re.fullmatch('[a-f0-9]{32}',args.token or ''):raise RuntimeError('Exact Windows payload owner required')
        case=c.plain(args.case);case.relative_to(c.ROOT.parent/'artifacts')
        if not case.name.startswith('amd-pixal3d-standalone-') or not case.name.endswith(args.token):raise ValueError('Foreign payload case')
        Runner(case,args.token,c.lock()).prepare_payload_body();return 0
    if not args.run:print('DRY_RUN: no live readings, WSL, downloads, environment, GPU or marker.');return 0
    if not args.commit:parser.error('--run requires exact --commit')
    return run(args.commit)

if __name__=='__main__':raise SystemExit(main())
