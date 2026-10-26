"""Standalone Windows owner. Default does no work; live requires deliberate --run."""
import argparse,ctypes,json,os,re,shutil,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common_v2 as c
import amd_pixal3d_memory_v2 as telemetry

FILES=['scripts/compatibility/'+name for name in ('Invoke-Pixal3DHipProofV2.ps1','amd_pixal3d_standalone_v2.py',
    'amd_pixal3d_proof_common_v2.py','amd_pixal3d_guest_v2.py','amd_pixal3d_hip_ops_v2.py','amd_pixal3d_memory_v2.py',
    'test_amd_pixal3d_standalone_v2.py','amd_pixal3d_proof_common.py',
    'amd_zombie_d_handoff.py','amd_zombie_ram_diagnostic.py','amd_forward_owned_job.py')]+[
    'docs/compatibility/amd-pixal3d-standalone-hip-contract.json','docs/compatibility/amd-pixal3d-standalone-hip-v2-contract.json',
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
    cfg=c.lock();entry=linux_path(Path(__file__).with_name('amd_pixal3d_guest_v2.py'))
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
        self.vhd_before=allocated_size(self.vhd);self.startup_active=False
    def memory(self):
        from amd_zombie_d_handoff import windows_memory
        return windows_memory()
    def live(self):
        row=self.memory();self.last_memory=row;D=shutil.disk_usage('D:/').free;output=shutil.disk_usage(self.case).free
        allocated=allocated_size(self.vhd)
        c.journal(self.memory_log,'sample',observer_pid=os.getpid(),**row,D_free_bytes=D,output_free_bytes=output,VHD_allocated_bytes=allocated)
        c.require_resources(row['available_bytes'],D,output)
        if allocated-self.vhd_before>self.cfg['guest_growth_limit_bytes']:raise RuntimeError('Original guest physical growth cap exceeded')
        if self.startup_active and row['available_bytes']<22*c.GIB:raise RuntimeError('WSL startup below 22 GiB; no guest GO')
        return row
    def observe(self,phase,stage):
        failure=None
        try:row=self.live()
        except RuntimeError as exc:
            row=getattr(self,'last_memory',None);failure=exc
        try:processes=telemetry.windows_processes()
        except OSError as exc:processes=dict(status='UNAVAILABLE',error=str(exc))
        c.journal(self.case/'startup-memory.jsonl',phase,stage=stage,host=row,Windows_WSL_process_observations=processes)
        if failure:raise failure
        return row
    def final_gate(self,stage):
        row=self.live();c.journal(self.case/'phases.jsonl','final-22GiB',stage=stage,**row);c.require_ram(row,22*c.GIB);return row
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
        if self.windows.exists():raise FileExistsError('Exclusive v2 Windows root already used; stop for review')
        sys.addaudithook(c.cache_policy(c.plain(self.cfg['cache_root'])))
        cache=c.verify_cache(self.cfg,self.live)
        downloads=c.plain(self.cfg['cache_root'])/'downloads';sources=c.plain(self.cfg['cache_root'])/'sources'
        kitchen=next(p for p in self.cfg['packages'] if p['name']=='comfy-kitchen')
        hip=c.hip_payload(downloads/kitchen['filename'])
        expansion=c.bounded_expansion([downloads/p['filename'] for p in self.cfg['packages']])
        if shutil.disk_usage('D:/').free<50*c.GIB+expansion:raise RuntimeError('Original D floor after expansion would fail')
        # Both proposed Windows paths were checked before creation. No archive/source copy.
        self.windows.mkdir(exist_ok=False)
        c.save(self.windows/'owner.json',dict(token=self.token,case=str(self.case),cache=str(downloads.parent)))
        c.save(self.case/'cache-verification.json',cache);c.save(self.case/'HIP-wheel.json',hip)
        c.save(self.case/'payload.json',dict(windows_root=str(self.windows),guest_downloads=linux_path(downloads),guest_sources=linux_path(sources),
            packages=self.cfg['packages'],sources=self.cfg['sources'],expansion_bytes=expansion,hip_payload=hip,cache_read_only=True,archives_copied=0,downloads=0))
    def cleanup_guest(self,stage):
        command=wsl_command(stage,linux_path(self.case),self.token,cleanup=True)
        with (self.case/stage/'cleanup-stdout.txt').open('xb') as out,(self.case/stage/'cleanup-stderr.txt').open('xb') as err:
            completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=self.cfg['cleanup_seconds'],creationflags=0x08000000)
        if completed.returncode:raise RuntimeError('Owned guest cleanup verification failed')
        return json.loads((self.case/stage/'cleanup.json').read_text())
    def invoke(self,stage):
        from amd_forward_owned_job import OwnedJob,ExtendedLimits
        leaf=self.case/stage
        if leaf.exists():raise FileExistsError('Stage evidence already exists; no retry')
        leaf.mkdir(exist_ok=False);command=wsl_command(stage,linux_path(self.case),self.token)
        job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=6*c.GIB
        if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            job.close();raise ctypes.WinError()
        child=None;receipt=dict(stage=stage,classification='FAIL',GO_sent=False,startup_classification='NOT_RUN');began=time.monotonic()
        try:
            self.startup_active=True
            receipt['startup_classification']='FAIL'
            receipt['before_WSL']=self.observe('immediately-before-WSL',stage)
            with (leaf/'stdout.txt').open('xb') as out,(leaf/'stderr.txt').open('xb') as err:
                child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=str(self.case),creationflags=0x08000000)
                job.attach(child);receipt['job_attached']=True
                ready=leaf/'ready.json';owner=None
                while owner is None:
                    self.observe('WSL-startup',stage)
                    if child.poll() is not None:raise RuntimeError('Guest exited before ready/GO')
                    if time.monotonic()-began>120:raise TimeoutError('Original CPU ready deadline')
                    if ready.exists():
                        try:owner=json.loads(ready.read_text())['owner']
                        except json.JSONDecodeError:pass
                    if owner is None:time.sleep(.2)
                if owner['token']!=self.token or owner['stage']!=stage or owner['pgid']!=owner['pid']:raise RuntimeError('Guest ownership handshake differs')
                receipt['ready_before_GO']=self.observe('guest-ready-before-GO',stage)
                # Last RAM read follows process inventory, immediately before GO.
                receipt['final_before_GO']=self.final_gate(stage)
                child.stdin.write(b'GO\n');child.stdin.close();receipt['GO_sent']=True
                receipt['startup_classification']='PASS';self.startup_active=False
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
            self.startup_active=False
            if child and child.stdin and not child.stdin.closed:child.stdin.close()
            # A client timeout must never bypass nonce/PGID Linux cleanup.
            # After GO, stop owned Linux work immediately on a failed guard.
            if child and not receipt['GO_sent']:
                try:child.wait(timeout=self.cfg['cleanup_seconds'])
                except subprocess.TimeoutExpired:receipt['startup_EOF_wait_timeout']=True
                except BaseException as exc:receipt['client_wait_error']=str(exc)
            try:
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
                    if not receipt.get('job_attached') and child.poll() is None:
                        try:child.kill();child.wait(timeout=5);receipt['unattached_owned_client_forced_cleanup']=True
                        except BaseException as exc:receipt['unattached_client_cleanup_error']=str(exc)
                    receipt['client_reap_error']=True
            try:receipt['after_owned_exit']=self.observe('after-owned-process-exit',stage)
            except BaseException as exc:receipt['recovery_observation_error']=str(exc)
            receipt['seconds']=time.monotonic()-began
            if (receipt.get('guest_cleanup',{}).get('natural_exit') is not True or receipt.get('cleanup_unverified')
                or any(k in receipt for k in ('error','cleanup_error','job_cleanup_error','client_reap_error','client_wait_error','startup_EOF_wait_timeout','measurement_error','recovery_observation_error'))
                or receipt.get('Windows_job_cleanup',{}).get('forced_job_cleanup') is not False):receipt['classification']='FAIL'
            c.save(leaf/'host-result.json',receipt)
        if receipt['classification']!='PASS':raise RuntimeError(stage+': '+str(receipt.get('error','cleanup failed')))
        if stage=='operators':accept_operators(result,receipt['guest_cleanup'])

def initial_summary(commit,case):
    return dict(classification='FAIL',commit=commit,case=str(case),operators={n:dict(status='NOT_RUN') for n in c.OPERATORS},
        stages={n:dict(status='NOT_RUN') for n in ('windows_admission','cache_verification','wsl_startup','provisioning','hip_operators')},
        model_weights=False,generation=False,automatic_retry=False,execution_started=False,next='STOP_FOR_REVIEW_NO_GENERATION')

def harvest(summary,case):
    startup_rows=[]
    for stage,label in [('provision','provisioning'),('operators','hip_operators')]:
        path=case/stage/'host-result.json'
        if path.exists():
            try:
                receipt=json.loads(path.read_text())
                startup_rows.append(dict(stage=stage,status=receipt['startup_classification'],GO_sent=receipt['GO_sent'],error=receipt.get('error')))
                summary['stages'][label]=dict(status=receipt['classification'] if receipt['GO_sent'] else 'NOT_RUN',GO_sent=receipt['GO_sent'],error=receipt.get('error'))
            except Exception as exc:summary['evidence_read_error']=str(exc)
    if startup_rows:
        summary['stages']['wsl_startup']=dict(status='FAIL' if any(r['status']=='FAIL' for r in startup_rows) else 'PASS',attempted_stages=startup_rows)
    partial=case/'operators/result.json'
    if partial.exists():
        try:
            if partial.stat().st_size>8*1024**2:raise ValueError('Bounded result required')
            result=json.loads(partial.read_text());summary['operators']=result.get('operators',summary['operators'])
            for key in ('memory_peak','memory_release','Linux_worker_peak_RSS_bytes','seconds','error','cleanup_error'):
                if key in result:summary['operator_'+key]=result[key]
        except Exception as exc:summary['evidence_read_error']=str(exc)

def run(commit):
    token=uuid.uuid4().hex;cfg=c.lock()
    case=c.plain(c.ROOT.parent/'artifacts')/('amd-pixal3d-standalone-v2-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+token)
    windows=c.plain(cfg['windows_root'])
    if case.exists() or windows.exists():raise FileExistsError('Exclusive v2 Windows/evidence path already exists; no retry')
    case.mkdir(exist_ok=False);runner=None;lease=None;summary=initial_summary(commit,case)
    try:
        if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-admin Windows PowerShell required')
        verify_publication(commit);desktop_absent();sys.addaudithook(c.cache_policy(c.plain(cfg['cache_root'])))
        if c.digest(sys.executable)!=cfg['windows_python_sha256']:raise ValueError('Standalone base Python changed')
        runner=Runner(case,token,cfg)
        def record(row):c.journal(case/'admission.jsonl','admission',**row)
        def admitted(_):
            summary['stages']['windows_admission']=dict(status='PASS');summary['execution_started']=True
            summary['stages']['cache_verification']=dict(status='FAIL')
            runner.prepare_payload();summary['stages']['cache_verification']=dict(status='PASS',archives_copied=0,downloads=0)
        c.admit_once(runner.memory,time.sleep,record,admitted)
        lease=c.CacheReadLease(cfg);lease.__enter__()
        sequence(runner.invoke)
        result=json.loads((case/'operators/result.json').read_text())
        summary.update(classification='PASS_MODEL_FREE_HIP',operators=result['operators'],next='GO_FOR_PREPARING_SEPARATE_SINGLE_IMAGE_HANDOFF_ONLY')
    except BaseException as exc:
        summary.update(error=type(exc).__name__+': '+str(exc),next='STOP_FOR_REVIEW_NO_GENERATION')
        if summary['stages']['windows_admission']['status']=='NOT_RUN':summary['stages']['windows_admission']=dict(status='FAIL',error=summary['error'])
    finally:
        harvest(summary,case)
        if summary['execution_started'] and runner is not None:
            try:
                summary['post_exit_host_memory']=runner.observe('final-post-exit','closeout')
                # Independent rehash; no cache mutation, archive copy or download at closeout.
                if summary['stages']['cache_verification']['status']=='PASS':
                    after=c.verify_cache(cfg,runner.live);before=json.loads((case/'cache-verification.json').read_text())
                    if before!=after:raise RuntimeError('Immutable cache changed during proof')
                    summary['cache_preserved']=True
            except Exception as exc:summary.update(classification='FAIL',post_exit_error=str(exc),next='STOP_FOR_REVIEW_NO_GENERATION')
        if lease:lease.__exit__(None,None,None)
        if summary.get('evidence_read_error'):summary['classification']='FAIL';summary['next']='STOP_FOR_REVIEW_NO_GENERATION'
        c.save(case/'summary.json',summary)
        with (case/'REPORT.txt').open('x',encoding='utf-8') as stream:
            stream.write(summary['classification']+'\n')
            for name,row in summary['stages'].items():stream.write(name+': '+row['status']+'\n')
            for name,row in summary['operators'].items():stream.write(name+': '+row['status']+'\n')
            for name in ('error','post_exit_error','evidence_read_error'):
                if name in summary:stream.write(name+': '+summary[name]+'\n')
            stream.write('No model weights, downloads or generation. '+summary['next']+'\nEvidence: '+str(case)+'\n')
        print(str(case/'REPORT.txt'));print(summary['classification'],flush=True)
    return 0 if summary['classification']=='PASS_MODEL_FREE_HIP' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--commit')
    parser.add_argument('--payload-child',action='store_true');parser.add_argument('--case');parser.add_argument('--token');args=parser.parse_args(argv)
    if args.payload_child:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned payload GO/EOF required')
        if os.name!='nt' or not args.case or not re.fullmatch('[a-f0-9]{32}',args.token or ''):raise RuntimeError('Exact Windows payload owner required')
        case=c.plain(args.case);case.relative_to(c.ROOT.parent/'artifacts')
        if not case.name.startswith('amd-pixal3d-standalone-v2-') or not case.name.endswith(args.token):raise ValueError('Foreign payload case')
        Runner(case,args.token,c.lock()).prepare_payload_body();return 0
    if not args.run:print('DRY_RUN: no live readings, WSL, downloads, environment, GPU or marker.');return 0
    if not args.commit:parser.error('--run requires exact --commit')
    return run(args.commit)

if __name__=='__main__':raise SystemExit(main())
