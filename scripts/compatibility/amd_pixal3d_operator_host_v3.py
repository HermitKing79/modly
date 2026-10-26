"""Operator-only v3 proposal. No work without explicit future user approval."""
import argparse,ctypes,hashlib,json,os,re,shutil,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c
import amd_pixal3d_memory_v2 as telemetry
from amd_pixal3d_standalone_v2 import desktop_absent,allocated_size,linux_path,accept_operators

FILES=['scripts/compatibility/'+n for n in ('Invoke-Pixal3DOperatorsV3.ps1','amd_pixal3d_operator_host_v3.py','amd_pixal3d_operator_guest_v3.py',
    'amd_pixal3d_operator_ops_v3.py','amd_pixal3d_operator_common_v3.py','test_amd_pixal3d_operator_v3.py','amd_pixal3d_wheel_payload.py','test_amd_pixal3d_wheel_payload.py','amd_pixal3d_standalone_v2.py',
    'amd_pixal3d_guest_v2.py','amd_pixal3d_memory_v2.py','amd_pixal3d_proof_common_v2.py','amd_pixal3d_proof_common.py','amd_zombie_d_handoff.py','amd_zombie_ram_diagnostic.py','amd_forward_owned_job.py')]+[
    'docs/compatibility/amd-pixal3d-operator-v3-contract.json','docs/compatibility/amd-pixal3d-standalone-hip-v2-contract.json',
    'docs/compatibility/amd-pixal3d-standalone-hip-contract.json','docs/compatibility/amd-pixal3d-native-single-image-proof-contract.json']

def verify_publication(commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('Exact published commit required')
    def git(*args):return subprocess.check_output(['git','-C',str(c.ROOT),*args],text=True,encoding='utf-8',timeout=15).strip()
    if git('rev-parse','HEAD')!=commit:raise ValueError('Checkout differs from proposed handoff')
    for name in FILES:
        actual=(c.ROOT/name).read_text().replace('\r\n','\n')
        published=subprocess.check_output(['git','-C',str(c.ROOT),'show',commit+':'+name],text=True,encoding='utf-8',timeout=15)
        if published.replace('\r\n','\n')!=actual:raise ValueError('Changed proof source: '+name)
    cfg=c.lock();c.validate_v2_evidence(cfg)

def wsl_command(stage,case,token,cleanup=False):
    if stage!='operators' or not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('Only approved operator route')
    if not re.fullmatch(r'/mnt/c/[A-Za-z0-9_./ ()+\-]+',case) or '..' in case.split('/'):raise ValueError('Exact evidence path required')
    cfg=c.lock();entry=linux_path(Path(__file__).with_name('amd_pixal3d_operator_guest_v3.py'))
    return [str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe'),'-d',cfg['distro'],'-u',cfg['guest_user'],
        '--exec','/usr/bin/env','MODLY_PIXAL_PROOF_TOKEN='+token,'PYTHONDONTWRITEBYTECODE=1',cfg['guest_python'],'-B','-I','-S',entry,
        '--cleanup' if cleanup else '--controller','--case',case,'--token',token]

class Runner:
    def __init__(self,case,token,cfg):
        self.case=case;self.token=token;self.cfg=cfg;self.startup_active=False;self.memory_log=case/'host-memory.jsonl'
        self.vhd=Path('D:/AI/Modly-WSL/Ubuntu24.04/ext4.vhdx')
        if not self.vhd.is_file():raise RuntimeError('Qualified VHD unavailable; growth guard required')
        self.vhd_before=allocated_size(self.vhd)
    def memory(self):
        from amd_zombie_d_handoff import windows_memory
        return windows_memory()
    def live(self):
        row=self.memory();self.last_memory=row;D=shutil.disk_usage('D:/').free;out=shutil.disk_usage(self.case).free;allocated=allocated_size(self.vhd)
        c.journal(self.memory_log,'sample',observer_pid=os.getpid(),**row,D_free_bytes=D,output_free_bytes=out,VHD_allocated_bytes=allocated)
        c.require_resources(row['available_bytes'],D,out)
        if allocated-self.vhd_before>self.cfg['guest_growth_limit_bytes']:raise RuntimeError('Original VHD growth cap failed')
        if self.startup_active:c.require_profile(row,self.cfg['operator_policy']['ready_bytes'])
        return row
    def observe(self,phase,stage):
        failure=None
        try:row=self.live()
        except RuntimeError as exc:row=getattr(self,'last_memory',None);failure=exc
        try:processes=telemetry.windows_processes()
        except OSError as exc:processes=dict(status='UNAVAILABLE',error=str(exc))
        c.journal(self.case/'startup-memory.jsonl',phase,stage=stage,host=row,Windows_WSL_process_observations=processes)
        if failure:raise failure
        return row
    def final_gate(self,stage):
        row=self.live();c.journal(self.case/'phases.jsonl','operator-only-final-admission',stage=stage,**row)
        c.require_profile(row,self.cfg['operator_policy']['ready_bytes']);return row
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
        child=None;receipt=dict(stage=stage,classification='FAIL',GO_sent=False,client_started=False,startup_classification='NOT_RUN');began=time.monotonic()
        try:
            self.startup_active=True
            receipt['startup_classification']='FAIL'
            receipt['before_WSL']=self.observe('immediately-before-WSL',stage)
            c.require_profile(receipt['before_WSL'],self.cfg['operator_policy']['prelaunch_bytes'])
            with (leaf/'stdout.txt').open('xb') as out,(leaf/'stderr.txt').open('xb') as err:
                child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=str(self.case),creationflags=0x08000000)
                receipt['client_started']=True;receipt['Windows_client_PID']=child.pid
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

def run(commit,approved=False):
    if not approved:raise RuntimeError('Separate user approval of operator-only profile required; no execution')
    cfg=c.lock();token=uuid.uuid4().hex;case=c.plain(c.ROOT.parent/'artifacts')/('amd-pixal3d-operators-v3-payload-auth-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+token)
    windows=c.plain(cfg['windows_root'])
    if case.exists() or windows.exists():raise FileExistsError('Exclusive v3 paths already used; no retry')
    case.mkdir(exist_ok=False);runner=None;lease=None
    summary=dict(classification='FAIL',commit=commit,model_weights=False,generation=False,provisioning=False,downloads=False,automatic_retry=False,
        explicit_user_profile_approval=True,operator_policy=cfg['operator_policy'],operators={n:dict(status='NOT_RUN') for n in c.OPERATORS},
        stages={n:dict(status='NOT_RUN') for n in ('operator_profile_admission','v2_environment_authentication','WSL_startup','HIP_operators')},next='STOP_FOR_REVIEW_NO_GENERATION')
    try:
        if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-admin PowerShell required')
        verify_publication(commit);desktop_absent()
        if c.digest(sys.executable)!=cfg['windows_python_sha256']:raise ValueError('Qualified Windows Python differs')
        sys.addaudithook(c.v2.cache_policy(c.plain(cfg['cache_root'])))
        runner=Runner(case,token,cfg)
        def admitted():
            summary['stages']['operator_profile_admission']=dict(status='PASS')
            windows.mkdir(exist_ok=False);c.save(windows/'owner.json',dict(token=token,case=str(case),profile='PROPOSED_OPERATOR_V3_EXPLICITLY_ACCEPTED'))
            c.save(case/'payload.json',dict(hip_payload=c.evidence_document(cfg,'HIP-wheel.json'),reused_v2_environment=cfg['guest_root'],provisioning=False,downloads=False))
        c.admit_operator(runner.memory,time.sleep,lambda row:c.journal(case/'admission.jsonl','operator-profile',**row),admitted)
        lease=c.CacheReadLease(cfg);lease.__enter__();runner.invoke('operators')
        result=json.loads((case/'operators/result.json').read_text());accept_operators(result,result['guest_cleanup'] if 'guest_cleanup' in result else json.loads((case/'operators/cleanup.json').read_text()))
        summary['classification']='PASS_MODEL_FREE_HIP';summary['next']='STOP_FOR_REVIEW_NO_GENERATION'
    except BaseException as exc:
        summary['error']=type(exc).__name__+': '+str(exc)
        if summary['stages']['operator_profile_admission']['status']=='NOT_RUN':summary['stages']['operator_profile_admission']=dict(status='FAIL',error=summary['error'])
    finally:
        leaf=case/'operators'
        try:
            path=leaf/'host-result.json'
            if path.exists():
                host=json.loads(path.read_text());summary['stages']['WSL_startup']=dict(status=host['startup_classification'],client_started=host['client_started'],GO_sent=host['GO_sent'],error=host.get('error'))
                summary['stages']['HIP_operators']=dict(status=host['classification'] if host['GO_sent'] else 'NOT_RUN',error=host.get('error'))
            auth=leaf/'environment-authentication.json'
            if auth.exists():summary['stages']['v2_environment_authentication']=json.loads(auth.read_text())
            result=leaf/'result.json'
            if result.exists():
                if result.stat().st_size>8*1024**2:raise ValueError('Bounded result required')
                row=json.loads(result.read_text());summary['operators']=row.get('operators',summary['operators'])
                for key in ('memory_peak','memory_release','Linux_worker_peak_RSS_bytes','seconds','error','cleanup_error','read_only_environment_preserved'):
                    if key in row:summary['operator_'+key]=row[key]
            if runner and summary['stages']['operator_profile_admission']['status']=='PASS':summary['post_exit_host_memory']=runner.observe('final-post-exit','closeout')
        except BaseException as exc:summary.update(classification='FAIL',evidence_or_closeout_error=str(exc))
        if lease:lease.__exit__(None,None,None)
        c.save(case/'summary.json',summary)
        with (case/'REPORT.txt').open('x',encoding='utf-8') as stream:
            stream.write(summary['classification']+'\nPROPOSED OPERATOR PROFILE; NEURAL22 GiB UNCHANGED\n')
            for name,row in summary['stages'].items():stream.write(name+': '+row['status']+'\n')
            for name,row in summary['operators'].items():stream.write(name+': '+row['status']+'\n')
            for key in ('error','evidence_or_closeout_error'):
                if key in summary:stream.write(key+': '+summary[key]+'\n')
            stream.write('No provisioning, downloads, models, inference, retry or generation continuation. Evidence: '+str(case)+'\n')
        print(case/'REPORT.txt');print(summary['classification'])
    return 0 if summary['classification']=='PASS_MODEL_FREE_HIP' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--commit');parser.add_argument('--approve-operator-profile-v3',action='store_true');args=parser.parse_args(argv)
    if not args.run:print('DRY_RUN: proposed v3 awaiting separate approval; no admission, WSL, environment changes or GPU');return 0
    if not args.approve_operator_profile_v3 or not args.commit:parser.error('Separate --approve-operator-profile-v3 and exact --commit required')
    return run(args.commit,approved=True)

if __name__=='__main__':raise SystemExit(main())
