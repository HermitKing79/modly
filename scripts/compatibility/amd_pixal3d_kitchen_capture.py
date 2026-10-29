"""Separately approved import-only forensics. Operators remain held. Default: no work."""
import argparse,atexit,ctypes,hashlib,json,os,re,signal,subprocess,sys,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c
from amd_pixal3d_operator_audit_v3 import AuditLog,exception_evidence
from amd_pixal3d_audit_rules_v3 import policy
from amd_pixal3d_owned_proc_v3 import ProcFS,process_rows,owned_members,signal_verified
from amd_pixal3d_operator_guest_v3 import baseline,child_environment,reap_child

CONTRACT=c.ROOT/'docs/compatibility/amd-pixal3d-kitchen-import-capture-contract.json'
PURPOSE='KITCHEN_IMPORT_FORENSICS_ONLY'
PREFIX='amd-pixal3d-kitchen-import-capture-v1-'
DENIED=23
EVIDENCE_FAILURE=24

def config():
    base=c.lock();spec=json.loads(CONTRACT.read_text())
    expected=dict(schema='modly.pixal3d.kitchen-import-capture.v1',purpose=PURPOSE,approval_pending=True,
        windows_root='D:/AI/Modly-Pixal3D-Kitchen-Import-Capture-v1',temporary_guest_root='/home/modly/pixal3d-kitchen-import-capture-v1',
        evidence_prefix=PREFIX,operator_hold_required=True,numerical_checks=False,model_loading=False,downloads=False,installations=False,automatic_retry=False)
    if spec!=expected or base['operator_execution_hold'] is not True:raise RuntimeError('Exact import-only scope and retained operator hold required')
    return dict(base,windows_root=spec['windows_root'],temporary_guest_root=spec['temporary_guest_root'],capture_spec=spec)

def validate_case(case,token):
    if not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('Exact nonce required')
    case=c.plain(case);case.relative_to(c.plain(c.ROOT.parent/'artifacts'))
    if not re.fullmatch(PREFIX+r'\d{8}T\d{6}Z-'+token,case.name):raise ValueError('Exclusive capture evidence path required')
    return case

class FirstDenial:
    """Persist once, then leave the process: imported code cannot swallow it."""
    def __init__(self,log,exit_process=None):self.log=log;self.exit=exit_process or os._exit;self.lock=threading.Lock()
    def __call__(self,event,args):
        with self.lock:
            try:self.log(event,args)
            except BaseException:self.exit(EVIDENCE_FAILURE);raise RuntimeError('Evidence exit unexpectedly returned')
            self.exit(DENIED);raise RuntimeError('Denial exit unexpectedly returned')

def capture_policy(root,case,sink):
    ordinary=policy(root,case,sink)
    immutable=[c.v2.cache_policy(c.plain(config()['guest_root'])),c.v2.cache_policy(Path('/mnt/d/AI/Modly-Pixal3D-Proof-v1')),c.v2.cache_policy(c.overlay.archive_path())]
    filesystem={'open','os.mkdir','os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate','os.utime','os.chmod','os.chown'}
    def audit(event,args):
        try:
            for guard in immutable:guard(event,args)
            ordinary(event,args)
        except Exception:
            if event in filesystem:sink(event,args)
            raise
    return audit

def child_command(mode,case,token,cfg):
    if mode not in ('--capture-worker','--release-observer'):raise ValueError('Only import capture or query-only cleanup')
    return [cfg['guest_root']+'/.venv/bin/python','-B','-I',str(Path(__file__).resolve()),mode,'--case',str(case),'--token',token]

def unraisable_handler(path,exit_process=None):
    leave=exit_process or os._exit
    def stop(args):
        try:c.save(path,exception_evidence(args.exc_value))
        finally:leave(EVIDENCE_FAILURE)
    return stop

def runtime(cfg,observer=False,phase=None):
    if phase:phase('torch-import','CPU')
    import importlib.metadata
    import torch
    rt=cfg['runtime'];root=c.plain(cfg['guest_root'])
    if (torch.__version__!=rt['torch'] or importlib.metadata.version('torch')!=rt['distribution'] or torch.version.git_version!=rt['git']
        or torch.version.hip!=rt['hip'] or torch.version.cuda is not None or c.digest(Path(torch.__file__).with_name('version.py'))!=rt['torch_version_file_sha256']
        or not Path(torch.__file__).resolve().is_relative_to(root/'.venv')):raise RuntimeError('Qualified Torch identity differs')
    if torch.cuda.is_initialized():raise RuntimeError('Unexpected GPU initialization during Torch import')
    def forbidden(*a,**kw):raise RuntimeError('Model loading, compilation or explicit numerical execution forbidden')
    torch.load=forbidden;torch.jit.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
    if phase:phase('query-only-device-admission','GPU')
    if torch.cuda.device_count()!=1:raise RuntimeError('One unmasked visible GPU required')
    props=torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(':')[0]!='gfx1201' or '9070 XT' not in props.name:raise RuntimeError('Qualified gfx1201 device required')
    free,total=torch.cuda.mem_get_info();floor=cfg['guards']['live_shape_free_VRAM_bytes'] if observer else cfg['guards']['initial_free_VRAM_bytes']
    if free<floor:raise RuntimeError('Existing VRAM admission/release floor failed')
    torch.cuda.set_per_process_memory_fraction(cfg['guards']['torch_allocator_cap_bytes']/total,0)
    return torch,dict(name=props.name,arch=props.gcnArchName,free_bytes=free,total_bytes=total)

def worker_context(case,token,cfg):
    if os.name!='posix' or Path(sys.prefix)!=Path(cfg['guest_root'])/'.venv' or sys.version.split()[0]!=cfg['runtime']['python']:raise RuntimeError('Qualified V2 Python required')
    if os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token or sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO/EOF required')
    ProcFS().require_pidfds()
    leaf=case/'capture';owner=json.loads(c.plain(leaf/'ownership.json').read_text());ready=json.loads(c.plain(leaf/'ready.json').read_text())
    if (owner.get('stage')!='capture' or owner.get('token')!=token or owner['pid']!=owner['pgid'] or os.getpgrp()!=owner['pgid']
        or ready.get('owner')!=owner or ready.get('purpose')!=PURPOSE or ready.get('pidfd_kernel_preflight') is not True or ready.get('environment_authenticated') is not True):raise RuntimeError('Authenticated capture handshake required')
    members=process_rows(owner['pgid']);owned_members(owner,members)
    if not any(r['pid']==owner['pid'] for r in members) or not any(r['pid']==os.getpid() for r in members):raise RuntimeError('Live controller/worker identities required')

def worker(case,token,observer=False):
    case=validate_case(case,token);cfg=config();leaf=case/'capture';temporary=c.plain(cfg['temporary_guest_root'])
    worker_context(case,token,cfg)
    sys.unraisablehook=unraisable_handler(leaf/('observer-unraisable.json' if observer else 'unraisable.json'))
    if observer:
        observer_log=AuditLog(leaf/'observer-denial.jsonl');observer_log.phase='query-only-release-observer'
        sys.addaudithook(capture_policy(temporary,case,FirstDenial(observer_log)))
        def query_phase(name,kind):
            observer_log.phase=name;c.journal(leaf/'capture-phases.jsonl',name,kind=kind,monotonic_start=time.monotonic())
        torch,gpu=runtime(cfg,True,query_phase)
        allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
        if allocated or reserved:raise RuntimeError('Query-only observer unexpectedly owns tensor allocation')
        c.save(leaf/'release-observer.json',dict(purpose=PURPOSE,gpu=gpu,allocated=allocated,reserved=reserved,numerical_checks=False))
        return 0
    log=AuditLog(leaf/'first-denial.jsonl');sink=FirstDenial(log)
    sys.addaudithook(capture_policy(temporary,case,sink))
    def phase(name,kind='CPU'):
        log.phase=name;c.journal(leaf/'capture-phases.jsonl',name,kind=kind,monotonic_start=time.monotonic())
    try:
        phase('packaging-overlay');overlay=c.overlay.activate()
        torch,gpu=runtime(cfg,phase=phase)
        # Observe state without calling gettempdir or changing its cache/env.
        temp=sys.modules.get('tempfile');before=dict(tempfile_loaded=temp is not None,tempdir=getattr(temp,'tempdir',None),
            loaded_relevant_modules=[n for n in ('torch._dynamo','torch._dynamo.package','torch.distributed.nn.jit.instantiator') if n in sys.modules],
            inductor_cache_env=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),compile_debug_env=os.environ.get('TORCH_COMPILE_DEBUG_DIR'))
        c.save(leaf/'before-kitchen.json',dict(purpose=PURPOSE,worker_pid=os.getpid(),overlay=overlay,gpu=gpu,observed_state=before))
        phase('Kitchen-import','GPU')
        import comfy_kitchen
        if not Path(comfy_kitchen.__file__).resolve().is_relative_to(Path(cfg['guest_root'])/'.venv'):raise RuntimeError('Foreign Kitchen origin')
        phase('import-completed-resource-release','GPU')
        free,total=torch.cuda.mem_get_info()
        if free<cfg['guards']['live_shape_free_VRAM_bytes']:raise RuntimeError('Existing live VRAM floor failed')
        torch.cuda.empty_cache();torch.cuda.synchronize()
        if torch.cuda.memory_allocated() or torch.cuda.memory_reserved():raise RuntimeError('Owned tensor allocation remains')
        c.save(leaf/'import-result.json',dict(purpose=PURPOSE,import_completed=True,numerical_checks=False,models_loaded=False,module=str(comfy_kitchen.__file__),overlay_origins=c.overlay.verify_loaded()))
        # Remain armed throughout Python atexit: imported cleanup may be the
        # first denied operation. Never close the evidence FD before teardown.
        atexit.register(lambda:phase('interpreter-teardown'))
        return 0
    except BaseException as exc:
        c.save(leaf/'worker-error.json',exception_evidence(exc));atexit.register(lambda:phase('interpreter-teardown-after-error'))
        return 2

def accept_capture(result,cleanup):
    if (result.get('classification')!='PASS' or result.get('purpose')!=PURPOSE or result.get('capture_status') not in ('FIRST_DENIAL_CAPTURED','IMPORT_COMPLETED_NO_DENIAL')
        or result.get('numerical_checks') is not False or result.get('models_loaded') is not False or result.get('operator_execution_hold') is not True
        or cleanup.get('natural_exit') is not True or cleanup.get('group_empty') is not True or result.get('release_observer',{}).get('numerical_checks') is not False
        or result.get('V2_preserved') is not True or result.get('natural_child_exit') is not True
        or result.get('release_observer',{}).get('allocated')!=0 or result.get('release_observer',{}).get('reserved')!=0
        or result.get('release_observer',{}).get('gpu',{}).get('free_bytes',-1)<3*c.GIB):
        raise RuntimeError('Incomplete import-only forensic evidence; no qualification')
    if result['capture_status']=='FIRST_DENIAL_CAPTURED':
        row=result.get('first_denial',{})
        if row.get('operation_allowed') is not False or not row.get('event') or not row.get('caller_frames'):raise RuntimeError('Actual denied event evidence required')
    elif result.get('import_result',{}).get('import_completed') is not True:raise RuntimeError('Completed import evidence required')

def last_phase(path):
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65536));complete=stream.read(65536).split(b'\n')[:-1]
    return json.loads(complete[-1]) if complete else None

def denial_record(leaf):
    path=leaf/'first-denial.jsonl'
    if not path.exists():return None
    if path.stat().st_size>65536:raise RuntimeError('First-denial evidence bound exceeded')
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    if not rows:return None
    if len(rows)!=1 or rows[0].get('operation_allowed') is not False:raise RuntimeError('Ambiguous first-denial evidence')
    return rows[0]

def controller(case,token):
    case=validate_case(case,token);cfg=config();leaf=case/'capture';temporary=c.plain(cfg['temporary_guest_root'])
    if os.name!='posix' or sys.prefix!='/usr' or sys.version.split()[0]!=cfg['runtime']['python'] or os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token:raise RuntimeError('Qualified system controller required')
    if os.getpgrp()!=os.getpid():os.setsid()
    ProcFS().require_pidfds()
    def observation(row):c.journal(leaf/'ownership-observations.jsonl','proc-check',**row)
    own=next(r for r in process_rows(os.getpgrp(),report=observation) if r['pid']==os.getpid())
    owner=dict(pid=own['pid'],pgid=own['pgid'],start_ticks=own['start_ticks'],token=token,stage='capture')
    if own['token']!=token or owner['pid']!=owner['pgid']:raise RuntimeError('Controller ownership differs')
    c.save(leaf/'ownership.json',owner)
    for path in (Path(cfg['guest_root']),Path('/mnt/d/AI/Modly-Pixal3D-Proof-v1'),c.overlay.archive_path()):sys.addaudithook(c.v2.cache_policy(c.plain(path)))
    record=dict(classification='FAIL',purpose=PURPOSE,numerical_checks=False,models_loaded=False,operator_execution_hold=True)
    start=time.monotonic();last=0;peak=0;child=None;before=None;auth_progress=None
    def resources(auth=False):
        nonlocal last,peak
        if time.monotonic()-last<.25:return
        last=time.monotonic();rows=process_rows(owner['pgid'],report=observation);owned_members(owner,rows);rss=sum(r['RSS_bytes'] for r in rows);peak=max(peak,rss)
        context=dict(authentication_progress=auth_progress.snapshot()) if auth and auth_progress is not None else {}
        disk=__import__('shutil').disk_usage('/home/modly').free;c.journal(leaf/'resources.jsonl','capture',RSS_bytes=rss,members=rows,guest_free_disk_bytes=disk,**context)
        if rss>6*c.GIB or disk<cfg['guest_disk_floor_bytes']:raise RuntimeError('Existing process/disk floor failed')
        if time.monotonic()-start>(120 if auth else cfg['operator_seconds']):raise TimeoutError('Existing authentication/overall deadline')
    def launch(mode):
        nonlocal child
        command=child_command(mode,case,token,cfg);c.journal(leaf/'commands.jsonl','child',argv=command)
        child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,env=child_environment(temporary,token))
        rows=process_rows(owner['pgid'],report=observation);owned_members(owner,rows)
        if not any(r['pid']==child.pid for r in rows):raise RuntimeError('Child identity not witnessed before GO')
        child.stdin.write(b'GO\n');child.stdin.close();began=time.monotonic()
        while child.poll() is None:
            resources();row=last_phase(leaf/'capture-phases.jsonl')
            if row:
                limit=120 if row['kind']=='CPU' else 180
                if time.monotonic()-row['monotonic_start']>limit:raise TimeoutError('Existing phase deadline')
            if time.monotonic()-began>cfg['operator_seconds']:raise TimeoutError('Existing child deadline')
            time.sleep(.25)
        code=child.wait(timeout=5);record.setdefault('child_exits',[]).append(dict(mode=mode,pid=child.pid,exit_code=code));return code
    try:
        if temporary.exists():raise FileExistsError('Exclusive capture scratch already used; no retry')
        before=c.environment_snapshot(cfg['guest_root']);c.save(leaf/'environment-before.json',before)
        auth_progress=c.AuthenticationProgress(lambda:resources(True))
        auth=c.authenticate_environment(cfg,auth_progress,lambda row:c.journal(leaf/'archive-authentication.jsonl','archive',**row));c.save(leaf/'environment-authentication.json',auth)
        c.save(leaf/'ready.json',dict(owner=owner,purpose=PURPOSE,pidfd_kernel_preflight=True,environment_authenticated=True))
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('No host GO/EOF')
        temporary.mkdir(mode=0o700,exist_ok=False)
        for name in ('home','tmp','hip-cache','comgr-cache','disabled-torch-extensions','disabled-triton-cache'):(temporary/name).mkdir(mode=0o700)
        c.save(temporary/'owner.json',dict(token=token,case=str(case),purpose=PURPOSE))
        code=launch('--capture-worker');denial=denial_record(leaf)
        if code==DENIED and denial is not None and not (leaf/'worker-error.json').exists():
            record.update(capture_status='FIRST_DENIAL_CAPTURED',first_denial=denial,termination='SELF_EXIT_ON_FIRST_DENIAL')
        elif code==0 and denial is None and (leaf/'import-result.json').exists():
            imported=json.loads((leaf/'import-result.json').read_text())
            if imported.get('purpose')!=PURPOSE or imported.get('import_completed') is not True or imported.get('numerical_checks') is not False:raise RuntimeError('Import completion evidence differs')
            record.update(capture_status='IMPORT_COMPLETED_NO_DENIAL',import_result=imported)
        else:raise RuntimeError('Capture failed/ambiguous; preserve evidence without retry')
        if any(r['pid']!=os.getpid() for r in process_rows(owner['pgid'])):raise RuntimeError('Capture child group not naturally empty')
        c.journal(leaf/'capture-phases.jsonl','query-only-release-observer',kind='CPU',monotonic_start=time.monotonic())
        if launch('--release-observer')!=0:raise RuntimeError('Query-only release observation failed')
        record['release_observer']=json.loads((leaf/'release-observer.json').read_text());record['classification']='PASS'
    except BaseException as exc:
        if auth_progress is not None:auth_progress.finish('failed')
        record.update(classification='FAIL',error=type(exc).__name__+': '+str(exc),controller_failure=exception_evidence(exc))
    finally:
        if auth_progress is not None:record['authentication_progress']=auth_progress.snapshot()
        reap_child(record,child)
        try:
            denial=denial_record(leaf)
            if denial is not None:record['observed_first_denial']=denial
        except BaseException as exc:record.update(classification='FAIL',denial_evidence_error=str(exc))
        try:
            after=c.environment_snapshot(cfg['guest_root']);c.save(leaf/'environment-after.json',after)
            record['V2_preserved']=before is not None and before==after
            if not record['V2_preserved']:raise RuntimeError('V2 snapshot changed; no repair')
            record['overlay_after']=c.overlay.authenticate();members=process_rows(owner['pgid'],report=observation);owned_members(owner,members)
            if any(r['pid']!=os.getpid() for r in members):raise RuntimeError('Remaining group member')
        except BaseException as exc:record.update(classification='FAIL',cleanup_or_preservation_error=str(exc))
        record.update(seconds=time.monotonic()-start,peak_sampled_RSS_bytes=peak,natural_child_exit=not record.get('child_forced_cleanup') and not record.get('child_exit_unverified'))
        c.save(leaf/'result.json',record)
    return 0 if record['classification']=='PASS' else 2

def cleanup(case,token):
    case=validate_case(case,token);leaf=case/'capture';owner=json.loads((leaf/'ownership.json').read_text())
    if owner['token']!=token or owner['stage']!='capture' or owner['pid']!=owner['pgid']:raise RuntimeError('Foreign capture owner')
    rows=process_rows(owner['pgid']);owned_members(owner,rows);forced=bool(rows)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if not rows:break
        signal_verified(owner,sig);began=time.monotonic()
        while time.monotonic()-began<5:
            rows=process_rows(owner['pgid']);owned_members(owner,rows)
            if not rows:break
            time.sleep(.05)
    if rows:raise RuntimeError('Capture group exit unverified')
    c.save(leaf/'cleanup.json',dict(group_empty=True,natural_exit=not forced,forced_owned_group_cleanup=forced,no_distro_termination=True));return 0

def host_wsl_command(case,token,cleanup_mode=False):
    from amd_pixal3d_standalone_v2 import linux_path
    cfg=config();validate_case(case,token)
    return [str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe'),'-d',cfg['distro'],'-u',cfg['guest_user'],'--exec','/usr/bin/env',
        'MODLY_PIXAL_PROOF_TOKEN='+token,'PYTHONDONTWRITEBYTECODE=1',cfg['guest_python'],'-B','-I','-S',linux_path(Path(__file__).resolve()),
        '--cleanup' if cleanup_mode else '--controller','--case',linux_path(case),'--token',token]

def runner_type():
    from amd_pixal3d_operator_host_v3 import Runner
    class CaptureRunner(Runner):
        def command(self,stage,cleanup=False):
            if stage!='capture':raise ValueError('Only capture route')
            return host_wsl_command(self.case,self.token,cleanup)
        def accept_result(self,stage,result,cleanup):accept_capture(result,cleanup)
    return CaptureRunner

def verify_publication(commit):
    from amd_pixal3d_operator_host_v3 import verify_publication as base_verify
    base_verify(commit)
    for name in ('scripts/compatibility/Invoke-Pixal3DKitchenImportCapture.ps1','scripts/compatibility/amd_pixal3d_kitchen_capture.py',
        'scripts/compatibility/test_amd_pixal3d_kitchen_capture.py','docs/compatibility/amd-pixal3d-kitchen-import-capture-contract.json'):
        expected=subprocess.check_output(['git','-C',str(c.ROOT),'show',commit+':'+name],text=True,encoding='utf-8',timeout=15)
        if expected.replace('\r\n','\n')!=(c.ROOT/name).read_text().replace('\r\n','\n'):raise RuntimeError('Changed capture source: '+name)

def run_host(commit,approved=False):
    if not approved:raise RuntimeError('Separate explicit approval of Kitchen-import-only capture required')
    cfg=config();verify_publication(commit)
    from amd_pixal3d_standalone_v2 import desktop_absent
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal Windows PowerShell required')
    desktop_absent()
    if c.digest(sys.executable)!=cfg['windows_python_sha256']:raise RuntimeError('Qualified Windows Python required')
    token=uuid.uuid4().hex;case=c.plain(c.ROOT.parent/'artifacts')/(PREFIX+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+token);windows=c.plain(cfg['windows_root'])
    if case.exists() or windows.exists():raise FileExistsError('Exclusive capture paths occupied; no retry')
    case.mkdir(exist_ok=False);runner=None;lease=None
    summary=dict(classification='FAIL',purpose=PURPOSE,commit=commit,admission='NOT_RUN',explicit_capture_approval=True,operator_execution_hold=True,numerical_checks=False,models_loaded=False,downloads=False,installations=False)
    try:
        runner=runner_type()(case,token,cfg)
        def admitted():
            summary['admission']='PASS';windows.mkdir(exist_ok=False);c.save(windows/'owner.json',dict(token=token,case=str(case),purpose=PURPOSE))
        c.admit_operator(runner.memory,time.sleep,lambda row:c.journal(case/'admission.jsonl','capture-profile',**row),admitted)
        lease=c.CacheReadLease(cfg);lease.__enter__();c.save(case/'overlay-authentication.json',c.overlay.hold_read_only(lease))
        runner.invoke('capture');summary.update(classification='CAPTURE_COMPLETE_REVIEW_REQUIRED',result=json.loads((case/'capture/result.json').read_text()))
    except BaseException as exc:
        summary['error']=type(exc).__name__+': '+str(exc)
        if summary['admission']=='NOT_RUN':summary['admission']='FAIL'
    finally:
        try:
            for name in ('result.json','host-result.json','ready.json'):
                path=case/'capture'/name
                if path.exists():
                    if path.stat().st_size>8*1024**2:raise RuntimeError('Bounded capture result required')
                    summary[name]=json.loads(path.read_text())
            if runner:summary['post_exit_host_memory']=runner.observe('post-capture-exit','capture')
        except BaseException as exc:summary.update(classification='FAIL',recovery_error=str(exc))
        if lease:lease.__exit__(None,None,None)
        c.save(case/'summary.json',summary)
        with (case/'REPORT.txt').open('x',encoding='utf-8') as f:
            f.write(summary['classification']+'\nKITCHEN IMPORT FORENSICS ONLY; OPERATOR HOLD RETAINED\n'+summary.get('error','')+'\n'+str(case)+'\n')
        print(case/'REPORT.txt')
    return 0 if summary['classification']=='CAPTURE_COMPLETE_REVIEW_REQUIRED' else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);routes=p.add_mutually_exclusive_group();routes.add_argument('--capture-kitchen-import',action='store_true');routes.add_argument('--controller',action='store_true');routes.add_argument('--capture-worker',action='store_true');routes.add_argument('--release-observer',action='store_true');routes.add_argument('--cleanup',action='store_true')
    p.add_argument('--approve-kitchen-import-capture',action='store_true');p.add_argument('--commit');p.add_argument('--case');p.add_argument('--token');a=p.parse_args(argv)
    if a.capture_kitchen_import:
        if not a.approve_kitchen_import_capture or not re.fullmatch('[a-f0-9]{40}',a.commit or ''):p.error('Separate capture approval and exact published commit required')
        return run_host(a.commit,True)
    if a.controller or a.capture_worker or a.release_observer or a.cleanup:
        if not a.case or not a.token:p.error('Exact owned case/nonce required')
        if a.cleanup:return cleanup(a.case,a.token)
        if a.controller:return controller(a.case,a.token)
        return worker(a.case,a.token,a.release_observer)
    print('DRY_RUN: capture only, separate approval pending; no admission, WSL, GPU or imports; operator hold retained');return 0

if __name__=='__main__':raise SystemExit(main())
