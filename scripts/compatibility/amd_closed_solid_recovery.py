"""RAM-SENSITIVE decode-only recovery; default inert, separate approval required."""
import argparse,contextlib,hashlib,json,os,re,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from types import ModuleType
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_handoff as original
import amd_zombie_neural_handoff as guard
ROOT=original.ROOT;UPSTREAM='853a9a28da970beeb53685f967de059bb90c7364'
SHAPE=original.CASE/'shape'
RECOVERY=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/closed-solid-box-decode-recovery-v1'
DECODE=RECOVERY/'decode';PRIVATE=ROOT.parent/'artifacts/amd-closed-solid-decode-recovery-v1'
RAW=ROOT.parent/'artifacts/amd-closed-solid-raw-recovery-v1'
RUN_ID='run-closed-solid-box-decode-recovery-v1'
FORWARDS=dict(dino=0,occupancy_sampling=0,occupancy_decode=0,shape=0,geometry_decode=1)
PREP=ROOT.parent/'artifacts/amd-closed-solid-decode-preparation-20261030-v1'
AUTH=PREP/'authority.json';AUTH_SHA='9c09a7ad4ad25f96bb7886c0228ccfe308f8892947f4bcba301598481a4422a8'
FILES=['scripts/compatibility/'+n for n in ('amd_closed_solid_recovery.py','amd_closed_solid_recovery_cpu.py','Invoke-ClosedSolidDecodeRecovery.ps1','test_amd_closed_solid_recovery.py')]
CRITICAL=list(dict.fromkeys(original.CRITICAL+FILES+['scripts/compatibility/'+n for n in ('amd_zombie_neural_handoff.py','amd_wsl_cold_backup.py','amd_wsl_cold_backup_inspect.ps1','amd_wsl_resize_operator.py','amd_wsl_resize_readiness.py','amd_wsl_post_resize_geometry.py')]))
digest=original.digest;plain=original.plain;save=original.save

def read_authority():
    if digest(plain(AUTH))!=AUTH_SHA:raise ValueError('Private recovery authority differs')
    return json.loads(AUTH.read_text())

def zero_forward_failure(row):
    if (row.get('forward_calls')!=0 or row.get('native_import')!='NOT_ATTEMPTED' or
        'CPU/disk admission refused' not in row.get('error','') or
        row['admission']['guest_disk_requirement']!=5*1024**3 or row['admission']['guest_disk_free']>=5*1024**3):
        raise ValueError('Original zero-forward pre-import disk refusal unverified')

def verify_map(root,rows,health=None):
    last=0
    for name,row in rows.items():
        if health and time.monotonic()-last>=1:health();last=time.monotonic()
        path=plain(root/name);before=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']) or digest(path)!=row['sha256']:
            raise ValueError('Preserved source identity/hash differs: '+name)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Source changed during verification')

def authority(health=None):
    a=read_authority();original.authority()
    manifest=plain(Path(a['input_manifest_path']))
    if digest(manifest)!=a['input_manifest_sha256']:raise ValueError('Original input manifest differs')
    verify_map(ROOT.parent,json.loads(manifest.read_text()),health)
    verify_map(ROOT.parent,a['support_files'],health)
    verify_map(ROOT.parent/'artifacts/amd-wsl-post-resize-capacity-v2',a['capacity_files'],health)
    final=json.loads((ROOT.parent/'artifacts/amd-wsl-post-resize-capacity-v2/final-result.json').read_text())
    linux=json.loads((ROOT.parent/'artifacts/amd-wsl-post-resize-capacity-v2/linux-capacity.json').read_text())
    from amd_wsl_post_resize_geometry import assess
    if (final['classification']!='PASS_SPACE_GATE' or final['measurement']!='PASS' or final['faults'] or not final['handles_closed'] or final['minimum_available_bytes']<22*1024**3 or final['commit']!='de8b72fc73c42ece44c45d2aab0511018e516433'
        or assess(linux)!=final['assessment'] or linux['available_bytes']<5*1024**3):raise ValueError('Authenticated capacity gate missing')
    from amd_wsl_resize_readiness import verify_backup
    cold=ROOT.parent/'artifacts/amd-wsl-cold-backup-v2'
    load=lambda name:json.loads((cold/name).read_text())
    verify_backup(load('final-result.json'),load('worker-result.json'),load('destination-hash-stdout.txt'),
        [load(n) for n in ('inventory-process.json','backup-worker-process.json','final-inventory-process.json')],
        json.loads((ROOT.parent/'artifacts/amd-wsl-resize-64g-v1/cold-identities.json').read_text()))
    zero_forward_failure(json.loads((original.CASE/'decode/run/guest/results.json').read_text()))
    return a

def require_fresh():
    for path in (PRIVATE,RECOVERY,RAW):
        plain(path)
        if path.exists():raise FileExistsError('One-shot recovery destination consumed: '+str(path))

def verify_publication(commit):
    original.verify_publication(commit)
    for name in (n for n in CRITICAL if n not in original.CRITICAL):
        if original.legacy.git(['show',commit+':'+name]).replace('\r\n','\n')!=(ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n'):raise ValueError('Recovery source differs: '+name)
    # Original input job and its complete original code closure keep their own identity.
    for name in original.CRITICAL:
        if original.legacy.git(['show',UPSTREAM+':'+name]).replace('\r\n','\n')!=(ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n'):raise ValueError('Original benchmark source changed: '+name)

def partition_files(files,a):
    return ([n for n in files if n not in a['private_repo']],[n for n in files if n in a['private_repo']])

def make_runtime(commit):
    d=original.b.load('_closed_solid_decode_only_runtime',Path(__file__).with_name('amd_generation_decode.py'))
    d.STATIC_FILES=list(dict.fromkeys(d.STATIC_FILES+CRITICAL))
    class RecoveryRuntime(d.Runtime):
        def preflight(self):
            authority(host_health)
            import amd_sparse_decoder_forward as sup
            proxy=ModuleType('amd_sparse_decoder_forward');proxy.__dict__.update(vars(sup))
            def verified(request,names=None,*,timeout_seconds=5):
                if request!=commit:raise ValueError('Foreign recovery execution commit')
                a=read_authority();code,private=partition_files(sup.COMMITTED_FILES if names is None else names,a)
                result=sup.verify_committed_files(request,list(dict.fromkeys(code+CRITICAL)),timeout_seconds=timeout_seconds)
                for name in private:
                    actual=digest(plain(ROOT/name))
                    if actual!=a['private_repo'][name]:raise ValueError('Private predecessor artifact differs')
                    result[name]=actual
                return result
            proxy.verify_committed_files=verified
            with d.module_binding('amd_sparse_decoder_forward',proxy):result=super().preflight()
            return dict(result,source_harness_commit=UPSTREAM,recovery_harness_commit=commit)
        def reserve_attempt(self,output):
            if self.evidence!=DECODE:raise ValueError('Exact recovery decode path required')
            return guard.reserve_guarded(self,super().reserve_attempt,output,PRIVATE/'final-pre-marker-admission.json')
    runtime=RecoveryRuntime(SHAPE,DECODE,commit,RUN_ID,tile_large_convolutions=True,final_support_profile=None)
    if runtime.bundle['count']!=3888 or runtime.bundle['source_job_sha256']!='498948bb64a88e1f869a2c01ed0f38c51346ae347bc1775aa263f1c8a69f7ea5':raise ValueError('Wrong original shape result')
    return runtime

def execute_once(runtime,cpu):
    if runtime.supervise()!=0:raise RuntimeError('Decode failed; no CPU tail or retry')
    if cpu()!=0:raise RuntimeError('CPU tail failed; neural PASS retained, no retry')

def host_health():
    row=original.windows_memory()
    if (row['available_bytes']<12*1024**3 or shutil.disk_usage('D:/').free<50*1024**3
        or shutil.disk_usage(PRIVATE if PRIVATE.exists() else PRIVATE.parent).free<2*1024**3):raise RuntimeError('Original live RAM/disk floors failed')
    return row

def create_private():
    PRIVATE.mkdir(exist_ok=False);(PRIVATE/'tmp').mkdir(exist_ok=False)

@contextlib.contextmanager
def held_backup(cold,path,expected):
    with cold.native_file(path,'verify') as f:
        before=dict(identity=cold.file_identity(f),metadata=cold.metadata(f))
        if before!=expected:raise ValueError('Accepted cold backup differs')
        save(PRIVATE/'backup-before.json',before);primary=None
        try:yield
        except BaseException as exc:primary=type(exc).__name__+': '+str(exc);raise
        finally:
            receipt=dict(operation_error=primary,fresh_whole_disk_hash=False)
            try:
                after=dict(identity=cold.file_identity(f),metadata=cold.metadata(f))
                receipt.update(unchanged=after==before,observed=after)
            except BaseException as exc:
                receipt.update(unchanged=False,verification_error=type(exc).__name__+': '+str(exc))
                save(PRIVATE/'backup-after.json',receipt);raise
            save(PRIVATE/'backup-after.json',receipt)
            if not receipt['unchanged']:raise ValueError('Accepted backup changed; original error: '+str(primary))

@contextlib.contextmanager
def owned_window():
    import amd_wsl_cold_backup as cold
    import amd_wsl_resize_operator as storage
    cold.authenticate_worker_python()
    a=cold.authority();previous=cold.CASE;cold.CASE=PRIVATE
    try:
        owner=cold.Controller();owner.health=host_health
        row=json.loads(owner.child(cold.helper_command('inventory'),'owned-window',30))
        cold.authenticate_inventory(row,a);storage.docker_quiet();save(PRIVATE/'owned-window.json',row)
        # This is the accepted backup only, never the running source disk.
        path=Path('E:/Modly-Cold-Backup-closed-solid-v2/ext4.vhdx')
        expected=json.loads((ROOT.parent/'artifacts/amd-wsl-resize-64g-v1/cold-identities.json').read_text())['backup']
        pins=cold.Pins()
        try:
            pins.add(path.parent)
            with held_backup(cold,path,expected):yield
        finally:pins.close()
    finally:cold.CASE=previous

def run(commit):
    import ctypes
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-administrator Windows required')
    if Path(sys.executable).resolve()!=original.PYTHON.resolve() or digest(original.PYTHON)!=original.PYTHON_SHA:raise ValueError('Qualified Windows Python required')
    if {'amd_generation_decode','amd_generation_shape','amd_generation_occupancy','amd_sparse_decoder_forward'}&set(sys.modules):raise ValueError('Fresh standalone process required')
    host_health();verify_publication(commit);authority(host_health);require_fresh();create_private()
    result=dict(classification='FAIL',harness_commit=commit,source_commit=UPSTREAM,maximum_forwards=FORWARDS,decode='NOT_ATTEMPTED',cpu='NOT_ATTEMPTED',ram_sensitive=True,raw_geometry_decision='UNASSESSED',geometry_approved=False)
    try:
        readings=[]
        def record(row):readings.append(row);save(PRIVATE/f'admission-{len(readings)}.json',row)
        rows=original.fresh_admission(original.windows_memory,time.sleep,record);original.validate_admission(rows,datetime.now(timezone.utc))
        save(PRIVATE/'admission.json',dict(readings=rows,mandatory_bytes=original.FLOOR,preferred_only_bytes=original.PREFERRED))
        if shutil.disk_usage(PRIVATE).free<2*1024**3:raise RuntimeError('Output disk floor')
        os.environ['WORKSPACE_DIR']=str(ROOT);sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        with owned_window():
            runtime=make_runtime(commit)
            def cpu():
                result['decode']='PASS';result['cpu']='STARTED'
                import amd_closed_solid_recovery_cpu as tail
                code=tail.future(commit)
                if code==0:
                    receipt=RAW/'closed-solid-closure.json';decision=json.loads(plain(receipt).read_text())
                    result.update(raw_geometry_decision=decision['recommendation'],raw_quality_receipt_sha256=digest(receipt))
                result['cpu']='PASS' if code==0 else 'FAIL';return code
            result['decode']='STARTED';execute_once(runtime,cpu);authority(host_health)
            result['classification']='DECODE_AND_RAW_CPU_EVIDENCE_COMPLETE_REVIEW_REQUIRED'
    except BaseException as exc:result.update(classification='FAIL',error=type(exc).__name__+': '+str(exc))
    finally:
        result.update(neural_marker_reserved=(DECODE/'decode-attempt-started.json').exists(),decoder_evidence=str(DECODE),raw_cpu_evidence=str(RAW),automatic_retry=False,geometry_repaired=False)
        save(PRIVATE/'recovery-result.pending.json',result);os.rename(PRIVATE/'recovery-result.pending.json',PRIVATE/'recovery-result.json')
        print('CLOSED_SOLID_RECOVERY='+result['classification']+'\nRAW_GEOMETRY_DECISION='+result['raw_geometry_decision'],flush=True)
    return 0 if result['classification']=='DECODE_AND_RAW_CPU_EVIDENCE_COMPLETE_REVIEW_REQUIRED' else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--approved-one-decode',action='store_true');p.add_argument('--harness-commit');a=p.parse_args(argv)
    if not a.run:print(json.dumps(dict(mode='DRY_RUN',ram_sensitive=True,restart_immediately_before_run=True,neural_execution=False,WSL_launched=False,marker_reserved=False,maximum_forwards=FORWARDS,admission_bytes=22*1024**3,preferred_only_bytes=24*1024**3,live_floor_bytes=12*1024**3,readings=3,spacing_seconds=4,run_id=RUN_ID)));return 0
    if not a.approved_one_decode or not re.fullmatch('[a-f0-9]{40}',a.harness_commit or ''):p.error('Execution HOLD: separate one-decode approval and exact commit required')
    return run(a.harness_commit)

if __name__=='__main__':raise SystemExit(main())
