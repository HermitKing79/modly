"""One fixed brick experiment, composed with the unchanged owned supervisor."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import re
import stat
import sys
import time
from types import ModuleType

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=Path(__file__).resolve().parent
PRIVATE=ROOT.parent/'artifacts/amd-brick-pbr-gpu-4f91b52'
DESTINATION=Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-gpu-4f91b52')
FIELD=DESTINATION/'field'
PROBE=DESTINATION/'probe'
WORKER=EVIDENCE/'worker.py'
FROZEN_PATH=ROOT/'docs/compatibility/amd-turret-texture-decode-resume-evidence/decode.py'
FROZEN_SHA='8e6d52b84a4be0aceb57f9b942e6bc83f8998decb931c287f40d2a31263b543d'
GEOMETRY=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output'
GEOMETRY_CONTRACT=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'
GEOMETRY_CONTRACT_SHA256='4676626d1061e6cd7090b560f952a974a584f8ea9e6218378b7902294b93bd4f'
SCRATCH='/home/modly/brick-pbr-gpu-4f91b52'
GIB=1024**3
PRELAUNCH_RAM=37*GIB//2
LIVE_RAM=12*GIB
IO_SECONDS=30
DEADLINES={'native':60,'cpu':120,'gpu':240,'host':480}
REFERENCES={'native':3067854848,'cpu':1143431168}
ACTIVE_PLAN_PATH=PROBE/'execution-plan.json'
ACTIVE_PLAN_SHA=None
HARNESS_COMMIT=None
_frozen=None
_original_admission=None
_profile=None
_bridge=None
_monitor=None
_last_resource=0.0
RESOURCE_SAMPLES=[]
STAGE_ACKS=[]
EMPTY_BYTECODE=PRIVATE/'forbidden-host-bytecode'
sys.dont_write_bytecode=True
sys.pycache_prefix=str(EMPTY_BYTECODE)
if os.path.lexists(EMPTY_BYTECODE):raise RuntimeError('Exclusive nonexistent source-cache prefix required')
sys.path[:0]=[str(ROOT/'scripts/compatibility'),str(ROOT/'experimental/extensions/trellis2-amd')]
from inference_texture import artifact as artifact_module
from inference_texture import profile as profiles, inputs, receipts, runtime


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,row):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(row,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1 or path.stat().st_size>1024**2:
        raise ValueError('Bounded plain JSON required')
    return json.loads(path.read_text(encoding='utf-8'))


def _canonical(path):
    raw=os.fspath(path)
    if (not isinstance(raw,str) or any(part in ('.','..') or part.endswith((' ','.'))
            for part in raw.replace('\\','/').split('/') if part) or raw.count(':')!=1):
        raise ValueError('Canonical approved C: path required')
    path=Path(raw)
    if not path.is_absolute() or path.drive!='C:' or path.as_posix()!=path.resolve().as_posix():
        raise ValueError('Canonical approved C: path required')
    for item in (path,*path.parents):
        if os.path.lexists(item):
            info=item.lstat()
            if (stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&1024
                    or (stat.S_ISREG(info.st_mode) and info.st_nlink!=1)):
                raise ValueError('Aliased execution paths are prohibited')
    return path


def _read_plain(path,maximum):
    path=_canonical(path);before=path.lstat()
    identity=lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_nlink)
    if not stat.S_ISREG(before.st_mode) or before.st_size>maximum:
        raise ValueError('Bounded plain execution input required')
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno());blob=stream.read(maximum+1);after=os.fstat(stream.fileno())
    if (len(blob)!=before.st_size or len(blob)>maximum or identity(before)!=identity(opened)
            or identity(before)!=identity(after) or identity(before)!=identity(path.lstat())):
        raise ValueError('Execution input changed during read')
    return blob


def to_guest(path):
    path=_canonical(path)
    if path.as_posix()==GEOMETRY_CONTRACT.as_posix():
        if hashlib.sha256(_read_plain(path,128*1024)).hexdigest()!=GEOMETRY_CONTRACT_SHA256:
            raise ValueError('Original geometry contract file identity differs')
    elif not any(path.is_relative_to(root) for root in (ROOT,PRIVATE,DESTINATION)):
        raise ValueError('Canonical approved C: path required')
    return '/mnt/c/'+path.as_posix()[3:]


def selected_profile():
    global _profile
    if _profile is None:
        _profile=profiles.authenticate(profiles.NAME,profiles.contracts.load(GEOMETRY/'run-contract.json').document())
    return profiles.require_authenticated(_profile)


def load_frozen():
    global _frozen,_original_admission
    if _frozen is None:
        blob=FROZEN_PATH.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=FROZEN_SHA:raise ValueError('Frozen supervisor identity differs')
        spec=importlib.util.spec_from_file_location('_brick_pbr_frozen_supervisor',FROZEN_PATH)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        exec(compile(blob,str(FROZEN_PATH),'exec',dont_inherit=True),module.__dict__)
        if Path(module.base.SCRIPT).resolve()!=ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py':
            raise ValueError('Shared helper resolution lost its original anchor')
        module.EV=EVIDENCE  # Even standalone preflight journals only this fresh experiment.
        _frozen=module;_original_admission=module.host_admission
    return _frozen


def require_source_cache():
    if (sys.pycache_prefix!=str(EMPTY_BYTECODE) or not sys.dont_write_bytecode
            or os.path.lexists(EMPTY_BYTECODE)):
        raise RuntimeError('Project imports must bypass all preexisting bytecode caches')


def assess_remaining(stage,available,*,recent=None):
    if stage not in REFERENCES or type(available) is not int:raise ValueError('Exact stage and actual RAM required')
    remaining=REFERENCES[stage]
    values=[row['available_ram_bytes'] for row in (recent or [])[-6:]]
    if any(type(value) is not int for value in values):raise ValueError('Measured RAM history required')
    downward=max([0,*[a-b for a,b in zip(values,values[1:])]])
    return dict(admitted=available-remaining-downward>=LIVE_RAM,
        source='closest-historical-whole-host-reference-not-bound',
        reference_remaining_bytes=remaining,remaining_reference_bytes=remaining,
        remaining_after_reference_bytes=available-remaining,live_floor_bytes=LIVE_RAM,
        recent_RAM_bytes=values,recent_maximum_downward_step_bytes=downward,
        remaining_after_reference_and_recent_step_bytes=available-remaining-downward,
        limitation='Empirical reference is not an upper bound; actual live monitoring remains mandatory.')


def validate_stage_ready(row,stage,plan,profile,plan_sha):
    expected=dict(schema='modly.amd.brick-pbr-stage-ready.v1',stage=stage,
        marker_sha256=plan['marker_sha256'],attestation_sha256=plan['attestation_sha256'],
        profile_sha256=profile.sha256,run_contract_sha256=profile.run_contract_sha256,plan_sha256=plan_sha)
    if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Stage identity differs')
    return True


def sample_process_memory():
    import psutil
    own=psutil.Process(os.getpid());rows=[];errors=[]
    try:processes=[own,*own.children(recursive=True)]
    except psutil.Error as exc:processes=[own];errors.append(type(exc).__name__)
    for proc in processes:
        try:
            memory=proc.memory_info()
            rows.append(dict(pid=proc.pid,ppid=proc.ppid(),name=proc.name(),rss_bytes=memory.rss,
                             private_bytes=getattr(memory,'private',None)))
        except psutil.NoSuchProcess:pass
        except psutil.Error as exc:errors.append(dict(pid=proc.pid,error=type(exc).__name__))
    vm=[]
    for proc in psutil.process_iter(['pid','name']):
        if str(proc.info['name']).lower() in ('vmmem','vmmemwsl'):
            try:vm.append(dict(pid=proc.pid,name=proc.info['name'],rss_bytes=proc.memory_info().rss))
            except psutil.Error as exc:vm.append(dict(pid=proc.pid,error=type(exc).__name__))
    return dict(windows_owned_host_and_descendants=rows,
        working_set_sum_bytes=sum(row['rss_bytes'] for row in rows),
        private_sum_bytes=sum(row['private_bytes'] or 0 for row in rows),
        vmmem_read_only_samples=vm,errors=errors,
        scope='Windows host and descendants only; shared pages may be counted repeatedly; Linux guest RSS is separate.')


def host_admission(stage='read-only-preflight'):
    global _last_resource
    load_frozen()
    row=_original_admission(stage)
    free=__import__('shutil').disk_usage(FIELD).free
    writable=os.access(FIELD,os.W_OK)
    if free<2*GIB or not writable:
        _frozen.base.journal(EVIDENCE,'destination-admission-failed',free_bytes=free,writable=writable)
        raise RuntimeError('Original destination 2GiB/write-access admission failed')
    row=dict(row,destination_free_bytes=free,destination_writable=writable,
             monotonic_seconds=time.monotonic(),process_memory=sample_process_memory())
    RESOURCE_SAMPLES.append(row);_last_resource=time.monotonic()
    _frozen.base.journal(EVIDENCE,'host-process-memory',**row)
    return row


def candidate_files():
    expected=read(EVIDENCE/'candidate-identities.json')
    required={path.relative_to(ROOT).as_posix() for path in (
        WORKER,Path(__file__).resolve(),FROZEN_PATH,*
        (ROOT/'experimental/extensions/trellis2-amd/inference_texture').glob('*.py'))}
    if not required.issubset(expected):raise ValueError('Candidate closure incomplete')
    for name,sha in expected.items():
        path=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or ':' in name:
            raise ValueError('Canonical repository-relative candidate required')
        to_guest(path)
        if digest(path)!=sha:raise ValueError('Candidate identity differs: '+name)
    return expected


def committed_blob_matches(name,blob,working):
    if blob==working:return True
    normalized=working.replace(b'\r\n',b'\n')
    if b'\r\n' not in working or b'\r' in normalized or blob!=normalized:return False
    eol=subprocess.check_output(['git','ls-files','--eol','--',name],cwd=ROOT,text=True).split()
    attrs=subprocess.check_output(['git','check-attr','-z','text','filter','working-tree-encoding','ident','--',name],cwd=ROOT)
    fields=attrs.decode().split('\0')[:-1]
    if len(fields)!=12 or any(fields[i]!=name for i in (0,3,6,9)):return False
    values={fields[i+1]:fields[i+2] for i in (0,3,6,9)}
    return (eol[:2]==['i/lf','w/crlf'] and values.get('text') in ('unspecified','auto','set')
        and all(values.get(key)=='unspecified' for key in ('filter','working-tree-encoding','ident'))
        and subprocess.check_output(['git','config','--get','core.autocrlf'],cwd=ROOT,text=True).strip()=='true')


def validate_committed(commit):
    if (type(commit) is not str or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit):
        raise ValueError('Exact published harness revision required')
    names=[*candidate_files(),(EVIDENCE/'candidate-identities.json').relative_to(ROOT).as_posix()]
    result=subprocess.run(['git','cat-file','--batch'],input=''.join(commit+':'+name+'\n' for name in names).encode(),
                          cwd=ROOT,stdout=subprocess.PIPE,check=True,timeout=15).stdout
    cursor=0
    for name in names:
        end=result.index(b'\n',cursor);header=result[cursor:end].split();cursor=end+1
        if len(header)!=3 or header[1]!=b'blob':raise ValueError('Candidate missing from commit: '+name)
        size=int(header[2]);blob=result[cursor:cursor+size];cursor+=size+1
        if not committed_blob_matches(name,blob,(ROOT/name).read_bytes()):raise ValueError('Committed bytes differ: '+name)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no','--',*names],cwd=ROOT,text=True):
        raise ValueError('Execution candidate has local edits')


def get_bridge():
    global _bridge
    if _bridge is None:
        plan=read(ACTIVE_PLAN_PATH)
        _bridge=receipts.ReceiptBridge(FIELD,PROBE,PROBE/'neural-attempt.marker',plan['marker_sha256'],
            profile=selected_profile(),attestation_path=PROBE/'ownership.json',
            attestation_sha256=plan['attestation_sha256'])
    return _bridge


def worker_command(script_linux,output_linux):
    base=load_frozen().base
    if script_linux!=to_guest(WORKER) or output_linux!=to_guest(EVIDENCE/'run/guest'):
        raise ValueError('Exact worker/output routing required')
    if type(ACTIVE_PLAN_SHA) is not str or len(ACTIVE_PLAN_SHA)!=64:raise ValueError('Exact plan hash required')
    command=base.worker_command(script_linux,output_linux)
    command[command.index('90s')]='450s'
    return command+['--plan',to_guest(ACTIVE_PLAN_PATH),'--plan-sha256',ACTIVE_PLAN_SHA]


def preflight():
    require_source_cache()
    if os.environ.get('WORKSPACE_DIR')!=str(ROOT):raise ValueError('Explicit WORKSPACE_DIR must equal checkout')
    expected=candidate_files();p=selected_profile()
    prepared=inputs.prepare(p);metadata=prepared['metadata'];del prepared;gc.collect()
    admission=host_admission()
    frozen=load_frozen()
    checkpoint=frozen.checkpoint_record(frozen.PREFIX_HOST.with_suffix('.safetensors'))
    if digest(frozen.PREFIX_HOST.with_suffix('.json'))!=profiles.CONFIG_SHA256:raise ValueError('Checkpoint config differs')
    proof=read(PROBE/'ownership.json')
    if proof.get('mode')!='production' or proof.get('profile_sha256')!=p.sha256:raise ValueError('Production ownership proof differs')
    owner=receipts._owner(FIELD,PROBE/'ownership.json',digest(PROBE/'ownership.json'))
    owner.host_verify(FIELD);owner.host_verify(PROBE)
    if any(FIELD.iterdir()):raise ValueError('Fresh field leaf must be empty')
    receipts._check_directory_resources(FIELD)
    ready=read(PRIVATE/'final-readiness.json')
    if ready.get('classification')!='PASS' or len(ready.get('readings',[]))!=3 or any(
            row['available_windows_ram_bytes']<PRELAUNCH_RAM for row in ready['readings']):
        raise ValueError('Fresh three-reading original RAM admission required')
    if any(path.exists() for path in (PROBE/'neural-attempt.marker',EVIDENCE/'neural-attempt.marker',EVIDENCE/'run',ACTIVE_PLAN_PATH)):
        raise ValueError('Exclusive attempt already reserved')
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        helper=frozen.base.load_sibling(name)
        if Path(helper.__file__).resolve()!=ROOT/'scripts/compatibility'/(name+'.py'):
            raise ValueError('Original helper path differs')
    return dict(profile_sha256=p.sha256,checkpoint=checkpoint,inputs=metadata,
                source_sha256=expected,host_admission=admission,weights_loaded=False,GPU_forward_calls=0)


def serialized_publication(value):
    # Match the unchanged genuine publisher, including Windows text newlines.
    receipts.artifact._canonical_field()._json_bytes(value,receipts.MAX_RECEIPT)
    buffer=io.BytesIO()
    with io.TextIOWrapper(buffer,encoding='utf-8',newline=None) as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();blob=buffer.getvalue()
    if len(blob)>receipts.MAX_RECEIPT:raise ValueError('Serialized execution record is oversized')
    return blob


def plan_worker(expected_sha256):
    blob=_read_plain(WORKER,128*1024)
    if hashlib.sha256(blob).hexdigest()!=expected_sha256:
        raise ValueError('Worker source changed before plan qualification')
    spec=importlib.util.spec_from_file_location('_brick_pbr_plan_qualification_worker',WORKER)
    module=importlib.util.module_from_spec(spec)
    exec(compile(blob,str(WORKER),'exec',dont_inherit=True),module.__dict__)
    return module


def prepare_attempt(output):
    if (_canonical(output)!=EVIDENCE/'run' or type(HARNESS_COMMIT) is not str
            or re.fullmatch('[0-9a-f]{40}',HARNESS_COMMIT) is None):
        raise ValueError('Exact fresh output/commit required')
    if (ACTIVE_PLAN_PATH!=PROBE/'execution-plan.json' or FIELD!=DESTINATION/'field'
            or PROBE!=DESTINATION/'probe'):
        raise ValueError('Exact execution control/output leaves required')
    reserved=(PROBE/'neural-attempt.marker',ACTIVE_PLAN_PATH,EVIDENCE/'neural-attempt.marker',Path(output))
    if any(os.path.lexists(path) or os.path.lexists(path.with_name(path.name+'.publishing')) for path in reserved):
        raise FileExistsError('Exclusive attempt path already exists')
    sources=candidate_files()
    plan_worker(sources[WORKER.relative_to(ROOT).as_posix()]).verify_manifest(
        EVIDENCE/'candidate-identities.json',digest(EVIDENCE/'candidate-identities.json'))
    if WORKER.parent!=EVIDENCE:raise ValueError('Worker and execution manifest directories differ')
    receipts._leaves(FIELD,PROBE)
    p=selected_profile();proof=PROBE/'ownership.json'
    prepared=inputs.prepare(p);del prepared;gc.collect()
    proof_record=read(proof)
    if (proof_record.get('mode')!='production' or proof_record.get('profile_sha256')!=p.sha256
            or proof_record.get('run_contract_sha256')!=p.run_contract_sha256):
        raise ValueError('Production ownership proof differs')
    # Validate every path used by the launcher, including prospective paths.
    for path in (WORKER,EVIDENCE/'run/guest',ACTIVE_PLAN_PATH,EVIDENCE/'neural-attempt.marker'):
        to_guest(path)
    marker=dict(schema='modly.amd.brick-pbr-receipt-marker.v1',mode='production',
                profile_sha256=p.sha256,artifact_root=FIELD.as_posix(),harness_commit=HARNESS_COMMIT)
    marker_sha=hashlib.sha256(serialized_publication(marker)).hexdigest()
    plan=dict(schema='modly.amd.brick-pbr-execution-plan.v1',harness_commit=HARNESS_COMMIT,
        manifest_path=to_guest(EVIDENCE/'candidate-identities.json'),manifest_sha256=digest(EVIDENCE/'candidate-identities.json'),
        scratch=SCRATCH,field=to_guest(FIELD),probe=to_guest(PROBE),marker_path=to_guest(PROBE/'neural-attempt.marker'),
        marker_sha256=marker_sha,attestation_path=to_guest(proof),attestation_sha256=digest(proof),
        geometry_contract_path=to_guest(GEOMETRY/'run-contract.json'),geometry_contract_file_sha256=digest(GEOMETRY/'run-contract.json'))
    if (plan['scratch']!='/home/modly/brick-pbr-gpu-4f91b52'
            or plan['geometry_contract_file_sha256']!=GEOMETRY_CONTRACT_SHA256
            or any(type(plan[key]) is not str or re.fullmatch('[0-9a-f]{64}',plan[key]) is None
                   for key in ('manifest_sha256','marker_sha256','attestation_sha256','geometry_contract_file_sha256'))):
        raise ValueError('Exact execution plan field required')
    plan_sha=hashlib.sha256(serialized_publication(plan)).hexdigest()
    evidence_marker=dict(marker,plan_sha256=plan_sha,marker_sha256=marker_sha)
    serialized_publication(evidence_marker)
    return marker,plan,evidence_marker,marker_sha,plan_sha


def reserve_attempt(output):
    global ACTIVE_PLAN_SHA
    marker,plan,evidence_marker,marker_sha,plan_sha=prepare_attempt(output)
    receipts._publish(PROBE/'neural-attempt.marker',marker)
    if digest(PROBE/'neural-attempt.marker')!=marker_sha:raise ValueError('Published marker serialization differs')
    receipts._publish(ACTIVE_PLAN_PATH,plan)
    if digest(ACTIVE_PLAN_PATH)!=plan_sha:raise ValueError('Published plan serialization differs')
    ACTIVE_PLAN_SHA=plan_sha
    save(EVIDENCE/'neural-attempt.marker',evidence_marker)
    output.mkdir(exist_ok=False)


class AsyncHostReceipt:
    def __init__(self,output,base,*,bridge=None,plan=None,plan_sha=None,monitor=None):
        self.bridge=get_bridge() if bridge is None else bridge
        self.base=base;self.plan=read(ACTIVE_PLAN_PATH) if plan is None else plan
        self.plan_sha=ACTIVE_PLAN_SHA if plan_sha is None else plan_sha
        self.monitor=monitor;self.done=False;self.future=None;self.started=None
        self.host=self.bridge.host_receipt();self.stage_index=0;self.stage_future=None;self.stage_started=None

    def poll_stages(self):
        if self.stage_index>=2:return
        stage=('native','cpu')[self.stage_index];path=self.bridge.receipt_dir/(stage+'-stage-ready.json')
        if not path.exists():return
        if self.stage_started is None:self.stage_started=time.monotonic()
        if time.monotonic()-self.stage_started>=IO_SECONDS:raise TimeoutError('30-second host stage acknowledgement')
        monitor=self.monitor or _monitor
        if monitor is None or not monitor.healthy_since(self.stage_started):return
        if self.stage_future is None:
            def qualify():
                self.bridge._context()
                row=receipts._decode(path)
                validate_stage_ready(row,stage,self.plan,self.bridge.profile,self.plan_sha)
                self.bridge.owner.host_verify(self.bridge.receipt_dir)
                return row
            self.stage_future=self.base.in_thread(qualify)
        if not self.stage_future.done():return
        self.stage_future.result()
        actual=host_admission(stage+'-host-release-barrier')
        assessment=assess_remaining(stage,actual['available_ram_bytes'],recent=RESOURCE_SAMPLES)
        if not assessment['admitted']:
            self.base.journal(EVIDENCE,'host-stage-reserve-refused',stage=stage,actual=actual,assessment=assessment)
            raise RuntimeError('Remaining host-memory reference would approach live floor')
        ack=dict(schema='modly.amd.brick-pbr-stage-go.v1',stage=stage,status='GO',healthy_observation=True,
            marker_sha256=self.plan['marker_sha256'],attestation_sha256=self.plan['attestation_sha256'],
            profile_sha256=self.bridge.profile.sha256,run_contract_sha256=self.bridge.profile.run_contract_sha256,
            plan_sha256=self.plan_sha,host_admission=actual,reserve_assessment=assessment)
        receipts._publish(self.bridge.receipt_dir/(stage+'-stage-go.json'),ack)
        STAGE_ACKS.append(ack);self.stage_index+=1;self.stage_future=None;self.stage_started=None

    def poll(self):
        self.poll_stages()
        if self.done:return
        path=self.bridge.receipt_dir/'artifact-ready.json'
        if self.future is None and path.exists():
            self.started=time.monotonic();self.future=self.base.in_thread(self.host.poll)
        if self.future is not None:
            if time.monotonic()-self.started>=IO_SECONDS:raise TimeoutError('30-second asynchronous host field verification')
            if self.future.done():
                if self.future.result() is not True:raise RuntimeError('Independent host field acknowledgement incomplete')
                self.done=True


def verify_after_exit(_directory):
    bridge=get_bridge();before=bridge.owner.host_verify(bridge.root,files=True)
    result=artifact_module.verify(bridge.root/artifact_module.ARTIFACT_NAME,profile=bridge.profile)
    after=bridge.owner.host_verify(bridge.root,files=True)
    if before!=after or result.get('verified') is not True:raise RuntimeError('Post-exit field/ownership readback failed')
    return dict(status='PASS',artifact_sha256=result['metadata']['artifact_sha256'],
                artifact_bytes=result['metadata']['artifact_bytes'],verified=True,independent_of_worker=True)


@contextmanager
def binding(name,value):
    old=sys.modules.get(name);sys.modules[name]=value
    try:yield
    finally:
        if old is None:sys.modules.pop(name,None)
        else:sys.modules[name]=old


def run(commit):
    global HARNESS_COMMIT,_monitor
    HARNESS_COMMIT=commit;validate_committed(commit);frozen=load_frozen()
    frozen.EV=EVIDENCE;frozen.SCRIPT=WORKER
    frozen.preflight=preflight;frozen.reserve_attempt=reserve_attempt
    frozen.worker_command=worker_command;frozen.host_admission=host_admission
    frozen.validate_qualified=lambda record:runtime.validate_result(record,profile=selected_profile())
    original_monitor=frozen.base.HealthMonitor
    class TrackingMonitor(original_monitor):
        def __init__(self,*args,**kwargs):
            global _monitor
            super().__init__(*args,**kwargs);_monitor=self
        def poll(self,now):
            super().poll(now)
            if now-_last_resource>=1:host_admission('observer-or-cleanup')
    frozen.base.HealthMonitor=TrackingMonitor
    shim=ModuleType('artifact_io');shim.HostReceipt=AsyncHostReceipt;shim.verify_on_host=verify_after_exit
    try:
        with binding('artifact_io',shim):code=frozen.supervise(EVIDENCE/'run')
    finally:frozen.base.HealthMonitor=original_monitor
    record=read(EVIDENCE/'run/results.json')
    forward=record.get('worker',{}).get('forward_calls',0)
    status='PASS' if code==0 and record.get('classification')=='QUALIFIED' else ('NO-GO' if forward else 'BLOCKED')
    save(EVIDENCE/'measured-host-summary.json',dict(classification=status,host=record,
        resource_samples=RESOURCE_SAMPLES,stage_acknowledgements=STAGE_ACKS,
        host_minimum_available_RAM_bytes=min((v['available_ram_bytes'] for v in RESOURCE_SAMPLES),default=None),
        windows_host_tree_peak_sampled_working_set_sum_bytes=max((v['process_memory']['working_set_sum_bytes'] for v in RESOURCE_SAMPLES),default=None),
        natural_exit=record.get('windows_job_cleanup',{}).get('forced_job_cleanup') is False))
    print('BRICK_GUIDED_PBR_RESULT='+status,flush=True)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--harness-commit',required=True)
    args=parser.parse_args()
    try:return run(args.harness_commit)
    except BaseException as exc:
        phases=EVIDENCE/'run/guest/phases.jsonl'
        entered=phases.exists() and 'texture-forward-entry' in phases.read_text(encoding='utf-8')
        status='NO-GO' if entered else 'BLOCKED'
        save(EVIDENCE/'unhandled-host-failure.json',dict(classification=status,error_type=type(exc).__name__,
            error=str(exc),traceback=__import__('traceback').format_exc(),resource_samples=RESOURCE_SAMPLES,
            genuine_forward_entry_observed=bool(entered),retry=False))
        print('BRICK_GUIDED_PBR_RESULT='+status,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
