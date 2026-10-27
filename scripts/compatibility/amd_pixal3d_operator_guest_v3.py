"""Operator-only controller. Existing environment read-only; no provisioning route."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c
from amd_pixal3d_guest_v2 import baseline,child_environment
from amd_pixal3d_owned_proc_v3 import owned_members,process_rows,signal_verified,ProcFS
from amd_pixal3d_memory_v2 import linux_memory

def validate_case(case,token):
    import re
    if os.name!='posix' or not re.fullmatch('[a-f0-9]{32}',token):raise RuntimeError('Linux/exact nonce required')
    case=c.plain(case);case.relative_to(c.plain(c.ROOT.parent/'artifacts'))
    if not case.name.startswith('amd-pixal3d-operators-v3-') or not case.name.endswith(token):raise ValueError('Exact v3 evidence path required')
    return case

def snapshot_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def harvest_worker(record,leaf):
    path=leaf/'child-result.json'
    if not path.exists():return None
    if path.stat().st_size>8*1024**2:raise RuntimeError('Bounded child result required')
    data=json.loads(path.read_text())
    if data.get('classification') not in ('PASS','FAIL'):raise RuntimeError('Invalid child classification')
    record['worker_result']=data
    for key in ('operators','dependency_overlay','worker_failure','audit_denial_records','memory_peak','memory_release','Linux_worker_peak_RSS_bytes'):
        if key in data:record[key]=data[key]
    if 'error' in data:record['worker_error']=data['error']
    if data['classification']!='PASS':record['classification']='FAIL'
    return data

def reap_child(record,child):
    if child is None:return
    try:
        if child.poll() is None:
            record['child_forced_cleanup']=True;record['classification']='FAIL'
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        else:child.wait(timeout=5)
        record['child_reaped']=True;record['child_exit_code']=child.returncode
    except BaseException as exc:record.update(classification='FAIL',child_cleanup_error=str(exc),child_exit_unverified=True)
    try:
        if child.stdin and not child.stdin.closed:child.stdin.close()
    except BaseException as exc:record.update(classification='FAIL',child_stdin_close_error=str(exc))

def controller(case,token):
    case=validate_case(case,token);cfg=c.lock()
    if cfg['operator_execution_hold']:raise RuntimeError('NO_GO: operator execution held pending identified audit event/path')
    leaf=c.plain(case/'operators');root=c.plain(cfg['guest_root']);temporary=c.plain(cfg['temporary_guest_root'])
    if os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token:raise RuntimeError('Exec-time nonce required')
    if os.getpgrp()!=os.getpid():os.setsid()
    if os.getpgrp()!=os.getpid():raise RuntimeError('Own group required')
    ProcFS().require_pidfds()
    def observe_proc(row):c.journal(leaf/'ownership-observations.jsonl','proc-check',**row)
    own=next(r for r in process_rows(os.getpgrp(),report=observe_proc) if r['pid']==os.getpid())
    if own['token']!=token:raise RuntimeError('Controller nonce unreadable or mismatched')
    owner=dict(pid=own['pid'],pgid=own['pgid'],start_ticks=own['start_ticks'],token=token,stage='operators')
    c.save(leaf/'ownership.json',owner)
    sys.addaudithook(c.v2.cache_policy(root))
    sys.addaudithook(c.v2.cache_policy(c.overlay.archive_path()))
    sys.addaudithook(c.v2.cache_policy(Path('/mnt/d/AI/Modly-Pixal3D-Proof-v1')))
    record=dict(classification='FAIL',stage='operators',child_started=False);before=None;child=None;began=time.monotonic();peak=0;last=0
    def resources(phase):
        nonlocal peak,last
        if time.monotonic()-last<.25:return
        last=time.monotonic();rows=process_rows(owner['pgid'],report=observe_proc);owned_members(owner,rows);resident=sum(r['RSS_bytes'] for r in rows);peak=max(peak,resident)
        free=__import__('shutil').disk_usage('/home/modly').free
        c.journal(leaf/'resources.jsonl',phase,RSS_bytes=resident,members=rows,guest_free_disk_bytes=free)
        if resident>6*c.GIB or free<cfg['guest_disk_floor_bytes']:raise RuntimeError('Original RSS/guest disk stop failed')
        allowed=120 if phase=='read-only-authentication' else cfg['operator_seconds']
        if time.monotonic()-began>=allowed:raise TimeoutError('Original CPU/operator overall deadline')
    try:
        if temporary.exists():raise FileExistsError('Exclusive v3 temporary path already used; no retry')
        before=c.environment_snapshot(root);c.save(leaf/'environment-before.json',before)
        try:
            auth=c.authenticate_environment(cfg,lambda:resources('read-only-authentication'),
                lambda row:c.journal(leaf/'wheel-payload-authentication.jsonl','authenticated-archive',**row))
            c.save(leaf/'environment-authentication.json',auth)
        except BaseException as exc:
            c.save(leaf/'environment-authentication.json',dict(status='FAIL',error=type(exc).__name__+': '+str(exc)));raise
        ready_memory=linux_memory();c.save(leaf/'linux-memory-ready.json',dict(utc=c.now(),**ready_memory))
        c.save(leaf/'ready.json',dict(owner=owner,existing_environment_authenticated=True,temporary_root_absent=True,linux_memory=ready_memory))
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('No exact host GO/EOF; no temporary environment or GPU imports')
        # Only new scratch directories are created. The v2 prefix remains read-only.
        temporary.mkdir(exist_ok=False)
        for name in ('home','tmp','hip-cache','comgr-cache','disabled-torch-extensions','disabled-triton-cache'):(temporary/name).mkdir()
        c.save(temporary/'owner.json',dict(token=token,case=str(case),immutable_environment=str(root)))
        command=[str(root/'.venv/bin/python'),'-B','-I',str(Path(__file__).with_name('amd_pixal3d_operator_ops_v3.py')),'--run','--case',str(case),'--token',token]
        c.journal(leaf/'phases.jsonl','GO',command=command,provisioning=False)
        child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,env=child_environment(temporary,token))
        child.stdin.write(b'GO\n');child.stdin.close();record['child_started']=True
        while child.poll() is None:
            resources('operators')
            phase_file=leaf/'operator-phases.jsonl'
            if phase_file.exists():
                with phase_file.open('rb') as stream:stream.seek(max(0,phase_file.stat().st_size-65536));complete=stream.read().split(b'\n')[:-1]
                if complete:
                    phase=json.loads(complete[-1]);allowed=120 if phase['kind']=='CPU' else 180
                    if time.monotonic()-phase['monotonic_start']>allowed:raise TimeoutError('Original CPU/GPU phase deadline')
            time.sleep(.25)
        record['child_exit_code']=child.returncode;data=harvest_worker(record,leaf)
        if child.returncode or not data or data['classification']!='PASS':raise RuntimeError('Operator child failed; no retry')
        record['classification']='PASS'
    except BaseException as exc:record['error']=type(exc).__name__+': '+str(exc)
    finally:
        reap_child(record,child)
        try:harvest_worker(record,leaf)
        except BaseException as exc:record.update(classification='FAIL',worker_evidence_error=str(exc))
        try:
            after=c.environment_snapshot(root);c.save(leaf/'environment-after.json',after)
            record['read_only_environment_preserved']=before is not None and before==after
            record['environment_snapshot_sha256']=dict(before=snapshot_digest(before),after=snapshot_digest(after))
            if before is None or before!=after:raise RuntimeError('Existing environment identity changed; no restoration/repair attempted')
            record['dependency_overlay_after']=c.overlay.authenticate(lambda:resources('operators'))
            record['linux_memory_at_controller_exit']=dict(utc=c.now(),**linux_memory())
        except BaseException as exc:record.update(classification='FAIL',environment_or_exit_error=str(exc))
        try:
            rows=process_rows(owner['pgid'],report=observe_proc);owned_members(owner,rows)
            members=[r for r in rows if r['pid']!=os.getpid()]
            if members:record.update(classification='FAIL',remaining_members=members)
        except BaseException as exc:record.update(classification='FAIL',ownership_exit_error=str(exc))
        record.update(seconds=time.monotonic()-began,max_sampled_Linux_process_tree_RSS_bytes=peak,RSS_polling_interval_seconds=.25,
            natural_child_exit=(child is None or record.get('child_reaped',False)) and not record.get('child_forced_cleanup',False),provisioning=False,downloads=False)
        c.save(leaf/'result.json',record)
    return 0 if record['classification']=='PASS' else 2

def cleanup(case,token):
    case=validate_case(case,token);leaf=case/'operators';owner=json.loads((leaf/'ownership.json').read_text())
    if owner['token']!=token or owner['stage']!='operators' or owner['pid']!=owner['pgid']:raise RuntimeError('Foreign ownership receipt')
    members=process_rows(owner['pgid']);owned_members(owner,members);forced=bool(members)
    if members:
        signal_verified(owner,signal.SIGTERM);began=time.monotonic()
        while process_rows(owner['pgid']) and time.monotonic()-began<5:time.sleep(.05)
        members=process_rows(owner['pgid']);owned_members(owner,members)
        if members:
            signal_verified(owner,signal.SIGKILL);began=time.monotonic()
            while process_rows(owner['pgid']) and time.monotonic()-began<5:time.sleep(.05)
    if process_rows(owner['pgid']):raise RuntimeError('Owned group exit unverified; no distro termination')
    c.save(leaf/'cleanup.json',dict(group_empty=True,natural_exit=not forced,forced_owned_group_cleanup=forced,no_distro_termination=True))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);route=parser.add_mutually_exclusive_group();route.add_argument('--controller',action='store_true');route.add_argument('--cleanup',action='store_true');parser.add_argument('--case');parser.add_argument('--token');args=parser.parse_args(argv)
    if not(args.controller or args.cleanup):print('DRY_RUN: operator-only guest; no work');return 0
    if not args.case or not args.token:parser.error('Exact case and token required')
    return cleanup(args.case,args.token) if args.cleanup else controller(args.case,args.token)

if __name__=='__main__':raise SystemExit(main())
