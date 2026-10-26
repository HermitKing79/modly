"""Linux-only owned controller/provisioner. Default is no work. Never a distro killer."""
import argparse,ctypes,hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common as c

def owned_members(owner,members):
    for row in members:
        if row['pgid']!=owner['pgid'] or row.get('token')!=owner['token']:raise RuntimeError('Foreign process-group member; refuse signal')
        if row['pid']==owner['pid'] and row['start_ticks']!=owner['start_ticks']:raise RuntimeError('Reused leader PID; refuse signal')
    return True

def process_rows(pgid):
    rows=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdecimal():continue
        try:
            parts=(item/'stat').read_text().rsplit(')',1)[1].split()
            if int(parts[2])!=pgid or parts[0]=='Z':continue
            env=(item/'environ').read_bytes().split(b'\0');prefix=b'MODLY_PIXAL_PROOF_TOKEN='
            tokens=[value[len(prefix):].decode('ascii') for value in env if value.startswith(prefix)]
            status=(item/'status').read_text();rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0)
            rows.append(dict(pid=int(item.name),pgid=pgid,start_ticks=int(parts[19]),token=tokens[0] if len(tokens)==1 else None,RSS_bytes=rss))
        except (FileNotFoundError,ProcessLookupError):continue
    return rows

def validate_case(case,token):
    if os.name!='posix' or not re.fullmatch('[a-f0-9]{32}',token):raise RuntimeError('Linux and exact nonce required')
    root=c.plain(c.ROOT.parent/'artifacts');case=c.plain(case);case.relative_to(root)
    if not case.name.startswith('amd-pixal3d-standalone-') or not case.name.endswith(token):raise ValueError('Exact owned evidence path required')
    return case

def baseline():
    root=Path('/opt/modly-wsl/venv');files=['pyvenv.cfg','lib/python3.12/site-packages/torch/version.py','lib/python3.12/site-packages/torch/__init__.py']
    return {name:c.digest(c.inside(root,name)) for name in files}

def child_environment(root,token):
    # Do not inherit user Python/pip/Triton overrides or device masks.
    return dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',HOME=str(root/'home'),TMPDIR=str(root/'tmp'),
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PIP_NO_CACHE_DIR='1',
        LD_LIBRARY_PATH='/opt/rocm-7.2.1/lib:/opt/rocm/lib:/usr/lib/wsl/lib',HSA_ENABLE_DXG_DETECTION='1',
        HIP_CACHE_DIR=str(root/'hip-cache'),AMD_COMGR_CACHE_DIR=str(root/'comgr-cache'),
        TORCH_EXTENSIONS_DIR=str(root/'disabled-torch-extensions'),TRITON_CACHE_DIR=str(root/'disabled-triton-cache'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MODLY_PIXAL_PROOF_TOKEN=token)

def release_owned_cache(root,hint=None,flush=None):
    """No global drop_caches: only flush/hint this new isolated file tree."""
    root=c.plain(root);hint=hint or (lambda fd:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED));flush=flush or os.fsync
    count=0;size=0
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file():continue
        path=c.plain(path);path.relative_to(root)
        with path.open('rb') as stream:flush(stream.fileno());hint(stream.fileno())
        count+=1;size+=path.stat().st_size
    return dict(files=count,logical_file_bytes=size,scope=str(root),physical_recovery_not_guaranteed=True)

def provision_child(case,token):
    cfg=c.lock();root=c.plain(cfg['guest_root']);payload=json.loads((case/'payload.json').read_text());leaf=case/'provision'
    if root.exists():raise FileExistsError('Approved guest root already exists; no retry/resume')
    if sys.version.split()[0]!=cfg['runtime']['python'] or sys.prefix!='/usr':raise RuntimeError('Qualified system Python changed')
    before=baseline()
    if before['lib/python3.12/site-packages/torch/version.py']!=cfg['runtime']['torch_version_file_sha256']:raise RuntimeError('Qualified baseline runtime identity changed')
    downloads=c.plain(payload['guest_downloads']);sources=c.plain(payload['guest_sources'])
    if str(downloads)!='/mnt/d/AI/Modly-Pixal3D-Proof-v1/downloads' or str(sources)!='/mnt/d/AI/Modly-Pixal3D-Proof-v1/sources':raise ValueError('Exact staged roots required')
    for row in cfg['packages']:c.verify(c.inside(downloads,row['filename']),row)
    c.hip_payload(downloads/next(p['filename'] for p in cfg['packages'] if p['name']=='comfy-kitchen'))
    expansion=c.bounded_expansion([downloads/p['filename'] for p in cfg['packages']])
    if shutil.disk_usage(root.parent).free<expansion+cfg['guest_disk_floor_bytes']:raise RuntimeError('Guest cannot retain original disk floor after expansion')
    root.mkdir(exist_ok=False)
    for name in ('home','tmp','hip-cache','comgr-cache','disabled-torch-extensions','disabled-triton-cache'):(root/name).mkdir()
    env=child_environment(root,token);venv=root/'.venv'
    commands=[['/usr/bin/python3','-B','-I','-m','venv','--copies',str(venv)]]
    c.journal(leaf/'commands.jsonl','venv',argv=commands[0])
    subprocess.run(commands[0],check=True,env=env,timeout=120,stdin=subprocess.DEVNULL)
    config=(venv/'pyvenv.cfg').read_text()
    if 'include-system-site-packages = false' not in config or not (venv/'bin/python').is_file():raise RuntimeError('Independent venv not created')
    requirement=root/'requirements.lock'
    with requirement.open('x',encoding='utf-8') as stream:
        for row in cfg['packages']:stream.write(row['name']+'=='+row['version']+' --hash=sha256:'+row['sha256']+'\n')
    command=[str(venv/'bin/python'),'-B','-I','-m','pip','--isolated','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
        '--only-binary=:all:','--require-hashes','--find-links',str(downloads),'--report',str(root/'pip-report.json'),'-r',str(requirement)]
    c.journal(leaf/'commands.jsonl','binary-only-isolated-install',argv=command)
    subprocess.run(command,check=True,env=env,timeout=150,stdin=subprocess.DEVNULL)
    report=json.loads((root/'pip-report.json').read_text())
    expected={p['name'].lower().replace('_','-'):p for p in cfg['packages']}
    actual={p['metadata']['name'].lower().replace('_','-'):p for p in report['install']}
    if set(actual)!=set(expected):raise RuntimeError('Unexpected installation closure')
    for name,p in actual.items():
        if p['metadata']['version']!=expected[name]['version'] or p['download_info']['archive_info']['hashes']['sha256']!=expected[name]['sha256']:raise RuntimeError('Installed dependency identity differs')
    for row in cfg['sources']:
        source=c.inside(sources,row['path']);c.verify(source,row)
        target=c.inside(root/'ComfyUI',row['path']);target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
        c.verify(target,row)
    after=baseline()
    if after!=before:raise RuntimeError('Original environment critical files changed')
    cache=release_owned_cache(root)
    import resource
    c.save(leaf/'child-result.json',dict(classification='PASS',isolated_prefix=str(venv),share_site_packages=False,
        runtime_packages={name:p['metadata']['version'] for name,p in actual.items()},baseline_critical_files_before=before,
        baseline_critical_files_after=after,whole_baseline_recursive_hash_claimed=False,installed_expansion_bytes=expansion,owned_cache_hint=cache,
        Linux_provision_child_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        Linux_largest_installer_child_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024))

def capture_child(record,leaf):
    path=leaf/'child-result.json'
    if path.exists():
        if path.stat().st_size>8*1024**2:raise RuntimeError('Bounded child result required')
        data=json.loads(path.read_text())
        if data.get('classification') not in ('PASS','FAIL'):raise RuntimeError('Invalid child classification')
        record.update(data)
        if 'error' in data:record['child_error']=data['error']
        return data
    return None

def controller(stage,case,token):
    case=validate_case(case,token);cfg=c.lock();leaf=c.plain(case/stage);root=c.plain(cfg['guest_root'])
    if os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token:raise RuntimeError('Exec-time nonce required')
    if os.getpgrp()!=os.getpid():os.setsid()
    if os.getpgrp()!=os.getpid():raise RuntimeError('Own session/group required')
    rows=process_rows(os.getpgrp());own=next(row for row in rows if row['pid']==os.getpid())
    owner=dict(pid=own['pid'],pgid=own['pgid'],start_ticks=own['start_ticks'],token=token,stage=stage)
    c.save(leaf/'ownership.json',owner)
    record=dict(classification='FAIL',stage=stage,child_started=False);child=None;began=time.monotonic();peak=0
    try:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('No exact host GO/EOF; no setup or GPU import')
        if stage=='provision':
            if root.exists():raise FileExistsError('Guest path already used')
            command=['/usr/bin/python3','-B','-I','-S',str(Path(__file__).resolve()),'--provision-child','--case',str(case),'--token',token]
            env=child_environment(root,token)
        elif stage=='operators':
            if not (root/'.venv/bin/python').is_file():raise RuntimeError('Isolated prefix missing')
            command=[str(root/'.venv/bin/python'),'-B','-I',str(Path(__file__).with_name('amd_pixal3d_hip_ops.py')),'--run','--case',str(case),'--token',token]
            env=child_environment(root,token)
        else:raise ValueError('Only approved stages')
        c.journal(leaf/'phases.jsonl','GO',command=command)
        child=subprocess.Popen(command,executable=command[0],stdin=subprocess.PIPE,env=env)
        child.stdin.write(b'GO\n');child.stdin.close();record['child_started']=True
        deadline=cfg['provision_seconds'] if stage=='provision' else cfg['operator_seconds']
        while child.poll() is None:
            current=process_rows(owner['pgid']);owned_members(owner,current)
            resident=sum(p['RSS_bytes'] for p in current);peak=max(peak,resident)
            free=shutil.disk_usage('/home/modly').free
            c.journal(leaf/'resources.jsonl','guest-process-tree',RSS_bytes=resident,members=current,guest_free_disk_bytes=free)
            if resident>6*c.GIB or free<cfg['guest_disk_floor_bytes']:raise RuntimeError('Original guest RSS/disk protection failed')
            if time.monotonic()-began>=deadline:raise TimeoutError('Guest overall deadline')
            phase_file=leaf/'operator-phases.jsonl'
            if stage=='operators' and phase_file.exists():
                with phase_file.open('rb') as stream:
                    stream.seek(max(0,phase_file.stat().st_size-65536));data=stream.read()
                complete=data.split(b'\n')[:-1]
                if complete:
                    phase=json.loads(complete[-1]);allowed=120 if phase['kind']=='CPU' else 180
                    if time.monotonic()-phase['monotonic_start']>allowed:raise TimeoutError('Original CPU/GPU-release phase deadline')
            time.sleep(.25)
        record['child_exit_code']=child.returncode
        data=capture_child(record,leaf)
        if child.returncode:raise RuntimeError('Child failed; no retry')
        if not data or data['classification']!='PASS':raise RuntimeError('Child result failed')
        record['classification']='PASS'
    except BaseException as exc:record.update(error=type(exc).__name__+': '+str(exc))
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            record['child_forced_cleanup']=True;record['classification']='FAIL'
        if child and child.stdin and not child.stdin.closed:child.stdin.close()
        members=[p for p in process_rows(owner['pgid']) if p['pid']!=os.getpid()]
        if members:record.update(classification='FAIL',remaining_members=members)
        record.update(max_sampled_Linux_process_tree_RSS_bytes=peak,RSS_polling_interval_seconds=.25,seconds=time.monotonic()-began,natural_child_exit=not record.get('child_forced_cleanup',False))
        c.save(leaf/'result.json',record)
    return 0 if record['classification']=='PASS' else 2

def cleanup(stage,case,token):
    case=validate_case(case,token);leaf=case/stage;owner=json.loads((leaf/'ownership.json').read_text())
    if owner['token']!=token or owner['stage']!=stage or owner['pid']!=owner['pgid']:raise RuntimeError('Foreign ownership receipt')
    members=process_rows(owner['pgid']);owned_members(owner,members);forced=bool(members)
    if members:
        os.killpg(owner['pgid'],signal.SIGTERM);began=time.monotonic()
        while process_rows(owner['pgid']) and time.monotonic()-began<5:time.sleep(.05)
        members=process_rows(owner['pgid']);owned_members(owner,members)
        if members:
            os.killpg(owner['pgid'],signal.SIGKILL);began=time.monotonic()
            while process_rows(owner['pgid']) and time.monotonic()-began<5:time.sleep(.05)
    if process_rows(owner['pgid']):raise RuntimeError('Owned group exit unverified; no distro termination')
    c.save(leaf/'cleanup.json',dict(group_empty=True,natural_exit=not forced,forced_owned_group_cleanup=forced,no_distro_termination=True))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controller',choices=('provision','operators'));parser.add_argument('--cleanup',choices=('provision','operators'))
    parser.add_argument('--provision-child',action='store_true');parser.add_argument('--case');parser.add_argument('--token');args=parser.parse_args(argv)
    if not (args.controller or args.cleanup or args.provision_child):print('DRY_RUN: guest no work');return 0
    if not args.case or not args.token:parser.error('Exact case and token required')
    case=validate_case(args.case,args.token)
    if args.cleanup:return cleanup(args.cleanup,case,args.token)
    if args.controller:return controller(args.controller,case,args.token)
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO required before provisioning')
    provision_child(case,args.token);return 0

if __name__=='__main__':raise SystemExit(main())
