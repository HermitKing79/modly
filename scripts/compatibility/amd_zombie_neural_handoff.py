"""One independent, bounded zombie shape decode; default is a no-work dry run.

Only the user runs --run after closing desktop apps. The historical decoder,
readers, resource policies, owned supervisor and cleanup remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT.parent/'artifacts/amd-zombie-neural-handoff-d77c7ef-v1'
EVIDENCE=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-standalone-d77c7ef/decode'
RUN_ID='run-zombie-standalone-d77c7ef-v1'
FLOOR=22*1024**3
PREFERRED=24*1024**3
PYTHON=Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe')
PYTHON_SHA='0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
FILES=['scripts/compatibility/'+name for name in ('Invoke-ZombieNeuralHandoff.ps1',
    'amd_zombie_neural_handoff.py','test_amd_zombie_neural_handoff.py','amd_zombie_ram_diagnostic.py')]


def save(path,row):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(row,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def fresh_admission(read,sleep,record):
    rows=[]
    for index in range(3):
        row=dict(read(),timestamp_utc=datetime.now(timezone.utc).isoformat())
        rows.append(row);record(row)
        if type(row['available_bytes']) is not int or row['available_bytes']<FLOOR:
            raise RuntimeError('NO_GO_ADMISSION: below 22 GiB; no runtime, marker or WSL launch')
        if index<2:sleep(4)
    return rows


def validate_admission(rows,now):
    if len(rows)!=3 or any(type(r['available_bytes']) is not int or r['available_bytes']<FLOOR for r in rows):
        raise ValueError('Three passing 22 GiB readings required')
    stamps=[datetime.fromisoformat(r['timestamp_utc']) for r in rows]
    if any((b-a).total_seconds()<4 for a,b in zip(stamps,stamps[1:])) or not 0<=(now-stamps[-1]).total_seconds()<60:
        raise ValueError('Fresh correctly spaced admission required')


def verify_publication(commit,root,files,git):
    if not re.fullmatch('[0-9a-f]{40}',commit) or git(['rev-parse','HEAD']).strip()!=commit:
        raise ValueError('Exact published checkout required')
    for name in files:
        committed=git(['show',commit+':'+name]).replace('\r\n','\n')
        if committed!= (root/name).read_text(encoding='utf-8').replace('\r\n','\n'):
            raise ValueError('Published source differs: '+name)


def reserve_guarded(runtime,original,output,receipt,*,disk_free=lambda p:shutil.disk_usage(p).free):
    if Path(output)!=runtime.evidence/'run':raise ValueError('Exact fresh owned run required')
    if disk_free(Path(receipt).parent)<2*1024**3:raise RuntimeError('Original 2 GiB output-volume floor failed')
    # This is the final check, immediately before the original exclusive marker.
    resource=runtime.host_admission()  # Original 12 GiB RAM and 50 GiB D: checks.
    passed=resource['available_ram_bytes']>=FLOOR
    save(receipt,dict(classification='ADMITTED' if passed else 'NO_GO_FINAL_RAM',
        stage='immediately-before-exclusive-neural-marker',host_resource=resource,
        required_prelaunch_ram_bytes=FLOOR,harness_commit=runtime.commit,
        run_contract_sha256=runtime.contract.sha256,neural_marker_reserved=False))
    if not passed:raise RuntimeError('Final 22 GiB gate failed; no neural marker reserved')
    return original(output)


def create_runtime(zombie,commit):
    # A private module copy routes only this attempt. Historical globals untouched.
    historical=zombie.historical
    route=historical.load_module('_amd_zombie_standalone_recovery',Path(zombie.__file__))
    route.EVIDENCE,route.RUN_ID=EVIDENCE,RUN_ID
    route.source_inputs=zombie.source_inputs
    runtime=route.make_runtime(commit)
    namespace=type(runtime).__init__.__globals__
    namespace['STATIC_FILES']=list(namespace['STATIC_FILES'])+FILES
    original=runtime.reserve_attempt
    runtime.reserve_attempt=lambda output:reserve_guarded(runtime,original,output,PRIVATE/'final-pre-marker-admission.json')
    return runtime


def run(commit):
    import ctypes
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-elevated Windows user required')
    if Path(sys.executable).resolve()!=PYTHON.resolve() or hashlib.sha256(PYTHON.read_bytes()).hexdigest()!=PYTHON_SHA:
        raise ValueError('Original qualified Windows interpreter required')
    git=shutil.which('git.exe')
    if not git:raise RuntimeError('Git executable missing')
    verify_publication(commit,ROOT,FILES,lambda args:subprocess.check_output(
        [git,*args],cwd=ROOT,text=True,encoding='utf-8',timeout=15))
    for path in (PRIVATE,EVIDENCE):
        for parent in (path,*path.parents):
            if parent.exists() and (parent.is_symlink() or parent.is_junction()):raise ValueError('Redirected output denied')
    if EVIDENCE.exists():raise FileExistsError('Single owned decoder destination is already consumed; no retry')
    PRIVATE.mkdir(exist_ok=False)
    print('Private handoff evidence: '+str(PRIVATE),flush=True)
    # Import only the stdlib measurement library; never execute its diagnostic.
    spec=importlib.util.spec_from_file_location('_standalone_windows_memory',Path(__file__).with_name('amd_zombie_ram_diagnostic.py'))
    memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    def record(row):
        save(PRIVATE/('admission-'+str(record.index)+'.json'),row);record.index+=1
        print('AVAILABLE_RAM_GIB='+str(row['available_bytes']/1024**3),flush=True)
    record.index=1
    rows=fresh_admission(memory.windows_memory,time.sleep,record)
    validate_admission(rows,datetime.now(timezone.utc))
    save(PRIVATE/'admission.json',dict(readings=rows,required_available_bytes=FLOOR,preferred_only_bytes=PREFERRED,
        harness_commit=commit,interpreter_pid=os.getpid(),interpreter_image=sys._base_executable))
    if shutil.disk_usage(PRIVATE).free<2*1024**3:raise RuntimeError('Original 2 GiB output-volume floor failed')
    os.environ['WORKSPACE_DIR']=str(ROOT)
    # Match the completed diagnostic's controlled startup: no sitecustomize/.pth.
    sys.path[:0]=[str(ROOT/'scripts/compatibility'),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
    import amd_zombie_decode as zombie
    runtime=create_runtime(zombie,commit)
    # This original supervisor performs preflight and final guarded reservation,
    # owns GO/EOF, WSL timeout/Job, health monitoring, cleanup and full readback.
    return runtime.supervise()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--harness-commit')
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(mode='DRY_RUN',neural_execution=False,marker_reserved=False,WSL_launched=False,
            admission_bytes=FLOOR,preferred_only_bytes=PREFERRED,samples=3,spacing_seconds=4,
            final_pre_marker_bytes=FLOOR,live_host_floor_bytes=12*1024**3,maximum_forwards=1,
            private=str(PRIVATE),evidence=str(EVIDENCE),run_id=RUN_ID)))
        return 0
    if not args.harness_commit or not re.fullmatch('[0-9a-f]{40}',args.harness_commit):
        parser.error('--run requires the exact published --harness-commit')
    try:return run(args.harness_commit)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())
