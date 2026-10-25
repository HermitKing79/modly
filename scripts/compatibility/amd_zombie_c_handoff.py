"""One user-run Zombie C pipeline. Default prints a plan and performs no work."""
import argparse
from datetime import datetime,timezone
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
REFERENCE='docs/compatibility/amd-zombie-c-reference-contract.json'
CASE=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-c-60862722-v1'
PRIVATE=ROOT.parent/'artifacts/amd-zombie-c-generation-60862722-v1'
COMPARISON=ROOT.parent/'artifacts/amd-zombie-c-comparison-60862722-v1'
PYTHON=Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe')
PYTHON_SHA='0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
FLOOR=22*1024**3
PREFERRED=24*1024**3
LIVE_FLOOR=12*1024**3
STAGES=('occupancy','shape','decode','comparison')
FORWARDS=dict(dino=1,occupancy_sampling=22,occupancy_decode=1,shape=21,geometry_decode=1)
CRITICAL=[REFERENCE,'docs/compatibility/amd-zombie-c-comparison-plan.json']+[
    'scripts/compatibility/'+name for name in ('Invoke-ZombieCGeneration.ps1','amd_zombie_c_handoff.py',
    'amd_zombie_c_stages.py','amd_zombie_c_compare.py','amd_zombie_c_compare_render.py',
    'test_amd_zombie_c_handoff.py','test_amd_zombie_c_stages.py','test_amd_zombie_c_compare.py',
    'amd_zombie_neural_handoff.py','amd_zombie_ram_diagnostic.py')]

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def plain(path):
    path=Path(path).absolute()
    if str(path).startswith('\\\\') or path!=path.resolve():raise ValueError('Canonical local path required')
    for p in (path,*path.parents):
        if p.exists() and (p.is_symlink() or p.is_junction()):raise ValueError('Redirected path denied')
    if path.is_file() and path.stat().st_nlink!=1:raise ValueError('Hardlinked file denied')
    return path

def save(path,row):
    with plain(path).open('x',encoding='utf-8') as f:
        json.dump(row,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def windows_memory():
    path=Path(__file__).with_name('amd_zombie_ram_diagnostic.py')
    spec=importlib.util.spec_from_file_location('_c_windows_memory',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.windows_memory()

def fresh_admission(read,sleep,record):
    rows=[]
    for i in range(3):
        row=dict(read(),timestamp_utc=datetime.now(timezone.utc).isoformat());rows.append(row);record(row)
        if type(row['available_bytes']) is not int or row['available_bytes']<FLOOR:
            raise RuntimeError('NO_GO_ADMISSION: below 22 GiB; no neural attempt')
        if i<2:sleep(4)
    return rows

def validate_admission(rows,now):
    if len(rows)!=3 or any(type(r['available_bytes']) is not int or r['available_bytes']<FLOOR for r in rows):
        raise ValueError('Three passing admission readings required')
    stamps=[datetime.fromisoformat(r['timestamp_utc']) for r in rows]
    if any((b-a).total_seconds()<4 for a,b in zip(stamps,stamps[1:])) or not 0<=(now-stamps[-1]).total_seconds()<60:
        raise ValueError('Fresh readings four seconds apart required')

def guard_reservation(read,record,reserve):
    row=read();value=row['available_bytes'];ok=type(value) is int and value>=FLOOR
    record(dict(classification='ADMITTED' if ok else 'NO_GO_FINAL_RAM',resource=row,
                required_available_bytes=FLOOR,timestamp_utc=datetime.now(timezone.utc).isoformat()))
    if not ok:raise RuntimeError('Final 22 GiB gate failed before reservation')
    return reserve()

def publication_files(files,private_inputs,critical):
    # Exactly these two hash-bound data files remain private. No code or other
    # dependency may bypass the inherited Git verifier.
    return sorted((set(files)-set(private_inputs))|set(critical))

def verify_private_inputs(root,authority):
    root=plain(root);paths=[]
    for key,sha_key in (('source_path','source_sha256'),('input_contract_path','input_contract_sha256')):
        value=authority[key]
        if not isinstance(value,str) or '\\' in value or ':' in value or '..' in value.split('/') or Path(value).is_absolute():
            raise ValueError('Only canonical repository-relative input paths admitted')
        path=plain(root/value);path.relative_to(root)
        limit=2*1024**2 if key=='source_path' else 65536
        if not path.is_file() or path.stat().st_size>limit or digest(path)!=authority[sha_key]:
            raise ValueError('Private input authority differs: '+key)
        paths.append(path)
    if paths[0].stat().st_size!=authority['source_bytes']:raise ValueError('Source bytes changed')
    return paths

def authority():
    path=plain(ROOT/REFERENCE)
    if path.stat().st_size>65536:raise ValueError('Bounded reference manifest required')
    row=json.loads(path.read_bytes())
    if (row['schema']!='modly.amd.zombie-c-reference.v1' or row['source_sha256']!='60862722c989772ad973870890601a6a95ed9a5cd5a9150a378e989db35ad437'
            or row['source_path']!=(CASE/'occupancy/source-image.png').relative_to(ROOT).as_posix()
            or row['input_contract_path']!=(CASE/'occupancy/input-contract.json').relative_to(ROOT).as_posix()):
        raise ValueError('Exact qualified Zombie C reference required')
    verify_private_inputs(ROOT,row);return row

def git(args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8',timeout=15)

def verify_publication(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or git(['rev-parse','HEAD']).strip()!=commit:
        raise ValueError('Exact published current checkout required')
    for name in CRITICAL:
        if git(['show',commit+':'+name]).replace('\r\n','\n')!=(ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n'):
            raise ValueError('Published source differs: '+name)

def require_fresh_outputs():
    for path in (PRIVATE,COMPARISON,CASE/'shape',CASE/'decode'):
        plain(path)
        if path.exists():raise FileExistsError('One-shot destination already consumed: '+str(path))
    prepared=plain(CASE/'occupancy')
    if {p.name for p in prepared.iterdir()}!={'source-image.png','input-contract.json'}:
        raise FileExistsError('Prepared occupancy leaf already contains attempt evidence; no retry')

def execute_once(stage):
    for name in STAGES:
        print('ZOMBIE_C_STAGE='+name,flush=True)
        if stage(name)!=0:raise RuntimeError('Stage failed; no retry or later stage: '+name)

def run(commit):
    import ctypes
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-elevated Windows user required')
    if Path(sys.executable).resolve()!=PYTHON.resolve() or digest(PYTHON)!=PYTHON_SHA:
        raise ValueError('Qualified Windows interpreter required')
    verify_publication(commit);authority();require_fresh_outputs()
    PRIVATE.mkdir(exist_ok=False)
    result=dict(classification='FAIL',harness_commit=commit,maximum_forwards=FORWARDS,neural_attempt_reserved=False)
    try:
        if shutil.disk_usage(PRIVATE).free<2*1024**3:raise RuntimeError('Original 2 GiB output disk floor failed')
        readings=[]
        def record(row):
            readings.append(row);save(PRIVATE/('admission-'+str(len(readings))+'.json'),row)
            print('AVAILABLE_RAM_GIB='+str(row['available_bytes']/1024**3),flush=True)
        rows=fresh_admission(windows_memory,time.sleep,record);validate_admission(rows,datetime.now(timezone.utc))
        save(PRIVATE/'admission.json',dict(readings=rows,required_bytes=FLOOR,preferred_only_bytes=PREFERRED))
        sys.path[:0]=[str(ROOT/'scripts/compatibility'),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
        os.environ['WORKSPACE_DIR']=str(ROOT)
        os.environ['MODLY_ZOMBIE_C_COMMIT']=commit
        import amd_zombie_c_stages as stages
        bridge=stages.Bridge(commit)
        guard_reservation(windows_memory,lambda row:save(PRIVATE/'pipeline-final-admission.json',row),
            lambda:save(PRIVATE/'pipeline-attempt-started.json',dict(harness_commit=commit,maximum_pipelines=1,maximum_forwards=FORWARDS,source_sha256=authority()['source_sha256'])))
        result['neural_attempt_reserved']=True
        execute_once(bridge.dispatch)
        result['classification']='PASS_GENERATION_AND_CPU_COMPARISON_READY_FOR_VISUAL_REVIEW'
        result['comparison_directory']=str(COMPARISON)
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(PRIVATE/'pipeline-result.json',result)
        print('ZOMBIE_C_PIPELINE_RESULT='+result['classification'],flush=True)
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--harness-commit')
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(mode='DRY_RUN',neural_execution=False,WSL_launched=False,marker_reserved=False,
            stages=list(STAGES),maximum_forwards=FORWARDS,admission_bytes=FLOOR,preferred_only_bytes=PREFERRED,
            admission_readings=3,spacing_seconds=4,final_pre_marker_bytes=FLOOR,live_host_floor_bytes=LIVE_FLOOR,
            fresh_conditioning_and_occupancy=True,case=str(CASE),private=str(PRIVATE),comparison=str(COMPARISON))))
        return 0
    if not re.fullmatch('[0-9a-f]{40}',args.harness_commit or ''):parser.error('--run requires exact published --harness-commit')
    try:return run(args.harness_commit)
    except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
