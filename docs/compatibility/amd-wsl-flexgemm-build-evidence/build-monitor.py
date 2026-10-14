"""One compiler tree,2400s budget, fixed RAM/disk floors; no automatic retry."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

S=Path('/home/modly/flexgemm-build-6dd94a85')
E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
helpers=E.parents[2]/'scripts/compatibility/amd_wsl_rocdxg_smoke.py'
spec=importlib.util.spec_from_file_location('existing_wsl_gate',helpers)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.await_go(sys.stdin.buffer)
GiB=1024**3
admission=json.loads((E/'resource-admission.json').read_text())
source=json.loads((E/'source-and-tools.json').read_text())
if not admission['admission_pass'] or json.loads((E/'toolchain-result.json').read_text())['status']!='installed-exactly':
    raise RuntimeError('Build admission absent')
for name,h in source['source_sha256'].items():
    if hashlib.sha256((S/'source'/name).read_bytes()).hexdigest()!=h:
        raise RuntimeError('Pinned source changed before build')
for d in ('home','tmp','artifacts'): (S/d).mkdir(exist_ok=False)
driver=S/'build-driver.py'
with driver.open('xb') as f:f.write((E/'build-driver.py').read_bytes())
env={**os.environ,'HOME':str(S/'home'),'TMPDIR':str(S/'tmp'),
     'XDG_CACHE_HOME':str(S/'home/cache'),'CCACHE_DIR':str(S/'home/ccache'),
     'HIP_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'',
     'ROCM_HOME':'/opt/rocm-7.2.1','ROCM_PATH':'/opt/rocm-7.2.1',
     'BUILD_TARGET':'rocm','GPU_ARCHS':'gfx1201','PYTORCH_ROCM_ARCH':'gfx1201',
     'MAX_JOBS':'1','PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1',
     'PYTHONDONTWRITEBYTECODE':'1',
     'LD_LIBRARY_PATH':'/opt/modly-wsl/venv/lib/python3.12/site-packages/torch/lib:/opt/rocm-7.2.1/lib:/usr/lib/wsl/lib'}
record={'environment':{k:env[k] for k in ('HOME','TMPDIR','ROCM_HOME','ROCM_PATH','BUILD_TARGET','GPU_ARCHS','PYTORCH_ROCM_ARCH','MAX_JOBS','PIP_NO_INDEX')},
        'peak_owned_build_file_bytes':0,'peak_compiler_tree_RSS_bytes':0,
        'minimum_guest_free_bytes':2**63,'minimum_D_free_bytes':2**63,
        'minimum_guest_available_RAM_bytes':2**63,'GPU_execution':False,
        'build_invocations':0,'retries':0}
def free(path):
    st=os.statvfs(path);return st.f_bavail*st.f_frsize
def snapshot(pid):
    files=sum(p.stat().st_size for directory in S.iterdir() if directory.name!='inputs'
              for p in ([directory] if directory.is_file() else directory.rglob('*')) if p.is_file())
    ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    procs={}
    for d in Path('/proc').iterdir():
        if not d.name.isdecimal():continue
        try:
            st=(d/'stat').read_text();rest=st[st.rfind(')')+2:].split()
            procs[int(d.name)]=(int(rest[1]),int(rest[21])*os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    owned={pid}
    while True:
        expanded=owned|{p for p,(parent,_) in procs.items() if parent in owned}
        if expanded==owned:break
        owned=expanded
    row={'seconds':time.monotonic()-start,'build_files_bytes':files,
         'tree_RSS_bytes':sum(procs.get(p,(0,0))[1] for p in owned),
         'guest_free_bytes':free('/'),'D_free_bytes':free('/mnt/d'),'guest_available_RAM_bytes':ram}
    record['peak_owned_build_file_bytes']=max(record['peak_owned_build_file_bytes'],files)
    record['peak_compiler_tree_RSS_bytes']=max(record['peak_compiler_tree_RSS_bytes'],row['tree_RSS_bytes'])
    for src,dest in [('guest_free_bytes','minimum_guest_free_bytes'),('D_free_bytes','minimum_D_free_bytes'),('guest_available_RAM_bytes','minimum_guest_available_RAM_bytes')]:
        record[dest]=min(record[dest],row[src])
    with (E/'build-resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    if files>3*GiB or row['tree_RSS_bytes']>8*GiB or ram<2*GiB or row['guest_free_bytes']<5*GiB or row['D_free_bytes']<50*GiB:
        raise RuntimeError('Admitted build resource boundary exceeded')
    if row['seconds']>2400:raise TimeoutError('Single native build2400s deadline')
    return row
start=time.monotonic();proc=None
try:
    snapshot(-1)
    with (E/'native-build-started.json').open('x',encoding='utf-8') as f:
        json.dump({'source_commit':source['commit'],'MAX_JOBS':1,'limit_seconds':2400},f)
    with (E/'build.stdout.txt').open('xb') as out,(E/'build.stderr.txt').open('xb') as err:
        command=['/opt/modly-wsl/venv/bin/python','-B',str(driver)]
        proc=subprocess.Popen(command,cwd=S/'source',env=env,stdin=subprocess.DEVNULL,
                              stdout=out,stderr=err,start_new_session=True)
        record['build_invocations']=1;record['command']=command
        while proc.poll() is None:
            snapshot(proc.pid);time.sleep(.5)
        record['exit']=proc.returncode;snapshot(proc.pid)
        if proc.returncode:raise RuntimeError('Pinned build failed: '+str(proc.returncode))
    record['status']='built';record['wheel']=json.loads((E/'backend-return.json').read_text())['wheel']
except BaseException:
    record['status']='stopped';record['error']=traceback.format_exc()
    raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    record['total_seconds']=time.monotonic()-start
    record['owned_build_process_exit']=proc is None or proc.poll() is not None
    with (E/'build-result.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)
