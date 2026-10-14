"""Install exactly the admitted61package payloads, with no repairs/retries."""
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import time
import traceback

E = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
S = Path('/home/modly/flexgemm-build-6dd94a85')
A = S / 'inputs'
GiB = 1024 ** 3
admission = json.loads((E / 'resource-admission.json').read_text())
pre = json.loads((E / 'preflight.json').read_text())
packages = pre['packages']
pinned = [p['Package'] + '=' + p['Version'] for p in packages]
def free(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize
def state():
    return dict(line.split('\t',1) for line in subprocess.check_output(
        ['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True).splitlines())
before = state()
record = {'before_packages': before, 'approved_packages': packages, 'install_started': False,
          'GPU_or_model_actions': False, 'repairs_or_retries': 0,
          'minimum_guest_free_bytes': free('/'), 'minimum_D_free_bytes': free('/mnt/d')}
def gate():
    guest = free('/'); disk = free('/mnt/d')
    record['minimum_guest_free_bytes'] = min(record['minimum_guest_free_bytes'], guest)
    record['minimum_D_free_bytes'] = min(record['minimum_D_free_bytes'], disk)
    if guest < 5 * GiB or disk < 50 * GiB:
        raise RuntimeError('Installation resource floor violated')
def command(args, label, limit):
    gate()
    start = time.monotonic()
    with (E / (label+'.stdout.txt')).open('xb') as out, (E / (label+'.stderr.txt')).open('xb') as err:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                start_new_session=True,env={**os.environ,'DEBIAN_FRONTEND':'noninteractive'})
        try:
            while proc.poll() is None:
                gate()
                if time.monotonic()-start > limit:
                    raise TimeoutError(label+' deadline')
                time.sleep(.5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    result={'command':args,'exit':proc.returncode,'seconds':time.monotonic()-start}
    record[label]=result
    if proc.returncode:raise RuntimeError(label+' failed: '+str(proc.returncode))
    gate()
try:
    if os.geteuid()!=0 or not admission['admission_pass'] or not A.exists():
        raise RuntimeError('Installation prerequisites absent')
    for row in packages:
        if row['previous_version'] is not None and row['Package'] not in ('libexpat1','libpciaccess0'):
            raise RuntimeError('Foreign upgrade')
    sim=subprocess.run(['apt-get','--simulate','--no-install-recommends','install',*pinned],
                       capture_output=True,text=True,timeout=30)
    actual=[]
    for line in sim.stdout.splitlines():
        if line.startswith('Inst '):
            m=re.match(r'Inst (\S+)(?: \[([^\]]+)\])? \((\S+) ',line)
            if not m:raise RuntimeError('Unparsed admission row')
            actual.append(list(m.groups()))
    expected=[[p['Package'],p['previous_version'],p['Version']] for p in packages]
    if sim.returncode or sorted(actual)!=sorted(expected):raise RuntimeError('APT drift before payload acquisition')
    partial=A/'partial';partial.mkdir(exist_ok=False)
    apt_uid=pwd.getpwnam('_apt').pw_uid
    os.chown(partial,apt_uid,0)
    common=['apt-get','-y','--no-install-recommends','--no-remove',
            '-o','Dir::Cache::archives='+str(A),'-o','APT::Keep-Downloaded-Packages=true']
    command([*common,'--download-only','install',*pinned],'download',600)
    payloads={}
    for f in A.glob('*.deb'):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            while chunk:=stream.read(1024*1024):h.update(chunk)
        payloads[h.hexdigest()]={'path':str(f),'bytes':f.stat().st_size}
    verified=[]
    for row in packages:
        match=payloads.get(row['SHA256'])
        if not match or match['bytes']!=int(row['Size']):raise RuntimeError('Unverified package payload: '+row['Package'])
        verified.append({**row,**match})
    record['verified_payloads']=verified
    if free('/')-admission['approved_extracted_size_bound_bytes']-admission['install_temporary_reserve_bytes']-admission['dpkg_metadata_misc_reserve_bytes']<5*GiB:
        raise RuntimeError('Measured post-download installation headroom insufficient')
    with (E/'toolchain-install-started.json').open('x',encoding='utf-8') as f:
        json.dump({'approved_base':'a233a753afbcf9789ffe13aec26d0f393b2a0e4f',
                   'packages':[p['Package']+'='+p['Version'] for p in packages]},f,indent=2)
    record['install_started']=True
    command([*common,'--no-download','install',*pinned],'install',600)
    after=state()
    allowed={p['Package']:p['Version'] for p in packages}
    changes={n:[before.get(n),v] for n,v in after.items() if before.get(n)!=v}
    if any(n not in allowed or allowed[n]!=v[1] for n,v in changes.items()) or set(changes)!=set(allowed) or set(before)-set(after):
        raise RuntimeError('Installed package changes exceed exact admitted contract')
    record.update(after_packages=after,actual_package_changes=changes,status='installed-exactly')
except BaseException:
    record['status']='blocked';record['error']=traceback.format_exc()
    raise
finally:
    record.update(final_guest_free_bytes=free('/'),final_D_free_bytes=free('/mnt/d'))
    with (E/'toolchain-result.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ('before_packages','after_packages','approved_packages','verified_payloads')}),flush=True)
