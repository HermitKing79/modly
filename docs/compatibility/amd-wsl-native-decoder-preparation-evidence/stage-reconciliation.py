"""Metadata-only recovery after a spelling check; never re-installs/downloads."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-native-decoder-preparation-evidence')
PRIOR=E.parent/'amd-wsl-flexgemm-build-evidence'
S=Path('/home/modly/native-decoder-preparation-6dd94a85');target=S/'site-packages'
staged={d.metadata['Name']:d.version for d in md.distributions(path=[str(target)])}
normalized={re.sub(r'[-_.]+','-',n).lower():v for n,v in staged.items()}
if normalized!={'numpy':'2.2.6','safetensors':'0.6.2','flex-gemm':'1.0.0'}:
    raise RuntimeError('Actual staged closure/version differs')
before=json.loads((PRIOR/'static-acceptance.json').read_text())
root=Path('/opt/modly-wsl/venv');rows=[]
for directory,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:
        f=Path(directory)/name;st=f.lstat()
        rows.append([f.relative_to(root).as_posix(),st.st_size,st.st_mtime_ns,st.st_mode])
vhash=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
packages=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
dpkg=dict(line.split('\t',1) for line in subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True).splitlines())
if vhash!=before['baseline_venv_metadata_sha256'] or packages!=before['python_packages'] or dpkg!=json.loads((PRIOR/'toolchain-result.json').read_text())['after_packages']:
    raise RuntimeError('Baseline drift')
admission=json.loads((E/'python-dependency-admission.json').read_text())
archives=[S/'inputs'/r['filename'] for r in admission['packages']]+[PRIOR/'flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl']
expanded=0
for f,expected in zip(archives,[r['sha256'] for r in admission['packages']]+['cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285']):
    if hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise RuntimeError('Staged archive changed')
    with zipfile.ZipFile(f) as z:
        expanded+=sum(i.file_size for i in z.infolist())
        for info in z.infolist():
            if info.is_dir() or '.dist-info/' in info.filename or '.data/' in info.filename:continue
            if (target/info.filename).read_bytes()!=z.read(info):raise RuntimeError('Staged package payload differs')
source=json.loads((PRIOR/'source-and-tools.json').read_text());native_source=Path(source['source_dir']);copies=[]
for name,h in source['source_sha256'].items():
    if not name.endswith(('hash.cu','submanifold_conv3d.py','subm_neighbor_map.cu')):continue
    data=(native_source/name).read_bytes()
    if hashlib.sha256(data).hexdigest()!=h:raise RuntimeError('Pinned native source changed')
    copy=E/('pinned-linux-'+Path(name).name+'.txt')
    with copy.open('xb') as f:f.write(data)
    copies.append({'source':name,'copy':str(copy),'sha256':h})
def free(path):
    st=os.statvfs(path);return st.f_bavail*st.f_frsize
ram={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
record={'baseline_venv_hash':vhash,'baseline_python_packages':packages,'baseline_dpkg':dpkg,
        'staging':str(S),'target':str(target),'staged_packages':staged,'canonical_staged_packages':normalized,
        'archive_expanded_bytes':expanded,'new_dependency_download_bytes':admission['sum_archive_bytes'],
        'source_copies':copies,'guest_free_bytes':free('/'),'D_free_bytes':free('/mnt/d'),'guest_RAM_bytes':ram,
        'GPU_or_model_actions':False,'baseline_installation':False,'package_installation_repeated':False,
        'prior_guard_failure':'MetadataName flex_gemm vs expectedflex-gemm; canonical[-_.] normalization confirmed',
        'all_staged_code_payloads_match_verified_wheels':True}
with (E/'staging.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:record[k] for k in ('staged_packages','canonical_staged_packages','archive_expanded_bytes',
    'new_dependency_download_bytes','guest_free_bytes','D_free_bytes','guest_RAM_bytes','package_installation_repeated')}),flush=True)
