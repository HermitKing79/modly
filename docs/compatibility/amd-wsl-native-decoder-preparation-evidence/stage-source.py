"""Pinned wheels to a fresh isolated target; baselinevenv neverinstalled into."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-native-decoder-preparation-evidence')
PRIOR=E.parent/'amd-wsl-flexgemm-build-evidence'
S=Path('/home/modly/native-decoder-preparation-6dd94a85')
root=Path('/opt/modly-wsl/venv');rows=[]
for directory,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:
        f=Path(directory)/name;st=f.lstat()
        rows.append([f.relative_to(root).as_posix(),st.st_size,st.st_mtime_ns,st.st_mode])
vhash=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
before=json.loads((PRIOR/'static-acceptance.json').read_text())
packages=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
dpkg=dict(line.split('\t',1) for line in subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True).splitlines())
if vhash!=before['baseline_venv_metadata_sha256'] or packages!=before['python_packages'] or dpkg!=json.loads((PRIOR/'toolchain-result.json').read_text())['after_packages']:
    raise RuntimeError('Pinned baseline changed')
def free(path):
    st=os.statvfs(path);return st.f_bavail*st.f_frsize
if free('/')<5*1024**3+512*1024**2 or free('/mnt/d')<50*1024**3:
    raise RuntimeError('Staging disk admission failed')
S.mkdir(exist_ok=False);(S/'inputs').mkdir();(S/'home').mkdir();(S/'tmp').mkdir()
admission=json.loads((E/'python-dependency-admission.json').read_text())
archives=[];expanded=0
for row in admission['packages']:
    with urllib.request.urlopen(row['URL'],timeout=30) as r:
        data=r.read(row['bytes']+1)
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise RuntimeError('Unverified dependency wheel')
    f=S/'inputs'/row['filename']
    with f.open('xb') as stream:stream.write(data)
    archives.append(f)
native=PRIOR/'flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'
if hashlib.sha256(native.read_bytes()).hexdigest()!='cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285':
    raise RuntimeError('Native wheel identity changed')
archives.append(native)
for f in archives:
    with zipfile.ZipFile(f) as z:
        if z.testzip() is not None:raise RuntimeError('Invalid archive CRC')
        expanded+=sum(i.file_size for i in z.infolist())
if expanded>256*1024**2 or free('/')-expanded-64*1024**2<5*1024**3:
    raise RuntimeError('Staged package expansion inadmissible')
command=['/opt/modly-wsl/venv/bin/python','-B','-m','pip','--isolated','--disable-pip-version-check',
         'install','--no-index','--no-deps','--no-compile','--target',str(S/'site-packages'),*map(str,archives)]
q=subprocess.run(command,capture_output=True,text=True,timeout=60)
if q.returncode:raise RuntimeError('Target-only installation failed: '+q.stderr)
staged={d.metadata['Name']:d.version for d in md.distributions(path=[str(S/'site-packages')])}
if {n.lower() for n in staged}!={'numpy','safetensors','flex-gemm'}:
    raise RuntimeError('Unexpected staged dependency closure')
source=json.loads((PRIOR/'source-and-tools.json').read_text())
native_source=Path(source['source_dir'])
selected=[n for n in source['source_sha256'] if n.endswith('hash.cu') or n.endswith('submanifold_conv3d.py') or n.endswith('subm_neighbor_map.cu')]
source_copies=[]
for name in selected:
    data=(native_source/name).read_bytes()
    if hashlib.sha256(data).hexdigest()!=source['source_sha256'][name]:raise RuntimeError('Pinned source changed')
    target=E/('pinned-linux-'+Path(name).name+'.txt')
    with target.open('xb') as f:f.write(data)
    source_copies.append({'source':name,'copy':str(target),'sha256':source['source_sha256'][name]})
ram={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
record={'baseline_venv_hash':vhash,'baseline_python_packages':packages,'baseline_dpkg':dpkg,
        'staging':str(S),'target':str(S/'site-packages'),'staged_packages':staged,
        'archive_expanded_bytes':expanded,'new_dependency_download_bytes':admission['sum_archive_bytes'],
        'command':command,'stdout':q.stdout,'stderr':q.stderr,'exit':q.returncode,'source_copies':source_copies,
        'guest_free_bytes':free('/'),'D_free_bytes':free('/mnt/d'),'guest_RAM_bytes':ram,
        'GPU_or_model_actions':False,'baseline_installation':False}
with (E/'staging.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:record[k] for k in ('target','staged_packages','archive_expanded_bytes',
                                      'new_dependency_download_bytes','guest_free_bytes','D_free_bytes','guest_RAM_bytes')}),flush=True)
