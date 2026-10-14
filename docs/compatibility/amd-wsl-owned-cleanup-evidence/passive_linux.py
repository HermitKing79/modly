"""Reuse prior metadata/wheel preservation checks; no AI/native import."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

E=Path(__file__).resolve().parent
DOCS=E.parent
S=Path('/home/modly/native-decoder-preparation-6dd94a85')
phase=sys.argv[1]
if phase not in ('before','after'):raise ValueError('before/after required')
def deny(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','numpy','safetensors'}:
        raise RuntimeError('Passive preservation prohibits AI/native imports')
sys.addaudithook(deny)
before=json.loads((DOCS/'amd-wsl-native-decoder-preparation-evidence/staging.json').read_text())
root=Path('/opt/modly-wsl/venv');rows=[]
for directory,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:
        p=Path(directory)/name;s=p.lstat()
        rows.append([p.relative_to(root).as_posix(),s.st_size,s.st_mtime_ns,s.st_mode])
h=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
packages=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
dpkg=dict(line.split('\t',1) for line in subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True,timeout=15).splitlines())
if h!=before['baseline_venv_hash'] or packages!=before['baseline_python_packages'] or dpkg!=before['baseline_dpkg']:
    raise RuntimeError('Baseline environment identity changed')
admission=json.loads((DOCS/'amd-wsl-native-decoder-preparation-evidence/python-dependency-admission.json').read_text())
archives=[(S/'inputs'/row['filename'],row['sha256']) for row in admission['packages']]
archives.append((DOCS/'amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl','cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285'))
payloads=0
for path,expected in archives:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Verified wheel changed')
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or '.dist-info/' in info.filename or '.data/' in info.filename:continue
            if (S/'site-packages'/info.filename).read_bytes()!=z.read(info):raise RuntimeError('Staged payload changed')
            payloads+=1
staged={re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in md.distributions(path=[str(S/'site-packages')])}
if staged!=before['canonical_staged_packages']:raise RuntimeError('Staged closure changed')
sources=json.loads((DOCS/'amd-wsl-launcher-path-correction-evidence/installed-source-before.json').read_text())['sources']
if not all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in sources.values()):raise RuntimeError('Installed Torch source changed')
trace=(DOCS/'amd-wsl-flexgemm-build-evidence/elf-library-trace.txt').read_text()
libraries={name:Path(path) for name,path in re.findall(r'(\S+) => (/\S+) \(',trace)}
accepted=json.loads((DOCS/'amd-wsl-flexgemm-build-evidence/static-acceptance.json').read_text())
if not set(accepted['needed_libraries']).issubset(libraries):raise RuntimeError('Dependency manifest incomplete')
librows={}
for name,path in libraries.items():
    target=path.resolve(strict=True);s=target.stat()
    librows[name]={'path':str(path),'resolved':str(target),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
verified=subprocess.run(['dpkg','--verify','rocdxg-roct','hip-runtime-amd','hsa-rocr'],capture_output=True,text=True,timeout=15)
if verified.returncode or verified.stdout or verified.stderr:raise RuntimeError('Protected ROCm payload verification changed')
def free(p):
    s=os.statvfs(p);return s.f_bavail*s.f_frsize
memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
if memory<4*1024**3 or free('/')<5*1024**3 or free('/mnt/d')<50*1024**3:raise RuntimeError('Existing RAM/disk floor refused')
if not Path('/dev/dxg').exists():raise RuntimeError('WSL DXG device absent')
if phase=='before' and Path('/home/modly/owned-cleanup-9a53e6b').exists():raise RuntimeError('Exclusive scratch already exists')
record={'status':'PASS','phase':phase,'baseline_venv_metadata_sha256':h,'baseline_venv_unchanged':True,
    'baseline_packages_and_dpkg_unchanged':True,'installed_source_hashes_unchanged':True,
    'staged_packages':staged,'staged_payloads_verified':payloads,'needed_libraries':accepted['needed_libraries'],
    'approved_library_paths':librows,'protected_ROCm_dpkg_verification':True,
    'RAM_available_bytes':memory,'guest_free_bytes':free('/'),'D_free_bytes':free('/mnt/d'),
    'os_release':Path('/etc/os-release').read_text(),'kernel':os.uname().release,'DXG_present':True,'AI_imports':False,'GPU_actions':False}
with (E/('linux-preservation-'+phase+'.json')).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({'status':'PASS','phase':phase,'staged_payloads_verified':payloads,'RAM_available_bytes':memory}))
