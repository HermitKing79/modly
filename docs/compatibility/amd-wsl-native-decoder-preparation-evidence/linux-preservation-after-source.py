"""Final metadata/source readback; no AI imports or installation."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
E=Path(__file__).resolve().parent
S=Path('/home/modly/native-decoder-preparation-6dd94a85')
def deny(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','numpy','safetensors'}:
        raise RuntimeError('Inventory prohibits native/AI imports')
sys.addaudithook(deny)
before=json.loads((E/'staging.json').read_text())
root=Path('/opt/modly-wsl/venv');rows=[]
for directory,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:
        f=Path(directory)/name;st=f.lstat()
        rows.append([f.relative_to(root).as_posix(),st.st_size,st.st_mtime_ns,st.st_mode])
h=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
packages=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
dpkg=dict(line.split('\t',1) for line in subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True,timeout=15).splitlines())
payloads=0
admission=json.loads((E/'python-dependency-admission.json').read_text())
archives=[S/'inputs'/r['filename'] for r in admission['packages']]+[E.parent/'amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl']
for f in archives:
    with zipfile.ZipFile(f) as z:
        for info in z.infolist():
            if info.is_dir() or '.dist-info/' in info.filename or '.data/' in info.filename:continue
            if (S/'site-packages'/info.filename).read_bytes()!=z.read(info):raise RuntimeError('Staged payload changed')
            payloads+=1
def free(path):
    st=os.statvfs(path);return st.f_bavail*st.f_frsize
footprint=sum(f.stat().st_size for f in S.rglob('*') if f.is_file() and not f.is_symlink())
record={'status':'PASS','baseline_venv_metadata_sha256':h,'baseline_venv_unchanged':h==before['baseline_venv_hash'],
        'baseline_python_packages_unchanged':packages==before['baseline_python_packages'],
        'baseline_dpkg_unchanged':dpkg==before['baseline_dpkg'],
        'staged_code_payloads_match_verified_wheels':True,'staged_payloads_checked':payloads,
        'isolated_scratch':str(S),'isolated_regular_file_bytes':footprint,
        'guest_free_bytes':free('/'),'D_free_bytes':free('/mnt/d'),
        'native_import_performed':False,'real_weights_loaded':False,'GPU_execution':False}
if not all(record[k] for k in ('baseline_venv_unchanged','baseline_python_packages_unchanged','baseline_dpkg_unchanged')):raise RuntimeError('Baseline preservation failed')
if record['guest_free_bytes']<5*1024**3 or record['D_free_bytes']<50*1024**3:raise RuntimeError('Disk floor failed')
with (E/'linux-preservation-after.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
