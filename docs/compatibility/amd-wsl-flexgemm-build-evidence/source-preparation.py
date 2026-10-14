"""Acquire one pinned source archive; qualify actual compiler/header identities."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import urllib.request

E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
S=Path('/home/modly/flexgemm-build-6dd94a85')
PIN='6dd94a859c26ee8246888502eada3dd8ad85532e'
URL='https://codeload.github.com/JeffreyXiang/FlexGEMM/tar.gz/'+PIN
if json.loads((E/'toolchain-result.json').read_text())['status']!='installed-exactly':
    raise RuntimeError('Toolchain not admitted')
with urllib.request.urlopen(URL,timeout=30) as r:
    data=r.read(32*1024**2+1)
if len(data)>32*1024**2:raise RuntimeError('Source archive exceeds32MiB')
archive=S/'inputs'/'FlexGEMM-6dd94a85.tar.gz'
with archive.open('xb') as f:f.write(data)
dest=S/'source';dest.mkdir(exist_ok=False)
total=0
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
    for m in tar:
        path=PurePosixPath(m.name)
        if path.is_absolute() or '..' in path.parts or path.parts[0]!='FlexGEMM-'+PIN:
            raise RuntimeError('Unexpected source path')
        if len(path.parts)==1:continue
        out=dest.joinpath(*path.parts[1:])
        if m.isdir():out.mkdir(parents=True,exist_ok=True)
        elif m.isfile():
            total+=m.size
            if total>64*1024**2 or m.size>16*1024**2:raise RuntimeError('Source expansion bound')
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f,tar.extractfile(m) as stream:shutil.copyfileobj(stream,f,1024*1024)
        else:raise RuntimeError('Nonregular source member')
hashes={f.relative_to(dest).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
        for f in dest.rglob('*') if f.is_file()}
expected={'setup.py':'9411c37c682d32417d5963b9dec3511ede1c89ef5c6014d54b1d73ea90028b6c',
          'pyproject.toml':'33bb082fc42516d6fd5fe44de6a5adb9c7190f1c24b1480902e88ce8b0b3d483'}
if any(hashes[n]!=h for n,h in expected.items()):raise RuntimeError('Source differs from independent audit')
tools={}
for cmd in (['g++','--version'],['/opt/rocm-7.2.1/bin/hipcc','--version'],
            ['/opt/rocm-7.2.1/llvm/bin/clang++','--version'],['ninja','--version']):
    q=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=True)
    tools[' '.join(cmd)]={'stdout':q.stdout,'stderr':q.stderr}
required=['/usr/include/python3.12/Python.h','/opt/rocm-7.2.1/include/hip/hip_runtime.h',
          '/opt/rocm-7.2.1/include/hipblas/hipblas.h','/opt/rocm-7.2.1/include/hipblaslt/hipblaslt.h',
          '/opt/rocm-7.2.1/include/hipsolver/hipsolver.h','/opt/rocm-7.2.1/include/hipsparse/hipsparse.h',
          '/opt/rocm-7.2.1/include/hipcub/hipcub.hpp','/opt/rocm-7.2.1/include/thrust/version.h']
headers={n:Path(n).exists() for n in required}
if not all(headers.values()):raise RuntimeError('Required native header absent')
record={'commit':PIN,'URL':URL,'source_dir':str(dest),'archive_bytes':len(data),
        'archive_sha256':hashlib.sha256(data).hexdigest(),'source_file_bytes':total,
        'source_sha256':hashes,'tools':tools,'headers':headers,
        'native_build_or_GPU_actions':False,'source_modified_by_hand':False}
with (E/'source-and-tools.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_sha256','tools')},indent=2),flush=True)
