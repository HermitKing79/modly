"""Wheel/ELF/loader-trace inspection only. Never dlopen the new extension."""
import base64
import csv
import hashlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
S=Path('/home/modly/flexgemm-build-6dd94a85')
result=json.loads((E/'build-result.json').read_text())
if result['status']!='built' or result['build_invocations']!=1:raise RuntimeError('No unique completed build')
wheel=S/'artifacts'/result['wheel']
if not wheel.name.endswith('-cp312-cp312-linux_x86_64.whl') or wheel.stat().st_size>16*1024**2:
    raise RuntimeError('Unexpected wheel identity/size')
raw=wheel.read_bytes();whash=hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(wheel) as z:
    if z.testzip() is not None:raise RuntimeError('Wheel CRC failure')
    names=z.namelist()
    if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>128*1024**2:
        raise RuntimeError('Unbounded wheel layout')
    if any(n.startswith('/') or '..' in Path(n).parts for n in names):raise RuntimeError('Invalid wheel path')
    native=[n for n in names if n.endswith('.so')]
    if len(native)!=1 or native[0]!='flex_gemm/kernels/cuda.cpython-312-x86_64-linux-gnu.so':
        raise RuntimeError('Unexpected native module layout')
    meta=z.read('flex_gemm-1.0.0.dist-info/METADATA').decode()
    info=z.read('flex_gemm-1.0.0.dist-info/WHEEL').decode()
    if 'Tag: cp312-cp312-linux_x86_64' not in info:raise RuntimeError('Wheel ABI tag mismatch')
    for n,h,size in csv.reader(io.StringIO(z.read('flex_gemm-1.0.0.dist-info/RECORD').decode())):
        if h:
            algorithm,value=h.split('=',1)
            data=z.read(n)
            actual=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            if algorithm!='sha256' or actual!=value or len(data)!=int(size):raise RuntimeError('Wheel RECORD mismatch')
    binary=z.read(native[0])
target=S/'qualification';target.mkdir(exist_ok=False)
so=target/Path(native[0]).name
with so.open('xb') as f:f.write(binary)
commands={}
def capture(argv,name,env=None):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=45,env=env)
    with (E/(name+'.txt')).open('x',encoding='utf-8') as f:f.write(r.stdout+r.stderr)
    commands[name]={'command':argv,'exit':r.returncode}
    if r.returncode:raise RuntimeError(name+' static tool failed')
    return r.stdout+r.stderr
header=capture(['readelf','-h',str(so)],'elf-header')
dynamic=capture(['readelf','-d',str(so)],'elf-dynamic')
symbols=capture(['nm','-D','--defined-only',str(so)],'elf-defined-symbols')
if 'ELF64' not in header or 'Advanced Micro Devices X86-64' not in header or 'PyInit_cuda' not in symbols:
    raise RuntimeError('Native CPython/Linux ELF mismatch')
needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
if any(n in needed for n in ('libcuda.so.1','libcudart.so','libcublas.so')):
    raise RuntimeError('Unexpected NVIDIA runtime dependency')
if not any(n.startswith('libamdhip64') for n in needed):raise RuntimeError('HIP runtime missing')
env={**os.environ,'LD_LIBRARY_PATH':'/opt/modly-wsl/venv/lib/python3.12/site-packages/torch/lib:/opt/rocm-7.2.1/lib:/usr/lib/wsl/lib'}
linked=capture(['ldd',str(so)],'elf-library-trace',env)
if 'not found' in linked or 'undefined symbol' in linked:raise RuntimeError('Unresolved shared library')
bindings=re.findall(r'm\.def\(\s*"([^"]+)"',(S/'source/flex_gemm/kernels/cuda/ext.cpp').read_text())
if len(bindings)!=18 or any(n.encode() not in binary for n in bindings):
    raise RuntimeError('Expected binding names not present in ELF')
if b'gfx1201' not in binary:raise RuntimeError('Compiled target metadata absent')
published=E/wheel.name
with published.open('xb') as f:f.write(raw)
if hashlib.sha256(published.read_bytes()).hexdigest()!=whash:raise RuntimeError('Preserved wheel copy failed')

# Exact baseline Python tree preservation: no Torch/native import here.
root=Path('/opt/modly-wsl/venv');rows=[]
for directory,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:
        f=Path(directory)/name;st=f.lstat()
        rows.append([f.relative_to(root).as_posix(),st.st_size,st.st_mtime_ns,st.st_mode])
pre=json.loads((E/'preflight.json').read_text())
vhash=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
packages=sorted([[d.metadata['Name'],d.version] for d in md.distributions(path=[str(root/'lib/python3.12/site-packages')])])
if vhash!=pre['identity']['venv_metadata_sha256'] or packages!=pre['identity']['python_packages']:
    raise RuntimeError('Baseline Python environment changed')
runtime=capture(['dpkg','--verify','rocdxg-roct','hip-runtime-amd','hsa-rocr'],'protected-runtime-dpkg-verify')
if runtime.strip():raise RuntimeError('Protected runtime payload changed')
after=dict(line.split('\t',1) for line in subprocess.check_output(['dpkg-query','-W','-f=${Package}\t${Version}\n'],text=True).splitlines())
if after!=json.loads((E/'toolchain-result.json').read_text())['after_packages']:
    raise RuntimeError('Package drift after admitted installation')
record={'wheel':str(wheel),'preserved_copy':str(published),'wheel_bytes':len(raw),'wheel_sha256':whash,
        'wheel_metadata':meta,'wheel_tag':info,'wheel_members':names,
        'native_member':native[0],'native_bytes':len(binary),'native_sha256':hashlib.sha256(binary).hexdigest(),
        'needed_libraries':needed,'source_binding_names':bindings,'all18_names_present':True,
        'PyInit_cuda_exported':True,'gfx1201_metadata_present':True,'cxx11_abi_from_compiler_and_Torch':1,
        'unresolved_shared_libraries':[],'commands':commands,'CPU_import_performed':False,
        'CPU_import_safety':'UNKNOWN: HIP registration constructors not proven free of HSA/GPU initialization; leave dlopen closed',
        'baseline_venv_metadata_sha256':vhash,'baseline_venv_matches':True,'python_packages':packages,
        'ROCDXG_HIP_HSA_payload_verification_passed':True,'approved_dpkg_after_matches':True,
        'rocdxg_version':after['rocdxg-roct'],'native_functions_called':0,'model_tensors_loaded':0,'GPU_actions':False,
        'final_guest_free_bytes':os.statvfs('/').f_bavail*os.statvfs('/').f_frsize,
        'final_D_free_bytes':os.statvfs('/mnt/d').f_bavail*os.statvfs('/mnt/d').f_frsize,
        'static_acceptance':'PASS'}
with (E/'static-acceptance.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:record[k] for k in ('wheel','wheel_bytes','wheel_sha256','native_bytes','needed_libraries',
    'all18_names_present','PyInit_cuda_exported','baseline_venv_matches','ROCDXG_HIP_HSA_payload_verification_passed',
    'CPU_import_performed','final_guest_free_bytes','final_D_free_bytes','static_acceptance')}),flush=True)
