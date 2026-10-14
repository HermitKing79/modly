"""Single offline backend invocation. Compiler execution allowed; GPU forbidden."""
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

S=Path('/home/modly/flexgemm-build-6dd94a85')
E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
os.chdir(S/'source')
resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2))
def forbidden(*args,**kwargs):
    raise RuntimeError('Offline CPU-only build prohibits network/GPU entry')
socket.socket.connect=forbidden
socket.socket.connect_ex=forbidden
socket.create_connection=forbidden
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'flex_gemm','triton','safetensors'}:
        raise RuntimeError('Build must not import model/native execution packages')
    if event=='subprocess.Popen':
        with (E/'compiler-commands.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'monotonic':time.monotonic(),'executable':str(args[0]),
                                'argv':args[1],'cwd':str(args[2])})+'\n');f.flush()
sys.addaudithook(audit)
import torch
if torch.cuda.is_initialized():raise RuntimeError('GPU unexpectedly initialized')
if torch.__version__!='2.9.1+rocm7.2.1.gitff65f5bc' or torch._C._GLIBCXX_USE_CXX11_ABI is not True:
    raise RuntimeError('Actual build Torch/ABI mismatch')
for name in ('init','_lazy_init','is_available','device_count','get_device_properties',
             'get_device_name','get_device_capability','synchronize'):
    setattr(torch.cuda,name,forbidden)
for name in ('_cuda_init','_cuda_getDeviceCount','_cuda_getDeviceProperties','_cuda_getArchFlags','_cuda_synchronize'):
    if hasattr(torch._C,name):setattr(torch._C,name,forbidden)
import setuptools.build_meta
wheel=setuptools.build_meta.build_wheel(str(S/'artifacts'))
with (E/'backend-return.json').open('x',encoding='utf-8') as f:
    json.dump({'wheel':wheel,'torch':torch.__version__,'cxx11_abi':True,
               'gpu_initialized':torch.cuda.is_initialized(),'native_bindings_called':0},f,indent=2)
