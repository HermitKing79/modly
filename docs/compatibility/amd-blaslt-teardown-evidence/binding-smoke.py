"""CPU-only exposure/provenance check; never invokes the native cleanup binding."""
import ctypes
import json
import os
from pathlib import Path
import sys

evidence=Path(__file__).resolve().parent;root=evidence.parents[2]
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_blaslt_teardown as teardown
from services.amd_offline_guard import block_external_actions
os.environ['WORKSPACE_DIR']=str(root)
scratch=evidence/'binding-scratch-3';scratch.mkdir(exist_ok=False)
os.environ.update(teardown.previous.loading.gpu_proof.scratch_environment(scratch))
sdk=[];block_external_actions(teardown.previous.local.VENV/'Scripts/rocm-sdk.exe',sdk,scratch)
def audit(event,args):
    if event.startswith('socket.'):raise RuntimeError('Binding smoke forbids network')
    if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors','.pt','.pth','.ckpt','.gguf','.bin'):
        raise RuntimeError('Binding smoke forbids checkpoint access')
sys.addaudithook(audit)
import torch
assert sys.version.split()[0]=='3.12.10' and not torch.cuda.is_initialized()
def forbidden(*args,**kwargs):raise RuntimeError('CPU binding smoke forbids GPU or model execution')
torch.cuda._lazy_init=forbidden;torch.load=forbidden
record=teardown.verify_runtime_binding(torch)
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetModuleHandleW.argtypes=[ctypes.c_wchar_p];kernel.GetModuleHandleW.restype=ctypes.c_void_p
kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint];kernel.GetModuleFileNameW.restype=ctypes.c_uint
loaded={};pins=teardown.verify_binding_evidence()
for name in ('torch_python.dll','torch_hip.dll'):
    handle=kernel.GetModuleHandleW(name);assert handle
    text=ctypes.create_unicode_buffer(32768);length=kernel.GetModuleFileNameW(handle,text,len(text));assert 0<length<len(text)
    path=Path(text.value).resolve();assert str(path) in pins['sources']
    loaded[name]={'path':str(path),'sha256':teardown.previous.local.digest(path)}
record.update(runtime=str(torch.__version__),hip=torch.version.hip,cuda=torch.version.cuda,
              loaded_native_modules=loaded,sdk_discovery=sdk,weights_loaded=False)
assert not torch.cuda.is_initialized()
# The guarded child may write only scratch; the parent saves this stdout record.
print(json.dumps(record))
