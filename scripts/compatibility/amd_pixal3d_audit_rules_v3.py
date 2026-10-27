"""Shared unchanged operator filesystem/import rules; no numeric entrypoint."""
from pathlib import Path
import os
import amd_pixal3d_operator_common_v3 as c

def policy(root,case,denied=None):
    def reject(message,event,args):
        if denied:denied(event,args)
        raise RuntimeError(message)
    def canonical(value,event,args):
        try:return c.plain(value)
        except (ValueError,OSError):
            if denied:denied(event,args)
            raise
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.fork','socket.connect','socket.bind'):raise RuntimeError('No compile, subprocess or network in operator worker')
        if event=='import':
            name=str(args[0])
            if name=='triton' or name.startswith('triton.'):raise ImportError('Global Triton disabled for this proof')
            if name=='comfy_kitchen.backends.cuda._C':raise ImportError('CUDA native extension disabled; HIP only')
            if name.split('.')[0] in ('pip','transformers','diffusers','trellis2','comfy','huggingface_hub','natten','spconv','nvdiffrast'):raise ImportError('Installer/model/foreign kernel module forbidden')
        if event=='ctypes.dlopen' and args[0] and any(n in str(args[0]).lower() for n in ('libcuda.so','libcublas','/nvidia/')):raise OSError('CUDA library disabled')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]));writing=any(v in str(args[1] or '') for v in 'wax+') or bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                path=canonical(path,event,args)
                if not (path.is_relative_to(root) or path.is_relative_to(case)):reject('Operator write outside isolated roots',event,args)
            elif path.suffix.lower() in ('.safetensors','.pt','.pth','.ckpt'):raise RuntimeError('No weights/checkpoints in model-free proof')
        if event=='os.mkdir':
            path=canonical(args[0],event,args)
            if not(path.is_relative_to(root) or path.is_relative_to(case)):reject('Foreign directory creation',event,args)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):reject('Destructive mutation forbidden in operator proof',event,args)
    return audit
