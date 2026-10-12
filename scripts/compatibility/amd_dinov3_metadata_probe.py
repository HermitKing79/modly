"""Bounded CPU/meta construction and full header match. No real tensor loading."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_local as local
import amd_sparse_decoder_loading as loading
from amd_forward_owned_job import OwnedJob
from services.amd_offline_guard import block_external_actions


def worker(scratch):
    record={'status':'FAIL','weights_loaded':False,'gpu_initialized':False,'sdk_discovery':[]}
    torch=None
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Missing owned job handshake')
        local.verify_sources()
        expected=local.identity()
        env=loading.gpu_proof.scratch_environment(scratch)
        for key in loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env)
        block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        os.environ.update(env)
        def audit(event,args):
            if event.startswith('socket.'):raise RuntimeError('CPU preflight prohibits network')
            if event=='open' and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt'):
                raise RuntimeError('CPU preflight prohibits checkpoint tensor access')
        sys.addaudithook(audit)
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Unexpected GPU initialization on Torch import')
        def forbidden(*a,**k):raise RuntimeError('CPU preflight forbids model loading/forward/GPU/compilation')
        torch.cuda._lazy_init=forbidden;torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        torch.nn.Module.__call__=forbidden
        import safetensors,safetensors.torch
        safetensors.safe_open=forbidden;safetensors.torch.load_file=forbidden
        from torch.utils._python_dispatch import TorchDispatchMode
        class CPUOnly(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                from torch.utils._pytree import tree_leaves
                values=tree_leaves((args,kwargs or {}))
                if any((isinstance(v,torch.Tensor) and v.device.type not in ('cpu','meta')) or
                       (isinstance(v,torch.device) and v.type not in ('cpu','meta')) for v in values):return forbidden()
                return func(*args,**(kwargs or {}))
        with CPUOnly():
            with torch.device('meta'):model=local.factory(expected['config'])
            actual=loading.tensor_manifest(model.state_dict())
            import amd_decoder_contract as policy
            policy.validate_tensors(expected['tensors'],actual,'meta')
            buffers=loading.tensor_manifest(dict(model.named_buffers()));local.contract.validate_buffers(buffers)
            cache=local.rope_cache()
            if cache.cache_info().currsize!=0:raise RuntimeError('Pre-existing DINO cache entries')
            ext=local.extractor(model)
            pipeline=local.pipeline_methods(ext)
            from alpha_input import preprocess_alpha_image
            image=local.fixture(None);processed=preprocess_alpha_image(pipeline,image)
            record.update(status='PASS',state=actual,buffers=buffers,strict_header_match=True,
                checkpoint_sha256=expected['checkpoint_sha256'],gpu_initialized=torch.cuda.is_initialized(),
                preprocessing={'input_mode':image.mode,'input_size':list(image.size),'output_mode':processed.mode,'output_size':list(processed.size)},
                model_class=type(model).__module__+'.'+type(model).__name__,attention_backend=model.config._attn_implementation,
                extractor_class=type(ext).__name__,expected_output=local.contract.expected_shape(expected['config']),
                cache_policy={'maxsize':cache.cache_parameters()['maxsize'],'entries':cache.cache_info().currsize})
    except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc())
    print('DINO_METADATA='+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1


def supervise(output):
    output.mkdir();scratch=output/'scratch';scratch.mkdir();child=None;job=OwnedJob();record={}
    started=time.monotonic()
    try:
        with (output/'stdout.txt').open('xb') as stdout,(output/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen([str(local.VENV/'Scripts/python.exe'),'-B','-I',str(Path(__file__).resolve()),'--worker','--output',str(scratch)],
                stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env={**os.environ,**loading.gpu_proof.scratch_environment(scratch)})
            job.attach(child);child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush();child.stdin.close()
            child.wait(timeout=120)
            record.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started,timed_out=False)
    except BaseException as exc:record.update(error=repr(exc),timed_out=isinstance(exc,subprocess.TimeoutExpired))
    finally:
        record['job_cleanup']=job.finish()
        if child is not None:loading.reap_owned_child(child,record)
    text=(output/'stdout.txt').read_text(encoding='utf-8')
    rows=[json.loads(line.split('=',1)[1]) for line in text.splitlines() if line.startswith('DINO_METADATA=')]
    if len(rows)==1:record['worker']=rows[0]
    else:record['error']='Missing/duplicate metadata result'
    (output/'results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    passed=(record.get('returncode')==0 and record.get('worker',{}).get('status')=='PASS'
            and record.get('worker_exit_verified') is True and not record['job_cleanup']['forced_job_cleanup'])
    print(json.dumps({'passed':passed,'process':{k:v for k,v in record.items() if k!='worker'},'worker_summary':{k:v for k,v in record.get('worker',{}).items() if k!='state'}}))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raise SystemExit(worker(args.output) if args.worker else supervise(args.output))
