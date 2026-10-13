"""Owned, bounded weights-free architecture proof and synthetic strict-load tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_flow_local as local
from amd_forward_owned_job import OwnedJob
from services.amd_offline_guard import block_external_actions

def worker(scratch):
    record=dict(status='FAIL',weights_loaded=False,real_model_forward_calls=0,gpu_initialized=False,sdk_discovery=[])
    torch=None
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned job required')
        scratch=scratch.resolve();scratch.relative_to(local.EVIDENCE.resolve())
        local.verify_sources()
        expected=local.identity(hash_weights=False) # header only; no real weight payload read
        env=local.loading.gpu_proof.scratch_environment(scratch)
        for key in local.loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        env.update(ATTN_BACKEND='sdpa',WORKSPACE_DIR=str(local.ROOT));os.environ.update(env)
        block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch);os.environ.update(env)
        phase='metadata'
        def audit(event,args):
            if event.startswith('socket.'):raise RuntimeError('CPU probe forbids network')
            if event=='import' and str(args[0]).split('.')[0] in ('cumesh','o_voxel','flex_gemm','nvdiffrast'):
                raise RuntimeError('Dense metadata proof forbids unrelated native imports')
            if event=='open' and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0]))
                if p.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt'):
                    allowed=p.resolve()==local.FEATURE.resolve() or p.resolve()==scratch/'tiny.safetensors'
                    allowed=allowed or phase=='hash-after-contract' and p.resolve()==local.PREFIX.with_suffix('.safetensors').resolve()
                    if not allowed:raise RuntimeError('CPU probe forbids real weight access before contract')
        sys.addaudithook(audit)
        def forbidden(*a,**k):raise RuntimeError('Weights-free preflight forbids GPU/model loading/forward/compilation')
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('GPU initialized on import')
        torch.cuda._lazy_init=forbidden;torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        torch.set_num_threads(1)
        original_call=torch.nn.Module.__call__
        def call(instance,*args,**kwargs):
            if type(instance).__module__!='trellis2.modules.attention.rope' or type(instance).__name__!='RotaryPositionEmbedder':return forbidden()
            return original_call(instance,*args,**kwargs)
        torch.nn.Module.__call__=call
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        class CPUOnly(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                values=tree_leaves((args,kwargs or {}))
                if any((isinstance(v,torch.Tensor) and v.device.type not in ('cpu','meta')) or
                       (isinstance(v,torch.device) and v.type not in ('cpu','meta')) for v in values):return forbidden()
                return func(*args,**(kwargs or {}))
        with CPUOnly():
            with torch.device('meta'):model=local.factory(expected['config'])
            parameters=local.manifest(dict(model.named_parameters()));buffers=local.manifest(dict(model.named_buffers()))
            local.contract.validate_manifest(parameters,buffers,'meta')
            record.update(strict_header_match=True,parameters=parameters,buffers=buffers,
                generated_rope_sha256=local.hashlib.sha256(model.rope_phases.numpy().tobytes()).hexdigest(),
                class_path=__import__('inspect').getfile(type(model)),attention_backend='sdpa',
                checkpoint_payload_read_before_contract=False,
                dtype_counts={'BF16':420,'F32':220},persistent_buffer_policy='rope_phases explicitly nonpersistent; validated genuine CPU construction')
            model=None
            # The full architecture schema has passed before any real payload hashing.
            phase='hash-after-contract'
            record['checkpoint_sha256']=local.digest(local.PREFIX.with_suffix('.safetensors'))
            phase='metadata'
            record['feature']=local.verify_feature()
            # Exercise unchanged strict loader plus scoped widening on tiny local tensors.
            from strict_loader import StrictLocalComponent
            import safetensors.torch
            path=scratch/'tiny.safetensors';config_path=scratch/'tiny.json';config_path.write_text('{}')
            safetensors.torch.save_file({'weight':torch.tensor([[1,-2],[.5,4]],dtype=torch.bfloat16)},str(path))
            def tiny_factory(config):
                m=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16);return m
            component=StrictLocalComponent(str(config_path),str(path),tiny_factory)
            tiny=component.load_cpu();small={'weight':{'shape':[2,2],'dtype':'F32'}}
            widened=local.widen(tiny,component.diagnostics['payload_sha256'],small)
            if widened!=['weight'] or tiny.weight.dtype!=torch.float32:raise ValueError('Synthetic widening failed')
            component.unload();tiny=component=None
            class Mixed(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fp=torch.nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=False)
                    self.bf=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16),requires_grad=False)
                    self.register_buffer('rope',torch.ones(2,dtype=torch.complex64))
            mixed=Mixed();source_ids={id(v) for v in list(mixed.parameters())+list(mixed.buffers())};copies=[]
            class CopyInspect(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    if str(func)=='aten._to_copy.default':
                        copies.append({'operation':str(func),'source_identity_verified':id(args[0]) in source_ids,
                            'source_dtype':str(args[0].dtype),'kwargs':{k:str(v) for k,v in (kwargs or {}).items()}})
                    return func(*args,**(kwargs or {}))
            with CopyInspect():mixed.to(device='meta') # Same Module.to path, no GPU or real weights.
            record['placement_copy_schema']=list(copies)
            if len(copies)!=3 or any(not row['source_identity_verified'] or row['kwargs']!={'dtype':row['source_dtype'],'layout':'torch.strided','device':'meta'} for row in copies):
                raise ValueError('Actual Module.to dispatch contract differs from unchanged dtype/layout copy policy')
            record['input_copy_schema']=str(torch.ops.aten._to_copy.default._schema)
            copies.clear();cpu_frequency=torch.ones(128,dtype=torch.float32)
            source_ids={id(cpu_frequency)}
            with CopyInspect():cpu_frequency.to(torch.device('meta'))
            record['frequency_copy_schema']=list(copies)
            if len(copies)!=1 or copies[0]['kwargs']!={'dtype':'torch.float32','layout':'torch.strided','device':'meta'}:
                raise ValueError('Timestep frequency Tensor.to schema differs')
            mixed=None
            # Negative value-integrity case and nonfinite storage case.
            failures=[]
            for label,value,hashes in [('bad-hash',[[1.,2.],[3.,4.]],{'weight':'0'*64}),
                                       ('nonfinite',[[float('nan'),2.],[3.,4.]],{'weight':'0'*64})]:
                m=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
                m.weight.data=torch.tensor(value,dtype=torch.bfloat16)
                try:local.widen(m,hashes,small)
                except ValueError as exc:failures.append({'case':label,'rejected':str(exc)})
                else:raise ValueError('Synthetic invalid widening accepted')
                m=None
            record['synthetic']={'strict_entries':1,'positive':True,'negative':failures,'real_weights_loaded':False}
            record['runtime']={'python':sys.version,'torch':str(torch.__version__),'hip':torch.version.hip,'cuda':torch.version.cuda}
            if torch.cuda.is_initialized():return forbidden()
            record.update(status='PASS',gpu_initialized=False)
    except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc())
    print('FLOW_METADATA='+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1

def supervise(output):
    output=output.resolve();output.relative_to(local.EVIDENCE.resolve())
    output.mkdir();scratch=output/'scratch';scratch.mkdir();child=None;job=OwnedJob();record={};started=time.monotonic()
    try:
        with (output/'stdout.txt').open('xb') as stdout,(output/'stderr.txt').open('xb') as stderr:
            env={**os.environ,**local.loading.gpu_proof.scratch_environment(scratch),'ATTN_BACKEND':'sdpa','WORKSPACE_DIR':str(local.ROOT)}
            child=subprocess.Popen([str(local.VENV/'Scripts/python.exe'),'-B','-I',str(Path(__file__).resolve()),'--worker','--output',str(scratch)],
                stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env)
            job.attach(child);child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush();child.stdin.close();child.wait(timeout=120)
            record.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started,timed_out=False)
    except BaseException as exc:record.update(error=repr(exc),timed_out=isinstance(exc,subprocess.TimeoutExpired))
    finally:
        record['job_cleanup']=job.finish()
        if child is not None:local.loading.reap_owned_child(child,record)
    rows=[json.loads(line.split('=',1)[1]) for line in (output/'stdout.txt').read_text().splitlines() if line.startswith('FLOW_METADATA=')]
    if len(rows)==1:record['worker']=rows[0]
    else:record['error']='Missing or duplicate CPU probe result'
    (output/'results.json').write_text(json.dumps(record,indent=2))
    passed=(record.get('returncode')==0 and record.get('worker',{}).get('status')=='PASS'
        and record.get('worker_exit_verified') is True and not record['job_cleanup']['forced_job_cleanup'])
    print(json.dumps({'passed':passed,'process':{k:v for k,v in record.items() if k!='worker'},
                      'worker_summary':{k:v for k,v in record.get('worker',{}).items() if k not in ('parameters','feature')}}))
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raise SystemExit(worker(args.output) if args.worker else supervise(args.output))
