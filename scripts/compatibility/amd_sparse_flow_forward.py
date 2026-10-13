"""Exactly one genuine image-conditioned dense sparse-structure flow forward."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import weakref
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_flow_local as local
import amd_sparse_flow_contract as contract
import amd_sparse_decoder_forward as supervisor
import amd_decoder_contract as strict_contract
import amd_forward_diagnostics as diagnostics
import amd_blaslt_teardown as teardown
from services.amd_offline_guard import block_external_actions

AUTHORIZE=b'EXECUTE_ONE_SPARSE_FLOW_FORWARD\n'
PROBE=local.EVIDENCE/'cpu-metadata-final-candidate/results.json'

def verify_preflight():
    local.verify_sources();teardown.verify_binding_evidence()
    probe=json.loads(PROBE.read_text())
    row=probe.get('worker',{})
    if (probe.get('returncode')!=0 or probe.get('worker_exit_verified') is not True
        or probe.get('job_cleanup',{}).get('forced_job_cleanup') is not False
        or row.get('status')!='PASS' or row.get('weights_loaded') is not False
        or row.get('gpu_initialized') is not False or row.get('strict_header_match') is not True
        or row.get('synthetic',{}).get('positive') is not True
        or len(row.get('placement_copy_schema',[]))!=3
        or any(not item.get('source_identity_verified') or item.get('kwargs')!={'dtype':item.get('source_dtype'),'layout':'torch.strided','device':'meta'} for item in row.get('placement_copy_schema',[]))
        or row.get('frequency_copy_schema')!=[{'operation':'aten._to_copy.default','source_identity_verified':True,
            'source_dtype':'torch.float32','kwargs':{'dtype':'torch.float32','layout':'torch.strided','device':'meta'}}]):
        raise RuntimeError('Successful CPU-only sparse-flow proof required')
    local.contract.validate_manifest(row['parameters'],row['buffers'],'meta')
    current=local.identity()
    if current['checkpoint_sha256']!=row['checkpoint_sha256']:raise RuntimeError('Checkpoint changed after metadata proof')
    if local.verify_feature()!=row['feature']:raise RuntimeError('Saved feature evidence changed')
    return current

def profile():
    files=list(supervisor.COMMITTED_FILES)
    files+=['scripts/compatibility/'+name+'.py' for name in ('amd_sparse_flow_forward','amd_sparse_flow_local',
        'amd_sparse_flow_contract','amd_sparse_flow_probe','amd_blaslt_teardown','amd_dinov3_conditioning',
        'amd_dinov3_local','amd_dinov3_contract','amd_feature_checkpoint')]
    files += [p.relative_to(local.ROOT).as_posix() for p in (PROBE,local.PINS,local.AUDIT,local.FEATURE,
        teardown.PINS,teardown.EVIDENCE/'binary-review.json',local.ROOT/'docs/compatibility/amd-dinov3-allocation-evidence/run/results.json')]
    def fresh(scratch):return {'policy':'fresh private caches; no historical artifacts executed','copied_entries':0}
    return SimpleNamespace(kind='sparse-flow',evidence=local.EVIDENCE,cpu_seconds=contract.CPU_SECONDS,
        gpu_seconds=contract.GPU_SECONDS,overall_seconds=contract.OVERALL_SECONDS,identity=verify_preflight,
        source_identity=lambda:(local.verify_sources(),teardown.verify_binding_evidence()),validate_config=lambda c:contract.validate_header(c,json.loads(local.AUDIT.read_text())['tensors']),
        classify=contract.classify,prepare_private_cache=fresh,worker_script=Path(__file__).resolve(),
        authorize=AUTHORIZE,authorize_placement=contract.authorize_placement,committed_files=files)

def describe(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device)}

def release(torch,component,refs,trace):
    """Called after caller has discarded every owned strong GPU reference."""
    component.unload();gc.collect()
    trace.call('synchronize:before-workspace-clear',lambda:torch.cuda.synchronize(0))
    trace.call('blas:clear-workspaces',torch._C._cuda_clearCublasWorkspaces)
    torch.cuda.empty_cache();torch.cuda.synchronize(0)
    row={**local.loading.gpu_memory(torch),'model_collected':refs[0]() is None,
         'owned_tensors_collected':all(ref() is None for ref in refs[1:]),'cpu_memory':local.loading.process_memory()}
    trace.emit('memory','release',memory=row)
    strict_contract.validate_release(row)
    if row['owned_tensors_collected'] is not True:raise RuntimeError('Sparse-flow owned tensors survive cleanup')
    return row

def worker(scratch):
    record={'status':'FAIL','stage':'startup','sdk_discovery':[],'calls':{},'load_calls':0}
    component=model=output=cpu_output=cpu_inputs=gpu_inputs=torch=None
    phase='cpu';module_names={};placement_sources=set();copy_source=None;in_timestep=False
    transfers={'x':0,'t':0,'cond':0,'output':0,'frequency':0};permit=contract.SingleForward();op_counts={}
    trace=diagnostics.Trace(lambda line:print(line,flush=True))
    def stage(name):
        record['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name,phase=phase)
    def forbidden(*a,**k):raise RuntimeError('Unapproved sparse-flow operation in phase '+phase)
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job handshake required')
        scratch=scratch.resolve();scratch.relative_to(local.EVIDENCE.resolve())
        env=local.loading.gpu_proof.scratch_environment(scratch)
        env.update(ATTN_BACKEND='sdpa',WORKSPACE_DIR=str(local.ROOT))
        for key in local.loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env)
        record['sdk_verification']=block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        os.environ.update(env)
        checkpoint=local.PREFIX.with_suffix('.safetensors').resolve()
        def audit(event,args):
            if event.startswith('socket.'):return forbidden()
            if event=='import' and contract.prohibited_import(str(args[0])):return forbidden()
            if event=='open' and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0]))
                if p.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt') and p.resolve() not in (checkpoint,local.FEATURE.resolve()):return forbidden()
        sys.addaudithook(audit)
        verified=verify_preflight();stage('runtime-import')
        import torch
        if (Path(sys.executable).resolve()!=local.VENV/'Scripts/python.exe' or sys.version.split()[0]!='3.12.10'
            or str(torch.__version__)!='2.13.0+rocm10.0.0' or torch.version.hip!='7.15.26333'
            or torch.version.cuda is not None or torch.cuda.is_initialized()):raise RuntimeError('Unexpected AMD runtime identity/context')
        record['runtime']={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
            'torch_path':torch.__file__,'hip':torch.version.hip,'cuda':torch.version.cuda,
            'git_version':torch.version.git_version,'dependencies':{n:importlib.metadata.version(n) for n in ('safetensors','numpy')}}
        trace.emit('state','runtime',runtime=record['runtime'])
        record['workspace_binding']=teardown.verify_runtime_binding(torch)
        torch.set_num_threads(1);original_init=torch.cuda._lazy_init;torch.cuda._lazy_init=forbidden
        torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        original_call=torch.nn.Module.__call__
        def module_call(instance,*args,**kwargs):
            nonlocal in_timestep
            if phase=='cpu' and type(instance).__module__=='trellis2.modules.attention.rope' and type(instance).__name__=='RotaryPositionEmbedder':
                return original_call(instance,*args,**kwargs)
            if phase!='forward' or id(instance) not in module_names:return forbidden()
            name=module_names[id(instance)]
            selected=(name=='' or name.startswith('blocks.') and (name.count('.')==1 or name.endswith(('.self_attn','.cross_attn'))))
            if selected:
                key=name or '<root>';record['calls'][key]=record['calls'].get(key,0)+1
                if record['calls'][key]!=1:return forbidden()
            previous=in_timestep
            if name=='t_embedder':in_timestep=True
            try:
                run=lambda:original_call(instance,*args,**kwargs)
                result=trace.call('module:'+(name or '<root>'),run,input=describe(args[0]) if args else {}) if selected else run()
                if name.startswith('blocks.') and name.count('.')==1:
                    memory=local.loading.gpu_memory(torch);trace.emit('memory','after-'+name,memory=memory)
                    if memory['allocated_bytes']>contract.MEMORY_LIMIT or memory['reserved_bytes']>contract.MEMORY_LIMIT:
                        raise RuntimeError('Sparse-flow memory budget exceeded')
                return result
            finally:in_timestep=previous
        torch.nn.Module.__call__=module_call
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        def gpu(values):return any((isinstance(v,torch.Tensor) and v.device.type=='cuda') or
            (isinstance(v,torch.device) and v.type=='cuda') or isinstance(v,str) and v.startswith('cuda') for v in tree_leaves(values))
        frequency=torch.exp(-__import__('numpy').log(10000)*torch.arange(128,dtype=torch.float32)/128)
        class BoundedDispatch(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};name=str(func);values=tree_leaves((args,kwargs));is_gpu=gpu(values)
                if not is_gpu:return func(*args,**kwargs)
                if any((isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.device.index!=0)
                    or isinstance(v,torch.device) and v.type=='cuda' and v.index not in (None,0) for v in values):return forbidden()
                if phase=='cpu':return forbidden()
                if phase=='placement':
                    if not strict_contract.gpu_operation_allowed(name):return forbidden()
                    if name=='aten._to_copy.default':
                        if (len(args)!=1 or id(args[0]) not in placement_sources or kwargs.get('device')!=torch.device('cuda:0')
                            or set(kwargs)-{'device','dtype','layout','memory_format'}
                            or kwargs.get('dtype',args[0].dtype)!=args[0].dtype
                            or kwargs.get('layout',torch.strided)!=torch.strided
                            or kwargs.get('memory_format',torch.preserve_format)!=torch.preserve_format):return forbidden()
                elif phase in ('inputs','output'):
                    key,source,target=copy_source
                    if (name!='aten._to_copy.default' or len(args)!=1 or args[0] is not source
                        or kwargs!={'device':torch.device(target)} or transfers[key]):return forbidden()
                    contract.validate_transfer(key,describe(source),target,phase)
                    transfers[key]=1
                elif phase=='forward':
                    if not name.startswith('aten.'):return forbidden()
                    if kwargs.get('dtype') is not None and kwargs['dtype'] not in (torch.float32,torch.bfloat16,torch.complex64):return forbidden()
                    if any(isinstance(v,torch.Tensor) and v.device.type=='cuda' and (v.is_floating_point() or v.is_complex())
                           and v.dtype not in (torch.float32,torch.bfloat16,torch.complex64) for v in values):return forbidden()
                    if name=='aten._to_copy.default':
                        if (set(kwargs)-{'dtype','device','layout','memory_format'} or kwargs.get('layout',torch.strided)!=torch.strided
                            or kwargs.get('memory_format',torch.preserve_format)!=torch.preserve_format):return forbidden()
                        source=args[0]
                        if source.device.type=='cpu':
                            if (not in_timestep or transfers['frequency'] or describe(source)!={'shape':[128],'dtype':'torch.float32','device':'cpu'}
                                or kwargs.get('device')!=torch.device('cuda:0')
                                or kwargs.get('dtype',torch.float32)!=torch.float32 or not torch.equal(source,frequency)):return forbidden()
                            transfers['frequency']=1
                        elif (kwargs.get('device',torch.device('cuda:0'))!=torch.device('cuda:0') or
                            kwargs.get('dtype',source.dtype)!=source.dtype and {kwargs['dtype'],source.dtype}!={torch.float32,torch.bfloat16}):return forbidden()
                    op_counts[name]=op_counts.get(name,0)+1
                    if len(op_counts)>128:return forbidden()
                elif phase=='release':
                    if not strict_contract.gpu_operation_allowed(name):return forbidden()
                else:return forbidden()
                if phase=='forward' and name in ('aten._scaled_dot_product_flash_attention.default','aten._scaled_dot_product_attention_math.default','aten._scaled_dot_product_efficient_attention.default'):
                    return trace.call('aten:'+name,lambda:func(*args,**kwargs),input=describe(args[0]))
                return func(*args,**kwargs)
        with BoundedDispatch():
            cls=local.import_class();from trellis2 import models;models.from_pretrained=forbidden
            import triton
            triton.compile=forbidden;triton.runtime.JITFunction.run=forbidden;triton.runtime.Autotuner.run=forbidden
            import safetensors.torch
            original_load=safetensors.torch.load_file
            def load_only(path,device='cpu',**kwargs):
                if Path(path).resolve()!=checkpoint or device!='cpu' or kwargs or record['load_calls']:return forbidden()
                record['load_calls']=1;return original_load(str(checkpoint),device='cpu')
            safetensors.torch.load_file=load_only
            import psutil
            if psutil.virtual_memory().available<verified['tensor_data_bytes']*4+1024**3:raise RuntimeError('Insufficient host strict-load margin')
            stage('strict-CPU-load');began=time.monotonic();before=local.loading.process_memory()
            component,model=local.load_cpu();feature=local.verify_feature()
            from safetensors.numpy import load_file
            import numpy as np
            saved=load_file(str(local.FEATURE))
            if set(saved)!={'cond'}:return forbidden()
            cpu_inputs={'x':torch.from_numpy(((np.arange(32768,dtype=np.int32)%257-128)/128).astype(np.float32)).reshape(1,8,16,16,16),
                        't':torch.tensor([500.],dtype=torch.float32),'cond':torch.from_numpy(saved['cond'].copy())}
            saved=None
            record['input']={k:{**describe(v),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in cpu_inputs.items()}
            if record['input']['cond']['sha256']!=feature['output_sha256']:raise RuntimeError('Loaded conditioning values differ from verified checkpoint')
            record['feature']=feature
            record['cpu_load']=dict(status='PASS',strict=True,assign=True,missing_keys=[],unexpected_keys=[],gpu_initialized=False,
                parameter_bytes=verified['tensor_data_bytes'],payload_entries=640,reversible_widening_count=220,feature_verified=True,
                tensors=local.manifest(dict(model.named_parameters())),buffers=local.manifest(dict(model.named_buffers())),
                elapsed_seconds=time.monotonic()-began,memory_before=before,memory_after=local.loading.process_memory(),loader_diagnostics=component.diagnostics.copy())
            contract.authorize_placement(record['cpu_load'],verified,verified['checkpoint_sha256'],verified['checkpoint_sha256'])
            if torch.cuda.is_initialized():return forbidden()
            print(supervisor.EVENT+json.dumps({'event':'cpu-ready','gate_one':record['cpu_load'],'checkpoint_sha256':verified['checkpoint_sha256']}),flush=True)
            stage('waiting-for-observer')
            if sys.stdin.readline().encode()!=AUTHORIZE:raise RuntimeError('GPU forward not authorized')
            phase='placement';torch.cuda._lazy_init=original_init;stage('GPU-metadata')
            torch.cuda.set_device(0);p=torch.cuda.get_device_properties(0)
            if 'RX 9070 XT' not in p.name or p.gcnArchName.split(':')[0]!='gfx1201':raise RuntimeError('Unexpected GPU')
            record['gpu']={'name':p.name,'architecture':p.gcnArchName,'total_bytes':p.total_memory};trace.emit('state','gpu',gpu=record['gpu'])
            torch.cuda.reset_peak_memory_stats(0);record['before_placement']=local.loading.gpu_memory(torch)
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<contract.MEMORY_LIMIT+1024**3:
                raise RuntimeError('Unexpected allocation or less than13GiB available VRAM')
            torch.cuda.set_per_process_memory_fraction(contract.MEMORY_LIMIT/p.total_memory,0)
            record['budget']={'allocator_limit_bytes':contract.MEMORY_LIMIT,'cpu_seconds':180,'gpu_seconds':240,'overall_seconds':480}
            trace.emit('memory','before-placement',memory=record['before_placement'])
            placement_sources={id(v) for v in list(model.parameters())+list(model.buffers())}
            stage('flow-placement');began=time.monotonic();model.to(device='cuda:0');torch.cuda.synchronize(0)
            contract.validate_manifest(local.manifest(dict(model.named_parameters())),local.manifest(dict(model.named_buffers())),'cuda:0')
            record['placement_seconds']=time.monotonic()-began;module_names={id(v):name for name,v in model.named_modules()}
            with torch.inference_mode():
                gpu_inputs={};phase='inputs';stage('input-transfer')
                for key in ('x','t','cond'):
                    copy_source=(key,cpu_inputs[key],'cuda:0')
                    gpu_inputs[key]=torch.ops.aten._to_copy.default(cpu_inputs[key],device=torch.device('cuda:0'))
                    if describe(gpu_inputs[key])!={**describe(cpu_inputs[key]),'device':'cuda:0'}:return forbidden()
                copy_source=None;phase='placement';torch.cuda.synchronize(0)
                record['before_forward']=local.loading.gpu_memory(torch);trace.emit('memory','before-forward',memory=record['before_forward'])
                if torch.is_autocast_enabled('cuda') or torch.is_grad_enabled():return forbidden()
                phase='forward';stage('single-forward');began=time.monotonic()
                output=permit.run(lambda:model(gpu_inputs['x'],gpu_inputs['t'],gpu_inputs['cond']))
                phase='placement';trace.call('synchronize:after-forward',lambda:torch.cuda.synchronize(0));record['forward_seconds']=time.monotonic()-began
                record['after_forward']=local.loading.gpu_memory(torch);trace.emit('memory','after-forward',memory=record['after_forward'])
                phase='output';stage('output-copy-validation');copy_source=('output',output,'cpu')
                cpu_output=torch.ops.aten._to_copy.default(output,device=torch.device('cpu'))
                copy_source=None;phase='placement';torch.cuda.synchronize(0);values=cpu_output.numpy()
                record['output']={**describe(output),'all_finite':bool(np.isfinite(values).all()),'count':values.size,
                    'requires_grad':output.requires_grad,'is_inference_tensor':output.is_inference(),'grad_enabled':torch.is_grad_enabled(),
                    'minimum':float(values.min()),'maximum':float(values.max()),'mean':float(values.mean()),'stddev':float(values.std()),
                    'representative_values':values.reshape(-1)[:8].tolist(),'sha256':hashlib.sha256(values.tobytes()).hexdigest()}
                contract.validate_output(record['output']);values=None
            stage('release');phase='release';refs=[weakref.ref(model),weakref.ref(output),*[weakref.ref(v) for v in gpu_inputs.values()]]
            model=output=gpu_inputs=cpu_inputs=cpu_output=None
            record['workspace_clear_calls']=1;record['release']=release(torch,component,refs,trace);component=None
            stage('complete');record['status']='PASS'
    except BaseException as exc:record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        record.update(forward_calls=permit.calls,transfers=transfers,gpu_operation_counts=op_counts)
        if component is not None:component.unload()
        component=model=output=cpu_output=cpu_inputs=gpu_inputs=copy_source=None;gc.collect()
        print(supervisor.RESULT+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--harness-commit')
    args=p.parse_args();teardown.verify_workspace()
    if args.worker:raise SystemExit(worker(args.output))
    verify_preflight();raise SystemExit(supervisor.supervise(args.output,args.harness_commit,profile=profile()))
