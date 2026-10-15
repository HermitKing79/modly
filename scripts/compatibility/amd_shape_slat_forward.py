"""Exactly one genuine sparse shape-SLat512 forward, isolated from application routes."""
import argparse, gc, hashlib, importlib.metadata, json, os, sys, time, traceback, weakref
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_slat_local as local
import amd_shape_slat_contract as contract
import amd_sparse_decoder_forward as supervisor
import amd_sparse_flow_forward as proven
import amd_decoder_contract as strict
import amd_forward_diagnostics as diagnostics
import amd_blaslt_teardown as teardown
from services.amd_offline_guard import block_external_actions
AUTHORIZE=b'EXECUTE_ONE_SHAPE_SLAT_FORWARD\n'
PROBE=local.EVIDENCE/'cpu-probe-corrected/results.json'
def restore_execution_environment(env):
    # Offline guard deliberately masks GPUs. Match the proven worker's restoration
    # while _lazy_init remains forbidden until the supervisor authorizes placement.
    os.environ.update(env)
    visibility={k:os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')}
    if any(v!='0' for v in visibility.values()):raise RuntimeError('Execution requires exactly the approved GPU0 visibility')
    return visibility
def verify_preflight():
    local.verify_sources();teardown.verify_workspace();teardown.verify_binding_evidence()
    p=json.loads(PROBE.read_text());r=p.get('worker',{})
    if (p.get('returncode')!=0 or p.get('worker_exit_verified') is not True or p.get('job_cleanup',{}).get('forced_job_cleanup') is not False
        or r.get('status')!='PASS' or r.get('weights_loaded') is not False or r.get('gpu_initialized') is not False
        or r.get('strict_header_match') is not True or r.get('pooling_torch_equal') is not True or r.get('cpu_sparse_layout_verified') is not True
        or r.get('synthetic',{}).get('strict_positive') is not True or len(r.get('synthetic',{}).get('negative',[]))!=4):raise RuntimeError('CPU-only architecture/strict-load proof incomplete')
    result=local.identity()
    if r['checkpoint_sha256']!=result['checkpoint_sha256'] or r['feature']!=result['feature'] or r['token_count']!=result['token_count']:raise ValueError('CPU evidence identity mismatch')
    return result
def profile():
    files=list(supervisor.COMMITTED_FILES)
    files += ['scripts/compatibility/'+n+'.py' for n in ('amd_shape_slat_forward','amd_shape_slat_local','amd_shape_slat_contract','amd_shape_slat_probe',
        'amd_sparse_flow_local','amd_sparse_flow_contract','amd_sparse_flow_forward','amd_blaslt_teardown','amd_dinov3_local','amd_dinov3_contract','amd_dinov3_conditioning','amd_feature_checkpoint')]
    files += [p.relative_to(local.ROOT).as_posix() for p in (PROBE,local.PINS,local.AUDIT,local.FEATURE,teardown.PINS,
        teardown.EVIDENCE/'binary-review.json',local.TURRET/'occupancy-logits.npy',local.TURRET/'occupancy-mask.npy',local.TURRET/'dino/run/results.json')]
    return SimpleNamespace(kind='shape-slat',evidence=local.EVIDENCE,cpu_seconds=180,gpu_seconds=240,overall_seconds=480,
        identity=verify_preflight,source_identity=lambda:(local.verify_sources(),teardown.verify_binding_evidence()),
        validate_config=lambda c:contract.validate_header(c,json.loads(local.AUDIT.read_text())['tensors']),classify=contract.classify,
        prepare_private_cache=lambda s:dict(policy='fresh private caches; historical artifacts never executed',copied_entries=0),
        worker_script=Path(__file__).resolve(),authorize=AUTHORIZE,authorize_placement=contract.authorize_placement,committed_files=files)
def describe(t):return dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
class CopyPolicy:
    """Production transfer validator, directly exercised by CPU-only tests."""
    def __init__(self,torch,n):
        self.contract=contract
        self.torch,self.n=torch,n;self.phase='cpu';self.active=None;self.placement=set();self.frequency=None;self.rope=None
        self.scopes=[];self.counts={k:0 for k in ('x','coords','batch_map','t','cond','output','output_coords','frequency','rope_frequency')}
    def reject(self):raise RuntimeError('Unapproved shape-SLat operation in phase '+self.phase)
    def check_copy(self,name,args,kwargs):
        torch=self.torch
        if len(args)!=1 or not isinstance(args[0],torch.Tensor):return self.reject()
        source=args[0]
        if self.phase in ('inputs','output'):
            if self.active is None:return self.reject()
            key,expected,target=self.active
            if name!='aten._to_copy.default' or source is not expected or kwargs!={'device':torch.device(target)} or self.counts[key]:return self.reject()
            self.contract.validate_transfer(key,describe(source),target,self.phase,self.n);self.counts[key]+=1;return
        allowed={'dtype','layout','device','memory_format'}
        if name=='aten.to.dtype_layout':allowed|={'pin_memory','non_blocking','copy'}
        if (set(kwargs)-allowed or kwargs.get('device',torch.device('cuda:0'))!=torch.device('cuda:0')
            or kwargs.get('layout',torch.strided) not in (None,torch.strided) or kwargs.get('memory_format') not in (None,torch.preserve_format)
            or kwargs.get('pin_memory') is not None and kwargs['pin_memory'] is not False
            or kwargs.get('non_blocking',False) is not False or kwargs.get('copy',False) is not False):return self.reject()
        dtype=kwargs.get('dtype',source.dtype)
        if self.phase=='placement':
            if name!='aten._to_copy.default' or id(source) not in self.placement or dtype!=source.dtype:return self.reject()
        elif self.phase=='forward' and source.device.type=='cpu':
            reference=key=None
            if 't_embedder' in self.scopes:reference=self.frequency;key='frequency'
            elif 'blocks.0.self_attn.rope' in self.scopes:reference=self.rope;key='rope_frequency'
            if key is None or source is not reference or self.counts[key] or kwargs.get('device')!=torch.device('cuda:0') or dtype not in (None,torch.float32):return self.reject()
            expected=torch.exp(-__import__('numpy').log(10000)*torch.arange(128,dtype=torch.float32)/128) if key=='frequency' else local.rope_frequencies()
            if source.dtype!=torch.float32 or source.shape!=expected.shape or not torch.equal(source,expected):return self.reject()
            self.counts[key]+=1
        elif self.phase=='forward':
            if source.device!=torch.device('cuda:0') or dtype not in (None,source.dtype) and {dtype,source.dtype}!={torch.float32,torch.bfloat16}:return self.reject()
        else:return self.reject()
def worker(scratch,sampling_factory=None,*,runtime=None):
    # A per-run adapter may supply independently validated image artifacts. The
    # historical proof retains its exact local inputs and checks by default.
    local=globals()['local'] if runtime is None else runtime.local
    record=dict(status='FAIL',stage='startup',sdk_discovery=[],calls={},load_calls=0,forward_calls=0)
    component=model=output=cpu_output=cpu_coords=inputs=gpu_inputs=sparse=torch=policy=None
    sampling=sampler_result=original_tensor=None
    active_contract=contract if sampling_factory is None else __import__('amd_shape_sampling_contract')
    permit=contract.SingleForward();trace=diagnostics.Trace(lambda line:print(line,flush=True),limit=512 if sampling_factory is None else 192);op_counts={};modules={}
    def forbidden(*a,**k):raise RuntimeError('Unapproved shape-SLat action')
    def stage(name):record['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name,phase=policy.phase if policy else 'cpu')
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job handshake required')
        scratch=scratch.resolve();scratch.relative_to(local.EVIDENCE.resolve())
        env=local.loading.gpu_proof.scratch_environment(scratch)
        for key in local.loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        env.update(ATTN_BACKEND='sdpa',WORKSPACE_DIR=str(local.ROOT));os.environ.update(env)
        record['sdk_verification']=block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        record['gpu_visibility']=restore_execution_environment(env)
        checkpoint=local.PREFIX.with_suffix('.safetensors').resolve()
        def audit(e,a):
            if e.startswith('socket.'):return forbidden()
            if e=='import' and active_contract.prohibited_import(str(a[0])):return forbidden()
            if e=='open' and isinstance(a[0],(str,bytes)):
                p=Path(os.fsdecode(a[0]))
                readable=(checkpoint,local.FEATURE.resolve()) if runtime is None else (checkpoint,local.FEATURE.resolve(),*runtime.readable_tensor_paths)
                if p.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt') and p.resolve() not in readable:return forbidden()
        sys.addaudithook(audit)
        verified=verify_preflight() if runtime is None else runtime.verify_preflight()
        record['token_count']=verified['token_count'];stage('runtime-import')
        import torch
        if torch.cuda.is_initialized():return forbidden()
        record['runtime']=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),torch_path=torch.__file__,hip=torch.version.hip,cuda=torch.version.cuda,
            git_version=torch.version.git_version,dependencies={n:importlib.metadata.version(n) for n in ('numpy','safetensors')})
        trace.emit('state','runtime',runtime=record['runtime']);record['workspace_binding']=teardown.verify_runtime_binding(torch)
        torch.set_num_threads(1);original_init=torch.cuda._lazy_init;torch.cuda._lazy_init=forbidden
        torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        if sampling_factory is not None:
            sampling=sampling_factory(torch)
            record['sampler']=dict(class_name=sampling.sampler_class.__name__,source=__import__('inspect').getfile(sampling.sampler_class),parameters=sampling.policy.PARAMS,sigma_min=sampling.policy.SIGMA_MIN)
        policy=CopyPolicy(torch,verified['token_count']) if sampling is None else sampling.copy_policy(verified['token_count'])
        original_call=torch.nn.Module.__call__
        def module_call(instance,*args,**kwargs):
            if policy.phase!='forward' or id(instance) not in modules:return forbidden()
            name=modules[id(instance)];policy.scopes.append(name)
            selected=name=='' or name.startswith('blocks.') and (name.count('.')==1 or name.endswith(('.self_attn','.cross_attn','.rope')))
            try:
                if sampling is not None and name=='':sampling.begin_model(args,kwargs,policy)
                if selected and not name.endswith('.rope'):
                    key=name or '<root>';record['calls'][key]=record['calls'].get(key,0)+1
                    if sampling is None and record['calls'][key]!=1:return forbidden()
                    if sampling is not None:sampling.module_enter(key)
                if selected and (sampling is None or name==''):
                    shape_input=describe(args[0].feats) if sampling is not None and name=='' else describe(args[0])
                    details=dict(input=shape_input)
                    if sampling is not None:details['invocation']=sampling.budget.active.copy()
                    result=trace.call('module:'+(name or '<root>'),lambda:original_call(instance,*args,**kwargs),**details)
                else:result=original_call(instance,*args,**kwargs)
                if name.startswith('blocks.') and name.count('.')==1:
                    memory=local.loading.gpu_memory(torch)
                    if sampling is None:trace.emit('memory','after-'+name,memory=memory)
                    else:sampling.block_memory(memory)
                    if memory['allocated_bytes']>contract.MEMORY_LIMIT or memory['reserved_bytes']>contract.MEMORY_LIMIT:raise RuntimeError('Shape-SLat memory budget exceeded')
                if sampling is not None and name=='':sampling.end_model(result,policy)
                return result
            finally:policy.scopes.pop()
        torch.nn.Module.__call__=module_call
        original_tensor=torch.tensor
        def timestep_tensor(*args,**kwargs):
            if sampling is None or policy.phase!='forward':return forbidden()
            sampling.next_timestep(args,kwargs)
            cpu=original_tensor(*args,device='cpu',dtype=torch.float32)
            previous=policy.phase;policy.phase='inputs';policy.active=('t',cpu,'cuda:0');policy.counts['t']=0
            try:
                placed=torch.ops.aten._to_copy.default(cpu,device=torch.device('cuda:0'));sampling.staged_timestep(cpu,placed);return placed
            finally:policy.active=None;policy.phase=previous
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        class BoundedDispatch(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};name=str(func);values=tree_leaves((args,kwargs))
                is_gpu=any(isinstance(v,torch.Tensor) and v.device.type=='cuda' or isinstance(v,torch.device) and v.type=='cuda' or isinstance(v,str) and v.startswith('cuda') for v in values)
                if name=='aten.to.dtype_layout' and is_gpu:policy.check_copy(name,args,kwargs)
                if not is_gpu:
                    result=func(*args,**kwargs)
                    if policy.phase=='forward' and 't_embedder' in policy.scopes and name=='aten.exp.default':
                        if policy.frequency is not None:return forbidden()
                        policy.frequency=result
                    return result
                if any(isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.device.index!=0 or isinstance(v,torch.device) and v.type=='cuda' and v.index not in (None,0) for v in values):return forbidden()
                if policy.phase in ('placement','release'):
                    if not strict.gpu_operation_allowed(name):return forbidden()
                    if name=='aten._to_copy.default':policy.check_copy(name,args,kwargs)
                elif policy.phase in ('inputs','output'):policy.check_copy(name,args,kwargs)
                elif policy.phase=='forward':
                    if not name.startswith('aten.'):return forbidden()
                    if name.startswith('aten.to.') and name!='aten.to.dtype_layout' and args and isinstance(args[0],torch.Tensor) and args[0].device.type=='cpu':return forbidden()
                    if name=='aten.copy_.default' and len(args)>1 and isinstance(args[1],torch.Tensor) and args[1].device.type=='cpu':return forbidden()
                    if any(isinstance(v,torch.Tensor) and v.device.type=='cuda' and (v.is_floating_point() or v.is_complex()) and v.dtype not in (torch.float32,torch.bfloat16,torch.complex64) for v in values):return forbidden()
                    if kwargs.get('dtype') is not None and kwargs['dtype'] not in (torch.float32,torch.bfloat16,torch.complex64,torch.int32,torch.int64):return forbidden()
                    if name=='aten._to_copy.default':policy.check_copy(name,args,kwargs)
                    op_counts[name]=op_counts.get(name,0)+1
                    if len(op_counts)>128:return forbidden()
                else:return forbidden()
                if policy.phase=='forward' and sampling is None and name in ('aten._scaled_dot_product_flash_attention.default','aten._scaled_dot_product_attention_math.default','aten._scaled_dot_product_efficient_attention.default'):
                    return trace.call('aten:'+name,lambda:func(*args,**kwargs),input=describe(args[0]))
                return func(*args,**kwargs)
        with BoundedDispatch():
            local.import_class();from trellis2 import models;models.from_pretrained=forbidden
            import triton
            triton.compile=forbidden;triton.runtime.JITFunction.run=forbidden;triton.runtime.Autotuner.run=forbidden
            import safetensors.torch
            original_load=safetensors.torch.load_file
            def load_only(path,device='cpu',**kwargs):
                if Path(path).resolve()!=checkpoint or device!='cpu' or kwargs or record['load_calls']:return forbidden()
                record['load_calls']=1;return original_load(str(checkpoint),device='cpu')
            safetensors.torch.load_file=load_only
            import psutil,numpy as np
            available=psutil.virtual_memory().available;required=verified['tensor_data_bytes']*4+1024**3
            record['ram_before_strict_load']=dict(available_bytes=available,required_bytes=required,admitted=available>=required)
            if available<required:raise RuntimeError('Insufficient host strict-load margin')
            stage('strict-CPU-load');began=time.monotonic();before=local.loading.process_memory()
            component,model=local.load_cpu();feature,pooled,coords=local.verify_inputs()
            from safetensors.numpy import load_file
            saved=load_file(str(local.FEATURE));from trellis2.modules.sparse import SparseTensor
            inputs={'x':torch.randn(len(coords),32,generator=torch.Generator(device='cpu').manual_seed(contract.SEED),dtype=torch.float32,device='cpu'),
                'coords':torch.from_numpy(coords.copy()),'cond':torch.from_numpy(saved['cond'].copy()),'t':torch.tensor([contract.TIMESTEP],dtype=torch.float32,device='cpu')}
            cpu_sparse=SparseTensor(inputs['x'],inputs['coords']);inputs['batch_map']=cpu_sparse.batch_boardcast_map.clone();cpu_sparse=None;saved=None
            if sampling is not None:
                inputs.pop('t');inputs['neg_cond']=torch.zeros_like(inputs['cond'])
                cpu_sparse=SparseTensor(inputs['x'],inputs['coords']);inputs['seqlen']=cpu_sparse.seqlen.clone();cpu_sparse=None
            record['input']={k:{**describe(v),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in inputs.items()}
            if runtime is None:
                if record['input']['cond']['sha256']!=feature['output_sha256'] or record['input']['coords']['sha256']!=json.loads(PROBE.read_text())['worker']['coords_sha256']:raise ValueError('Saved feature/coordinate values differ')
                if sampling is not None:
                    old=json.loads((local.PINS.parent/'run/results.json').read_text())['worker']['input']['x']['sha256']
                    if record['input']['x']['sha256']!=old:raise ValueError('Canonical CPU noise differs from verified seed contract')
            else:record['generation_inputs']=runtime.validate_inputs(record['input'],feature,coords)
            record['provenance']=dict(seed=contract.SEED,timestep=contract.TIMESTEP,conditioning=feature,checkpoint_sha256=verified['checkpoint_sha256'],config_sha256=verified['config_sha256'],source_sha256=local.digest(local.SOURCE/'trellis2/models/structured_latent_flow.py'),
                occupancy_logits_sha256=local.LOGITS_SHA,occupancy_mask_sha256=local.MASK_SHA,pooling='max_pool3d((decoded>0).float(),2,2,0)>0.5; argwhere[:,[0,2,3,4]].int()',sparse_attention='explicit existing PyTorch SDPA branch; no FlashAttention fallback',sparse_convolution='none; no convolution used')
            if sampling is not None:
                record['provenance'].pop('timestep');record['provenance']['timestep_schedule']=active_contract.times()
            record['cpu_load']=dict(status='PASS',strict=True,assign=True,missing_keys=[],unexpected_keys=[],gpu_initialized=False,parameter_bytes=verified['tensor_data_bytes'],payload_entries=640,
                reversible_widening_count=220,feature_verified=True,input_token_count=len(coords),tensors=local.manifest(dict(model.named_parameters())),buffers=local.manifest(dict(model.named_buffers())),
                elapsed_seconds=time.monotonic()-began,memory_before=before,memory_after=local.loading.process_memory(),loader_diagnostics=component.diagnostics.copy())
            contract.authorize_placement(record['cpu_load'],verified,verified['checkpoint_sha256'],verified['checkpoint_sha256'])
            if torch.cuda.is_initialized():return forbidden()
            print(supervisor.EVENT+json.dumps(dict(event='cpu-ready',gate_one=record['cpu_load'],checkpoint_sha256=verified['checkpoint_sha256'])),flush=True);stage('waiting-for-observer')
            authorization=(AUTHORIZE if sampling is None else __import__('amd_shape_sampling').AUTHORIZE) if runtime is None else runtime.authorize
            if sys.stdin.readline().encode()!=authorization:raise RuntimeError('GPU attempt not authorized')
            policy.phase='placement';torch.cuda._lazy_init=original_init;stage('GPU-metadata');torch.cuda.set_device(0);p=torch.cuda.get_device_properties(0)
            if 'RX 9070 XT' not in p.name or p.gcnArchName.split(':')[0]!='gfx1201':raise RuntimeError('Unexpected GPU')
            record['gpu']=dict(name=p.name,architecture=p.gcnArchName,total_bytes=p.total_memory);trace.emit('state','gpu',gpu=record['gpu'])
            torch.cuda.reset_peak_memory_stats(0);record['before_placement']=local.loading.gpu_memory(torch)
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<contract.MEMORY_LIMIT+1024**3:raise RuntimeError('VRAM admission failed')
            torch.cuda.set_per_process_memory_fraction(contract.MEMORY_LIMIT/p.total_memory,0)
            record['budget']=dict(allocator_limit_bytes=contract.MEMORY_LIMIT,cpu_seconds=180,gpu_seconds=240,overall_seconds=480,required_ram_bytes=required)
            trace.emit('memory','before-placement',memory=record['before_placement'])
            policy.placement={id(v) for v in model.parameters()};policy.rope=model.blocks[0].self_attn.rope.freqs
            stage('shape-placement');began=time.monotonic();model.to(device='cuda:0');torch.cuda.synchronize(0)
            contract.validate_manifest(local.manifest(dict(model.named_parameters())),local.manifest(dict(model.named_buffers())),'cuda:0')
            record['placement_seconds']=time.monotonic()-began;modules={id(v):name for name,v in model.named_modules()}
            with torch.inference_mode():
                policy.phase='inputs';stage('input-transfer');gpu_inputs={}
                for key in (('x','coords','cond','t','batch_map') if sampling is None else ('x','coords','cond','neg_cond','batch_map','seqlen')):
                    policy.active=(key,inputs[key],'cuda:0');gpu_inputs[key]=torch.ops.aten._to_copy.default(inputs[key],device=torch.device('cuda:0'))
                policy.active=None;policy.phase='placement';torch.cuda.synchronize(0)
                sparse=SparseTensor(gpu_inputs['x'],gpu_inputs['coords'],shape=torch.Size([1,32]))
                sparse.register_spatial_cache('layout',[slice(0,len(coords))]);sparse.register_spatial_cache('batch_boardcast_map',gpu_inputs['batch_map'])
                if sampling is not None:sparse.register_spatial_cache('seqlen',gpu_inputs['seqlen'])
                record['before_forward']=local.loading.gpu_memory(torch);trace.emit('memory','before-forward',memory=record['before_forward'])
                if torch.is_autocast_enabled('cuda') or torch.is_grad_enabled():return forbidden()
                policy.phase='forward';stage('single-forward' if sampling is None else 'sampling-sequence');began=time.monotonic()
                if sampling is None:output=permit.run(lambda:model(sparse,gpu_inputs['t'],gpu_inputs['cond']))
                else:
                    torch.tensor=timestep_tensor
                    try:sampler_result=sampling.run(model,sparse,gpu_inputs,trace,lambda:torch.cuda.synchronize(0),lambda:local.loading.gpu_memory(torch))
                    finally:torch.tensor=original_tensor
                    output=sampler_result.samples
                policy.phase='placement';trace.call('synchronize:after-forward',lambda:torch.cuda.synchronize(0));record['forward_seconds']=time.monotonic()-began
                record['after_forward']=local.loading.gpu_memory(torch);trace.emit('memory','after-forward',memory=record['after_forward'])
                if not isinstance(output,SparseTensor):raise ValueError('Genuine model did not return SparseTensor')
                policy.phase='output';stage('output-copy-validation');policy.active=('output',output.feats,'cpu')
                cpu_output=torch.ops.aten._to_copy.default(output.feats,device=torch.device('cpu'))
                policy.active=('output_coords',output.coords,'cpu');cpu_coords=torch.ops.aten._to_copy.default(output.coords,device=torch.device('cpu'))
                policy.active=None;policy.phase='placement';torch.cuda.synchronize(0);values=cpu_output.numpy();out_coords=cpu_coords.numpy()
                record['output']={**describe(output.feats),'sparse_shape':list(output.shape),'coords_unchanged':np.array_equal(out_coords,coords),'all_finite':bool(np.isfinite(values).all()),'count':values.size,
                    'requires_grad':output.feats.requires_grad,'is_inference_tensor':output.feats.is_inference(),'grad_enabled':torch.is_grad_enabled(),'minimum':float(values.min()),'maximum':float(values.max()),
                    'mean':float(values.mean()),'stddev':float(values.std()),'representative_values':values.reshape(-1)[:8].tolist(),'sha256':hashlib.sha256(values.tobytes()).hexdigest()}
                contract.validate_output(record['output'],len(coords));contract.validate_coords(out_coords,len(coords))
                if sampling is None:
                    for name,array in [('shape-flow-output.npy',values),('shape-flow-coordinates.npy',out_coords)]:
                        with (scratch/name).open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
                        readback=np.load(scratch/name,allow_pickle=False)
                        if not np.array_equal(readback,array):raise ValueError('Output readback differs')
                    record['artifacts']={name:dict(path=str(scratch/name),file_bytes=(scratch/name).stat().st_size,sha256=local.digest(scratch/name)) for name in ('shape-flow-output.npy','shape-flow-coordinates.npy')}
                else:
                    if runtime is None:import amd_shape_sampling as entry
                    else:entry=runtime.entry
                    import amd_shape_sampling_artifact as artifact
                    if runtime is not None:artifact=runtime.artifact
                    normalization=entry.verify_sampler()['normalization'];normalized=artifact.denormalize(values,np.array(normalization['mean'],dtype=np.float32),np.array(normalization['std'],dtype=np.float32))
                    sequence=sampling.report();sequence={**sequence,'models_sha256':hashlib.sha256(json.dumps(sequence['models'],sort_keys=True).encode()).hexdigest()};sequence.pop('models')
                    provenance=json.loads(json.dumps(dict(harness_commit=os.environ['MODLY_SAMPLING_HARNESS_COMMIT'],checkpoint=record['provenance'],input=record['input'],parameters=active_contract.PARAMS,
                        sigma_min=active_contract.SIGMA_MIN,seed=active_contract.SEED,sampler_sources=entry.sources(),normalization=normalization,sequence=sequence,raw_output=record['output'])))
                    if runtime is not None:provenance['generation_inputs']=record['generation_inputs']
                    record['artifacts']=artifact.write(scratch,values,normalized,out_coords,provenance,noise=inputs['x'].numpy());normalized=None
                values=out_coords=readback=array=None
            stage('release');policy.phase='release';refs=[weakref.ref(model),weakref.ref(output),weakref.ref(sparse),*[weakref.ref(v) for v in gpu_inputs.values()]]
            sparse.clear_spatial_cache();output.clear_spatial_cache()
            if sampling is not None:refs+=sampling.issued;sampling.drop_references();sampler_result=None
            model=output=sparse=gpu_inputs=inputs=cpu_output=cpu_coords=None;policy.rope=policy.frequency=policy.active=None;policy.placement.clear();modules.clear()
            record['workspace_clear_calls']=1;record['release']=proven.release(torch,component,refs,trace);component=None
            stage('complete');record['status']='PASS'
    except BaseException as e:record.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    finally:
        if original_tensor is not None and torch is not None:torch.tensor=original_tensor
        record.update(forward_calls=permit.calls if sampling is None else sampling.budget.calls,transfers=(policy.counts if sampling is None else sampling.transfers(policy.counts)) if policy else {},gpu_operation_counts=op_counts)
        if sampling is not None:record['sampling']=sampling.report();sampling.drop_references()
        if component is not None:component.unload()
        component=model=output=cpu_output=cpu_coords=inputs=gpu_inputs=sparse=policy=sampling=sampler_result=None;gc.collect()
        print(supervisor.RESULT+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--harness-commit');args=p.parse_args()
    teardown.verify_workspace()
    raise SystemExit(worker(args.output) if args.worker else supervisor.supervise(args.output,args.harness_commit,profile=profile()))
