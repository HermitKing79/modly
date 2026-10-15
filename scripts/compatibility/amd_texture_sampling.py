"""One genuine 12-step texture sample; preserved shape inputs, no geometry stages."""
import argparse,gc,hashlib,importlib.metadata,json,os,sys,time,traceback,weakref
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_slat_forward as existing
import amd_texture_sampling_local as local
import amd_texture_sampling_contract as contract
import amd_texture_sampling_artifact as artifact
from amd_texture_sampling_control import TextureSamplingControl
from services.amd_offline_guard import block_external_actions
supervisor,proven,strict,diagnostics,teardown=existing.supervisor,existing.proven,existing.strict,existing.diagnostics,existing.teardown
describe=existing.describe
ARTIFACT_ROOT=Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-4f845cd')
AUTHORIZE=b'EXECUTE_ONE_CANONICAL_12_POSITIVE_TEXTURE_SEQUENCE\n'
TUNABLE_ENV={key:'0' for key in ('PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')}

def verify_tunable(torch):
    environment={key:os.environ.get(key) for key in TUNABLE_ENV}
    flags={key:getattr(torch.cuda.tunable,key)() for key in ('is_enabled','tuning_is_enabled','record_untuned_is_enabled')}
    if environment!=TUNABLE_ENV or any(type(v) is not bool or v for v in flags.values()):raise RuntimeError('Actual TunableOp policy must remain entirely disabled')
    return dict(environment=environment,flags=flags,gpu_initialized=torch.cuda.is_initialized())

def make_provenance(record,sampling,commit,*,runtime=None):
    local=globals()['local'] if runtime is None else runtime.local
    result=json.loads(json.dumps(dict(harness_commit=commit,checkpoint=record['provenance'],input=record['input'],parameters=contract.PARAMS,
        sigma_min=contract.SIGMA_MIN,seed=contract.SEED,sampler_sources=local.verify_sampler()['sampler_sources'],shape_normalization=contract.SHAPE_NORMALIZATION,
        texture_normalization=contract.TEXTURE_NORMALIZATION,sampling=sampling.report(),raw_output=record['output'],runtime=record['runtime'],
        measurements={key:record[key] for key in ('forward_seconds','before_placement','after_forward','budget')},
        tunable_import=record['tunable_import'],tunable_before_forward=record['tunable_before_forward'])))
    if runtime is not None:result['generation_inputs']=record['generation_inputs']
    return result

def verify_preflight():
    local.verify_sources();teardown.verify_workspace();teardown.verify_binding_evidence()
    pins=json.loads(local.PINS.read_text())
    if pins!=local.verify_sampler():raise RuntimeError('Pinned texture source/sampler contract differs')
    artifact.approved_root(ARTIFACT_ROOT);artifact.approved_root(ARTIFACT_ROOT/'latent')
    persistence=json.loads((local.EVIDENCE/'persistence-preflight.json').read_text())
    if persistence.get('synthetic') is not True or persistence.get('torch_imported') is not False:raise RuntimeError('CPU-only output persistence qualification missing')
    if artifact.verify(Path(persistence['artifact_path']),ARTIFACT_ROOT/'persistence-probe',persistence['metadata'])!=dict((k,v) for k,v in persistence.items() if k not in ('synthetic','torch_imported')):raise RuntimeError('Persistence identity differs')
    qualification=json.loads((local.EVIDENCE/'cpu-qualification.json').read_text())
    if qualification.get('passed') is not True or qualification.get('gpu_initialized') is not False or qualification.get('weights_loaded') is not False:raise RuntimeError('Focused CPU qualification incomplete')
    return local.identity()

def profile():
    p=existing.profile();p.kind='texture-sampling';p.evidence=local.EVIDENCE;p.worker_script=Path(__file__).resolve();p.scratch_directory=ARTIFACT_ROOT
    p.maximum_forwards=12;p.maximum_steps=12;p.maximum_sequences=1;p.authorize=AUTHORIZE;p.authorize_placement=contract.authorize_placement
    p.identity=verify_preflight;p.source_identity=lambda:(local.verify_sources(),local.verify_sampler(),teardown.verify_binding_evidence())
    p.validate_config=lambda c:contract.validate_header(c,local.identity(hash_weights=False)['tensors']);p.classify=contract.classify
    p.committed_files += ['scripts/compatibility/'+n+'.py' for n in ('amd_texture_sampling','amd_texture_sampling_control','amd_texture_sampling_contract','amd_texture_sampling_local','amd_texture_sampling_artifact','amd_shape_sampling_control','amd_shape_sampling_contract','amd_shape_sampling_artifact')]
    p.committed_files += [str(v.relative_to(local.ROOT).as_posix()) for v in (local.PINS,local.AUDIT,local.SOURCE_AUDIT,local.SHAPE_RAW,local.SHAPE_DENORMALIZED,local.COORDS,local.SHAPE_PROVENANCE,local.EVIDENCE/'cpu-qualification.json',local.EVIDENCE/'persistence-preflight.json')]
    return p

def worker(scratch,*,runtime=None):
    local=globals()['local'] if runtime is None else runtime.local
    artifact=globals()['artifact'] if runtime is None else runtime.artifact
    ARTIFACT_ROOT=globals()['ARTIFACT_ROOT'] if runtime is None else runtime.artifact_root
    sampling_factory=(lambda torch:TextureSamplingControl(torch,local.import_sampler())) if runtime is None else runtime.factory
    record=dict(status='FAIL',stage='startup',sdk_discovery=[],calls={},load_calls=0,forward_calls=0)
    component=model=output=cpu_output=cpu_coords=inputs=gpu_inputs=sparse=torch=policy=None
    sampling=sampler_result=original_tensor=None
    active_contract=contract
    permit=existing.contract.SingleForward();trace=diagnostics.Trace(lambda line:print(line,flush=True),limit=256);op_counts={};modules={}
    def forbidden(*a,**k):raise RuntimeError('Unapproved shape-SLat action')
    def stage(name):record['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name,phase=policy.phase if policy else 'cpu')
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job handshake required')
        scratch=scratch.resolve(strict=True)
        if scratch!=ARTIFACT_ROOT.resolve(strict=True):raise RuntimeError('Foreign permanent scratch root')
        artifact.approved_root(scratch);teardown.verify_workspace()
        env=local.loading.gpu_proof.scratch_environment(scratch)
        for key in local.loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        env.update(ATTN_BACKEND='sdpa',WORKSPACE_DIR=str(scratch));env.update(TUNABLE_ENV);os.environ.update(env)
        record['sdk_verification']=block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        env['WORKSPACE_DIR']=str(local.ROOT);record['gpu_visibility']=existing.restore_execution_environment(env)
        checkpoint=local.PREFIX.with_suffix('.safetensors').resolve()
        def audit(e,a):
            if e.startswith('socket.'):return forbidden()
            if e=='import' and active_contract.prohibited_import(str(a[0])):return forbidden()
            if e=='open' and isinstance(a[0],(str,bytes)):
                p=Path(os.fsdecode(a[0]))
                readable=(checkpoint,local.FEATURE.resolve(),(ARTIFACT_ROOT/'latent/texture-latent.safetensors').resolve(),(ARTIFACT_ROOT/'persistence-probe/texture-latent.safetensors').resolve())
                if runtime is not None:readable+=runtime.readable_tensor_paths
                if p.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt') and p.resolve() not in readable:return forbidden()
        sys.addaudithook(audit)
        verified=verify_preflight() if runtime is None else runtime.verify_preflight()
        record['token_count']=verified['token_count'];stage('runtime-import')
        import torch
        if torch.cuda.is_initialized():return forbidden()
        record['runtime']=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),torch_path=torch.__file__,hip=torch.version.hip,cuda=torch.version.cuda,
            git_version=torch.version.git_version,dependencies={n:importlib.metadata.version(n) for n in ('numpy','safetensors')})
        trace.emit('state','runtime',runtime=record['runtime']);record['workspace_binding']=teardown.verify_runtime_binding(torch);record['tunable_import']=verify_tunable(torch)
        torch.set_num_threads(1);original_init=torch.cuda._lazy_init;torch.cuda._lazy_init=forbidden
        torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        if sampling_factory is not None:
            sampling=sampling_factory(torch)
            record['sampler']=dict(class_name=sampling.sampler_class.__name__,source=__import__('inspect').getfile(sampling.sampler_class),parameters=sampling.policy.PARAMS,sigma_min=sampling.policy.SIGMA_MIN)
        policy=sampling.copy_policy(verified['token_count'])
        original_call=torch.nn.Module.__call__
        def module_call(instance,*args,**kwargs):
            if policy.phase!='forward' or id(instance) not in modules:return forbidden()
            name=modules[id(instance)];policy.scopes.append(name)
            selected=name=='' or name.startswith('blocks.') and (name.count('.')==1 or name.endswith(('.self_attn','.cross_attn','.rope')))
            try:
                if sampling is not None and name=='':sampling.begin_model(args,kwargs,policy)
                if sampling is not None and name=='input_layer':sampling.validate_concat(args[0])
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
            available=psutil.virtual_memory().available;required=contract.REQUIRED_RAM
            record['ram_before_strict_load']=dict(available_bytes=available,required_bytes=required,admitted=available>=required)
            if available<required:raise RuntimeError('Insufficient host strict-load margin')
            stage('strict-CPU-load');began=time.monotonic();before=local.loading.process_memory()
            component,model=local.load_cpu();feature,shape_normalized,coords=local.verify_inputs()
            from safetensors.numpy import load_file
            saved=load_file(str(local.FEATURE));from trellis2.modules.sparse import SparseTensor
            inputs={'x':torch.randn(len(coords),32,generator=torch.Generator(device='cpu').manual_seed(contract.SEED),dtype=torch.float32,device='cpu'),
                'coords':torch.from_numpy(coords.copy()),'cond':torch.from_numpy(saved['cond'].copy()),'shape_cond':torch.from_numpy(shape_normalized.copy())}
            cpu_sparse=SparseTensor(inputs['x'],inputs['coords']);inputs['batch_map']=cpu_sparse.batch_boardcast_map.clone();inputs['seqlen']=cpu_sparse.seqlen.clone();cpu_sparse=None;saved=None
            record['input']={k:{**describe(v),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in inputs.items()}
            if (record['input']['cond']['sha256']!=feature['output_sha256'] or record['input']['coords']['sha256']!=verified['saved_inputs'][local.COORDS.name]['payload_sha256']
                or record['input']['shape_cond']['sha256']!=verified['shape_condition_sha256']):raise ValueError('Saved conditioning/coordinate identity differs')
            if runtime is not None:record['generation_inputs']=runtime.validate_inputs(record['input'],feature,shape_normalized,coords)
            record['provenance']=dict(seed=contract.SEED,conditioning=feature,checkpoint_sha256=verified['checkpoint_sha256'],config_sha256=verified['config_sha256'],
                source_sha256=local.digest(local.SOURCE/'trellis2/models/structured_latent_flow.py'),pipeline_sha256=verified['pipeline_sha256'],saved_inputs=verified['saved_inputs'],
                shape_provenance_sha256=verified['shape_provenance_sha256'],shape_condition_sha256=verified['shape_condition_sha256'],timestep_schedule=contract.times(),
                sparse_attention='explicit existing PyTorch SDPA; no fallback',sparse_convolution='none')
            record['cpu_load']=dict(status='PASS',strict=True,assign=True,missing_keys=[],unexpected_keys=[],gpu_initialized=False,parameter_bytes=verified['tensor_data_bytes'],payload_entries=640,
                reversible_widening_count=220,feature_verified=True,input_token_count=len(coords),tensors=local.manifest(dict(model.named_parameters())),buffers=local.manifest(dict(model.named_buffers())),
                elapsed_seconds=time.monotonic()-began,memory_before=before,memory_after=local.loading.process_memory(),loader_diagnostics=component.diagnostics.copy())
            contract.authorize_placement(record['cpu_load'],verified,verified['checkpoint_sha256'],verified['checkpoint_sha256'])
            if torch.cuda.is_initialized():return forbidden()
            print(supervisor.EVENT+json.dumps(dict(event='cpu-ready',gate_one=record['cpu_load'],checkpoint_sha256=verified['checkpoint_sha256'])),flush=True);stage('waiting-for-observer')
            authorization=AUTHORIZE
            if sys.stdin.readline().encode()!=authorization:raise RuntimeError('GPU attempt not authorized')
            policy.phase='placement';torch.cuda._lazy_init=original_init;stage('GPU-metadata');torch.cuda.set_device(0);p=torch.cuda.get_device_properties(0)
            if 'RX 9070 XT' not in p.name or p.gcnArchName.split(':')[0]!='gfx1201':raise RuntimeError('Unexpected GPU')
            record['gpu']=dict(name=p.name,architecture=p.gcnArchName,total_bytes=p.total_memory);trace.emit('state','gpu',gpu=record['gpu'])
            torch.cuda.reset_peak_memory_stats(0);record['before_placement']=local.loading.gpu_memory(torch)
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<13*1024**3:raise RuntimeError('VRAM admission failed')
            allowance=min(contract.MEMORY_LIMIT,record['before_placement']['available_bytes']-contract.FREE_RESERVE-contract.OPAQUE_HEADROOM)
            if allowance<=0:raise RuntimeError('No allocator allowance after fixed device reserves')
            torch.cuda.set_per_process_memory_fraction(allowance/p.total_memory,0)
            record['budget']=dict(allocator_limit_bytes=contract.MEMORY_LIMIT,effective_allocator_limit_bytes=allowance,cpu_seconds=180,gpu_seconds=240,overall_seconds=480,required_ram_bytes=required,free_reserve_bytes=contract.FREE_RESERVE,opaque_workspace_headroom_bytes=contract.OPAQUE_HEADROOM)
            trace.emit('memory','before-placement',memory=record['before_placement'])
            policy.placement={id(v) for v in model.parameters()};policy.rope=model.blocks[0].self_attn.rope.freqs
            stage('texture-placement');began=time.monotonic();model.to(device='cuda:0');torch.cuda.synchronize(0)
            contract.validate_manifest(local.manifest(dict(model.named_parameters())),local.manifest(dict(model.named_buffers())),'cuda:0')
            record['placement_seconds']=time.monotonic()-began;modules={id(v):name for name,v in model.named_modules()}
            with torch.inference_mode():
                policy.phase='inputs';stage('input-transfer');gpu_inputs={}
                for key in (('x','coords','cond','t','batch_map') if sampling is None else ('x','coords','cond','shape_cond','batch_map','seqlen')):
                    policy.active=(key,inputs[key],'cuda:0');gpu_inputs[key]=torch.ops.aten._to_copy.default(inputs[key],device=torch.device('cuda:0'))
                policy.active=None;policy.phase='placement';torch.cuda.synchronize(0)
                sparse=SparseTensor(gpu_inputs['x'],gpu_inputs['coords'],shape=torch.Size([1,32]))
                sparse.register_spatial_cache('layout',[slice(0,len(coords))]);sparse.register_spatial_cache('batch_boardcast_map',gpu_inputs['batch_map'])
                if sampling is not None:sparse.register_spatial_cache('seqlen',gpu_inputs['seqlen'])
                shape_sparse=sparse.replace(gpu_inputs['shape_cond']);gpu_inputs['shape_cond']=shape_sparse;shape_sparse=None
                record['tunable_before_forward']=verify_tunable(torch)
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
                    normalized=contract.denormalize_texture(values) if runtime is None else runtime.denormalize_texture(values)
                    provenance=make_provenance(record,sampling,os.environ['MODLY_SAMPLING_HARNESS_COMMIT'],runtime=runtime)
                    stage('permanent-artifact-write')
                    record['artifacts']=artifact.write(ARTIFACT_ROOT/'latent',values,normalized,out_coords,provenance,noise=inputs['x'].numpy())
                    record['live_artifact_verification']=artifact.verify(Path(record['artifacts']['artifact_path']),ARTIFACT_ROOT/'latent',record['artifacts']['metadata'])
                    trace.emit('state','permanent-artifact-verified',artifact={'path':record['artifacts']['artifact_path'],'sha256':record['artifacts']['metadata']['artifact_sha256'],'bytes':record['artifacts']['metadata']['artifact_bytes']})
                    normalized=None
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();teardown.verify_workspace();os.environ.update(TUNABLE_ENV)
    if args.worker:raise SystemExit(worker(args.output))
    verify_preflight()
    import psutil
    available=psutil.virtual_memory().available
    with (local.EVIDENCE/'pre-execution-admission.json').open('x') as stream:json.dump(dict(available_host_RAM_bytes=available,required_bytes=contract.REQUIRED_RAM,admitted=available>=contract.REQUIRED_RAM,torch_imported='torch' in sys.modules),stream,indent=2)
    if available<contract.REQUIRED_RAM:raise RuntimeError('12GiB available host RAM required before attempt')
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=args.harness_commit or ''
    rc=supervisor.supervise(args.output,args.harness_commit,profile=profile())
    result=json.loads((args.output/'results.json').read_text());post=dict(classification=result['classification'],worker_exit_verified=result['process'].get('worker_exit_verified'),verified=False)
    try:
        if result['worker'].get('artifacts'):
            expected=result['worker']['artifacts'];post['artifact']=artifact.verify(Path(expected['artifact_path']),ARTIFACT_ROOT/'latent',expected['metadata']);post['verified']=post['artifact']==expected
        if result['classification']=='PASS' and (not post['verified'] or post['worker_exit_verified'] is not True):raise RuntimeError('Post-exit persistence or process termination incomplete')
    except BaseException as exc:post.update(classification='UNKNOWN',error=repr(exc));rc=1
    with (local.EVIDENCE/'post-exit-artifact-verification.json').open('x') as stream:json.dump(post,stream,indent=2)
    print(json.dumps(post));raise SystemExit(rc)
