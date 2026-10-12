"""One offline float32 DINOv3 conditioning pass, supervised by the proven harness."""
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
from types import SimpleNamespace,MethodType
import weakref

sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_local as local
import amd_dinov3_contract as contract
import amd_sparse_decoder_forward as supervisor
import amd_sparse_decoder_loading as loading
import amd_decoder_contract as strict_contract
import amd_forward_diagnostics as diagnostics
from services.amd_offline_guard import block_external_actions

AUTHORIZE=b'EXECUTE_ONE_DINOV3_CONDITIONING\n'


def profile():
    files=list(supervisor.COMMITTED_FILES)
    files += ['scripts/compatibility/'+name+'.py' for name in (
        'amd_dinov3_conditioning','amd_dinov3_local','amd_dinov3_contract','amd_dinov3_metadata_probe')]
    files += ['experimental/extensions/trellis2-amd/alpha_input.py',
              local.AUDIT.relative_to(local.ROOT).as_posix(),local.PINS.relative_to(local.ROOT).as_posix(),
              (local.EVIDENCE/'cpu-metadata/results.json').relative_to(local.ROOT).as_posix()]
    def prepared(scratch):
        # This model uses a different convolution/attention workload. No historic
        # cache is assumed eligible; all first-use products stay in fresh scratch.
        return {'policy':'fresh private caches; no historical entries executed','copied_entries':0}
    return SimpleNamespace(evidence=local.EVIDENCE,cpu_seconds=120,gpu_seconds=180,overall_seconds=360,
        identity=local.identity,source_identity=local.verify_sources,validate_config=contract.expected_shape,
        classify=contract.classify,prepare_private_cache=prepared,worker_script=Path(__file__).resolve(),
        authorize=AUTHORIZE,committed_files=files)


def verify_preflight():
    local.verify_sources();current=local.identity()
    probe=json.loads((local.EVIDENCE/'cpu-metadata/results.json').read_text())
    gate=probe.get('worker',{})
    if (probe.get('returncode')!=0 or probe.get('worker_exit_verified') is not True
            or probe.get('job_cleanup',{}).get('forced_job_cleanup') is not False
            or gate.get('status')!='PASS' or gate.get('weights_loaded') is not False
            or gate.get('gpu_initialized') is not False or gate.get('strict_header_match') is not True
            or current['checkpoint_sha256']!=gate.get('checkpoint_sha256')
            or current['checkpoint_sha256']!=json.loads(local.PINS.read_text())['checkpoint_sha256']):
        raise RuntimeError('Missing or changed CPU-only DINO compatibility proof')
    strict_contract.validate_tensors(current['tensors'],gate['state'],'meta')
    contract.validate_buffers(gate['buffers'])
    return current


def worker(scratch,*,allocation_evidence=None,workspace_teardown=False):
    record={'status':'FAIL','stage':'startup','sdk_discovery':[],'calls':{},'load_calls':0}
    component=model=feature=pipeline=output=cpu_output=cond=processed=image=torch=None
    phase='cpu';allowed_modules={};placement_sources=set();input_source=None
    normalizing=False;constant_copy=None;normalization_copies={'mean':0,'std':0}
    permit=contract.SingleConditioning();transfers={'input':0,'output':0};cache=None
    operation_counts={}
    allocation_observer=None
    trace=diagnostics.Trace(lambda line:print(line,flush=True))
    def stage(name):
        record['stage']=name;print('FORWARD_STAGE='+name,flush=True);trace.emit('stage',name)
    def describe(tensor):return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device)}
    def count(name):
        record['calls'][name]=record['calls'].get(name,0)+1
        if record['calls'][name]!=1:raise RuntimeError('Duplicate DINO operation: '+name)
    try:
        if type(workspace_teardown) is not bool or workspace_teardown and allocation_evidence is None:
            raise RuntimeError('Workspace teardown requires the explicit exclusive evidence profile')
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned job must contain worker before preparation')
        evidence=local.EVIDENCE if allocation_evidence is None else allocation_evidence
        scratch=scratch.resolve();scratch.relative_to(evidence.resolve())
        feature_path=scratch/'conditioning.safetensors'
        env=loading.gpu_proof.scratch_environment(scratch)
        for key in loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env)
        record['sdk_verification']=block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        os.environ.update(env)
        checkpoint=(local.MODEL/'model.safetensors').resolve()
        def audit(event,args):
            if event.startswith('socket.'):raise RuntimeError('Conditioning proof forbids network')
            if event=='import' and any(str(args[0]).startswith(prefix) for prefix in ('cumesh','o_voxel','nvdiffrast','trellis2.models','trellis2.pipelines')):
                raise RuntimeError('Only the inspected standalone DINO extractor may import')
            if event=='open' and isinstance(args[0],(str,bytes)):
                path=Path(os.fsdecode(args[0]))
                feature_access=(allocation_observer is not None and phase=='feature-save' and path.resolve()==feature_path)
                if path.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.ckpt','.gguf') and path.resolve()!=checkpoint and not feature_access:
                    raise RuntimeError('Only the audited DINO checkpoint may be read')
        sys.addaudithook(audit)
        verified=verify_preflight();stage('runtime-import')
        import torch
        Path(sys.executable).resolve().relative_to(local.VENV)
        Path(torch.__file__).resolve().relative_to(local.VENV/'Lib/site-packages')
        if (sys.version.split()[0]!='3.12.10' or torch.__version__!='2.13.0+rocm10.0.0'
                or not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized()):
            raise RuntimeError('Unexpected AMD runtime or pre-existing GPU context')
        record['runtime']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                          'torch_path':torch.__file__,'hip':torch.version.hip,'cuda':torch.version.cuda,
                          'dependencies':{name:importlib.metadata.version(name) for name in ('transformers','torchvision','safetensors','numpy','pillow')}}
        if record['runtime']['dependencies']['transformers']!='4.56.0' or record['runtime']['dependencies']['torchvision']!='0.28.0+rocm10.0.0':
            raise RuntimeError('Conditioning dependency identity changed')
        trace.emit('state','runtime',runtime=record['runtime'])
        if workspace_teardown:
            from amd_blaslt_teardown import verify_runtime_binding
            record['workspace_binding']=verify_runtime_binding(torch)
        torch.set_num_threads(1)
        original_init=torch.cuda._lazy_init
        def forbidden(*a,**k):raise RuntimeError('Unapproved initialization/loading/compilation/model call')
        torch.cuda._lazy_init=forbidden;torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        original_call=torch.nn.Module.__call__
        def module_call(instance,*args,**kwargs):
            nonlocal normalizing
            if phase!='conditioning' or id(instance) not in allowed_modules:return forbidden()
            name=allowed_modules[id(instance)]
            if name in ('embeddings','rope_embeddings') or name.startswith('layer.') and name.count('.')==1:
                count(name)
                return trace.call('module:'+name,lambda:original_call(instance,*args,**kwargs),
                                  input=describe(args[0]) if args and isinstance(args[0],torch.Tensor) else {})
            if instance is model:return forbidden()  # Genuine extractor calls submodules, not root.forward.
            if name=='normalize':
                if normalizing:return forbidden()
                normalizing=True
                try:return trace.call('normalize',lambda:original_call(instance,*args,**kwargs))
                finally:normalizing=False
            return original_call(instance,*args,**kwargs)
        torch.nn.Module.__call__=module_call
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        def gpu(values):return any((isinstance(v,torch.Tensor) and v.device.type=='cuda') or
            (isinstance(v,torch.device) and v.type=='cuda') or (isinstance(v,str) and v.startswith('cuda')) for v in tree_leaves(values))
        class BoundedDispatch(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};is_gpu=gpu((args,kwargs))
                if is_gpu:
                    if any((isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.device.index!=0)
                            or (isinstance(v,torch.device) and v.type=='cuda' and v.index not in (None,0))
                            for v in tree_leaves((args,kwargs))):return forbidden()
                if is_gpu and phase=='cpu':return forbidden()
                if is_gpu and phase=='placement':
                    if (not strict_contract.gpu_operation_allowed(str(func)) or
                        str(func)=='aten._to_copy.default' and (not args or id(args[0]) not in placement_sources)):
                        return forbidden()
                if is_gpu and phase=='output':
                    if (str(func)!='aten._to_copy.default' or len(args)!=1 or args[0] is not output
                            or kwargs!={'device':torch.device('cpu')} or transfers['output']):return forbidden()
                    contract.validate_transfer('output',describe(output),'cpu',{},phase);transfers['output']=1
                if is_gpu and phase=='release' and not strict_contract.gpu_operation_allowed(str(func)):return forbidden()
                if is_gpu and phase not in ('placement','conditioning','output','release'):return forbidden()
                if phase=='conditioning' and is_gpu and str(func)=='aten._to_copy.default':
                    if not args or len(args)!=1 or kwargs!={'device':torch.device('cuda:0')}:return forbidden()
                    if normalizing and constant_copy is not None and args[0] is constant_copy:
                        if describe(constant_copy)!={'shape':[3],'dtype':'torch.float32','device':'cpu'}:return forbidden()
                    else:
                        if args[0] is not input_source or transfers['input']:return forbidden()
                        contract.validate_transfer('input',describe(input_source),'cuda:0',{},phase);transfers['input']=1
                if is_gpu and phase=='conditioning' and not str(func).startswith('aten.'):return forbidden()
                if is_gpu and phase=='conditioning':
                    operation_counts[str(func)]=operation_counts.get(str(func),0)+1
                    if len(operation_counts)>128:raise RuntimeError('Unexpected conditioning operation diversity')
                floating=[v for v in tree_leaves((args,kwargs)) if isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.is_floating_point()]
                if any(v.dtype!=torch.float32 for v in floating) or kwargs.get('dtype') in (torch.float16,torch.bfloat16,torch.float64):return forbidden()
                if allocation_observer is not None:
                    allocation_observer.observe([v for v in tree_leaves((args,kwargs)) if isinstance(v,torch.Tensor)],str(func)+':argument')
                if phase=='conditioning' and str(func) in ('aten.convolution.default','aten._convolution.default','aten._scaled_dot_product_flash_attention.default','aten._scaled_dot_product_attention_math.default','aten._scaled_dot_product_efficient_attention.default'):
                    result=trace.call('aten:'+str(func),lambda:func(*args,**kwargs),input=describe(args[0]) if args else {})
                else:result=func(*args,**kwargs)
                if allocation_observer is not None:
                    allocation_observer.observe([v for v in tree_leaves(result) if isinstance(v,torch.Tensor)],str(func)+':return')
                return result
        with BoundedDispatch():
            import transformers,huggingface_hub,safetensors.torch
            transformers.PreTrainedModel.from_pretrained=classmethod(forbidden)
            huggingface_hub.hf_hub_download=forbidden;huggingface_hub.snapshot_download=forbidden
            import triton
            triton.compile=forbidden;triton.runtime.JITFunction.run=forbidden;triton.runtime.Autotuner.run=forbidden
            original_load=safetensors.torch.load_file
            def load_only(path,device='cpu',**kwargs):
                if Path(path).resolve()!=checkpoint or device!='cpu' or kwargs or record['load_calls']:return forbidden()
                record['load_calls']=1;return original_load(str(checkpoint),device='cpu')
            safetensors.torch.load_file=load_only
            from strict_loader import StrictLocalComponent
            import psutil
            if psutil.virtual_memory().available<verified['tensor_data_bytes']*4+1024**3:
                raise RuntimeError('Insufficient host memory margin for strict checkpoint validation')
            stage('strict-CPU-load');began=time.monotonic();cpu_before=loading.process_memory()
            component=StrictLocalComponent(str(local.MODEL/'config.json'),str(checkpoint),local.factory)
            model=component.load_cpu()
            record['attention_backend']=model.config._attn_implementation
            state=loading.tensor_manifest(model.state_dict());strict_contract.validate_tensors(verified['tensors'],state,'cpu')
            contract.validate_buffers(loading.tensor_manifest(dict(model.named_buffers())))
            if torch.cuda.is_initialized():raise RuntimeError('CPU model loading initialized GPU')
            record['cpu_load']={'status':'PASS','elapsed_seconds':time.monotonic()-began,'strict':True,'assign':True,
                'missing_keys':[],'unexpected_keys':[],'gpu_initialized':False,'parameter_bytes':verified['tensor_data_bytes'],
                'tensors':state,'memory_before':cpu_before,'memory_after':loading.process_memory(),'loader_diagnostics':component.diagnostics.copy()}
            feature=local.extractor(model);pipeline=local.pipeline_methods(feature);cache=local.rope_cache()
            if cache.cache_info().currsize!=0:return forbidden()
            from alpha_input import preprocess_alpha_image
            image=local.fixture(scratch/'input.png');processed=preprocess_alpha_image(pipeline,image)
            record['input']={'mode':image.mode,'size':list(image.size),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                'png_sha256':local.digest(scratch/'input.png'),'crop_mode':processed.mode,'crop_size':list(processed.size),
                'resolution':512,'alpha_only':True,'rmbg_model':None}
            print(supervisor.EVENT+json.dumps({'event':'cpu-ready','gate_one':record['cpu_load'],'checkpoint_sha256':verified['checkpoint_sha256']}),flush=True)
            stage('waiting-for-observer')
            if sys.stdin.readline().encode()!=AUTHORIZE:raise RuntimeError('GPU conditioning not authorized by supervisor')
            phase='placement';torch.cuda._lazy_init=original_init;stage('GPU-metadata')
            torch.cuda.set_device(0);properties=torch.cuda.get_device_properties(0)
            if 'RX 9070 XT' not in properties.name or properties.gcnArchName.split(':')[0]!='gfx1201':raise RuntimeError('Unexpected AMD GPU')
            record['gpu']={'name':properties.name,'architecture':properties.gcnArchName,'total_bytes':properties.total_memory}
            trace.emit('state','gpu',gpu=record['gpu'])
            torch.cuda.reset_peak_memory_stats(0);record['before_placement']=loading.gpu_memory(torch)
            trace.emit('memory','before-placement',**record['before_placement'])
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<verified['tensor_data_bytes']+2*1024**3:
                raise RuntimeError('Unexpected baseline allocation or insufficient VRAM margin')
            if allocation_evidence is not None:
                from amd_dinov3_allocation import AllocationDiagnostics
                if torch.cuda.device_count()!=1:raise RuntimeError('Allocator diagnostics require exactly one visible GPU')
                allocation_observer=AllocationDiagnostics(scratch)
                trace.call('allocator:enable-history',lambda:allocation_observer.begin(torch))
            torch.backends.cudnn.benchmark=False
            placement_sources={id(v) for v in list(model.parameters())+list(model.buffers())}
            stage('DINO-placement');began=time.monotonic();model.to(device='cuda:0');torch.cuda.synchronize(0)
            strict_contract.validate_tensors(verified['tensors'],loading.tensor_manifest(model.state_dict()),'cuda:0')
            record['placed_buffers']=loading.tensor_manifest(dict(model.named_buffers()))
            if record['placed_buffers']!={'rope_embeddings.inv_freq':{'shape':[16],'dtype':'F32','device':'cuda:0','is_meta':False}}:
                raise RuntimeError('DINO nonpersistent-buffer placement mismatch')
            record['placement_seconds']=time.monotonic()-began
            if allocation_observer is not None:
                allocation_observer.observe(list(model.parameters())+list(model.buffers()),'DINO:placed-state')
            allowed_modules={id(v):name for name,v in model.named_modules()}
            allowed_modules.update({id(v):'normalize' for v in feature.transform.transforms})
            original_cuda=torch.Tensor.cuda
            def exact_cuda(tensor,*args,**kwargs):
                nonlocal input_source
                if phase!='conditioning' or args or kwargs:return forbidden()
                if tensor.device.type=='cpu':
                    contract.validate_transfer('input',describe(tensor),'cuda:0',{},phase)
                    if transfers['input']:return forbidden()
                    input_source=tensor
                    return torch.ops.aten._to_copy.default(tensor,device=torch.device('cuda:0'))
                if describe(tensor)!={'shape':[1,3,512,512],'dtype':'torch.float32','device':'cuda:0'}:return forbidden()
                return tensor  # The second genuine .cuda() is already on device 0.
            torch.Tensor.cuda=exact_cuda
            original_as_tensor=torch.as_tensor
            def exact_as_tensor(data,*args,**kwargs):
                nonlocal constant_copy
                if not gpu(kwargs):return original_as_tensor(data,*args,**kwargs)
                expected={'mean':[.485,.456,.406],'std':[.229,.224,.225]}
                name=next((key for key,value in expected.items() if type(data) is list and data==value),None)
                if (phase!='conditioning' or not normalizing or args or name is None
                        or kwargs!={'dtype':torch.float32,'device':torch.device('cuda:0')}
                        or normalization_copies[name] or constant_copy is not None
                        or name=='std' and normalization_copies['mean']!=1):return forbidden()
                normalization_copies[name]=1
                constant_copy=original_as_tensor(data,dtype=torch.float32,device='cpu')
                try:return torch.ops.aten._to_copy.default(constant_copy,device=torch.device('cuda:0'))
                finally:constant_copy=None
            torch.as_tensor=exact_as_tensor
            extractor_id=id(feature);original_extractor=type(feature).__call__
            def extractor_call(instance,*args,**kwargs):
                if id(instance)!=extractor_id or phase!='conditioning':return forbidden()
                count('extractor');return trace.call('DinoV3FeatureExtractor',lambda:original_extractor(instance,*args,**kwargs))
            type(feature).__call__=extractor_call
            original_features=feature.extract_features
            def features_call(instance,*args,**kwargs):
                if id(instance)!=extractor_id or phase!='conditioning':return forbidden()
                count('features');return trace.call('extract_features',lambda:original_features(*args,**kwargs),input=describe(args[0]))
            feature.extract_features=MethodType(features_call,feature)
            with torch.inference_mode():
                record['before_conditioning']=loading.gpu_memory(torch);trace.emit('memory','before-conditioning',**record['before_conditioning'])
                stage('single-conditioning');phase='conditioning';began=time.monotonic()
                cond=permit.run(lambda:pipeline.get_cond(processed,512,include_neg_cond=False))
                if set(cond)!={'cond'}:raise RuntimeError('Unexpected conditioning return keys')
                output=cond['cond'];phase='placement'
                trace.call('synchronize:after-conditioning',lambda:torch.cuda.synchronize(0))
                record['conditioning_seconds']=time.monotonic()-began
                record['after_conditioning']=loading.gpu_memory(torch);trace.emit('memory','after-conditioning',**record['after_conditioning'])
                stage('output-copy-validation');phase='output'
                cpu_output=torch.ops.aten._to_copy.default(output,device=torch.device('cpu'))
                phase='placement';torch.cuda.synchronize(0)
                values=cpu_output.numpy()
                import numpy as np
                record['output']={**describe(output),'conditioning_width':output.shape[-1],'count':values.size,
                    'all_finite':bool(np.isfinite(values).all()),'minimum':float(values.min()),'maximum':float(values.max()),
                    'mean':float(values.mean()),'stddev':float(values.std()),'representative_values':values.reshape(-1)[:8].tolist(),
                    'sha256':hashlib.sha256(values.tobytes()).hexdigest(),'requires_grad':output.requires_grad,
                    'is_inference_tensor':output.is_inference(),'grad_enabled':torch.is_grad_enabled()}
                contract.validate_output(record['output']);contract.validate_counts(record['calls']);values=None
                if allocation_observer is not None and not workspace_teardown:
                    stage('feature-checkpoint');phase='feature-save'
                    from amd_feature_checkpoint import write
                    published=json.loads((evidence/'run/committed-harness.json').read_text())['commit']
                    pins=local.verify_sources()
                    provenance=dict(component='DINOv3ViTModel',checkpoint_sha256=verified['checkpoint_sha256'],
                        config_sha256=verified['config_sha256'],input_pixel_sha256=record['input']['pixel_sha256'],
                        input_png_sha256=record['input']['png_sha256'],harness_commit=published,
                        extractor_source_sha256=pins[str(local.SOURCE/'trellis2/modules/image_feature_extractor.py')],
                        pipeline_source_sha256=pins[str(local.SOURCE/'trellis2/pipelines/trellis2_image_to_3d.py')],
                        runtime='Python 3.12.10; PyTorch '+str(torch.__version__)+'; HIP '+str(torch.version.hip))
                    record['feature_checkpoint']=trace.call('feature-checkpoint:write-verify',lambda:write(feature_path,scratch,cpu_output.numpy(),provenance))
                    if record['feature_checkpoint']['output_sha256']!=record['output']['sha256']:
                        raise RuntimeError('Feature checkpoint differs from validated output')
                    trace.emit('state','feature-checkpoint',checkpoint=record['feature_checkpoint'])
                    phase='placement'
            if permit.calls!=1 or transfers!={'input':1,'output':1} or normalization_copies!={'mean':1,'std':1}:
                raise RuntimeError('Execution/transfer counts invalid')
            stage('release');phase='release'
            refs=[weakref.ref(v) for v in (model,output)]
            before_clear=cache.cache_info().currsize
            if before_clear!=1:raise RuntimeError('Unexpected DINO coordinate-cache occupancy')
            cache.cache_clear()
            component.unload();feature.model=None;pipeline.image_cond_model=None
            component=model=feature=pipeline=output=cpu_output=cond=processed=image=input_source=None
            original_features=None;gc.collect();torch.cuda.synchronize(0)
            if workspace_teardown:
                stage('BLAS-workspace-release');record['workspace_clear_calls']=1
                trace.call('blas:clear-workspaces',torch._C._cuda_clearCublasWorkspaces)
            torch.cuda.empty_cache();torch.cuda.synchronize(0)
            record['release']={**loading.gpu_memory(torch),'model_collected':refs[0]() is None,
                'output_collected':refs[1]() is None,'rope_cache_entries_before':before_clear,'rope_cache_entries_after':cache.cache_info().currsize,
                'cpu_memory':loading.process_memory()}
            trace.emit('memory','release',**record['release'])
            if allocation_observer is not None:
                if workspace_teardown and not record['release']['allocated_bytes'] and not record['release']['reserved_bytes']:
                    record['allocation_diagnosis']=trace.call('allocator:disable-history',lambda:allocation_observer.disable_history(torch))
                else:
                    stage('allocator-snapshot')
                    record['allocation_diagnosis']=trace.call('allocator:snapshot-release',lambda:allocation_observer.capture(
                        torch,record['release']['allocated_bytes'],record['release']['reserved_bytes']))
                trace.emit('state','allocator-release',diagnosis={key:value for key,value in record['allocation_diagnosis'].items() if key!='summary'},
                           allocation_totals={key:value for key,value in record['allocation_diagnosis'].get('summary',{}).items() if key not in ('active_blocks','interpretation')})
            strict_contract.validate_release(record['release'])
            if not record['release']['output_collected'] or record['release']['rope_cache_entries_after']:raise RuntimeError('Incomplete DINO cleanup')
            stage('complete');record['status']='PASS'
    except BaseException as exc:record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        record['conditioning_calls']=permit.calls;record['transfer_calls']=transfers
        record['normalization_copies']=normalization_copies
        record['gpu_operation_counts']=operation_counts
        # On error do not attempt GPU recovery, synchronize or another forward.
        # The unchanged owned-job supervisor verifies process/tree termination.
        if component is not None:component.unload()
        component=model=feature=pipeline=output=cpu_output=cond=processed=image=input_source=None
        gc.collect();print(supervisor.RESULT+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--harness-commit')
    args=p.parse_args()
    if args.worker:raise SystemExit(worker(args.output))
    verify_preflight();raise SystemExit(supervisor.supervise(args.output,args.harness_commit,profile=profile()))
