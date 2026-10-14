"""One real saved-turret parent decode. Output is outside the Git checkout."""
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import weakref
from admission import AdmissionHooks, AdmissionStop, GIB, MAX_ROWS
from runtime import NativeBoundary, validate_tensor_contract

ROOT=Path(__file__).resolve().parents[3]
ARTIFACTS=ROOT.parent/'artifacts/amd-wsl-turret-neural-a9c9ad7b'
MAX_OUTPUT=256*1024**2

def detach_failure(exc):
    detail={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
    kind,message=type(exc),str(exc)
    exc.__traceback__=None
    return detail,(kind,message)

def validate_output(tensors,torch):
    coords=tensors['coordinates'];feats=tensors['features'];n=coords.shape[0]
    if (not 0<n<=MAX_ROWS or tuple(coords.shape)!=(n,4) or coords.dtype!=torch.int32
        or tuple(feats.shape)!=(n,7) or feats.dtype!=torch.float32):
        raise RuntimeError('Neural output shape/dtype/row contract')
    def coordinate_ok(c,resolution):
        return (c.device.type=='cpu' and c.dtype==torch.int32 and c.ndim==2 and c.shape[1]==4
            and bool((c[:,0]==0).all()) and bool((c[:,1:]>=0).all())
            and bool((c[:,1:]<resolution).all()) and torch.unique(c,dim=0).shape[0]==c.shape[0])
    if not coordinate_ok(coords,512):raise RuntimeError('Invalid final sparse support')
    stages=[]
    for i in range(4):
        c=tensors[f'subdivision_{i}_coordinates'];s=tensors[f'subdivision_{i}_logits']
        if (not 0<c.shape[0]<=MAX_ROWS or tuple(s.shape)!=(c.shape[0],8)
            or s.dtype!=torch.bfloat16 or not coordinate_ok(c,32*2**i)):
            raise RuntimeError('Subdivision output contract')
        stages.append({'stage':i,'parents':c.shape[0],'selected_children':int((s>0).sum()),'logits_dtype':str(s.dtype)})
    if stages[0]['parents']!=3743 or any(stages[i]['selected_children']!=(stages[i+1]['parents'] if i<3 else n) for i in range(4)):
        raise RuntimeError('Subdivision masks/output support linkage failed')
    if len(tensors)!=10 or any(t.device.type!='cpu' or not t.is_contiguous() for t in tensors.values()):
        raise RuntimeError('CPU output artifact tensor closure differs')
    if any(not bool(torch.isfinite(t).all()) for t in tensors.values() if t.is_floating_point()):
        raise RuntimeError('Nonfinite neural output/subdivision logits')
    size=sum(t.numel()*t.element_size() for t in tensors.values())
    if size>MAX_OUTPUT:raise RuntimeError('Output exceeds256MiB payload limit')
    return {'rows':n,'features_shape':[n,7],'coordinates_shape':[n,4],
        'features_dtype':'torch.float32','coordinate_dtype':'torch.int32',
        'coordinate_order':'[batch,D,H,W], original C2S parent/child order',
        'subdivisions':stages,'payload_bytes':size,'finite':True,
        'feature_min':float(feats.min()),'feature_max':float(feats.max()),
        'feature_mean':float(feats.double().mean()),'feature_std':float(feats.double().std())}

def preserve_output(tensors,torch,provenance):
    from safetensors.torch import save_file,load_file
    from safetensors import safe_open
    summary=validate_output(tensors,torch)
    ARTIFACTS.mkdir(parents=True,exist_ok=False)
    path=ARTIFACTS/'turret-neural-output.safetensors'
    save_file(tensors,str(path),metadata={'provenance':json.dumps(provenance,sort_keys=True),'summary':json.dumps(summary,sort_keys=True)})
    if path.stat().st_size>MAX_OUTPUT+128*1024:raise RuntimeError('Artifact file size exceeded')
    with path.open('rb') as f:
        header=int.from_bytes(f.read(8),'little')
    if not 0<header<=128*1024:raise RuntimeError('Unbounded output header')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if f.metadata()['provenance']!=json.dumps(provenance,sort_keys=True):raise RuntimeError('Artifact provenance readback failed')
    restored=load_file(str(path),device='cpu')
    if set(restored)!=set(tensors) or any(not torch.equal(tensors[k],restored[k]) for k in tensors):
        raise RuntimeError('Byte-exact tensor artifact readback failed')
    # All finite BF16/F32 values have exact equality here; also hash raw tensor bytes.
    raw={k:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in tensors.items()}
    if any(raw[k]!=hashlib.sha256(restored[k].view(torch.uint8).numpy().tobytes()).hexdigest() for k in tensors):
        raise RuntimeError('Raw tensor byte readback failed')
    restored=None;gc.collect()
    summary.update(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   tensor_payload_sha256=raw,verified=True)
    with (ARTIFACTS/'provenance.json').open('x') as f:json.dump({'provenance':provenance,'output':summary},f,indent=2)
    return summary

def run_neural(model,torch,cuda,record,output,gpu_started):
    import amd_wsl_rocdxg_smoke as base
    import amd_shape_decoder_preflight as inputs
    from trellis2.modules import sparse as sp
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.modules.sparse.conv import config as conv_config
    import flex_gemm.ops.spconv as spconv
    x_cpu=c_cpu=x=y=subs=cpu_tensors=None
    hooks=native=None;handles=[];tensor_refs=[]
    failure=None
    def emit(event,**details):
        if time.monotonic()-gpu_started>=240:raise TimeoutError('240-second GPU stage deadline')
        record['last_event']=event
        base.journal(output,event,elapsed_gpu_seconds=time.monotonic()-gpu_started,**details)
    def event_hook(event):
        kind=event.pop('event');emit(kind,**event)
    def memory():
        free,total=torch.cuda.mem_get_info(0)
        return dict(free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
    try:
        hooks=AdmissionHooks(model,torch,emit=event_hook,memory=memory).install()
        native=NativeBoundary(cuda,record['bindings'],hooks,torch,conv_config,spconv,emit).install()
        spconv.set_algorithm('explicit_gemm');spconv.set_hashmap_ratio(2.0)
        a=inputs.read_npy(inputs.SAMPLE/'shape-latent-denormalized.npy',inputs.INPUTS['shape-latent-denormalized.npy'])
        c=inputs.read_npy(inputs.SAMPLE/'shape-latent-coordinates.npy',inputs.INPUTS['shape-latent-coordinates.npy'])
        x_cpu=torch.from_numpy(a);c_cpu=torch.from_numpy(c)
        validate_tensor_contract(x_cpu,c_cpu)
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        # Include initial FP32 projection, cast and retained input, plus final F32
        # normalization/output. Last conv2's downstream allowance also covers it.
        hooks._admit('model-placement-and-from-latent',{'visible_bytes':parameter_bytes+3743*(32*4+4*4+1024*16)})
        emit('model-placement-entry',memory=memory(),parameter_bytes=parameter_bytes)
        model.to(device=torch.device('cuda',0))
        if any(p.device!=torch.device('cuda',0) for p in model.parameters()):raise RuntimeError('Decoder GPU placement incomplete')
        x=sp.SparseTensor(x_cpu.to('cuda:0'),c_cpu.to('cuda:0'),shape=torch.Size([1,32]))
        x.clear_spatial_cache()  # Canonical fresh saved-latent cache; no guide masks.
        validate_tensor_contract(x.feats,x.coords)
        tensor_refs += [weakref.ref(x.feats),weakref.ref(x.coords)]
        torch.cuda.synchronize(0)
        emit('model-placement-synchronized',memory=memory(),rows=x.feats.shape[0])
        def projection_pre(m,args):emit('from-latent-entry',rows=args[0].feats.shape[0])
        def projection_post(m,args,result):emit('from-latent-host-return',rows=result.feats.shape[0],dtype=str(result.feats.dtype),host_return_is_not_sync=True)
        handles.append(model.from_latent.register_forward_pre_hook(projection_pre))
        handles.append(model.from_latent.register_forward_hook(projection_post))
        native.phase='neural'
        record['forward_calls']+=1
        if record['forward_calls']!=1:raise RuntimeError('Forward retry prohibited')
        emit('neural-forward-entry',rows=3743,guide_subs=None,parent='SparseUnetVaeDecoder',low_vram=model.low_vram)
        forward_started=time.monotonic()
        with torch.inference_mode():
            y,subs=SparseUnetVaeDecoder.forward(model,x,guide_subs=None,return_subs=True)
        emit('neural-forward-host-return',rows=y.feats.shape[0],host_return_is_not_sync=True)
        torch.cuda.synchronize(0)
        record['forward_synchronized_seconds']=time.monotonic()-forward_started
        emit('neural-forward-synchronized',rows=y.feats.shape[0],memory=memory())
        hooks.assert_complete()
        if len(subs)!=4 or y.feats.dtype!=torch.float32:raise RuntimeError('Parent neural return type/precision differs')
        tensor_refs += [weakref.ref(y.feats),weakref.ref(y.coords)]
        cpu_tensors={'features':y.feats.detach().cpu().contiguous(),'coordinates':y.coords.detach().cpu().contiguous()}
        for i,sub in enumerate(subs):
            tensor_refs += [weakref.ref(sub.feats),weakref.ref(sub.coords)]
            cpu_tensors[f'subdivision_{i}_logits']=sub.feats.detach().cpu().contiguous()
            cpu_tensors[f'subdivision_{i}_coordinates']=sub.coords.detach().cpu().contiguous()
        sub=None
        provenance={'source':record['source_sha256'],'checkpoint_sha256':record['checkpoint']['freshly_streamed_sha256'],
            'inputs':record['saved_inputs'],'input_file_sha256':inputs.INPUTS,
            'native_binary_sha256':'91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01',
            'algorithm':'explicit_gemm','BF16_policy':'216 body tensors after original strict F16/F32 verification',
            'runtime':record['runtime'],'forward_calls':1,'convolutions':40,'resolution':512,'mesh_extraction':False}
        record['output']=preserve_output(cpu_tensors,torch,provenance)
        emit('output-readback-verified',output_sha256=record['output']['sha256'],bytes=record['output']['bytes'])
    except BaseException as exc:
        # Preserve original diagnostics, then detach forward/hook tracebacks before
        # collection. Those frames otherwise keep the actual sparse tensors alive.
        record['neural_failure'],failure=detach_failure(exc)
    finally:
        record['convolutions_completed']=hooks.convolution_completed if hooks else 0
        record['subdivision_operations']=list(hooks.subdivision_records) if hooks else []
        record['native_map_calls']=native.calls if native else 0
        if native:native.remove()
        if hooks:hooks.remove()
        for h in handles:h.remove()
        handles.clear();native=hooks=None
        if x:x.clear_spatial_cache()
        if y:y.clear_spatial_cache()
        if subs:
            for sub in subs:sub.clear_spatial_cache()
        sub=None
        x_cpu=c_cpu=x=y=subs=cpu_tensors=None;gc.collect()
        record['GPU_input_output_tensors_collected']=all(ref() is None for ref in tensor_refs)
        if not record['GPU_input_output_tensors_collected']:raise RuntimeError('Owned input/output tensor references survived')
    if failure:raise failure[0](failure[1])
