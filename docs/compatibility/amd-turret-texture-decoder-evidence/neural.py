"""One original guided texture forward. No shape decode, extraction or baking."""
import gc,hashlib,json,time,traceback,weakref
from admission import AdmissionHooks,AdmissionStop,FINAL_ROWS
from chunked_final import FinalConvolutionPatch
from runtime import NativeBoundary,validate_tensor_contract

def detach_failure(exc):
    detail={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
    kind,message=type(exc),str(exc);exc.__traceback__=None
    return detail,(kind,message)

def run_neural(model,torch,cuda,record,output,gpu_started):
    import numpy as np
    import amd_wsl_rocdxg_smoke as base
    import amd_texture_decoder_inputs as inputs
    from trellis2.modules import sparse as sp
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.modules.sparse.conv import config as conv_config
    import flex_gemm.ops.spconv as spconv
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
    prepared=x_cpu=c_cpu=x=y=guide_subs=raw_cpu=coords_cpu=raw=features=coords=None
    hooks=native=patch=None;handles=[];tensor_refs=[];failure=None
    def emit(event,**details):
        if time.monotonic()-gpu_started>=240:raise TimeoutError('240-second texture GPU stage deadline')
        record['last_event']=event;base.journal(output,event,elapsed_gpu_seconds=time.monotonic()-gpu_started,**details)
    def event_hook(event):
        kind=event.pop('event');emit(kind,**event)
    def memory():
        free,total=torch.cuda.mem_get_info(0)
        return dict(free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
    try:
        prepared=inputs.read_inputs();record['guided_inputs']=prepared['metadata']
        hooks=AdmissionHooks(model,torch,emit=event_hook,memory=memory,guidance=prepared).install()
        native=NativeBoundary(cuda,record['bindings'],hooks,torch,conv_config,spconv,emit).install()
        patch=FinalConvolutionPatch(SubMConv3dFunction,model.blocks[3][-1].conv2,hooks,torch,emit).install()
        spconv.set_algorithm('explicit_gemm');spconv.set_hashmap_ratio(2.0)
        x_cpu=torch.from_numpy(prepared['denormalized'].copy());c_cpu=torch.from_numpy(prepared['coords'].copy())
        validate_tensor_contract(x_cpu,c_cpu)
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        guidance_bytes=sum(g['logits_bf16'].nbytes+g['coords'].nbytes for g in prepared['guides'])
        hooks._admit('texture-model-guides-and-from-latent',{'visible_bytes':parameter_bytes+guidance_bytes+3743*(32*4+4*4+1024*16)})
        emit('texture-model-placement-entry',memory=memory(),parameter_bytes=parameter_bytes,guidance_bytes=guidance_bytes)
        model.to(device=torch.device('cuda',0))
        if any(p.device!=torch.device('cuda',0) for p in model.parameters()):raise RuntimeError('Texture GPU placement incomplete')
        x=sp.SparseTensor(x_cpu.to('cuda:0'),c_cpu.to('cuda:0'),shape=torch.Size([1,32]));x.clear_spatial_cache()
        validate_tensor_contract(x.feats,x.coords);tensor_refs.extend((weakref.ref(x.feats),weakref.ref(x.coords)))
        guide_subs=[]
        for g in prepared['guides']:
            bits=torch.from_numpy(g['logits_bf16'].copy());logits=bits.view(torch.bfloat16);parent_coords=torch.from_numpy(g['coords'].copy())
            if hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()!=hashlib.sha256(g['logits_bf16'].tobytes()).hexdigest():raise RuntimeError('Original guide BF16 bytes changed')
            sub=sp.SparseTensor(logits.to('cuda:0'),parent_coords.to('cuda:0'))
            guide_subs.append(sub);tensor_refs.extend((weakref.ref(sub.feats),weakref.ref(sub.coords)))
        bits=logits=parent_coords=sub=g=None
        torch.cuda.synchronize(0);emit('texture-model-placement-synchronized',memory=memory(),rows=3743,guides=4)
        handles.append(model.from_latent.register_forward_pre_hook(lambda m,args:emit('from-latent-entry',rows=args[0].feats.shape[0])))
        native.phase='neural';record['forward_calls']+=1
        if record['forward_calls']!=1:raise RuntimeError('Texture forward retry prohibited')
        emit('texture-forward-entry',rows=3743,guide_subs=4,return_subs=False,runtime_dtype='source F16 body/F32 norms and projections')
        forward_started=time.monotonic()
        with torch.inference_mode():
            y=SparseUnetVaeDecoder.forward(model,x,guide_subs=guide_subs,return_subs=False)
        emit('texture-forward-host-return',rows=y.feats.shape[0],host_return_is_not_sync=True)
        torch.cuda.synchronize(0);record['forward_synchronized_seconds']=time.monotonic()-forward_started
        emit('texture-forward-synchronized',rows=y.feats.shape[0],memory=memory())
        hooks.assert_complete()
        if tuple(y.feats.shape)!=(FINAL_ROWS,6) or y.feats.dtype!=torch.float32 or tuple(y.coords.shape)!=(FINAL_ROWS,4) or y.coords.dtype!=torch.int32:raise RuntimeError('Texture output shape/dtype differs')
        if not torch.isfinite(y.feats).all().item():raise RuntimeError('Genuine raw texture field contains nonfinite values')
        tensor_refs.extend((weakref.ref(y.feats),weakref.ref(y.coords)))
        raw_cpu=y.feats.detach().cpu().contiguous();coords_cpu=y.coords.detach().cpu().contiguous()
        raw=raw_cpu.numpy();coords=coords_cpu.numpy()
        hooks.guide_order.finalize(coords)
        if not np.array_equal(coords,prepared['geometry_coords']):raise RuntimeError('Texture support/order differs from preserved geometry')
        features=np.ascontiguousarray(raw*np.float32(.5)+np.float32(.5))
        if not np.isfinite(features).all():raise RuntimeError('Transformed genuine texture field is nonfinite')
        record['output_validation']={'rows':FINAL_ROWS,'raw_shape':[FINAL_ROWS,6],'features_shape':[FINAL_ROWS,6],'coordinates_shape':[FINAL_ROWS,4],
            'all_finite':True,'geometry_coordinate_order_exact':True,'original_transform':'output * F32(0.5) + F32(0.5), CPU elementwise; no clamp','clamped':False}
        from artifact_io import write_output
        provenance={'mode':'production','source':record['source_sha256'],'checkpoint_sha256':record['checkpoint']['freshly_streamed_sha256'],
            'inputs':prepared['metadata'],'native_binary_sha256':'91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01',
            'algorithm':'explicit_gemm','runtime_dtype_policy':'208 original F16 body parameters;76 exact F32 source norms/projections; no BF16 substitution',
            'runtime':dict(record['runtime']),'forward_calls':1,'convolutions':40,'resolution':512,'mesh_extraction':False,'texture_baking':False,
            'guidance_parent_rows':[3743,17015,79169,378891],'guidance_child_rows':[17015,79169,378891,1842968],
            'final_convolution':{'source_equivalent_row_chunking':True,'rows':FINAL_ROWS,'tile_rows':32768,'chunks':patch.chunks,'global_native_neighbor_map':True,'global_input_indices_preserved':True},
            'measurements':{'forward_synchronized_seconds':record['forward_synchronized_seconds'],'memory_before_persistence':memory(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}}
        # Pin from the loader export rather than duplicating model identity strings.
        from texture_decoder_loader import CONFIG_SHA256
        provenance['config_sha256']=CONFIG_SHA256
        provenance=json.loads(json.dumps(provenance,allow_nan=False))
        record['output']=write_output(raw,features,coords,provenance,output)
        emit('texture-field-readback-verified',output_sha256=record['output']['metadata']['artifact_sha256'],bytes=record['output']['metadata']['artifact_bytes'])
    except BaseException as exc:record['neural_failure'],failure=detach_failure(exc)
    finally:
        record['convolutions_completed']=hooks.convolution_completed if hooks else 0
        record['subdivision_operations']=list(hooks.subdivision_records) if hooks else []
        record['native_map_calls']=native.calls if native else 0
        record['final_convolution_chunks']=patch.chunks if patch else 0;record['final_convolution_rows']=patch.rows_completed if patch else 0
        if patch:patch.remove()
        record['final_convolution_patch_restored']=patch is None or SubMConv3dFunction.__dict__[patch.name] is patch.original_descriptor
        patch=None
        if native:native.remove()
        if hooks:hooks.remove()
        for h in handles:h.remove()
        handles.clear();h=None;native=hooks=None
        if x:x.clear_spatial_cache()
        if y:y.clear_spatial_cache()
        if guide_subs:
            for sub in guide_subs:sub.clear_spatial_cache()
        sub=None
        prepared=x_cpu=c_cpu=x=y=guide_subs=raw_cpu=coords_cpu=raw=features=coords=None;gc.collect()
        record['GPU_input_output_tensors_collected']=all(ref() is None for ref in tensor_refs)
        if not record['GPU_input_output_tensors_collected']:raise RuntimeError('Owned texture input/output references survived')
    if failure:raise failure[0](failure[1])
