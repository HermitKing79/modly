"""Composable genuine guided F16 forward; importing this module executes none."""
import gc
import hashlib
import json
import time
import traceback
from types import SimpleNamespace
import weakref
from fractions import Fraction

from . import inputs
from .profile import require_authenticated, CHECKPOINT_SHA256, CONFIG_SHA256
from .admission import AdmissionHooks, INITIAL_FREE_REQUIRED, LARGE_TILED_SITES, require
from .tiling import LargeSupportConvolutionPatch, FinalConvolutionPatch
from .runtime import NativeBoundary, validate_configured_model, validate_tensor_contract
from .source_identity import qualify_function

GPU_DEADLINE = 240.0


def _validate_dependencies(model, dependencies, torch=None):
    """Injection supports composition, never permission to replace the learned forward."""
    forward = dependencies.decoder_forward
    model_type = type(model)
    require(model_type.__name__ == 'SparseUnetVaeDecoder'
            and model_type.__module__ == 'trellis2.models.sc_vaes.sparse_unet_vae'
            and forward is model_type.forward and forward.__name__ == 'forward'
            and forward.__module__ == model_type.__module__,
            'genuine source-pinned guided decoder forward required')
    source_sha = 'c9ce51db70bb1374c8bdc858129f6ff1170af1447e5210d434b7ebb289dbd2ce'
    qualify_function(forward, source_sha, 'SparseUnetVaeDecoder', 'forward', defaults=(None, False))
    torch = dependencies.torch if torch is None else torch
    require(forward.__globals__.get('torch') is torch
            and forward.__globals__.get('F') is torch.nn.functional,
            'decoder Torch/functional globals differ from qualified runtime')
    chunked = forward.__globals__.get('chunked_apply')
    qualify_function(chunked, source_sha, 'chunked_apply')
    require(chunked.__globals__ is forward.__globals__,
            'original finalization helper runtime namespace differs')
    sparse_type = dependencies.sparse.SparseTensor
    require(sparse_type.__name__ == 'SparseTensor' and sparse_type.__module__ == 'trellis2.modules.sparse.basic',
            'genuine source-pinned SparseTensor required')
    initializer = sparse_type.__init__
    qualify_function(initializer, 'e867e429b9c2fe154000725596b626273fb567025b9c232f3cc79140f00605b7',
                     'SparseTensor', '__init__')
    require(initializer.__globals__.get('torch') is torch
            and initializer.__globals__.get('Fraction') is Fraction
            and initializer.__globals__.get('config') is dependencies.sparse_config
            and dependencies.sparse_config.CONV == 'flex_gemm'
            and dependencies.sparse_config.DEBUG is False
            and sparse_type.SparseTensorData is None,
            'SparseTensor runtime globals/backend differ from original qualified path')
    require(dependencies.owner.__name__ == 'SubMConv3dFunction'
            and dependencies.owner.__module__ == 'flex_gemm.ops.spconv.submanifold_conv3d',
            'genuine explicit-GEMM descriptor owner required')


def _dependencies():
    # A future qualified worker calls this only after its original native gate.
    # CPU qualification never invokes this function or imports these packages.
    from trellis2.modules import sparse
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.modules.sparse.conv import config
    from trellis2.modules.sparse import config as sparse_config
    import flex_gemm.ops.spconv as spconv
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
    return SimpleNamespace(sparse=sparse, decoder_forward=SparseUnetVaeDecoder.forward,
                           conv_config=config, sparse_config=sparse_config,
                           spconv=spconv, owner=SubMConv3dFunction)


def run_neural(model, torch, cuda, record, output, gpu_started, *, profile, prepared,
               write_output, emit=None, dependencies=None):
    """Execute once inside an already-qualified owned worker; no CLI launches it.

    ``write_output(raw, features, coords, provenance, output, profile=profile)``
    must durably serialize, independently reopen and acknowledge before returning
    verified=True and independently_verified_before_release=True. The caller
    owns checkpoint/native qualification, model unloading and zero-VRAM release.
    """
    import numpy as np
    profile = require_authenticated(profile)
    inputs.require_prepared(prepared, profile)
    validate_configured_model(model, torch)
    require(callable(write_output) and callable(emit or (lambda *_a, **_k: None)),
            'explicit durable writer/consumer callback required')
    require(type(gpu_started) in (int, float) and gpu_started <= time.monotonic(),
            'original GPU stage start time required')
    require(record.get('decoder_cpu') == record.get('native_import') == 'PASS'
            and record.get('runtime_dtype_counts') == {'F16': 208, 'F32': 76}
            and record.get('runtime_dtype_changes') == {}
            and record.get('checkpoint', {}).get('freshly_streamed_sha256') == CHECKPOINT_SHA256,
            'genuine strict F16/native/checkpoint qualification required before forward')
    deps = _dependencies() if dependencies is None else dependencies
    external_emit = emit or (lambda _event, **_details: None)
    record.update(pbr_profile=profile.document(), profile_sha256=profile.sha256,
                  run_contract_sha256=profile.run_contract_sha256)
    hooks = native = large = final = None
    handles, tensor_refs = [], []
    x_cpu = c_cpu = x = y = guide_subs = bits = logits = parent_coords = sub = g = None
    raw_cpu = coords_cpu = raw = features = coords = None
    failure = None

    def event(event_name, **details):
        elapsed = time.monotonic()-gpu_started
        if elapsed >= GPU_DEADLINE:
            raise TimeoutError('240-second texture GPU stage deadline')
        record['last_event'] = event_name
        external_emit(event_name, elapsed_gpu_seconds=elapsed, **details)

    def hook_event(details):
        details = dict(details)
        name = details.pop('event')
        event(name, **details)

    def memory():
        free, total = torch.cuda.mem_get_info(0)
        return dict(free_bytes=free, total_bytes=total,
                    allocated_bytes=torch.cuda.memory_allocated(0),
                    reserved_bytes=torch.cuda.memory_reserved(0))

    try:
        initial_memory = memory()
        require(initial_memory['free_bytes'] >= INITIAL_FREE_REQUIRED,
                'original 13GiB initial free VRAM admission failed')
        require(torch.cuda.current_stream(0).cuda_stream == 0,
                'original default CUDA stream required')
        _validate_dependencies(model, deps, torch)
        cache_type = getattr(deps.owner, '_sparse_submanifold_conv_forward').__globals__['SubMConv3dNeighborCache']
        hooks = AdmissionHooks(model, torch, profile=profile, guidance=prepared, cache_type=cache_type,
                               emit=hook_event, memory=memory).install()
        native = NativeBoundary(cuda, record['bindings'], hooks, torch,
                                deps.conv_config, deps.spconv, event, profile=profile).install()
        modules = dict((name, dict(model.named_modules())[name]) for name in LARGE_TILED_SITES)
        large = LargeSupportConvolutionPatch(deps.owner, modules, hooks, torch, event,
                                             profile=profile).install()
        modules = None
        final = FinalConvolutionPatch(deps.owner, model.blocks[3][-1].conv2, hooks,
                                      torch, event, profile=profile).install()
        deps.spconv.set_algorithm('explicit_gemm')
        deps.spconv.set_hashmap_ratio(2.0)
        x_cpu = torch.from_numpy(prepared['denormalized'].copy())
        c_cpu = torch.from_numpy(prepared['coords'].copy())
        validate_tensor_contract(x_cpu, c_cpu, profile=profile)
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        guidance_bytes = sum(g['logits_bf16'].nbytes+g['coords'].nbytes for g in prepared['guides'])
        require(parameter_bytes == 948567448, 'original F16/F32 model parameter bytes differ')
        hooks._admit('texture-model-guides-and-from-latent',
            {'visible_bytes': parameter_bytes+guidance_bytes+profile.initial_rows*(32*4+4*4+1024*16)})
        event('texture-model-placement-entry', memory=memory(), parameter_bytes=parameter_bytes,
              guidance_bytes=guidance_bytes)
        model.to(device=torch.device('cuda', 0))
        require(all(p.device == torch.device('cuda', 0) for p in model.parameters()),
                'texture GPU placement incomplete')
        x = deps.sparse.SparseTensor(x_cpu.to('cuda:0'), c_cpu.to('cuda:0'), shape=torch.Size([1, 32]))
        x.clear_spatial_cache()
        validate_tensor_contract(x.feats, x.coords, profile=profile)
        tensor_refs.extend((weakref.ref(x.feats), weakref.ref(x.coords)))
        guide_subs = []
        for g in prepared['guides']:
            bits = torch.from_numpy(g['logits_bf16'].copy())
            logits = bits.view(torch.bfloat16)
            parent_coords = torch.from_numpy(g['coords'].copy())
            require(hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()
                    == hashlib.sha256(g['logits_bf16']).hexdigest(), 'original guide BF16 bytes changed')
            sub = deps.sparse.SparseTensor(logits.to('cuda:0'), parent_coords.to('cuda:0'))
            guide_subs.append(sub)
            tensor_refs.extend((weakref.ref(sub.feats), weakref.ref(sub.coords)))
        bits = logits = parent_coords = sub = g = None
        torch.cuda.synchronize(0)
        event('texture-model-placement-synchronized', memory=memory(), rows=profile.initial_rows, guides=4)
        handles.append(model.from_latent.register_forward_pre_hook(
            lambda _m, args: event('from-latent-entry', rows=args[0].feats.shape[0])))
        native.phase = 'neural'
        record['forward_calls'] = record.get('forward_calls', 0)+1
        require(record['forward_calls'] == 1, 'texture forward retry prohibited')
        event('texture-forward-entry', rows=profile.initial_rows, guide_subs=4, return_subs=False,
              runtime_dtype='source F16 body/F32 norms and projections')
        began = time.monotonic()
        with torch.inference_mode():
            y = deps.decoder_forward(model, x, guide_subs=guide_subs, return_subs=False)
        event('texture-forward-host-return', rows=y.feats.shape[0], host_return_is_not_sync=True)
        torch.cuda.synchronize(0)
        record['forward_synchronized_seconds'] = time.monotonic()-began
        event('texture-forward-synchronized', rows=y.feats.shape[0], memory=memory())
        hooks.assert_complete()
        native.assert_complete()
        large.assert_complete()
        final.assert_complete()
        require(tuple(y.feats.shape) == (profile.final_rows, 6) and y.feats.dtype == torch.float32
                and tuple(y.coords.shape) == (profile.final_rows, 4) and y.coords.dtype == torch.int32
                and torch.isfinite(y.feats).all().item(), 'genuine raw field shape/dtype/finite contract differs')
        tensor_refs.extend((weakref.ref(y.feats), weakref.ref(y.coords)))
        raw_cpu = y.feats.detach().cpu().contiguous()
        coords_cpu = y.coords.detach().cpu().contiguous()
        raw, coords = raw_cpu.numpy(), coords_cpu.numpy()
        record['guide_order_validation'] = hooks.guide_order.finalize(coords)
        require(np.array_equal(coords, prepared['geometry_coords']), 'texture/geometry support/order differs')
        features = np.ascontiguousarray(raw*np.float32(.5)+np.float32(.5))
        require(np.isfinite(features).all(), 'transformed genuine field is nonfinite')
        record['output_validation'] = dict(rows=profile.final_rows, raw_shape=[profile.final_rows, 6],
            features_shape=[profile.final_rows, 6], coordinates_shape=[profile.final_rows, 4],
            all_finite=True, geometry_coordinate_order_exact=True,
            original_transform='output * F32(0.5) + F32(0.5), CPU elementwise; no clamp', clamped=False)
        metadata = dict(prepared['metadata'])
        metadata.pop('profile', None)  # Embedded separately as canonical JSON by the field adapter.
        provenance = dict(mode='production', source=record['source_sha256'],
            checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=CONFIG_SHA256, inputs=metadata,
            native_binary_sha256='91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01',
            algorithm='explicit_gemm', runtime_dtype_policy='208 original F16 body parameters;76 exact F32 source norms/projections; no BF16 substitution',
            runtime=dict(record['runtime']), forward_calls=1, convolutions=40, resolution=512,
            mesh_extraction=False, texture_baking=False, guidance_parent_rows=list(profile.parent_rows),
            guidance_child_rows=list(profile.child_rows), large_convolutions=list(large.records),
            final_convolution=dict(final.records[0]),
            measurements=dict(forward_synchronized_seconds=record['forward_synchronized_seconds'],
                memory_before_persistence=memory(), peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
        provenance = json.loads(json.dumps(provenance, allow_nan=False))
        record['output'] = write_output(raw, features, coords, provenance, output, profile=profile)
        require(record['output'].get('verified') is True
                and record['output'].get('independently_verified_before_release') is True,
                'field not independently acknowledged before tensor release')
        event('texture-field-readback-verified', output_sha256=record['output']['metadata']['artifact_sha256'],
              bytes=record['output']['metadata']['artifact_bytes'])
    except BaseException as exc:
        record['neural_failure'] = dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        failure = (type(exc), str(exc))
        exc.__traceback__ = None
    finally:
        record['convolutions_completed'] = hooks.convolution_completed if hooks else 0
        record['subdivision_operations'] = list(hooks.subdivision_records) if hooks else []
        record['native_map_calls'] = native.calls if native else 0
        record['native_map_records'] = list(native.records) if native else []
        record['large_convolution_records'] = [dict(row) for row in large.records] if large else []
        record['large_convolution_chunks'] = large.chunks if large else 0
        record['final_convolution_record'] = dict(final.records[0]) if final and final.records else None
        record['final_convolution_chunks'] = final.chunks if final else 0
        record['final_convolution_rows'] = final.rows_completed if final else 0
        for name, patch in (('final', final), ('large', large)):
            if patch:
                patch.remove()
            record[name+'_convolution_patch_restored'] = (patch is None or
                patch.owner.__dict__[patch.name] is patch.original_descriptor)
        patch = final = large = None
        if native:
            native.remove()
        record['native_bindings_restored'] = native is None or all(
            getattr(cuda, name) is original for name, original in native.originals.items())
        if hooks:
            hooks.remove()
        record['convolution_hooks_removed'] = hooks is None or not hooks.handles
        for handle in handles:
            handle.remove()
        handles.clear()
        handle = native = hooks = None
        for value in (x, y, *(guide_subs or [])):
            if value is not None:
                value.clear_spatial_cache()
        value = None
        x_cpu = c_cpu = x = y = guide_subs = bits = logits = parent_coords = sub = g = None
        raw_cpu = coords_cpu = raw = features = coords = None
        gc.collect()
        record['GPU_input_output_tensors_collected'] = all(ref() is None for ref in tensor_refs)
        if not record['GPU_input_output_tensors_collected']:
            raise RuntimeError('owned texture input/output tensor references survived')
    if failure:
        raise failure[0](failure[1])
