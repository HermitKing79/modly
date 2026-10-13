"""Explicit local audited assets; reuse strict loader with narrowly scoped BF16 widening."""
import importlib, inspect, json, os
from pathlib import Path
import amd_sparse_flow_local as old
import amd_shape_slat_contract as contract
ROOT,SOURCE,VENV,loading,digest,manifest=old.ROOT,old.SOURCE,old.VENV,old.loading,old.digest,old.manifest
EVIDENCE=ROOT/'docs/compatibility/amd-shape-slat-512-forward-evidence'
PREFIX=Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16')
AUDIT=ROOT/'docs/compatibility/amd-offline-model-audit-evidence/final-assets/slat_flow_img2shape_dit_1_3B_512_bf16.metadata.json'
TURRET=ROOT/'docs/compatibility/amd-object-conditioned-occupancy-evidence'
FEATURE=TURRET/'conditioning.safetensors'
FEATURE_SHA='78a0b54210568624639c8cacb7a803d83d4b838b782ea54e7aceab749d47e17a'
LOGITS_SHA='5ec5e5ceb114e477177df91ba0de444dc54a9c70103f4086701b754c76a9e3c2'
MASK_SHA='b30f83758f79a0a512e15e303da9392a4b95d15b50fcd41071ca85da40f191c5'
PINS=EVIDENCE/'source-contract.json'
def sources():
    return {str(p):digest(p) for p in (SOURCE/'trellis2').rglob('*.py')}
def verify_sources():
    current=sources()
    if current!=json.loads(PINS.read_text())['sources']:raise ValueError('AMD source identity changed')
    return current
def verify_inputs():
    import numpy as np
    import amd_feature_checkpoint as feature
    row=json.loads((TURRET/'dino/run/results.json').read_text())['worker']['feature_checkpoint']
    result=feature.verify(FEATURE,FEATURE.parent,{k:row['provenance'][k] for k in feature.FIELDS},row['output_sha256'])
    if result['file_sha256']!=FEATURE_SHA:raise ValueError('Saved turret feature hash changed')
    if digest(TURRET/'occupancy-logits.npy')!=LOGITS_SHA or digest(TURRET/'occupancy-mask.npy')!=MASK_SHA:raise ValueError('Saved native occupancy hash changed')
    pooled,coords=contract.reduce_occupancy(np.load(TURRET/'occupancy-logits.npy',allow_pickle=False),np.load(TURRET/'occupancy-mask.npy',allow_pickle=False))
    contract.validate_coords(coords,len(coords))
    return result,pooled,coords
def identity(*,hash_weights=True):
    from services.amd_model_metadata import json_metadata,safetensors_metadata
    config=json_metadata(PREFIX.with_suffix('.json'));current=safetensors_metadata(PREFIX.with_suffix('.safetensors'))
    audited=json.loads(AUDIT.read_text())
    for key in ('file_bytes','mtime_ns','header_bytes','header_sha256','tensor_count','parameter_count','tensors','tensor_data_bytes'):
        if current[key]!=audited[key]:raise ValueError('Audited shape checkpoint changed: '+key)
    if config!=audited['config'] or digest(SOURCE/'trellis2/models/structured_latent_flow.py')!=audited['source_module_sha256']:raise ValueError('Audited shape configuration/source changed')
    contract.validate_header(config,current['tensors'])
    feature,pooled,coords=verify_inputs()
    return dict(current,config=config,config_sha256=digest(PREFIX.with_suffix('.json')),
        checkpoint_sha256=digest(PREFIX.with_suffix('.safetensors')) if hash_weights else None,token_count=len(coords),feature=feature)
def import_class():
    if os.environ.get('ATTN_BACKEND')!='sdpa':raise RuntimeError('Explicit PyTorch SDPA required')
    old.sys.path.insert(0,str(SOURCE))
    cls=importlib.import_module('trellis2.models.structured_latent_flow').SLatFlowModel
    if Path(inspect.getfile(cls)).resolve()!=(SOURCE/'trellis2/models/structured_latent_flow.py').resolve():raise ValueError('Model provenance differs')
    from trellis2.modules.sparse import config
    # This setter is the existing source API; select its explicit PyTorch-only else branch.
    # No exception-driven FlashAttention fallback and no sparse convolution in this model.
    config.set_conv_backend('none');config.set_attn_backend('sdpa')
    if config.CONV!='none' or config.ATTN!='sdpa':raise ValueError('Explicit sparse PyTorch backend selection failed')
    return cls
def rope_frequencies():
    import torch
    from trellis2.modules.sparse.attention.rope import SparseRotaryPositionEmbedder
    with torch.device('cpu'):return SparseRotaryPositionEmbedder(128).freqs
def factory(config,*,storage=False):
    import torch
    model=import_class()(**config['args'])
    contract.validate_manifest(manifest(dict(model.named_parameters())),manifest(dict(model.named_buffers())),'meta')
    # Real source's freqs are ordinary unlearned attributes, not state_dict buffers.
    # Meta construction cannot materialize these; re-run only their genuine CPU constructor.
    for block in model.blocks:block.self_attn.rope.freqs=rope_frequencies()
    if storage:
        for p in model.parameters():p.data=p.data.to(torch.bfloat16)
    return model
def load_cpu():
    from strict_loader import StrictLocalComponent
    component=StrictLocalComponent(str(PREFIX.with_suffix('.json')),str(PREFIX.with_suffix('.safetensors')),lambda c:factory(c,storage=True))
    try:
        model=component.load_cpu();widened=old.widen(model,component.diagnostics['payload_sha256'],contract.schema())
        contract.validate_manifest(manifest(dict(model.named_parameters())),manifest(dict(model.named_buffers())),'cpu')
        component.diagnostics.update(runtime_fp32_widened=widened,ordinary_frequency_policy='30 genuine deterministic CPU F32 [21] attributes; shared sparse cache generates phases during forward')
        return component,model
    except BaseException:component.unload();raise
