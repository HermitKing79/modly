"""Proof-only strict storage adapter; genuine dense source, explicit local assets."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent),str(ROOT/'experimental/extensions/trellis2-amd')]
import amd_sparse_flow_contract as contract
import amd_sparse_decoder_loading as loading
from services.amd_model_metadata import json_metadata,safetensors_metadata
from amd_dinov3_local import digest,SOURCE,VENV

EVIDENCE=ROOT/'docs/compatibility/amd-sparse-flow-evidence'
PREFIX=Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16')
AUDIT=ROOT/'docs/compatibility/amd-offline-model-audit-evidence/final-assets/ss_flow_img_dit_1_3B_64_bf16.metadata.json'
FEATURE=ROOT/'docs/compatibility/amd-dinov3-allocation-evidence/conditioning.safetensors'
FEATURE_SHA='3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a'
PINS=EVIDENCE/'source-contract.json'

def sources():
    names=['__init__.py','pipelines/__init__.py','models/sparse_structure_flow.py','models/__init__.py','modules/utils.py','modules/norm.py',
        'modules/attention/__init__.py','modules/attention/config.py','modules/attention/modules.py',
        'modules/attention/full_attn.py','modules/attention/rope.py','modules/attention/proj_attention.py',
        'modules/transformer/__init__.py','modules/transformer/modulated.py','modules/transformer/blocks.py',
        'modules/sparse/__init__.py','modules/sparse/config.py','modules/sparse/basic.py',
        'modules/sparse/linear.py','modules/sparse/conv/__init__.py','modules/sparse/conv/conv.py']
    return {str(SOURCE/'trellis2'/name):digest(SOURCE/'trellis2'/name) for name in names}

def verify_sources():
    pins=json.loads(PINS.read_text())
    if sources()!=pins['sources']:raise RuntimeError('Pinned sparse-flow source identity changed')
    return pins['sources']

def identity(*,hash_weights=True):
    config=json_metadata(PREFIX.with_suffix('.json'));current=safetensors_metadata(PREFIX.with_suffix('.safetensors'))
    old=json.loads(AUDIT.read_text())
    for key in ('file_bytes','mtime_ns','header_bytes','header_sha256','tensor_count','parameter_count','tensors','tensor_data_bytes'):
        if current[key]!=old[key]:raise ValueError('Sparse-flow audited metadata changed: '+key)
    if config!=old['config'] or old['source_module_sha256']!=digest(SOURCE/'trellis2/models/sparse_structure_flow.py'):
        raise ValueError('Sparse-flow configuration/source identity changed')
    contract.validate_header(config,current['tensors'])
    return dict(current,config=config,checkpoint_sha256=digest(PREFIX.with_suffix('.safetensors')) if hash_weights else None,
                config_sha256=digest(PREFIX.with_suffix('.json')))

def manifest(state):
    dtypes={'torch.float32':'F32','torch.bfloat16':'BF16','torch.complex64':'C64'}
    return {k:{'shape':list(v.shape),'dtype':dtypes.get(str(v.dtype),str(v.dtype)),
               'device':str(v.device),'is_meta':v.is_meta} for k,v in state.items()}

def verify_feature():
    import amd_feature_checkpoint as feature
    old=json.loads((ROOT/'docs/compatibility/amd-dinov3-allocation-evidence/run/results.json').read_text())['worker']['feature_checkpoint']
    provenance={k:old['provenance'][k] for k in feature.FIELDS}
    result=feature.verify(FEATURE,FEATURE.parent,provenance,old['output_sha256'])
    if result['file_sha256']!=FEATURE_SHA or result['file_sha256']!=old['file_sha256']:
        raise ValueError('Saved DINO feature checkpoint hash differs')
    return result

def import_class():
    import os
    if os.environ.get('ATTN_BACKEND')!='sdpa':raise RuntimeError('Explicit genuine SDPA backend required')
    sys.path.insert(0,str(SOURCE))
    cls=importlib.import_module('trellis2.models.sparse_structure_flow').SparseStructureFlowModel
    if Path(inspect.getfile(cls)).resolve()!=(SOURCE/'trellis2/models/sparse_structure_flow.py').resolve():
        raise RuntimeError('Sparse-flow class provenance mismatch')
    from trellis2.modules.attention import config
    if config.BACKEND!='sdpa':raise RuntimeError('Dense attention backend differs')
    return cls

def rope():
    import torch
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    with torch.device('cpu'):
        coords=torch.stack(torch.meshgrid(*(torch.arange(16,dtype=torch.int64) for _ in range(3)),indexing='ij'),dim=-1).reshape(-1,3)
        generated=RotaryPositionEmbedder(128,3)(coords)
        freqs=1/(10000**(torch.arange(21,dtype=torch.float32)/21))
        angles=(coords.float().unsqueeze(-1)*freqs).reshape(4096,63)
        reference=torch.cat((torch.polar(torch.ones_like(angles),angles),torch.ones(4096,1,dtype=torch.complex64)),dim=-1)
    if (generated.shape!=(4096,64) or generated.dtype!=torch.complex64 or generated.device.type!='cpu'
            or not torch.isfinite(generated).all() or not torch.equal(generated,reference)):
        raise ValueError('Procedural RoPE differs from exact CPU source reference')
    return generated

def factory(config,*,storage=False):
    import torch
    contract.validate_header(config,{k:{**v,'dtype':'BF16'} for k,v in contract.schema().items()})
    cls=import_class();model=cls(**config['args'])
    # Only this deterministic, nonlearned buffer is absent from the checkpoint.
    # Source persistence changes explicitly for this proof; global loader stays strict.
    model.register_buffer('rope_phases',rope(),persistent=False)
    parameters=manifest(dict(model.named_parameters()))
    buffers=manifest(dict(model.named_buffers()))
    contract.validate_manifest(parameters,buffers,'meta')
    if storage:
        for name,p in model.named_parameters():
            if contract.schema()[name]['dtype']=='F32':p.data=p.data.to(torch.bfloat16)
    return model

def widen(model,payload_hashes,schema=None):
    """Finite reversible widening only; raw bytes were already strictly checked."""
    import torch
    schema=contract.schema() if schema is None else schema
    params=dict(model.named_parameters())
    if set(params)!=set(schema) or set(payload_hashes)!=set(schema):raise ValueError('Widening keys differ')
    widened=[]
    for name,row in schema.items():
        p=params[name]
        if p.device.type!='cpu' or p.is_meta or p.dtype!=torch.bfloat16 or list(p.shape)!=row['shape']:
            raise ValueError('Invalid storage-stage parameter: '+name)
        if not torch.isfinite(p).all():raise ValueError('Nonfinite checkpoint parameter: '+name)
        if row['dtype']=='F32':
            p.data=p.data.float()
            raw=p.detach().to(torch.bfloat16).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest()!=payload_hashes[name]:raise ValueError('Widening does not preserve checkpoint value: '+name)
            widened.append(name)
        elif row['dtype']!='BF16':raise ValueError('Unapproved runtime parameter dtype: '+name)
    return widened

def load_cpu():
    from strict_loader import StrictLocalComponent
    component=StrictLocalComponent(str(PREFIX.with_suffix('.json')),str(PREFIX.with_suffix('.safetensors')),
                                   lambda config:factory(config,storage=True))
    try:
        model=component.load_cpu()
        widened=widen(model,component.diagnostics['payload_sha256'])
        contract.validate_manifest(manifest(dict(model.named_parameters())),
                                   manifest(dict(model.named_buffers())),'cpu')
        if not __import__('torch').equal(model.rope_phases,rope()):raise ValueError('RoPE changed while loading')
        component.diagnostics.update(runtime_fp32_widened=widened,storage_schema='640 BF16 parameters',
            generated_buffer_policy='rope_phases only: genuine deterministic CPU construction; explicitly nonpersistent')
        return component,model
    except BaseException:
        component.unload()
        raise
