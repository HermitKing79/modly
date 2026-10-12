"""Explicit offline DINO construction; real upstream methods, no remote factories."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent),str(ROOT/'experimental/extensions/trellis2-amd')]
import amd_dinov3_contract as contract
from services.amd_model_metadata import json_metadata,safetensors_metadata

SOURCE=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
VENV=Path('D:/AI/Trellis2AMD/venv')
MODEL=Path('D:/AI/Trellis2AMD/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m')
EVIDENCE=ROOT/'docs/compatibility/amd-dinov3-conditioning-evidence'
AUDIT=ROOT/'docs/compatibility/amd-offline-model-audit-evidence/final-assets/dinov3.metadata.json'
PINS=EVIDENCE/'source-contract.json'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def sources():
    names=[SOURCE/'trellis2/modules/image_feature_extractor.py', SOURCE/'trellis2/pipelines/trellis2_image_to_3d.py']
    site=VENV/'Lib/site-packages'
    names += [site/name for name in (
        'transformers/models/dinov3_vit/modeling_dinov3_vit.py',
        'transformers/models/dinov3_vit/configuration_dinov3_vit.py',
        'transformers/pytorch_utils.py','torchvision/transforms/transforms.py',
        'torchvision/transforms/_functional_tensor.py')]
    return {str(path):digest(path) for path in names}


def verify_sources():
    expected=json.loads(PINS.read_text())['sources']
    actual=sources()
    if actual!=expected:raise RuntimeError('Inspected DINOv3/conditioning source identity changed')
    return actual


def identity():
    old=json.loads(AUDIT.read_text());config=json_metadata(MODEL/'config.json')
    current=safetensors_metadata(MODEL/'model.safetensors')
    fields=('file_bytes','mtime_ns','header_bytes','header_sha256','tensor_count','parameter_count',
            'tensors','metadata_only_fixture','tensor_data_bytes')
    if config!=old['config'] or any(current[key]!=old[key] for key in fields):
        raise ValueError('DINOv3 checkpoint/configuration differs from the audited local asset')
    contract.expected_shape(config)
    if current['tensor_count']!=415 or current['parameter_count']!=303129600 or current['dtypes']!={'F32':415}:
        raise ValueError('Unexpected DINO checkpoint tensor identity')
    return dict(current,config=config,config_sha256=digest(MODEL/'config.json'),
                checkpoint_sha256=digest(MODEL/'model.safetensors'))


def factory(config):
    # StrictLocalComponent invokes this under torch.device('meta'). No factory
    # accepts a repository ID. Replace only the nonpersistent RoPE submodule
    # with its real CPU initializer, leaving all 415 persistent entries on meta.
    contract.expected_shape(config)
    import torch
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel,DINOv3ViTRopePositionEmbedding
    model=DINOv3ViTModel(DINOv3ViTConfig(**config))
    with torch.device('cpu'):
        model.rope_embeddings=DINOv3ViTRopePositionEmbedding(model.config)
    import amd_sparse_decoder_loading as loading
    contract.validate_buffers(loading.tensor_manifest(dict(model.named_buffers())))
    return model


def rope_cache():
    from transformers.models.dinov3_vit.modeling_dinov3_vit import get_patches_center_coordinates
    wrapper=get_patches_center_coordinates
    closed=inspect.getclosurevars(wrapper).nonlocals
    cached=closed.get('func_with_cache')
    if (set(closed)!={'func','func_with_cache'} or cached is None
            or cached.__wrapped__ is not wrapper.__wrapped__ or closed['func'] is not wrapper.__wrapped__
            or cached.cache_parameters()!=dict(maxsize=32,typed=False)
            or not callable(cached.cache_clear)):
        raise RuntimeError('Unverified ownership/cleanup of DINO patch-coordinate cache')
    return cached


def extractor(model):
    verify_sources()
    path=SOURCE/'trellis2/modules/image_feature_extractor.py'
    spec=importlib.util.spec_from_file_location('_modly_inspected_image_features',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from torchvision import transforms
    # The four assignments mirror the inspected constructor; its downloading
    # from_pretrained call is replaced by the strictly loaded local model only.
    obj=module.DinoV3FeatureExtractor.__new__(module.DinoV3FeatureExtractor)
    obj.model_name=str(MODEL);obj.model=model.eval();obj.image_size=512
    obj.transform=transforms.Compose([transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    return obj


def pipeline_methods(feature_extractor):
    # Bind ONLY these unmodified inspected upstream method bodies. Importing the
    # full pipeline would pull unrelated native geometry components. No models,
    # samplers or dummy successful pipeline implementations are created here.
    import numpy as np
    import torch
    from PIL import Image
    from typing import List,Union
    verify_sources();path=SOURCE/'trellis2/pipelines/trellis2_image_to_3d.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Trellis2ImageTo3DPipeline')
    wanted={'preprocess_image','get_cond'}
    functions=[node for node in cls.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in wanted]
    if {node.name for node in functions}!=wanted:raise ValueError('Upstream conditioning methods missing')
    namespace=dict(np=np,torch=torch,Image=Image,List=List,Union=Union)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    shell=types.SimpleNamespace(rembg_model=None,low_vram=False,image_cond_model=feature_extractor)
    for name in wanted:setattr(shell,name,types.MethodType(namespace[name],shell))
    return shell


def fixture(path):
    from PIL import Image
    pixels=bytearray(64*64*4)
    for y in range(8,56):
        for x in range(8,56):
            offset=(y*64+x)*4;pixels[offset:offset+4]=bytes((x*4,y*4,(x+y)*2,255))
    image=Image.frombytes('RGBA',(64,64),bytes(pixels))
    if path is not None:image.save(path)
    return image
