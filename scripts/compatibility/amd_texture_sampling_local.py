"""Pinned local texture flow; shared strict BF16 load and reversible F32 widening."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

import amd_shape_slat_local as previous
import amd_texture_sampling_contract as contract

ROOT, SOURCE, VENV = previous.ROOT, previous.SOURCE, previous.VENV
loading, digest, manifest = previous.loading, previous.digest, previous.manifest
widen = previous.old.widen
import_class = previous.import_class
rope_frequencies = previous.rope_frequencies
EVIDENCE = ROOT / 'docs/compatibility/amd-turret-texture-sampling-evidence'
PINS = EVIDENCE / 'source-contract.json'
PREFIX = previous.PREFIX.parent / 'slat_flow_imgshape2tex_dit_1_3B_512_bf16'
PIPELINE = PREFIX.parent.parent / 'pipeline.json'
AUDIT = ROOT / 'docs/compatibility/amd-turret-texture-route-evidence/local-assets.json'
SOURCE_AUDIT = AUDIT.parent / 'independent-texture-source-audit.json'
SHAPE_EVIDENCE = ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence'
SHAPE_RAW = SHAPE_EVIDENCE / 'shape-latent-raw.npy'
SHAPE_DENORMALIZED = SHAPE_EVIDENCE / 'shape-latent-denormalized.npy'
COORDS = SHAPE_EVIDENCE / 'shape-latent-coordinates.npy'
SHAPE_PROVENANCE = SHAPE_EVIDENCE / 'shape-latent-provenance.json'
FEATURE, FEATURE_SHA = previous.FEATURE, previous.FEATURE_SHA
CHECKPOINT_SHA = '8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d'
CONFIG_SHA = '3093796257a62f838d2e5578ac4580ca3b7d88d10b87d348065d3fdfcc4c2077'
PIPELINE_SHA = '222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6'
AUDIT_SHA = '9f764cece1ae4630ec437b2a45d8619ff894c3312190e86813350a1490c2d481'
SOURCE_AUDIT_SHA = '4b17ec4c7fa357ab835421c563cc1de02f86c874bd7df1a3057b6001c670b47c'
SHAPE_PROVENANCE_SHA = '4834412e700434f5645cd919dd0d75e9209b6b5d67cc7cb765e98b76890f9a81'
SAMPLER_PINS = SHAPE_EVIDENCE / 'source-contract.json'


def pinned_json(path, sha256):
    if digest(path) != sha256:
        raise ValueError('Pinned texture-route evidence identity changed: ' + str(path))
    return json.loads(path.read_text())


def sources():
    return previous.sources()


def verify_sources():
    current = previous.verify_sources()
    audit = pinned_json(SOURCE_AUDIT, SOURCE_AUDIT_SHA)
    for relative, expected in audit['source_sha256'].items():
        path = (PIPELINE if relative == 'pipeline.json' else PREFIX.parent / Path(relative).name
                if relative.startswith('ckpts/') else SOURCE / relative)
        if digest(path) != expected:
            raise ValueError('Audited texture source/configuration changed: ' + relative)
    return current


def dependencies():
    site = VENV / 'Lib/site-packages'
    wanted = {'torch', 'numpy', 'safetensors', 'tqdm', 'easydict'}
    return dict(versions={d.metadata['Name'].lower(): d.version
                          for d in importlib.metadata.distributions(path=[str(site)])
                          if d.metadata['Name'].lower() in wanted},
                pure_entry_hashes={name: digest(site / (name + '/__init__.py')) for name in ('tqdm', 'easydict')})


def verify_sampler():
    pinned_json(AUDIT, AUDIT_SHA)
    pins = json.loads(SAMPLER_PINS.read_text())
    current = {path: digest(Path(path)) for path in pins['sampler_sources']}
    deps = dependencies()
    if current != pins['sampler_sources'] or deps != pins['dependencies'] or digest(PIPELINE) != PIPELINE_SHA:
        raise ValueError('Pinned texture sampler/dependencies/pipeline identity changed')
    pipeline = json.loads(PIPELINE.read_text())['args']
    expected_params = {key: list(value) if isinstance(value, tuple) else value
                       for key, value in contract.PARAMS.items() if key != 'dino_lock'}
    expected = dict(name='FlowEulerGuidanceIntervalSampler', args={'sigma_min': contract.SIGMA_MIN}, params=expected_params)
    if (pipeline['tex_slat_sampler'] != expected
            or pipeline['shape_slat_normalization'] != contract.SHAPE_NORMALIZATION
            or pipeline['tex_slat_normalization'] != contract.TEXTURE_NORMALIZATION
            or pipeline['models']['tex_slat_flow_model_512'] != 'ckpts/' + PREFIX.name):
        raise ValueError('Canonical texture sampler/model/normalizations differ')
    contract.require_count(12, 12, 0)
    return dict(sampler_sources=current, dependencies=deps, pipeline_sha256=PIPELINE_SHA,
                shape_normalization=pipeline['shape_slat_normalization'], normalization=pipeline['tex_slat_normalization'])


def verify_inputs():
    import numpy as np
    import amd_feature_checkpoint as feature
    from amd_shape_sampling_artifact import denormalize

    assets = pinned_json(AUDIT, AUDIT_SHA)
    provenance = pinned_json(SHAPE_PROVENANCE, SHAPE_PROVENANCE_SHA)
    expected = provenance['checkpoint']['conditioning']
    result = feature.verify(FEATURE, FEATURE.parent,
                            {key: expected['provenance'][key] for key in feature.FIELDS}, expected['output_sha256'])
    if result['file_sha256'] != FEATURE_SHA or result != expected:
        raise ValueError('Saved shape/object conditioning provenance changed')
    arrays = {}
    for path in (SHAPE_RAW, SHAPE_DENORMALIZED, COORDS):
        row = assets['saved_inputs'][path.name]
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Saved texture-conditioning input identity changed: ' + path.name)
        array = np.load(path, allow_pickle=False)
        if (str(array.dtype) != row['dtype'] or list(array.shape) != row['shape']
                or not array.flags.c_contiguous or not np.isfinite(array).all()
                or hashlib.sha256(array.tobytes()).hexdigest() != row['payload_sha256']):
            raise ValueError('Saved texture-conditioning input payload changed: ' + path.name)
        arrays[path.name] = array
    raw, denormalized, coords = (arrays[path.name] for path in (SHAPE_RAW, SHAPE_DENORMALIZED, COORDS))
    contract.validate_coords(coords, contract.TOKEN_COUNT)
    mean, std = contract.normalization_arrays(contract.SHAPE_NORMALIZATION, contract.SHAPE_NORMALIZATION)
    if not np.array_equal(denormalize(raw, mean, std), denormalized):
        raise ValueError('Saved shape latent denormalization no longer reproduces byte-exactly')
    if provenance['raw_output']['sha256'] != assets['saved_inputs'][SHAPE_RAW.name]['payload_sha256']:
        raise ValueError('Saved shape sample provenance differs')
    return result, contract.normalize_shape(denormalized), coords


def identity(*, hash_weights=True, input_verifier=None):
    from services.amd_model_metadata import json_metadata, safetensors_metadata
    assets = pinned_json(AUDIT, AUDIT_SHA)
    audited = assets['model_checkpoints'][PREFIX.name]
    config = json_metadata(PREFIX.with_suffix('.json'))
    current = safetensors_metadata(PREFIX.with_suffix('.safetensors'))
    comparisons = {'file_bytes': 'bytes', 'mtime_ns': 'mtime_ns', 'header_bytes': 'header_bytes',
                   'header_sha256': 'header_with_8byte_prefix_sha256', 'tensor_count': 'tensor_count',
                   'parameter_count': 'parameter_count', 'dtypes': 'dtypes'}
    for key, audited_key in comparisons.items():
        if current[key] != audited[audited_key]:
            raise ValueError('Audited texture checkpoint changed: ' + key)
    if (config != audited['config'] or digest(PREFIX.with_suffix('.json')) != CONFIG_SHA
            or current['tensor_data_bytes'] != 2584605760):
        raise ValueError('Audited texture configuration/payload size changed')
    contract.validate_header(config, current['tensors'])
    verify_sources()
    sampler = verify_sampler()
    if input_verifier is None:
        feature, normalized, coords = verify_inputs()
        input_rows,provenance_sha=assets['saved_inputs'],SHAPE_PROVENANCE_SHA
    else:
        inputs=input_verifier()
        feature,normalized,coords=inputs['feature'],inputs['normalized'],inputs['coords']
        input_rows,provenance_sha=inputs['saved_inputs'],inputs['shape_provenance_sha256']
    checkpoint_sha256 = digest(PREFIX.with_suffix('.safetensors')) if hash_weights else None
    if hash_weights and checkpoint_sha256 != CHECKPOINT_SHA:
        raise ValueError('Complete texture checkpoint hash changed')
    return dict(current, config=config, config_sha256=CONFIG_SHA, checkpoint_sha256=checkpoint_sha256,
                token_count=len(coords), feature=feature, pipeline_sha256=PIPELINE_SHA,
                shape_condition_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                saved_inputs=input_rows, shape_provenance_sha256=provenance_sha,
                shape_normalization=sampler['shape_normalization'], normalization=sampler['normalization'])


def import_sampler():
    verify_sampler()
    sys.path.insert(0, str(SOURCE))
    cls = importlib.import_module('trellis2.pipelines.samplers.flow_euler').FlowEulerGuidanceIntervalSampler
    if Path(inspect.getfile(cls)).resolve() != (SOURCE / 'trellis2/pipelines/samplers/flow_euler.py').resolve():
        raise ValueError('Texture sampler class provenance differs')
    return cls


def validate_rope(model):
    import torch
    reference = rope_frequencies()
    if len(model.blocks) != 30:
        raise ValueError('Texture RoPE block count differs')
    for block in model.blocks:
        value = block.self_attn.rope.freqs
        if (list(value.shape) != [21] or value.dtype != torch.float32 or str(value.device) != 'cpu'
                or value.is_meta or not torch.equal(value, reference)):
            raise ValueError('Texture ordinary RoPE frequency attribute differs from genuine CPU constructor')


def factory(config, *, storage=False):
    import torch
    contract.validate_header(config, {key: {**row, 'dtype': 'BF16'} for key, row in contract.schema().items()})
    with torch.device('meta'):
        model = import_class()(**config['args'])
    contract.validate_manifest(manifest(dict(model.named_parameters())), manifest(dict(model.named_buffers())), 'meta')
    for block in model.blocks:
        block.self_attn.rope.freqs = rope_frequencies()
    validate_rope(model)
    if storage:
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    return model


def load_cpu(*,identity_verifier=None):
    from strict_loader import StrictLocalComponent
    (identity if identity_verifier is None else identity_verifier)(hash_weights=False)
    component = StrictLocalComponent(str(PREFIX.with_suffix('.json')), str(PREFIX.with_suffix('.safetensors')),
                                     lambda config: factory(config, storage=True))
    try:
        model = component.load_cpu()
        widened = widen(model, component.diagnostics['payload_sha256'], contract.schema())
        if len(widened) != 220:
            raise ValueError('Texture source-dtype widening requires exactly220 verified BF16→F32 parameters')
        contract.validate_manifest(manifest(dict(model.named_parameters())), manifest(dict(model.named_buffers())), 'cpu')
        validate_rope(model)
        component.diagnostics.update(runtime_fp32_widened=widened, storage_schema='640 BF16 parameters',
                                     ordinary_frequency_policy='30 genuine deterministic CPU F32 [21] attributes; shared sparse cache generates phases during forward')
        return component, model
    except BaseException:
        component.unload()
        raise
