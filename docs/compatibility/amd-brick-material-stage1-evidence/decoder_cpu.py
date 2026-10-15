"""Exact brick guided-PBR CPU preparation only. There is no execution entrypoint.

Reuses authenticated geometry contracts, original guide readers and byte planners.
The historical texture runtime and artifact bounds remain unchanged.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILE = 'brick-1957891-v1'
CONTRACT_SHA = '82bf47920f667a3a445d5e70297a0057fae78a073d7ae9c903f0827319a8dfdd'
TEXTURE_SHA = '4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289'
GEOMETRY_SHA = 'd4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861'
COORDS_SHA = 'e350ce386f21a9ee8473fda12a8839bf8c37ae0c1375c34c56cea88f81859520'
TEXTURE = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688/latent/texture-latent.safetensors')
GEOMETRY = ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors'
GEOMETRY_MODULES = 'experimental/extensions/trellis2-amd/inference_decoder/'
HISTORICAL = 'docs/compatibility/amd-turret-texture-decode-resume-evidence/'
PINS = {
    'api/services/amd_model_metadata.py': 'ca7b9f01ea09cb1820208e22f9bb5605bf8cf67159ea0dead9be7036eaf2ab96',
    'api/services/amd_runtime.py': '16499c8c8b5ef3f133ee872bade91a4332f4195d757da662022511c5049bf756',
    'experimental/extensions/trellis2-amd/strict_loader.py': '0a4b2aab6fb34a034be1725d4514d71f17c59376151b549ab631ebc8b5c8fc79',
    'scripts/compatibility/amd_texture_decoder_inputs.py': '87bd6437412df561100b174e945c2fa040a9c02ee1ac2ea8c70a325b8f1a3d85',
    'scripts/compatibility/amd_texture_sampling_artifact.py': '5eaf35c08cf2e96f46dfe8bb32fa835c079ef34ce5a322a0afd411ee7cf9474c',
    'scripts/compatibility/amd_texture_field_artifact.py': '77d9592662653413e9b40f81741e47e2a23739e9e5d013d48b6cefff1831d3b5',
    GEOMETRY_MODULES+'run_contract.py': '0998c6e0c9a8d5c262b6f258f60fb11d82e8b51aec2c5fd48c771c33ef999f57',
    GEOMETRY_MODULES+'admission.py': '6f4831747c6973d860a55c1988af5105f4fadb53b79e50164a86ffe81f73f21a',
    GEOMETRY_MODULES+'artifact_io.py': '10a2792b47ab24ec64cf5d911c37fa12ac789a1d7134c5221d0479ced1bdb989',
    HISTORICAL+'admission.py': '99c1078cbcb4b026feeec3f085976586f1118812bdd3d4b5fbe0e76c47cbbbbe',
    'experimental/extensions/trellis2-amd/texture_decoder_loader.py': '409bf691bb3d7f3638d40cb46f02250111249af31f19a4377345da84ec2fb988',
}


def guard(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {
        'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cumesh'}:
        raise RuntimeError('CPU preparation forbids model/runtime/native AI imports')
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
        raise RuntimeError('CPU preparation forbids network and child execution')


sys.addaudithook(guard)
for relative, expected in PINS.items():
    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
        raise ValueError('CPU preparation source pin differs: '+relative)
sys.path[:0] = [str(ROOT/'scripts/compatibility'), str(ROOT/'api'),
               str(ROOT/'experimental/extensions/trellis2-amd')]
import numpy as np
import amd_texture_decoder_inputs as inputs
import amd_texture_sampling_artifact as latent
import amd_texture_field_artifact as field
import texture_decoder_loader as loader


def module(name, relative):
    path = (ROOT/relative).resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise RuntimeError('Foreign CPU dependency module: '+name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


contracts = module('run_contract', GEOMETRY_MODULES+'run_contract.py')
geometry_artifact = module('_brick_pbr_geometry_artifact', GEOMETRY_MODULES+'artifact_io.py')
geometry_plans = module('_brick_pbr_geometry_plans', GEOMETRY_MODULES+'admission.py')
historical = module('_brick_pbr_historical_plans', HISTORICAL+'admission.py')


def authenticate(profile, document):
    if type(profile) is not str or profile != PROFILE:
        raise ValueError('Exact explicit brick-1957891-v1 CPU preparation opt-in required')
    contract = contracts.parse(document)
    if contract.sha256 != CONTRACT_SHA or contract.final_support_profile != profile:
        raise ValueError('Exact authenticated brick geometry run contract required')
    return contract


def prepare(profile):
    contract = authenticate(profile, inputs._decode((GEOMETRY.parent/'run-contract.json').read_bytes()))
    tm = latent.verify(TEXTURE, TEXTURE.parent, token_count=4430, maximum_tokens=4430)['metadata']
    if (tm['artifact_sha256'], tm['artifact_bytes']) != (TEXTURE_SHA, 1814784):
        raise ValueError('Exact genuine brick texture latent required')
    _, ta, tr, _ = inputs._container(TEXTURE, TEXTURE_SHA, 1814784,
        latent.layouts(4430, maximum_tokens=4430), {k:v['sha256'] for k,v in tm['tensors'].items()})
    gm = inputs._decode((GEOMETRY.parent/'provenance.json').read_bytes())
    layout = {'features': ('F32', (1957891, 7)), 'coordinates': ('I32', (1957891, 4)),
              **{f'subdivision_{i}_coordinates': ('I32', (n, 4)) for i,n in enumerate(contracts.BRICK_PARENT_ROWS)},
              **{f'subdivision_{i}_logits': ('BF16', (n, 8)) for i,n in enumerate(contracts.BRICK_PARENT_ROWS)}}
    _, ga, gr, metadata = inputs._container(GEOMETRY, GEOMETRY_SHA, 104148612,
                                           layout, gm['output']['tensor_payload_sha256'])
    summary = inputs._decode(metadata['summary'])
    if gm['run_contract'] != contract.document() or inputs._decode(metadata['provenance']) != gm['provenance']:
        raise ValueError('Saved geometry sidecar/header/run contract differs')
    geometry_artifact.verify_file(GEOMETRY, gm['provenance'], summary,
                                 gm['output']['tensor_payload_sha256'], contract)
    sources = inputs.verify_source_identity()
    if any(gm['provenance']['source'].get(str(inputs.platform_path(inputs.SOURCE_ROOT, platform='linux')/name)) != sha
           for name,sha in sources.items()):
        raise ValueError('Saved guide source identities differ')
    guides = tuple(dict(coords=ga[f'subdivision_{i}_coordinates'],
                        logits_bf16=ga[f'subdivision_{i}_logits'],
                        logits_f32=inputs.bf16_to_f32(ga[f'subdivision_{i}_logits'])) for i in range(4))
    if tr['coords']['sha256'] != COORDS_SHA or not np.array_equal(ta['coords'], guides[0]['coords']):
        raise ValueError('Texture latent and original geometry guide coordinate order differs')
    contracts.validate_chain([(g['coords'],g['logits_f32']) for g in guides],
                             ga['coordinates'], 4430, contract=contract)
    loader._verify_source(loader.SOURCE)
    return dict(contract=contract, guides=guides, texture_coords=ta['coords'], geometry_coords=ga['coordinates'],
                proof=dict(parent_rows=list(contracts.BRICK_PARENT_ROWS), child_rows=[21391,97313,438684,1957891],
                    initial_coordinates_sha256=COORDS_SHA, texture_sha256=TEXTURE_SHA, geometry_sha256=GEOMETRY_SHA,
                    all_four_masks_and_original_child_orders_verified=True, guide_BF16_bytes_preserved=True,
                    source_identity=sources, geometry_payloads=gr, texture_payloads=tr))


def checkpoint_metadata():
    """Fresh config/header only; no checkpoint tensor payload or constructor."""
    from services.amd_model_metadata import safetensors_metadata
    prefix = Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16')
    if hashlib.sha256(prefix.with_suffix('.json').read_bytes()).hexdigest() != loader.CONFIG_SHA256:
        raise ValueError('Original texture decoder config differs')
    row = safetensors_metadata(prefix.with_suffix('.safetensors'))
    schema = loader.schema()
    if (row['header_sha256'] != loader.HEADER_SHA256 or row['file_bytes'] != 948458812
            or row['bytes_read'] != 26800 or row['tensor_data_bytes'] != loader.CHECKPOINT_PAYLOAD_BYTES
            or row['tensor_count'] != 284 or row['dtypes'] != {'F16':284}
            or set(row['tensors']) != set(schema)
            or any(row['tensors'][k]['shape'] != v['shape'] for k,v in schema.items())):
        raise ValueError('Original texture F16 header schema differs')
    return {k:v for k,v in row.items() if k not in ('tensors','metadata')}


def memory_profile(contract):
    contract = authenticate(PROFILE, contract.document())
    runtime_schema = loader.schema('cpu')
    model_bytes = sum(prod(v['shape'])*{'F16':2,'F32':4}[v['dtype']] for v in runtime_schema.values())
    plans, c2s = [], []
    parents = contracts.BRICK_PARENT_ROWS
    children = (*parents[1:], contracts.BRICK_FINAL_ROWS)
    def annotate(site, plan):
        padded = (plan['visible_bytes']*5+3)//4
        return dict(site=site, **plan, padded_bytes=padded,
                    maximum_retained_bytes=historical.ALLOCATOR_LIMIT-historical.OPAQUE_HEADROOM-padded,
                    minimum_free_bytes=padded+historical.FREE_RESERVE)
    for stage, count in enumerate(historical.COUNTS[:4]):
        n, ci = parents[stage], historical.WIDTHS[stage]
        for block in range(count):
            plans.append(annotate(f'blocks.{stage}.{block}.conv', historical.conv_plan(
                n,ci,ci,(32*2**stage)**3,cached=not(stage==0 and block==0),block_extra=n*32*ci)))
        co = historical.WIDTHS[stage+1]*8
        plans.append(annotate(f'blocks.{stage}.{count}.conv1', historical.conv_plan(
            n,ci,co,(32*2**stage)**3,cached=True,block_extra=n*(16*ci+4*co))))
        ci = historical.WIDTHS[stage+1]
        plan = (geometry_plans.chunked_conv_plan(children[stage],32768,512**3,contract=contract)
                if stage==3 else historical.conv_plan(children[stage],ci,ci,(64*2**stage)**3,
                                                      block_extra=children[stage]*(16*ci+4*ci)))
        plans.append(annotate(f'blocks.{stage}.{count}.conv2', plan))
        plan = (geometry_plans.final_c2s_plan(n,ci*8,historical.WIDTHS[stage],children[stage],ci,contract=contract)
                if stage==3 else historical.c2s_plan(n,ci*8,historical.WIDTHS[stage],children[stage],ci))
        c2s.extend(dict(stage=stage,call=call,**plan) for call in (1,2))
    decisive = dict(next(p for p in plans if p['site']=='blocks.3.4.conv1'))
    retained = model_bytes+2*parents[-1]*128*2
    decisive.update(unavoidable_retained_lower_bound_bytes=retained,
                    minimum_cap_excess_bytes=retained-decisive['maximum_retained_bytes'])
    return dict(runtime_model_parameter_bytes=model_bytes,
        runtime_dtype_counts={dtype:sum(v['dtype']==dtype for v in runtime_schema.values()) for dtype in ('F16','F32')},
        caps=dict(allocator_limit_bytes=historical.ALLOCATOR_LIMIT,initial_free_required_bytes=historical.INITIAL_FREE_REQUIRED,
                  free_reserve_bytes=historical.FREE_RESERVE,opaque_headroom_bytes=historical.OPAQUE_HEADROOM,
                  padding_numerator=5,padding_denominator=4,tile_maximum=historical.TILE_ROWS,gpu_deadline_seconds=240),
        convolution_plans=plans,c2s_plans=c2s,expected_native_map_calls=5,decisive_conv1=decisive,
        final=plans[-1],final_only_tiling_admissible=retained<=decisive['maximum_retained_bytes'],
        lower_bound_source='Pinned SparseResBlockC2S3d._forward keeps x for later self.updown(x); h is a distinct norm1/SiLU result passed to conv1. LayerNorm32 restores input F16. Both F16[438684,128] feature tables coexist at conv1 entry.',
        interpretation='Predicted refusal by the unchanged conservative admission formula; no GPU OOM or physical peak was measured.',
        spatial_volume_assumption='Nominal cubical grid upper bounds, not measured rectangular sparse spatial shapes. Both remain below2**32; the same32bit hash-key byte formula applies.',
        output_payload_bytes=1957891*64)


def source_equivalence():
    def function(relative, name):
        return next(n for n in ast.parse((ROOT/relative).read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name==name)
    matches = {name: ast.dump(function(HISTORICAL+'admission.py',name)) ==
                     ast.dump(function(GEOMETRY_MODULES+'admission.py',name))
               for name in ('conv_plan','_c2s_bytes')}
    if not all(matches.values()):
        raise ValueError('Reused historical two-byte planner source differs')
    return dict(source_pins=PINS, original_two_byte_formula_AST_equal=matches,
                numerical_F16_equivalence_rerun=False,
                note='CPU byte arithmetic reuse only; geometry runtime remains BF16 and cannot be a drop-in F16 PBR runner.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=[PROFILE])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepared = prepare(args.profile)
    report = dict(schema='modly.amd.brick-guided-pbr-cpu-preparation.v1',
        classification='PASS_CPU_INPUTS_AND_PLANNING',execution_readiness='BLOCKED',
        run_contract_sha256=prepared['contract'].sha256,profile=args.profile,
        authenticated_inputs=prepared['proof'],memory=memory_profile(prepared['contract']),
        source_equivalence=source_equivalence(),checkpoint_header_only=checkpoint_metadata(),
        model_loaded=False,GPU_executed=False,WSL_executed=False,
        torch_imported='torch' in sys.modules,synthetic_output_created=False,
        blockers=['Final-only tiling exceeds the unchanged cap at blocks.3.4.conv1 even before coords/guides/caches.',
                  'Historical input/hooks/final patch/output receipt/host acceptance pin turret counts and paths.',
                  'Historical production field writer and independent reader cap rows at1842968.',
                  'Geometry six-site large tiling is BF16; preserve F16 using the existing texture helper and qualify its integration.'],
        next_milestone='One targeted CPU brick guided-decoder integration: central authenticated opt-in across input/guide hooks, F16 six-site tiling, output persistence and independent acceptance; retain all historical defaults and caps. No GPU attempt authorized by this report.')
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(classification=report['classification'],execution_readiness='BLOCKED',
                         final_only_tiling_admissible=False,output=str(args.output),torch_imported=False)))


if __name__ == '__main__':
    main()
