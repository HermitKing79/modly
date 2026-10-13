"""Static AST/reference and saved execution comparison. Never execute model code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore',category=SyntaxWarning)
root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2')
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Static comparison prohibits model/GPU imports')
sys.addaudithook(audit)
refs={r['path']:r for file in ('canonical-sources.json','canonical-support.json') for r in json.loads((evidence/file).read_text())['sources']}
parsed={k:(ast.parse(v['content']),ast.parse((source/k).read_text())) for k,v in refs.items()}
class NoDocs(ast.NodeTransformer):
    def generic_visit(self,node):
        node=super().generic_visit(node)
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
        return node
def normalize(node):return ast.dump(NoDocs().visit(copy.deepcopy(node)),include_attributes=False)
def select(tree,cls,name):
    scope=tree if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in scope.body if isinstance(n,ast.FunctionDef) and n.name==name)
checks=[]
for path,cls,name in [('pipelines/trellis2_image_to_3d.py','Trellis2ImageTo3DPipeline','preprocess_image'),
    ('modules/image_feature_extractor.py','DinoV3FeatureExtractor','__call__'),
    ('models/sparse_structure_vae.py','SparseStructureDecoder','forward'),
    ('models/sparse_structure_vae.py','ResBlock3d','forward'),('models/sparse_structure_vae.py','UpsampleBlock3d','forward'),
    ('modules/spatial.py',None,'pixel_shuffle_3d'),
    ('pipelines/samplers/flow_euler.py','FlowEulerSampler','_inference_model'),
    ('pipelines/samplers/flow_euler.py','FlowEulerSampler','_get_model_prediction'),
    ('pipelines/samplers/flow_euler.py','FlowEulerSampler','_v_to_xstart_eps'),
    ('pipelines/samplers/flow_euler.py','FlowEulerSampler','sample_once'),
    ('pipelines/samplers/classifier_free_guidance_mixin.py','ClassifierFreeGuidanceSamplerMixin','_inference_model'),
    ('pipelines/samplers/guidance_interval_mixin.py','GuidanceIntervalSamplerMixin','_inference_model')]:
    a,b=parsed[path];equal=normalize(select(a,cls,name))==normalize(select(b,cls,name))
    assert equal,(path,cls,name)
    checks.append(dict(path=path,class_name=cls,method=name,docstring_removed_ast_equal=equal))
# Only verbosity differs in the Euler loop. Do not normalize any numerical operations.
a,b=parsed['pipelines/samplers/flow_euler.py'];a=copy.deepcopy(select(a,'FlowEulerSampler','sample'));b=select(b,'FlowEulerSampler','sample')
for node in ast.walk(a):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tqdm':node.keywords=[k for k in node.keywords if k.arg!='disable']
assert normalize(a)==normalize(b)
checks.append(dict(path='pipelines/samplers/flow_euler.py',method='FlowEulerSampler.sample',equal_after_only_tqdm_disable_keyword_removed=True))
# Dense SDPA active branch is exactly the canonical branch; backend differences are outside this comparison.
a,b=parsed['modules/attention/full_attn.py']
def sdpa_branch(tree):
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scaled_dot_product_attention'][-1]
    return next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=="config.BACKEND == 'sdpa'")
assert normalize(ast.Module(body=sdpa_branch(a).body,type_ignores=[]))==normalize(ast.Module(body=sdpa_branch(b).body,type_ignores=[]))
for name in ('modules/attention/modules.py','modules/attention/rope.py','modules/spatial.py'):
    assert normalize(parsed[name][0])==normalize(parsed[name][1])
# Pinned source adds conditioning modes; for its default cross mode the effective forward matches.
a,b=parsed['models/sparse_structure_flow.py'];a=select(a,'SparseStructureFlowModel','forward');b=copy.deepcopy(select(b,'SparseStructureFlowModel','forward'))
for i,node in enumerate(b.body):
    if isinstance(node,ast.If) and ast.unparse(node.test)=="hasattr(self, 'image_attn_mode')":
        b.body[i]=ast.parse('cond = manual_cast(cond, self.dtype)').body[0]
assert normalize(a)==normalize(b)
# Decoder normalization default bypasses its optional chunking wrapper, and keeps the same arithmetic.
for cls in ('LayerNorm32','GroupNorm32'):
    a,b=parsed['modules/norm.py'];a=select(a,cls,'forward');b=copy.deepcopy(select(b,cls,'_forward'));b.name='forward'
    assert normalize(a)==normalize(b)
sampling=json.loads((root/'docs/compatibility/amd-sparse-structure-sampling-evidence/run/results.json').read_text())['worker']
decoder=json.loads((root/'docs/compatibility/amd-sparse-occupancy-evidence/run/results.json').read_text())['worker']
dino=json.loads((root/'docs/compatibility/amd-dinov3-allocation-evidence/run/results.json').read_text())['worker']
canonical=json.loads((evidence/'official-pipeline.json').read_text())
local_pipeline=root.parent # Local config is the audited genuine model directory, not a downloaded replacement.
local_path=Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/pipeline.json')
assert hashlib.sha256(local_path.read_bytes()).hexdigest()==canonical['sha256']=='222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6'
params=canonical['configuration']['args']['sparse_structure_sampler']['params']
assert sampling['sampler']['parameters']==dict(params,dino_lock=0.0)
assert sampling['sampler']['sigma_min']==canonical['configuration']['args']['sparse_structure_sampler']['args']['sigma_min']
assert sampling['forward_calls']==22 and sampling['sampling']['positive_calls']==12 and sampling['sampling']['negative_calls']==10
metadata=json.loads((evidence/'official-model-metadata.json').read_text())
expected_weights={row['model']:{item['rfilename']:item['lfs']['sha256'] for item in row['siblings'] if 'lfs' in item} for row in metadata}
assert sampling['latent_checkpoint']['provenance']['model_checkpoint_sha256']==expected_weights['microsoft/TRELLIS.2-4B']['ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors']
assert decoder['checkpoint_identity']['checkpoint_sha256']==expected_weights['microsoft/TRELLIS-image-large']['ckpts/ss_dec_conv3d_16l8_fp16.safetensors']
assert dino['feature_checkpoint']['provenance']['checkpoint_sha256']==expected_weights['facebook/dinov3-vitl16-pretrain-lvd1689m']['model.safetensors']
assert dino['input']['resolution']==512 and dino['output']['shape']==[1,1029,1024]
assert all(dino['calls']['layer.'+str(i)]==1 for i in range(24))
assert sampling['cpu_load']['reversible_widening_count']==220
assert all(sampling['cpu_load']['tensors']['blocks.'+str(i)+'.modulation']['dtype']=='F32' for i in range(30))
row=dict(classification='UNKNOWN',canonical_commit='75fbf0183001ed9876c8dbb35de6b68552ee08bd',canonical_model_revision=metadata[0]['revision'],
    official_weight_lfs_hashes_match_saved_execution=True,local_pipeline_exact_official_bytes=True,
    exact_ast_checks=checks,dense_sdpa_active_branch_equal=True,attention_module_and_rope_ast_equal=True,
    cross_mode_flow_forward_equal_after_explicit_inactive_mode_selection=True,normalization_active_arithmetic_equal=True,
    conditioning=dict(resolution=512,feature_shape=[1,1029,1024],class_and_register_tokens=5,patch_tokens=1024,
        all24_layers_executed_once=True,local_default2048_overridden_to512=True,preprocessing_ast_equal=True,
        huggingface_layer_path_compatibility='model.layer or model.model.layer; operator loop unchanged',neg_cond='zeros_like(cond)'),
    sampling=dict(parameters=sampling['sampler']['parameters'],sigma_min=sampling['sampler']['sigma_min'],
        noise='one standard Gaussian CPU float32 draw; canonical also draws CPU noise then transfers; same seed policy',
        timestep='1000*t; exact Euler rescaling;12 steps/22 calls',
        dino_lock='0.0: local mixin falls through to genuine Euler path',
        latent='sampler.samples copied without transpose, scale, clipping or shape-SLat normalization'),
    numerical_difference_review=dict(modulation='canonical adds F32 modulation to BF16 timestep then rounds; local casts modulation first. Audited checkpoint modulation is BF16 widened reversibly to F32, so the early cast preserves every operand value. Not a demonstrated defect for these weights.',
        fp8='local extra branches inactive; no FP8 model or parameter used',
        sdpa='same source contract; GPU numerical equivalence is not established by static inspection'),
    missing_observation='No object-conditioned sparse result or matched reference backend output exists for this fixture.',
    torch_imported='torch' in sys.modules,gpu_initialized=False,weights_loaded=False,canonical_source_executed=False,
    source_hashes={name:dict(url=r['url'],canonical_sha256=hashlib.sha256(r['content'].encode()).hexdigest(),local_sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()) for name,r in refs.items()})
with (evidence/'contract-comparison.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps({k:row[k] for k in ('classification','official_weight_lfs_hashes_match_saved_execution','local_pipeline_exact_official_bytes','dense_sdpa_active_branch_equal','attention_module_and_rope_ast_equal','cross_mode_flow_forward_equal_after_explicit_inactive_mode_selection','torch_imported')}))
