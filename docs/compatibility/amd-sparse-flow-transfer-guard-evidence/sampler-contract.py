"""Read-only source/config audit and rational Euler schedule; no AI imports."""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2')
config_path=Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/pipeline.json')
config=json.loads(config_path.read_text())
audit=json.loads((root/'docs/compatibility/amd-offline-model-audit-evidence/final-assets/audit.json').read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert config==audit['pipelines']['pipeline.json']['config']
assert digest(config_path)==audit['pipelines']['pipeline.json']['sha256']
sampler=config['args']['sparse_structure_sampler']
assert sampler=={'name':'FlowEulerGuidanceIntervalSampler','args':{'sigma_min':1e-5},
    'params':{'steps':12,'guidance_strength':7.5,'guidance_rescale':0.7,'guidance_interval':[.6,1.0],'rescale_t':5.0}}
names=('pipelines/samplers/flow_euler.py','pipelines/samplers/classifier_free_guidance_mixin.py',
       'pipelines/samplers/guidance_interval_mixin.py','pipelines/trellis2_image_to_3d.py')
files={};snippets={}
for name in names:
    text=(source/name).read_text();tree=ast.parse(text)
    files[name]={'sha256':digest(source/name),'path':str(source/name)}
    keep={'FlowEulerGuidanceIntervalSampler','_inference_model','sample','sample_once','get_cond',
          'sample_sparse_structure'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in keep:
            if name.endswith('flow_euler.py') and node.lineno>420:continue
            snippets[name+':'+str(node.lineno)]={'name':node.name,'line':node.lineno,
                'code':ast.get_source_segment(text,node)}
rows=[]
for i in range(12):
    t=Fraction(12-i,12);scaled=5*t/(1+4*t)
    nxt=Fraction(11-i,12);scaled_next=5*nxt/(1+4*nxt)
    calls=2 if Fraction(3,5)<=scaled<=1 else 1
    rows.append({'step':i+1,'t':float(scaled),'t_exact':str(scaled),'model_t':float(1000*scaled),
                 'next_t':float(scaled_next),'forwards':calls,'roles':['positive','zero-negative'] if calls==2 else ['positive']})
assert sum(r['forwards'] for r in rows)==22
record={'config_path':str(config_path),'config_sha256':digest(config_path),'audited_config_match':True,
    'sampler':sampler,'explicit_dino_lock':0.0,'steps':rows,'total_forwards':22,
    'source_files':files,'source_excerpts':snippets,
    'execution':{'torch_imported':False,'sampler_imported':False,'sampling':False,'model_weights_loaded':False}}
with (evidence/'sampler-contract.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_excerpts','source_files')}))
