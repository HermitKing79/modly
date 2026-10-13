"""Final passive publication checks; no imports of AI code or model tensors."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Final passive verification prohibits AI/native GPU imports')
sys.addaudithook(audit)
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args,cwd=root):return subprocess.run(['git',*args],cwd=cwd,capture_output=True,check=True,timeout=15).stdout.decode().strip()
before=json.loads((evidence/'git-before.json').read_text())
for name,sha in before['historical_files'].items():assert digest(root/name)==sha,name
for record in ('cpu-tests.json','independent-review.json'):
    for name,sha in json.loads((evidence/record).read_text())['production_sha256'].items():
        assert digest(root/'scripts/compatibility'/name)==sha,name
local=dict(line.split(' ',1) for line in git('for-each-ref','--format=%(refname:short) %(objectname)','refs/heads').splitlines())
remote={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in git('ls-remote','--heads','origin').splitlines()}
continuation='codex/amd-sparse-occupancy-decode'
for key,sha in before['local_refs'].items():
    if key!=continuation:assert local[key]==sha,key
for key,sha in before['remote_refs'].items():
    if key!=continuation:assert remote[key]==sha,key
assert local[continuation]==remote[continuation]==git('rev-parse','HEAD')=='fe02093614605a0621c738b0e7827b287660fd14'
assert local['main']==remote['main']=='f11a4d2bd9fe994b9d30f76a36cb6c18ac8733ea'
assert local['codex/amd-sparse-structure-sampling']==remote['codex/amd-sparse-structure-sampling']=='26fc3aa988c57f082542a80636a63f436218f3fb'
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
assert git('rev-parse','HEAD',cwd=source)==before['amd_source_head']
assert git('status','--short','--untracked-files=all',cwd=source)==before['amd_source_status']
for name,sha in before['amd_modified_files'].items():assert digest(source/name)==sha,name
for prefix,suffix in [('inventory','.json'),('venv-metadata','.json.gz'),('model-cache-metadata','.json.gz')]:
    assert (evidence/(prefix+'-before'+suffix)).read_bytes()==(evidence/(prefix+'-after'+suffix)).read_bytes(),prefix
assert digest(root/'docs/compatibility/amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors')=='8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2'
result=json.loads((evidence/'run/results.json').read_text());analysis=json.loads((evidence/'independent-result-analysis.json').read_text())
assert result['classification']==analysis['classification']=='PASS'
for name,row in analysis['artifacts'].items():assert digest(evidence/name)==row['sha256']
preview=json.loads((evidence/'voxel-preview-independent.json').read_text())
assert digest(evidence/'voxel-preview.obj')==preview['obj_sha256'] and digest(evidence/'voxel-preview.png')==preview['png_sha256']
row=dict(verified=True,classification='PASS',branch=git('branch','--show-current'),execution_harness_commit=git('rev-parse','HEAD'),
    historical_files_verified=len(before['historical_files']),historical_local_refs_verified=len([k for k in before['local_refs'] if k!=continuation]),historical_remote_refs_verified=len([k for k in before['remote_refs'] if k!=continuation]),
    main_preserved=True,sampling_branch_preserved=True,installed_inventories_byte_identical=True,raw_environment_model_metadata_byte_identical=True,
    amd_source_changes_preserved=True,approved_execution_code_unchanged=True,input_latent_unchanged=True,occupancy_and_preview_hashes_verified=True,
    focused_cpu_tests=json.loads((evidence/'cpu-tests.json').read_text())['tests'],torch_imported='torch' in sys.modules,additional_gpu_execution=False)
with (evidence/'final-verification.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
paths=[p for p in sorted(evidence.rglob('*')) if p.is_file() and not p.is_relative_to(evidence/'run/scratch') and p.name!='SHA256SUMS']
paths.append(root/'docs/compatibility/amd-sparse-occupancy-decode.md')
with (evidence/'SHA256SUMS').open('x',encoding='utf-8',newline='\n') as f:
    for path in paths:f.write(digest(path)+'  '+path.relative_to(root).as_posix()+'\n')
print(json.dumps(row))
