"""Reuse existing passive inventory policy; all writes are new evidence files."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent;phase=sys.argv[1]
assert phase in ('before','after')
sys.path.insert(0,str(root/'scripts/compatibility'))
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):raise RuntimeError('Passive inventory prohibits AI imports')
sys.addaudithook(audit)
import amd_bounded_gpu_proof as old
args=SimpleNamespace(amd_venv=Path('D:/AI/Trellis2AMD/venv'),modly_venv=Path('C:/Users/wattz/Documents/Modly/extensions/trellis2/venv'),amd_source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD'),extension_source=root.parent/'gguf-source')
inventory=old.snapshot(args,evidence/('venv-metadata-'+phase+'.json.gz'))
roots=json.loads(gzip.decompress((root/'docs/compatibility/amd-sparse-slab-diagnosis-evidence/model-cache-metadata-after.json.gz').read_bytes()))
raw={};inventory['model_cache_metadata']={}
for name in roots:
    base=Path(name);rows=[]
    for directory,dirs,files in os.walk(base,followlinks=False):
        for child in dirs+files:
            p=Path(directory)/child;s=p.lstat();rows.append([p.relative_to(base).as_posix(),s.st_size,s.st_mtime_ns,s.st_mode])
    raw[name]=sorted(rows);encoded=json.dumps(old.normalize_metadata_rows(raw[name]),separators=(',',':')).encode()
    inventory['model_cache_metadata'][name]=dict(entries=len(rows),normalized_sha256=hashlib.sha256(encoded).hexdigest())
with (evidence/('model-cache-metadata-'+phase+'.json.gz')).open('xb') as f:f.write(gzip.compress(json.dumps(raw).encode(),mtime=0))
with (evidence/('inventory-'+phase+'.json')).open('x',encoding='utf-8') as f:json.dump(inventory,f,indent=2)
baseline=(root/'docs/compatibility/amd-sparse-structure-sampling-evidence/resume-inventory-after.json') if phase=='before' else evidence/'inventory-before.json'
assert inventory==json.loads(baseline.read_text()),'Installed environment/source metadata changed'
print(json.dumps(dict(phase=phase,inventories_equal=True,torch_imported='torch' in sys.modules)))
