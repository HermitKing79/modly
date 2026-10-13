"""After-only passive snapshot using established inventory policy; no AI imports."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
sys.path.insert(0,str(root/'scripts/compatibility'))
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','trellis2','triton','cumesh','o_voxel','flex_gemm'):
        raise RuntimeError('Passive inventory prohibits framework imports')
sys.addaudithook(audit)
import amd_bounded_gpu_proof as previous
args=SimpleNamespace(amd_venv=Path('D:/AI/Trellis2AMD/venv'),
    modly_venv=Path('C:/Users/wattz/Documents/Modly/extensions/trellis2/venv'),
    amd_source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD'),
    extension_source=root.parent/'gguf-source')
after=previous.snapshot(args,evidence/'venv-metadata-after.json.gz')
old=json.loads(gzip.decompress((evidence/'model-cache-metadata-before.json.gz').read_bytes()))
raw={};after['model_cache_metadata']={}
for name in old:
    directory_root=Path(name);entries=[]
    for directory,dirs,files in os.walk(directory_root,followlinks=False):
        for child in dirs+files:
            path=Path(directory)/child;info=path.lstat()
            entries.append([path.relative_to(directory_root).as_posix(),info.st_size,info.st_mtime_ns,info.st_mode])
    raw[name]=sorted(entries)
    encoded=json.dumps(previous.normalize_metadata_rows(raw[name]),separators=(',',':')).encode()
    after['model_cache_metadata'][name]=dict(entries=len(entries),normalized_sha256=hashlib.sha256(encoded).hexdigest())
with (evidence/'model-cache-metadata-after.json.gz').open('xb') as f:f.write(gzip.compress(json.dumps(raw).encode(),mtime=0))
with (evidence/'inventory-after.json').open('x',encoding='utf-8') as f:json.dump(after,f,indent=2)
before=json.loads((evidence/'inventory-before.json').read_text())
print(json.dumps(dict(inventories_equal=before==after,torch_imported='torch' in sys.modules)))
assert before==after,'Installed inventory changed; structurally reconcile before claiming preservation'
