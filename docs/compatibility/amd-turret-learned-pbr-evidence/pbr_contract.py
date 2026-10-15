"""Non-colour F32 channel contract. No rendering or model imports."""
import hashlib,json
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-turret-learned-colour-evidence'
INPUT=ROOT.parent/'artifacts/amd-turret-learned-colour-fa3a98c/learned-colour.blend'
INPUT_SHA='b5c91db6b4c9717da987ab7c6d65ba495f78b0fcb87cd10e0ad3944dc1e17632'
FIELD=Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-owner-067f668/field/texture-field.safetensors')
FIELD_SHA='7b24c946765d8e5de9718dbc358ce35141a56d836aa54fe7f8945cb1acbf7a80'
OUT=ROOT.parent/'artifacts/amd-turret-learned-pbr-3485263'
BLEND=OUT/'learned-pbr.blend';PNG=OUT/'preview-side.png'
N=1842968;F=4012360
CHANNELS={'Trellis_metallic_raw':3,'Trellis_roughness_raw':4,'Trellis_alpha_raw':5}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def array_digest(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def raw_channels(features):
 if features.dtype!=np.float32 or features.ndim!=2 or features.shape[1]!=6 or not np.isfinite(features).all():raise ValueError('Finite F32[N,6] material features required')
 return {name:np.ascontiguousarray(features[:,index]) for name,index in CHANNELS.items()}
def save(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load_features():
 if digest(FIELD)!=FIELD_SHA:raise ValueError('Genuine material identity mismatch')
 with FIELD.open('rb') as f:
  n=int.from_bytes(f.read(8),'little')
  if not 0<n<=65536:raise ValueError('Bounded safetensors header required')
  h=json.loads(f.read(n))
 row=h['features'];a,b=row['data_offsets']
 if row['dtype']!='F32' or row['shape']!=[N,6] or not 0<=a<b<=FIELD.stat().st_size-8-n or b-a!=N*6*4:raise ValueError('Genuine field schema mismatch')
 coords=h['coordinates']
 if coords['dtype']!='I32' or coords['shape']!=[N,4]:raise ValueError('Original coordinate layout changed')
 c,d=coords['data_offsets'];actual=np.memmap(FIELD,mode='r',dtype='<i4',offset=8+n+c,shape=(N,4))
 if d-c!=N*4*4 or array_digest(actual)!='c87fa21d6f863aced647b48e0bb2edebe32523d36c742e350d39ccfea3fc0934':raise ValueError('Original row correspondence changed')
 return np.memmap(FIELD,mode='r',dtype='<f4',offset=8+n+a,shape=(N,6))
