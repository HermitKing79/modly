"""Paths and preservation for the one bounded CPU LOD qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
E=Path(__file__).resolve().parent
ROOT=E.parents[2]
ART=ROOT.parent/'artifacts'
OUT=ART/'amd-turret-lods-1f947fa'
COLOUR=ART/'amd-turret-learned-colour-fa3a98c'
SOURCE=COLOUR/'turret-learned-colour.ply'
BLEND=COLOUR/'learned-colour.blend'
COMPACT=ART/'amd-turret-compact-glb-d3eec48/turret-learned-colour-compact.glb'
spec=importlib.util.spec_from_file_location('preserved_glb_contract',E.parent/'amd-turret-compact-colour-glb-evidence/contract.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
SOURCES=dict(legacy.SOURCES)
SOURCES.update({SOURCE:'d79f6fb38d23e2f0c7bcc492153290d67767466411f71dc19fcc5c6540ceeb72',
 COMPACT:'6f3a71eb1dc683a83a83ba0b1c6e0add87391ef1d7ee4a9e0faf4092eb3f6c2d',
 COLOUR/'preview-front.png':'262aa6f754954c09a9d17fab77f6003f425ac1dec0331b5ee956add9363d7955',
 COLOUR/'preview-side.png':'ff3319685760ef59351b4e6f96295675e589bdef883d15d6a603c79ba8b9325e'})

def digest(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,record):
 with Path(path).open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')

def preserve():
 result=[]
 for path,expected in SOURCES.items():
  actual=digest(path);assert actual==expected,str(path)
  result.append({'path':str(path),'bytes':path.stat().st_size,'sha256':actual})
 return result
