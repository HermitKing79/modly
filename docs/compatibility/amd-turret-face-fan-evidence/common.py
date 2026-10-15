"""Reuse preservation identities and owned output paths; no model dependencies."""
import importlib.util
from pathlib import Path
E=Path(__file__).resolve().parent
ROOT=E.parents[2]
spec=importlib.util.spec_from_file_location('prior_lod_common',E.parent/'amd-turret-lod-candidates-evidence/common.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OUT=ROOT.parent/'artifacts/amd-turret-face-fans-ad7a436'
SOURCE=old.SOURCE
LINEAR=ROOT.parent/'artifacts/amd-turret-coloured-glb-357e541/source-linear-RGB.npy'
SOURCES=dict(old.SOURCES)
SOURCES[LINEAR]='ce010634db30e056c1d67d799b9e1ccdf09c9a1e14c5e6e14193a273bb6f29f8'
digest=old.digest
save=old.save
def preserve():
 result=[]
 for path,expected in SOURCES.items():
  actual=digest(path);assert actual==expected,str(path)
  result.append({'path':str(path),'bytes':path.stat().st_size,'sha256':actual})
 return result
