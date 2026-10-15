"""Reuse pinned preservation and bounded patch measurement code."""
import importlib.util
import json
from pathlib import Path
E=Path(__file__).resolve().parent
ROOT=E.parents[2]
PRIOR=E.parent/'amd-turret-face-fan-evidence'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
old=module('planar_preservation',PRIOR/'common.py')
OUT=ROOT.parent/'artifacts/amd-turret-planar-e32915d'
digest=old.digest
save=old.save
SOURCE=old.SOURCE
LINEAR=old.LINEAR
PATCH=json.loads((PRIOR/'forward-patch.json').read_text())
def preserve():
 records=old.preserve()
 for metadata in (PATCH['artifacts']['source'],PATCH['patch_extraction']['provenance']):
  path=Path(metadata['path'])
  assert path.stat().st_size==metadata['bytes'] and digest(path)==metadata['sha256']
  records.append(metadata)
 return records
