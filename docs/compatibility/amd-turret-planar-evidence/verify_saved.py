"""Independent archive/hash readback only; no Blender or modifiers."""
import json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
core=c.module('planar_readback_stats',c.PRIOR/'core.py')
host=json.loads((c.E/'host.json').read_text())
assert host['completed'] and host['returncode']==0 and not host['job_cleanup']['forced_job_cleanup']
assert host['job_cleanup']['owned_tree_exit_verified'] and host['job_cleanup']['active_after_cleanup']==0
for name,expected in host['scripts'].items():
 path=c.PRIOR/name.split('/',1)[1] if name.startswith('prior/') else c.E/name
 assert c.digest(path)==expected
result=json.loads((c.E/'results.json').read_text());records=[]
for case in result['cases']:
 report=json.loads((c.E/(case['name']+'.json')).read_text())
 metas=[report['source']]+[a['artifact'] for a in report['angles']]
 arrays=[]
 for meta in metas:
  path=Path(meta['path']);assert path.parent==c.OUT and path.stat().st_size==meta['bytes'] and c.digest(path)==meta['sha256']
  with np.load(path,allow_pickle=False) as data:
   part={k:data[k] for k in data.files}
  stats,_=core.stats(part['xyz'],part['faces'])
  assert np.isfinite(part['rgba']).all() and stats['zero_area_triangles']==0
  arrays.append(part)
 if case['name']=='saved-forward':
  with np.load(c.PATCH['artifacts']['source']['path'],allow_pickle=False) as prior:
   for key in ('xyz','faces','rgba'):assert prior[key].tobytes()==arrays[0][key].tobytes()
 before=arrays[0];remaining=[]
 for actual,item in zip(arrays[1:],report['angles']):
  stats,_=core.stats(actual['xyz'],actual['faces'])
  assert stats==item['after'] and len(actual['faces'])==item['effective_triangle_count']
  original={point.tobytes():i for i,point in enumerate(before['xyz'])}
  assert len(original)==len(before['xyz'])
  ids=[original.get(point.tobytes()) for point in actual['xyz']]
  exact=all(i is not None for i in ids)
  remaining.append({'angle':item['angle_degrees'],'triangles':len(actual['faces']),
   'surviving_vertex_positions_are_original_rows':exact,
   'surviving_vertex_RGBA_byte_exact':exact and actual['rgba'].tobytes()==before['rgba'][ids].tobytes()})
 records.append({'case':case['name'],'all_artifacts_hash_readback_verified':True,'surviving_values':remaining})
receipt={'decision':result['decision'],'saved_cases':records,'preserved_sources':c.preserve(),
 'exact_executed_script_hashes_verified':True,'natural_empty_Job':True,'no_additional_Blender_execution':True}
c.save(c.E/'verification.json',receipt);c.save(c.OUT/'verification.json',receipt)
print(json.dumps({'decision':receipt['decision'],'cases':records,'sources':len(receipt['preserved_sources'])}))
