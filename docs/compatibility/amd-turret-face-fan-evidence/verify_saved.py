"""Read-only verification of saved bounded arrays, exact scripts and sources."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import core

def main():
 hosts={p:json.loads((c.E/(p+'-host.json')).read_text()) for p in ('synthetic','real')}
 for host in hosts.values():
  assert host['completed'] and host['returncode']==0 and host['job_cleanup']['owned_tree_exit_verified']
  assert not host['job_cleanup']['forced_job_cleanup']
 for name,expected in hosts['real']['scripts'].items():assert c.digest(c.E/name)==expected
 # Preserve exact phase1 code; phase2 added only Blender-ingest assertions/flag.
 raw=(c.E/'worker.py').read_bytes()
 start=raw.index(b' # Verify the actual Blender ingestion boundary')
 end=raw.index(b' mod=obj.modifiers.new',start)
 old=raw[:start]+raw[end:]
 marker=b"  'actual_Blender_ingest_XYZ_triangles_RGBA_sheet_labels_byte_exact':True,"
 lines=old.splitlines(keepends=True);assert sum(line.startswith(marker) for line in lines)==1
 old=b''.join(line for line in lines if not line.startswith(marker))
 assert hashlib.sha256(old).hexdigest()==hosts['synthetic']['scripts']['worker.py']
 snapshot=c.E/'worker-synthetic-executed.py.txt'
 with snapshot.open('xb') as f:f.write(old)
 for name,expected in hosts['synthetic']['scripts'].items():
  if name!='worker.py':assert c.digest(c.E/name)==expected
 cases=[]
 for name in ('synthetic','forward-patch'):
  result=json.loads((c.E/(name+'.json')).read_text())
  loaded={}
  for label,meta in result['artifacts'].items():
   p=Path(meta['path']);assert p.parent==c.OUT and p.stat().st_size==meta['bytes'] and c.digest(p)==meta['sha256']
   with np.load(p,allow_pickle=False) as data:loaded[label]={k:data[k] for k in data.files}
   part=loaded[label];stats,_=core.stats(part['xyz'],part['faces'])
   assert stats['zero_area_triangles']==0 and np.isfinite(part['rgba']).all()
  source,prepared=loaded['source'],loaded['separated']
  origin=prepared['original_vertex_row'];pf=prepared['faces'];sf=source['faces']
  assert np.array_equal(origin[pf],sf)
  assert prepared['xyz'][pf].tobytes()==source['xyz'][sf].tobytes()
  assert prepared['rgba'][pf].tobytes()==source['rgba'][sf].tobytes()
  assert np.array_equal(prepared['face_sheet_ids'],source['face_sheet_ids'])
  cases.append({'name':name,'decision':result['decision'],'all_saved_artifacts_hash_verified':True,
    'pre_decimation_ordered_triangle_and_corner_colour_readback_exact':True})
 real=json.loads((c.E/'real.json').read_text());assert real['decision']=='NO-GO' and len(real['cases'])==1
 assert real['full_mesh_separations']==real['full_mesh_decimations']==real['GLB_exports']==0
 result={'decision':'NO-GO','cases':cases,'saved_synthetic_worker_sha256':c.digest(snapshot),
  'worker_phase_change':'Only added exact actual-Blender XYZ/triangles/RGBA/face-sheet input assertions and diagnostic flag before real phase; no synthetic rerun',
  'boundary_field_clarification':'227 bad-edge descendant boundaries include7 pre-existing crop boundaries; net new boundary edges294-74=220',
  'metric_clarification':'Complete-source geometry/RGB metrics are null after6missinglabels; matched-only RGB errors pass limits. NO-GO follows fragmentation, missing sheets, retained/reverse geometry and silhouettes.',
  'sources_unchanged':c.preserve(),'full_master_topology_operations':0}
 c.save(c.E/'verification.json',result);c.save(c.OUT/'verification.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
