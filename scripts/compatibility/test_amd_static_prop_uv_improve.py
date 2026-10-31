import sys,unittest,json,tempfile
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_uv_improve as p
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(p.owner.CPU_SITE));import numpy as np;cls.np=np
 def test_historical_hold_and_limits(self):
  self.assertTrue(p.old.LIVE_EXECUTION_HOLD);self.assertNotEqual(p.BASE,p.HISTORICAL)
  self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180)
  self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3))
  for s in ('render','bake','wrap','neural','wsl','retry'):self.assertNotIn(s,p.STAGES)
 def test_closed_tetra_uv_cuts_are_disks(self):
  np=self.np;f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
  uv=np.array([[[0.,0],[1,0],[0,1]]])+np.arange(4)[:,None,None]*2
  r=p.topology(np,uv,f)
  self.assertEqual(len(r['charts']),4);self.assertTrue(all(c['disk'] for c in r['charts']))
  self.assertEqual(r['coordinate_anomalies'],[])
  seams={tuple(sorted((int(face[k]),int(face[(k+1)%3])))) for face in f for k in range(3)}
  plan=p.topology(np,None,f,seams)
  self.assertEqual([c['euler'] for c in plan['charts']],[1]*4)
 def fixture_measure(self,distort=False,permute=False,invalid=None,audit=False):
  from unittest.mock import patch
  np=self.np;v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]],'f4');f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],'u4');uv=[]
  for i,t in enumerate(v.astype('f8')[f]):
   a,b=t[1]-t[0],t[2]-t[0];l=np.linalg.norm(a);x=b@a/l;h=np.linalg.norm(np.cross(a,b))/l
   coords=np.array([[0,0],[l,0],[x,h]])*.05
   if distort:coords*=np.array([2,.5])
   uv.append(coords+np.array([.05+.5*(i%2),.05+.5*(i//2)]))
  uv=np.array(uv,'f4')
  if invalid=='nan':uv[:]=np.nan
  if invalid=='zero':uv[:]=0
  with tempfile.TemporaryDirectory(dir=p.BASE) as folder:
   root=Path(folder);(root/'inputs').mkdir();np.save(root/'inputs/final-f.npy',f)
   seams={tuple(sorted((int(face[k]),int(face[(k+1)%3])))) for face in f for k in range(3)};top=p.topology(np,None,f,seams)
   (root/'seam-plan.json').write_text(json.dumps(dict(charts=top['charts'],seam_edges=list(seams))))
   (root/'strategy.json').write_text(json.dumps(dict(maximum_triangle_directional_stretch_ratio=1.1)))
   if permute:f=f[::-1,[1,2,0]];uv=uv[::-1,[1,2,0]]
   with patch.object(p,'BASE',root):
    d=p.measure(np,uv,v,f)
    if not audit:return d
    (root/'asset').mkdir();(root/'uv-run').mkdir();np.save(root/'inputs/final-v.npy',v);np.save(root/'asset/uv.npy',uv)
    d['uv_file_sha256']=p.owner.sha(root/'asset/uv.npy');(root/'asset/uv-diagnostics.json').write_text(json.dumps(d,allow_nan=False))
    (root/'uv-run/host-result.json').write_text(json.dumps(dict(classification='FAIL',worker_returncode=1,cleanup=dict(owned_tree_exit_verified=True,forced_job_cleanup=False))))
    message='ValueError: UV qualification refused: '+json.dumps({k:d[k] for k in ('classification','failures','density_ratio')})
    (root/'uv-run/stderr.txt').write_text(message);(root/'uv-run/stdout.txt').write_text('')
    p.uv_audit();return json.loads((root/'uv-audit.json').read_text())
 def test_intrinsic_isometric_uv_and_permuted_source_mapping(self):
  a=self.fixture_measure();b=self.fixture_measure(permute=True)
  self.assertEqual(a['classification'],'PASS_UV');self.assertEqual(a['classification'],b['classification'])
  self.assertAlmostEqual(a['density_ratio'],1,places=5);self.assertLess(a['directional_stretch_maximum'],1.00001)
 def test_equal_area_stretch_does_not_pass(self):
  d=self.fixture_measure(distort=True)
  self.assertAlmostEqual(d['density_ratio'],1,places=5)
  self.assertIn('DIRECTIONAL_STRETCH',d['failures']);self.assertEqual(d['classification'],'NO_GO_UV')
 def test_invalid_uv_diagnostics_and_audit_are_preserved(self):
  for kind in ('nan','zero'):
   d=self.fixture_measure(invalid=kind);self.assertEqual(d['classification'],'NO_GO_UV');json.dumps(d,allow_nan=False)
   a=self.fixture_measure(invalid=kind,audit=True);self.assertEqual(a['classification'],'NO_GO_UV');self.assertIsNone(a['independent_float64_density_ratio'])
 def test_export_refuses_uv_nogo_and_no_render_route(self):
  from unittest.mock import patch
  with patch.object(p.old,'success'),patch.object(Path,'read_text',return_value='{"classification":"NO_GO_UV"}'):
   with self.assertRaises(ValueError):p.require_uv_pass()
  code=Path(__file__).with_name('amd_static_prop_uv_improve_blender.py').read_text()
  self.assertEqual(code.count('bpy.ops.uv.unwrap('),1);self.assertNotIn('bpy.ops.render',code)
 def test_numeric_agreement_never_waives_verdict_or_identity(self):
  self.assertTrue(p.same_measurements({'metric':[1.0],'classification':'PASS'},{'metric':[1+1e-12],'classification':'PASS'}))
  self.assertFalse(p.same_measurements({'classification':'PASS'},{'classification':'NO_GO'}))
  self.assertFalse(p.same_measurements({'sha':'original'},{'sha':'changed'}))
if __name__=='__main__':unittest.main()
