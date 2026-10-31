import sys,unittest,json,math
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_recovery as r
class RecoveryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(r.owner.CPU_SITE));import numpy as np;cls.np=np
 def fixture(self):
  np=self.np;v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]],'f4');f=np.array([[0,1,2],[1,3,2]],'u4');return v,f,v[f][:,:,:2].copy()
 def test_all_diagnostics_survive_density_failure(self):
  v,f,uv=self.fixture();uv[1]=[[.8,.8],[1,.8],[.8,1]]
  d=r.diagnostics(self.np,uv,v,f)
  self.assertEqual(d['classification'],'NO_GO_UV');self.assertEqual(d['failures'],['AREA_DENSITY'])
  self.assertAlmostEqual(d['density_ratio'],25,places=4);self.assertEqual(d['charts'],2)
  self.assertEqual(d['seam_edges'],1);self.assertEqual(d['continuous_overlap_pairs'],[])
  self.assertEqual(len(d['triangles']),2);self.assertTrue(d['boundary_segments'])
  self.assertIn('density_units',d)
 def test_nonfinite_overlap_and_zero_are_not_density_waivers(self):
  np=self.np;v,f,uv=self.fixture()
  for bad in [np.full_like(uv,np.nan),np.zeros_like(uv),np.stack([uv[0],uv[0]])]:
   d=r.diagnostics(np,bad,v,f);self.assertEqual(d['classification'],'NO_GO_UV');json.dumps(d,allow_nan=False)
   self.assertNotEqual(d['failures'],['AREA_DENSITY'])
 def test_valid_layout_and_truthful_summary(self):
  v,f,uv=self.fixture();d=r.diagnostics(self.np,uv,v,f)
  self.assertEqual(d['classification'],'PASS_UV');self.assertAlmostEqual(d['density_ratio'],1)
  text=r.uv_summary(d);self.assertIn('1',text);self.assertNotIn('not preserved',text)
 def test_skipped_overlap_is_never_atlas_coverage(self):
  np=self.np;v,f,uv=self.fixture();f=np.stack([f[0],f[0],f[0]]);uv=np.stack([uv[0]*0,uv[0],uv[0]])
  d=r.diagnostics(np,uv,v,f)
  self.assertFalse(d['overlap_test_complete']);self.assertIsNone(d['coverage_fraction']);self.assertAlmostEqual(d['sum_uv_triangle_area'],1)
  uv[0]=uv[1]+2;d=r.diagnostics(np,uv,v,f);self.assertIsNone(d['coverage_fraction'])
 def test_historical_hold_and_fresh_routes(self):
  import amd_static_prop as old
  self.assertTrue(old.LIVE_EXECUTION_HOLD)
  self.assertNotEqual(r.BASE,r.HISTORICAL)
  self.assertEqual(r.owner.CAP,6*1024**3);self.assertEqual(r.owner.DEADLINE,180)
  for s in ('bake','wrap','neural','wsl','retry'):self.assertNotIn(s,r.STAGES)
  self.assertEqual(r.main([]),0)
 def test_current_failure_requires_saved_diagnostics_and_exact_reason(self):
  from unittest.mock import patch
  host=dict(classification='FAIL',worker_returncode=1,cleanup=dict(forced_job_cleanup=False,owned_tree_exit_verified=True))
  d=dict(classification='NO_GO_UV',failures=['AREA_DENSITY'],density_ratio=25.,uv_file_sha256='pinned')
  line='ValueError: UV qualification refused: '+json.dumps({k:d[k] for k in ('classification','failures','density_ratio')})
  def call(h,err,diag):
   with patch.object(Path,'read_text',side_effect=[json.dumps(h),err,json.dumps(diag)]),patch.object(Path,'exists',return_value=False),patch.object(r.owner,'sha',return_value='pinned'):return r.check_export_failure()
  self.assertEqual(call(host,line+'\n\nBlender trailer\n',d),d)
  for err in ['ValueError: Unequal UV area density','ValueError: CPU flat unmodified render required',line.replace('25.0','26.0')]:
   with self.assertRaises(ValueError):call(host,err,d)
  with self.assertRaises(ValueError):call(dict(host,error='RAM floor'),line,d)
 def test_equal_settings_and_single_GO_owner(self):
  renderer=Path(__file__).with_name('amd_static_prop_blender.py').read_text()
  wrapper=Path(__file__).with_name('amd_static_prop_recovery_blender.py').read_text()
  self.assertIn('angle_limit=math.radians(66),island_margin=.01,area_weight=0.',renderer)
  self.assertIn('data.shade_flat()',renderer);self.assertNotIn('stdin.buffer.read',wrapper)
  self.assertEqual(renderer.count("sys.stdin.buffer.read(4)"),1)
if __name__=='__main__':unittest.main()
