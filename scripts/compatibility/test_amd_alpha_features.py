import json,sys,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_alpha_features_proof as p
class GuardTests(unittest.TestCase):
 def test_no_reconstruction_route(self):
  for s in ('wrap','simplify','neural','wsl','retry'):
   self.assertNotIn(s,p.STAGES)
   with self.assertRaises(ValueError):p.command(s)
 def test_exact_windows_command(self):
  c=p.command('census')
  self.assertTrue(p.spawn_allowed('census',(c[0],subprocess.list2cmdline(c),str(p.BASE),{})))
  for exe,args,cwd in [('wsl.exe',c,str(p.BASE)),(c[0],c+['extra'],str(p.BASE)),(c[0],c,str(p.PRIOR))]:self.assertFalse(p.spawn_allowed('census',(exe,args,cwd,{})))
 def test_review_stage_and_hash(self):
  row=dict(classification='APPROVED',allowed_stages=['census'],files={'x':'pinned'})
  with patch.object(Path,'read_text',return_value=json.dumps(row)),patch.object(p.owner,'sha',return_value='changed'):
   with self.assertRaises(ValueError):p.reviewed('census')
  with patch.object(Path,'read_text',return_value=json.dumps(row)),patch.object(p.owner,'sha',return_value='pinned'):
   with self.assertRaises(ValueError):p.reviewed('wrap')
 def test_inert_and_limits(self):
  self.assertEqual(p.main([]),0);self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180)
  self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3))
class MeasurementTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(p.owner.CPU_SITE))
  import numpy as np
  import amd_alpha_features_result as result
  cls.np=np;cls.result=result
 def test_area_unsigned_orientation_and_degenerate_rejection(self):
  np=self.np;t=np.array([[[0.,0,0],[2,0,0],[0,1,0]]])
  n,a=self.result.geometry(np,t)
  self.assertAlmostEqual(a[0],.0001);self.assertEqual(n.tolist(),[[0,0,1]])
  n2,a2=self.result.geometry(np,t[:,[0,2,1]])
  self.assertEqual(n2.tolist(),[[0,0,-1]]);self.assertEqual(a.tolist(),a2.tolist())
  with self.assertRaises(ValueError):self.result.geometry(np,np.zeros((1,3,3)))
 def test_render_keeps_full_geometry_occlusion(self):
  np=self.np
  vertices=np.array([[-1.,-1,0],[1,-1,0],[0,1,0],[-1,-1,1],[1,-1,1],[0,1,1]])
  faces=np.array([[0,1,2],[3,4,5]])
  camera=dict(center=[0,0,0],view=[0,0,1],right=[1,0,0],up=[0,1,0],span=3)
  first=self.result.raster(np,vertices,faces,camera,{0},True)
  second=self.result.raster(np,vertices,faces,camera,set(),True)
  self.assertTrue(np.array_equal(first,second))
  budget=dict(triangles=0,pixels=0)
  visible=self.result.raster(np,vertices,faces,camera,{1},True,budget)
  self.assertFalse(np.array_equal(first,visible))
  self.assertTrue(json.dumps(budget))
 def test_outcome_union_tampering_rejected(self):
  import copy
  np=self.np;out=np.array([(0,0,1,1),(1,1,0,4)],dtype=[('face','<u4'),('region','<i4'),('raw','u1'),('final','u1')])
  rows=[dict(face=0,region=0,raw_mask=1,final_mask=1,incidences=[dict(corner=0)]),dict(face=1,region=1,raw_mask=0,final_mask=4,incidences=[dict(corner=2)])]
  patches=[[0]*4 for _ in range(51)];patches[0]=[2,0,0,1];patches[1]=[2,0,1,0]
  summary=dict(tracked_faces=2,source_faces=3,transitions=[4,0,1,1],patch_transitions=patches)
  self.result.validate_index(np,out,rows,summary)
  for key,value in [('raw_mask',0),('face',2),('incidences',[])]:
   bad=copy.deepcopy(rows);bad[0][key]=value
   with self.assertRaises(ValueError):self.result.validate_index(np,out,bad,summary)
 def test_independent_distances_interior_edge_vertex(self):
  from amd_mesh_aware_result import point_triangle_distances
  np=self.np;t=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]*3)
  got=point_triangle_distances(np,np.array([[.2,.2,2],[.5,-1,0],[-1,-1,0]]),t)
  self.assertTrue(np.allclose(got,[2,1,2**.5]))
 def test_png_export_readback_and_tampering(self):
  import tempfile
  from amd_mesh_aware_result import png
  with tempfile.TemporaryDirectory(prefix='modly-alpha-feature-test-',dir=p.BASE) as directory:
   path=Path(directory)/'fixture.png';png(path,self.np,self.np.zeros((2,3,3),dtype='u1'))
   self.assertEqual(self.result.verify_png(path,3,2)['decoded_pixels'],6)
   data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
   with self.assertRaises(ValueError):self.result.verify_png(path,3,2)
if __name__=='__main__':unittest.main()
