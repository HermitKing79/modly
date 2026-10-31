import sys,unittest,json,tempfile
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_appearance as p
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(p.owner.CPU_SITE));import numpy as np;cls.np=np
 def test_limits_and_scope(self):
  self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180);self.assertTrue(p.old.LIVE_EXECUTION_HOLD)
  self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3));self.assertEqual(len(p.RENDERS),4)
  for s in ('export','uv','wrap','neural','bake','wsl','retry'):self.assertNotIn(s,p.STAGES)
 def test_ray_two_sided_and_nearest(self):
  np=self.np;v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1],[1,0,-1],[0,1,-1]])
  f=np.array([[0,1,2],[3,5,4]])
  self.assertEqual(p.ray(np,v,f,[.2,.2,2],[0,0,-1])['face'],0)
  self.assertEqual(p.ray(np,v,f,[.2,.2,-2],[0,0,1])['face'],1)
  self.assertIsNone(p.ray(np,v,f,[2,2,2],[0,0,-1]))
 def test_full_camera_basis(self):
  np=self.np;p.basis(np,[0,0,1],[1,0,0],[0,1,0])
  with self.assertRaises(ValueError):p.basis(np,[0,0,1],[1,0,0],[0,-1,0])
  with self.assertRaises(ValueError):p.basis(np,[0,0,1],[1,0,0],[1,1,0])
 def test_png_measurement(self):
  from amd_mesh_aware_result import png
  np=self.np
  with tempfile.TemporaryDirectory(dir=p.BASE) as folder:
   path=Path(folder)/'white.png';png(path,np,np.full((4,4,3),255,'u1'));r=p.image_stats(np,path)
   self.assertEqual(r['all_255_fraction'],1);self.assertEqual(r['near_white_fraction'],1);self.assertLess(r['luminance_std'],1e-10)
 def test_continuation_requires_pilot(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory(dir=p.BASE) as folder:
   root=Path(folder);(root/'continuation-review.json').write_text(json.dumps(dict(classification='APPROVED',parent=p.PARENT,allowed_stages=list(p.RENDERS[1:]),files={},pilot_useful=False)))
   with patch.object(p,'BASE',root),patch.object(p.old,'success'):
    with self.assertRaisesRegex(ValueError,'Useful pilot'):p.approved(p.RENDERS[1])
 def test_renderer_has_no_export_or_normal_edit_route(self):
  import ast
  tree=ast.parse(Path(p.__file__).with_name('amd_static_prop_appearance_blender.py').read_text())
  calls=[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)]
  self.assertNotIn('bpy.ops.export_scene.gltf',calls)
  self.assertFalse(any('unwrap' in x or 'normals_split_custom_set' in x for x in calls))
  self.assertEqual(calls.count('bpy.ops.render.render'),1) # one fixed loop, no retries
 def test_verifier_rejects_changed_normals_uv_and_camera(self):
  from unittest.mock import patch
  for changed in ('normals','UV','camera'):
   with self.subTest(changed=changed),tempfile.TemporaryDirectory(dir=p.BASE) as folder:
    root=Path(folder);(root/p.RENDERS[0]).mkdir();(root/'render-plan.json').write_text(json.dumps(dict(cameras=dict(detail37={'span':1}))))
    before=dict(source=dict(normals='original',UV='original'));after=json.loads(json.dumps(before))
    if changed!='camera':after['source'][changed]='modified'
    row=dict(camera={'span':2 if changed=='camera' else 1},geometry_modified=False,GLB_sha256=p.GLB_SHA,before=before,after=after)
    (root/p.RENDERS[0]/'appearance.json').write_text(json.dumps(row))
    with patch.object(p,'BASE',root),patch.object(p.old,'success'):
     with self.assertRaises(ValueError):p.verify()
if __name__=='__main__':unittest.main()
