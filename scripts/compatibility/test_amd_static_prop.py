import sys,unittest,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop as p
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(p.owner.CPU_SITE));import numpy as np;cls.np=np
 def test_overlap_exact_subpixel_and_boundary(self):
  a=[(0.,0.),(1.,0.),(0.,1.)]
  self.assertFalse(p.overlap(a,[(1.,0.),(1.,1.),(0.,1.)]))
  self.assertTrue(p.overlap(a,[(.1,.1),(.10000001,.1),(.1,.10000001)]))
  self.assertTrue(p.overlap(a,a))
 def test_uv_reassignment_and_reversal(self):
  pos=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)];uv=[(0.,0.),(1.,0.),(0.,1.)]
  self.assertEqual(p.corner_key(pos,uv),p.corner_key(pos[1:]+pos[:1],uv[1:]+uv[:1]))
  self.assertNotEqual(p.corner_key(pos,uv),p.corner_key(pos,[uv[1],uv[0],uv[2]]))
  self.assertNotEqual(p.corner_key(pos,uv),p.corner_key(pos[::-1],uv[::-1]))
 def test_inert_routes_and_limits(self):
  self.assertEqual(p.main([]),0)
  self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180)
  for stage in ('wrap','bake','neural','wsl','retry'):self.assertNotIn(stage,p.STAGES)
  self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3))
  self.assertTrue(p.LIVE_EXECUTION_HOLD)
  with self.assertRaises(RuntimeError):p.main(['--stage','export'])
  with self.assertRaises(RuntimeError):p.main(['--stage','matte-detail8'])
 def test_uv_continuous_overlap_and_density(self):
  np=self.np;v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]],'f4');f=np.array([[0,1,2],[1,3,2]])
  uv=v[f][:,:,:2]
  self.assertEqual(p.uv_check(np,uv,v,f)['continuous_overlap_pairs'],0)
  bad=uv.copy();bad[1]=bad[0]
  with self.assertRaises(ValueError):p.uv_check(np,bad,v,f)
  collapsed=uv.copy();collapsed[0,1]=collapsed[0,0]
  with self.assertRaises(ValueError):p.uv_check(np,collapsed,v,f)
 def test_neutral_material_tampering(self):
  import copy
  good=dict(pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1],metallicFactor=0,roughnessFactor=.35))
  p.validate_material(good)
  for name,val in [('baseColorFactor',[.5,1,1,1]),('metallicFactor',1),('roughnessFactor',.7),('roughnessFactor',float('nan'))]:
   bad=copy.deepcopy(good);bad['pbrMetallicRoughness'][name]=val
   with self.assertRaises(ValueError):p.validate_material(bad)
 def test_owned_scratch_delete_scope(self):
  import tempfile,os
  from unittest.mock import patch
  with tempfile.TemporaryDirectory(dir=p.BASE) as directory:
   base=Path(directory);scratch=base/'scratch'/'stage';scratch.mkdir(parents=True)
   with patch.object(p,'BASE',base):guard=p.audit('stage')
   for target,args in [(scratch,(str(scratch),-1)),(base/'scratch'/'other',(str(base/'scratch'/'other'),-1)),(scratch/'x',(str(scratch/'x'),123))]:
    with self.assertRaises((ValueError,RuntimeError)):guard('os.remove',args)
   target=scratch/'owned';guard('open',(str(target),'w',os.O_WRONLY|os.O_CREAT));target.write_text('owned')
   guard('os.remove',(str(target),-1))
 def test_only_exact_density_failure_allows_appearance(self):
  from unittest.mock import patch
  import copy
  r=dict(classification='FAIL',worker_returncode=1,cleanup=dict(forced_job_cleanup=False,owned_tree_exit_verified=True))
  err='stats=p.uv_check(np,uv,v,f)\nValueError: Unequal UV area density\n'
  with patch.object(p,'success'),patch.object(Path,'exists',return_value=False):
   with patch.object(Path,'read_text',side_effect=[json.dumps(r),err]):p.appearance_admission()
   for bad in [dict(r,error='resource fault'),dict(r,worker_returncode=2)]:
    with patch.object(Path,'read_text',side_effect=[json.dumps(bad),err]):
     with self.assertRaises(ValueError):p.appearance_admission()
   with patch.object(Path,'read_text',side_effect=[json.dumps(r),err.replace('Unequal UV area density','Different error')]):
    with self.assertRaises(ValueError):p.appearance_admission()
  p.appearance_admission() # Actual immutable host/stderr, including Blender's wrapper trailer.
 def test_glb_rejects_external_buffer_and_transform(self):
  import tempfile,struct
  doc=dict(buffers=[dict(byteLength=4)],scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0,scale=[2,2,2])])
  with tempfile.TemporaryDirectory(dir=p.BASE) as directory:
   path=Path(directory)/'bad.glb'
   for external in (False,True):
    if external:doc['buffers'][0]['uri']='foreign.bin'
    data=json.dumps(doc).encode();data+=b' '*((-len(data))%4);blob=struct.pack('<II',len(data),0x4e4f534a)+data+struct.pack('<II',4,0x004e4942)+b'\0'*4
    path.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(blob))+blob)
    with self.assertRaises(ValueError):p.glb(self.np,path)
   doc=dict(buffers=[dict(byteLength=4)],scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0)],meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,NORMAL=0,TEXCOORD_0=0),indices=0)])],accessors=[dict(bufferView=0,componentType=5126,type='VEC3',count=3)],bufferViews=[dict(buffer=0,byteLength=4)])
   data=json.dumps(doc).encode();data+=b' '*((-len(data))%4);blob=struct.pack('<II',len(data),0x4e4f534a)+data+struct.pack('<II',4,0x004e4942)+b'\0'*4
   path.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(blob))+blob)
   with self.assertRaisesRegex(ValueError,'bounds'):p.glb(self.np,path)
if __name__=='__main__':unittest.main()
