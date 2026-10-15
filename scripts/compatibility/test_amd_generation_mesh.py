"""Focused run-artifact reader and real CPU converter integration tests."""
import hashlib,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
import amd_generation_mesh as mesh

def fixture(path,wrong_dtype=False,varied_logits=False,permuted=False):
 coords=np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]],dtype='<i4')
 if permuted:coords[[1,2]]=coords[[2,1]]
 feats=np.zeros((4,7),'<f4');feats[0,5]=1
 tensors={'features':feats,'coordinates':coords}
 for i in range(4):
  tensors[f'subdivision_{i}_coordinates']=np.zeros((1,4),'<i4')
  mask=np.full((1,8),0xc000 if varied_logits else 0,dtype='<u2')
  mask[0,:4 if i==3 else 1]=0x4000 if varied_logits else 0x3f80
  tensors[f'subdivision_{i}_logits']=mask
 header={};payload=bytearray()
 for name,a in tensors.items():
  dtype='BF16' if name.endswith('logits') else 'I32' if name.endswith('coordinates') else 'F32'
  header[name]={'dtype':dtype,'shape':list(a.shape),'data_offsets':[len(payload),len(payload)+a.nbytes]};payload.extend(a.tobytes())
 if wrong_dtype:header['features']['dtype']='F16'
 raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
 path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
 return hashlib.sha256(path.read_bytes()).hexdigest()

class MeshTests(unittest.TestCase):
 def setUp(self):
  self.allowed=Path(__file__).resolve().parents[3]/'artifacts/amd-repeatable-generation-v1-6ea0abb/cpu-mesh-tests'
  self.allowed.mkdir(exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=self.allowed);self.root=Path(self.temp.name)
 def tearDown(self):
  self.root.resolve().relative_to(self.allowed.resolve());self.temp.cleanup()
 def test_source_derived_converter_and_independent_PLY_readback(self):
  source=self.root/'neural.safetensors';sha=fixture(source)
  result=mesh.extract(source,sha,self.root/'mesh')
  self.assertEqual((result['vertices'],result['triangles']),(4,2))
  self.assertEqual(result['input_sha256'],sha)
  self.assertTrue(result['PLY_readback_exact'])
  self.assertEqual(result['learned_subdivisions'],[1,1,1,4])
 def test_wrong_hash_rejected_before_output_creation(self):
  source=self.root/'neural.safetensors';fixture(source)
  with self.assertRaisesRegex(ValueError,'hash'):mesh.extract(source,'0'*64,self.root/'mesh')
  self.assertFalse((self.root/'mesh').exists())
 def test_wrong_feature_precision_rejected(self):
  source=self.root/'neural.safetensors';sha=fixture(source,True)
  with self.assertRaises(ValueError):mesh.read_neural(source,sha)
 def test_refuse_existing_output(self):
  source=self.root/'neural.safetensors';sha=fixture(source);out=self.root/'mesh';out.mkdir()
  with self.assertRaises(FileExistsError):mesh.extract(source,sha,out)
 def test_raw_nonbinary_BF16_logits_use_original_strict_positive_rule(self):
  source=self.root/'neural.safetensors';sha=fixture(source,varied_logits=True)
  arrays,counts=mesh.read_neural(source,sha)
  self.assertEqual(counts,[1,1,1,4])
  self.assertEqual(int(arrays['subdivision_0_logits'][0,0]),0x4000)
 def test_permuted_child_coordinates_rejected_before_output(self):
  source=self.root/'neural.safetensors';sha=fixture(source,permuted=True)
  with self.assertRaisesRegex(ValueError,'child.*order'):mesh.extract(source,sha,self.root/'mesh')
  self.assertFalse((self.root/'mesh').exists())
 def test_measured_initial_support_bound_and_rejection(self):
  for count in (4430,4431):
   xyz=np.stack(np.unravel_index(np.arange(count),(32,32,32)),axis=1).astype('<i4')
   coords=np.column_stack((np.zeros(count,'<i4'),xyz));tensors={}
   for i in range(4):
    tensors[f'subdivision_{i}_coordinates']=coords.copy()
    logits=np.zeros((count,8),'<u2');logits[:,0]=0x3f80
    tensors[f'subdivision_{i}_logits']=logits;coords=coords.copy();coords[:,1:]*=2
   tensors.update(features=np.zeros((count,7),'<f4'),coordinates=coords)
   header={};payload=bytearray()
   for name,array in tensors.items():
    dtype='BF16' if name.endswith('logits') else 'I32' if name.endswith('coordinates') else 'F32'
    header[name]={'dtype':dtype,'shape':list(array.shape),'data_offsets':[len(payload),len(payload)+array.nbytes]};payload.extend(array.tobytes())
   raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
   path=self.root/f'support-{count}.safetensors';path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
   if count==4430:
    arrays,counts=mesh.read_neural(path,mesh.digest(path));self.assertEqual(counts,[count]*4)
    for array in arrays.values():array._mmap.close()
   else:
    with self.assertRaisesRegex(ValueError,'parent envelope'):mesh.read_neural(path,mesh.digest(path))

if __name__=='__main__':unittest.main()
