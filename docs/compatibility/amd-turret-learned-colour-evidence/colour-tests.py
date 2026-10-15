"""Three proportional CPU tests for the derived preview writer only."""
import importlib.util, tempfile, unittest
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('colour',E/'colour_preview.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_clamp_quantize_only_derived_copy(self):
  x=np.array([[-.1,.5,1.2],[0,1,1/255]],np.float32);old=x.tobytes()
  np.testing.assert_array_equal(m.quantize_rgb(x),np.array([[0,128,255],[0,255,1]],np.uint8))
  self.assertEqual(x.tobytes(),old)
 def test_nonfinite_or_wrong_layout_rejected(self):
  for x in (np.array([[0,float('nan'),1]],np.float32),np.zeros((2,4),np.float32),np.zeros((2,3),np.float64)):
   with self.assertRaises(ValueError):m.quantize_rgb(x)
 def test_coloured_ply_retains_original_payloads(self):
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'original.ply';b=Path(t)/'coloured.ply'
   xyz=np.array([[0.,-0.,1.],[.1,.2,.3],[-1.,.5,2.]],dtype='<f4')
   faces=np.array([(3,(2,0,1))],dtype=[('count','u1'),('indices','<i4',(3,))])
   h=b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
   a.write_bytes(h+xyz.tobytes()+faces.tobytes());original=a.read_bytes()
   rgb=np.array([[0,127,255],[12,34,56],[250,11,40]],np.uint8)
   m.append_colours(a,b,rgb)
   header,offset,nv,nf=m.ply_header(b,True)
   rec=np.memmap(b,mode='r',offset=offset,dtype=m.COLOURED_DTYPE,shape=(nv,))
   self.assertEqual(rec['xyz'].tobytes(),xyz.tobytes());self.assertEqual(rec['rgb'].tobytes(),rgb.tobytes())
   self.assertEqual(b.read_bytes()[offset+nv*15:],faces.tobytes());self.assertEqual(a.read_bytes(),original)
   del rec
if __name__=='__main__':unittest.main(verbosity=2)
