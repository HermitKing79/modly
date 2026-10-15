import importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('pbr_contract.py');s=importlib.util.spec_from_file_location('pbr_contract',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_genuine_channel_order_and_no_clamp(self):
  x=np.array([[.1,.2,.3,-.02,1.04,1.008],[.4,.5,.6,.9,.12,.99]],np.float32);old=x.tobytes();v=m.raw_channels(x)
  for name,index in [('Trellis_metallic_raw',3),('Trellis_roughness_raw',4),('Trellis_alpha_raw',5)]:
   self.assertEqual(v[name].tobytes(),x[:,index].tobytes());self.assertTrue(v[name].flags.c_contiguous)
  self.assertEqual(x.tobytes(),old);self.assertLess(v['Trellis_metallic_raw'][0],0);self.assertGreater(v['Trellis_roughness_raw'][0],1)
 def test_derived_arrays_do_not_mutate_source(self):
  x=np.zeros((2,6),np.float32);v=m.raw_channels(x);v['Trellis_alpha_raw'][0]=1;self.assertEqual(x.sum(),0)
 def test_invalid_dtype_shape_or_nonfinite_rejected(self):
  for x in [np.zeros((2,5),np.float32),np.zeros((2,6),np.float64),np.full((1,6),np.nan,np.float32)]:
   with self.assertRaises(ValueError):m.raw_channels(x)
if __name__=='__main__':unittest.main(verbosity=2)
