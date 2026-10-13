import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import amd_shape_sampling_artifact as a
class ArtifactTests(unittest.TestCase):
    def test_complete_readback_and_exact_normalization(self):
        x=np.arange(4*32,dtype=np.float32).reshape(4,32)/128
        coords=np.array([[0,0,0,i] for i in range(4)],dtype=np.int32)
        mean=np.linspace(-1,1,32,dtype=np.float32);std=np.linspace(.5,2,32,dtype=np.float32)
        y=a.denormalize(x,mean,std)
        np.testing.assert_array_equal(y,x*std[None]+mean[None])
        with tempfile.TemporaryDirectory() as d:
            rows=a.write(Path(d),x,y,coords,{'fixture':True},noise=x)
            self.assertEqual(set(rows),{'shape-latent-raw.npy','shape-latent-denormalized.npy','shape-latent-coordinates.npy','shape-latent-provenance.json','shape-initial-noise.npy'})
            with self.assertRaises(FileExistsError):a.write(Path(d),x,y,coords,{'fixture':True})
    def test_reject_nonfinite_dtype_shape_and_coordinate_changes(self):
        x=np.ones((4,32),np.float32);m=np.zeros(32,np.float32);s=np.ones(32,np.float32)
        for bad in (x.astype(np.float16),x[:,:31],np.full_like(x,np.nan)):
            with self.assertRaises(ValueError):a.denormalize(bad,m,s)
        with self.assertRaises(ValueError):a.denormalize(x,m,np.zeros(32,np.float32))
if __name__=='__main__':unittest.main()
