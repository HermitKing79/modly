"""Exclusive actual NPY staging, fixed CPU-only supervisor and tiny integration."""
import sys
sys.dont_write_bytecode=True
import tempfile,unittest,importlib.util
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from npy_stream import create,sync
from inputs import P
spec=importlib.util.spec_from_file_location('test_exact_full_sampled_host',E/'run_host.py')
host_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(host_module);make_host=host_module.make_host
TESTROOT=P/'unit-scratch';TESTROOT.mkdir(exist_ok=True)
class IntegrationTests(unittest.TestCase):
    def test_real_exclusive_NPY_writer_reopens_exact_F32_bits(self):
        with tempfile.TemporaryDirectory(dir=TESTROOT) as t:
            p=Path(t)/'linear.npy';a=create(p,'<f4',(5,3));expected=np.arange(15,dtype=np.float32).reshape(5,3)/15;a[:]=expected;sync(p,a);del a
            r=np.load(p,mmap_mode='r',allow_pickle=False);np.testing.assert_array_equal(r.view(np.uint32),expected.view(np.uint32));del r
            with self.assertRaises(FileExistsError):create(p,'<f4',(5,3))
            with self.assertRaises(ValueError):create(Path(t)/'bad.npy','<f8',(5,3))
    def test_only_original_limits_and_two_CPU_stages(self):
        h=make_host();self.assertEqual(h.CAP,6*1024**3);self.assertEqual(h.DEADLINE,180);self.assertEqual(h.E,E)
        self.assertEqual(h.BASE,'3db265ca500cfe46909d12f52c151924763cd2f6');self.assertEqual(h.BRANCH,'codex/amd-brick-full-sampled-glb');self.assertEqual(h.FOLDERS,{'export':'export','verify':'verify'})
    def test_actual_writer_and_independent_reader_agree_on_tiny_fixture(self):
        from stream_writer import write_glb
        from verify_stream import verify
        v=np.array([[0.,-0.,0.],[1,0,0],[0,1,0],[0,0,1]],np.float32);f=np.array([[0,1,2],[0,3,1],[0,2,3]],np.int32);c=np.full_like(v,.2);hashes={'fixture':'1'*64}
        with tempfile.TemporaryDirectory(dir=TESTROOT) as t:
            path=Path(t)/'fixture.glb';write_glb(path,v,f,c,hashes,batch_rows=2);got=verify(path,v,f,c,hashes,batch_rows=2)
            np.testing.assert_array_equal(got['linear_COLOR0'].view(np.uint32),c.view(np.uint32));self.assertFalse(got['linear_COLOR0'].flags.writeable);del got
if __name__=='__main__':unittest.main()
