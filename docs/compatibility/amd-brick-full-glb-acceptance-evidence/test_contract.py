import sys
sys.dont_write_bytecode=True
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from contract import compaction,verify_geometry
class Contract(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0,0,0],[1,0,0],[99,99,99],[0,1,0],[0,0,1]],dtype='<f4');self.f=np.array([[0,1,3],[0,1,3],[3,1,4]],dtype='<i4')
        self.ids,self.inv=compaction(self.f,len(self.v));self.a=self.v[self.ids];self.b=self.inv[self.f]
    def test_exact_compaction(self):
        np.testing.assert_array_equal(self.ids,[0,1,3,4]);np.testing.assert_array_equal(self.inv,[0,1,-1,2,3]);verify_geometry(self.v,self.f,self.a,self.b)
    def test_winding_and_order_and_multiplicity_rejected(self):
        for b in (self.b[:,::-1].copy(),self.b[::-1].copy(),self.b[:-1]):
            with self.assertRaises(ValueError):verify_geometry(self.v,self.f,self.a,b)
    def test_extra_merge_or_position_rejected(self):
        for a in (self.a[:-1],self.a+np.float32(.000001)):
            with self.assertRaises(ValueError):verify_geometry(self.v,self.f,a,self.b)
    def test_foreign_index(self):
        for f in (np.array([[-1,0,1]],dtype='<i4'),np.array([[0,1,5]],dtype='<i4')):
            with self.assertRaises(ValueError):compaction(f,5)
    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):compaction(self.f.astype('f4'),5)
    def test_actual_owned_commands_and_limits(self):
        import run_host
        for stage in ('import','render'):
            cmd=run_host.worker_command(stage,Path('C:/fixed'))
            self.assertEqual(cmd[0],str(run_host.BLENDER));self.assertIn('--factory-startup',cmd);self.assertIn('--disable-autoexec',cmd);self.assertEqual(cmd[-2:],['--output','C:\\fixed'])
        for stage in ('verify','inspect'):
            cmd=run_host.worker_command(stage,Path('C:/fixed'));self.assertEqual(cmd[1:3],['-I','-B']);self.assertNotIn('blender.exe',cmd[0])
        self.assertEqual((run_host.CAP,run_host.DEADLINE),(6*1024**3,180))
        with self.assertRaises(ValueError):run_host.worker_command('retry',Path('C:/fixed'))
if __name__=='__main__':unittest.main(verbosity=2)
