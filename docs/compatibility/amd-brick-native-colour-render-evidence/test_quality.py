import sys
sys.dont_write_bytecode=True
import unittest,json,ast,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from quality import decode
class Quality(unittest.TestCase):
    def test_byte_decode_endpoints_and_monotonicity(self):
        a=np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,axis=1);b=decode(a)
        self.assertEqual(b.dtype,np.dtype('<f4'));self.assertTrue((np.diff(b[:,0])>0).all());self.assertEqual(float(b[0,0]),0);self.assertEqual(float(b[-1,0]),1)
        self.assertEqual(float(b[128,0]),float(np.float32(((128/255+.055)/1.055)**2.4)))
    def test_bad_decode_schema(self):
        for a in (np.zeros((2,3),dtype='f4'),np.zeros((2,4),dtype='u1')):
            with self.assertRaises(ValueError):decode(a)
    def test_no_reachable_import_command(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('native_exact_host_under_test',Path(__file__).with_name('run_host.py'))
        run_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(run_host)
        for stage in ('import','verify','export','retry'):
            with self.assertRaises(ValueError):run_host.worker_command(stage,Path('C:/fixed'))
        self.assertEqual(run_host.worker_command('qualify',Path('C:/fixed'))[1:3],['-I','-B'])
        self.assertIn('--factory-startup',run_host.worker_command('render',Path('C:/fixed')))
        self.assertEqual((run_host.CAP,run_host.DEADLINE),(6*1024**3,180))
        source=Path(__file__).with_name('render_worker.py').read_text()
        self.assertNotIn('bpy.ops.import_scene',source);self.assertIn('bpy.ops.wm.open_mainfile',source)
    def test_exact_original_code_pins_before_any_marker(self):
        import inputs
        self.assertEqual(inputs.digest(inputs.OLD/'contract.py'),inputs.PINS[inputs.OLD/'contract.py'])
        for p in Path(__file__).resolve().parent.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
if __name__=='__main__':unittest.main(verbosity=2)
