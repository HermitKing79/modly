import sys
sys.dont_write_bytecode=True
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_host import make_host,E,P


class InputRouteTests(unittest.TestCase):
    def test_real_supervisor_adapter_uses_correct_paths_limits_and_no_renderer_stage(self):
        host=make_host()
        self.assertEqual(host.E,E);self.assertEqual(host.P,P)
        self.assertEqual(host.BASE,'1012a6c04976085c5cf8d4c6911d9010d0e8cc59')
        self.assertEqual(host.BRANCH,'codex/amd-brick-full-colour-glb')
        self.assertEqual(host.CAP,6*1024**3);self.assertEqual(host.DEADLINE,180)
        self.assertEqual(host.FOLDERS,{'export':'suitability'})
        self.assertTrue(callable(host.authenticate))
        self.assertEqual(host.HELPER,E.parent/'amd-brick-pbr-exposure-evidence/render_host.py')


if __name__=='__main__':unittest.main()
