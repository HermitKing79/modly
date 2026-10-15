"""Only the changed bootstrap order and unchanged computational sources."""
import ast,hashlib
from pathlib import Path
import unittest
E=Path(__file__).resolve().parent
OLD=E.parent/'amd-turret-texture-decoder-evidence'
class BootstrapTests(unittest.TestCase):
 def test_shared_bootstrap_prepares_cache_and_stage_before_writer(self):
  tree=ast.parse((E/'decode.py').read_text())
  function=next((x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_bootstrap'),None)
  self.assertIsNotNone(function,'Production needs a shared bootstrap exercised by actual WSL CPU acceptance')
  text=ast.get_source_segment((E/'decode.py').read_text(),function)
  self.assertLess(text.index('scratch.mkdir'),text.index('os.environ.update(HOME='))
  self.assertLess(text.index('os.environ.update(HOME='),text.index('sys.path[:0]'))
  self.assertLess(text.index('sys.path[:0]'),text.index('from artifact_io import check_directory'))
  worker=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
  text=ast.get_source_segment((E/'decode.py').read_text(),worker)
  self.assertLess(text.index('base.await_go('),text.index('prepare_bootstrap('))
  self.assertLess(text.index('prepare_bootstrap('),text.index('import torch'))
 def test_computational_files_are_byte_identical(self):
  for name in ('neural.py','runtime.py','admission.py','chunked_final.py'):
   self.assertEqual((E/name).read_bytes(),(OLD/name).read_bytes(),name)
if __name__=='__main__':unittest.main(verbosity=2)
