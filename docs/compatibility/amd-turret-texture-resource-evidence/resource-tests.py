"""Only failed-resource persistence and unchanged runtime floors; no AI imports."""
import importlib.util,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resource_candidate',E/'decode.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ResourceTests(unittest.TestCase):
 def exercise(self,ram,disk,failed):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   def measure(ptr): ptr._obj.available=ram;return 1
   kernel=types.SimpleNamespace(GlobalMemoryStatusEx=measure)
   with patch.object(m,'EV',root),patch.object(m.ctypes,'windll',types.SimpleNamespace(kernel32=kernel),create=True),patch.object(m.shutil,'disk_usage',return_value=types.SimpleNamespace(free=disk)):
    if failed:
     with self.assertRaisesRegex(RuntimeError,'Host RAM/D-disk floor failed'):m.host_admission('native-qualified-CPU-preparation-entry')
    else:m.host_admission('pre-marker')
   rows=[json.loads(x) for x in (root/'phases.jsonl').read_text().splitlines()]
   self.assertEqual(len(rows),1);v=rows[0]
   self.assertEqual(v['phase'],'host-resource-admission')
   self.assertEqual(v['available_ram_bytes'],ram);self.assertEqual(v['D_free_bytes'],disk)
   self.assertEqual(v['required_host_ram_bytes'],12*m.GiB);self.assertEqual(v['required_D_free_bytes'],50*m.GiB)
   self.assertEqual(v['failed_conditions'],failed)
   self.assertEqual(v['execution_stage'],'native-qualified-CPU-preparation-entry' if failed else 'pre-marker')
   self.assertTrue(v['utc']);self.assertEqual(v['admitted'],not failed)
 def test_ram_failure_is_durable(self):self.exercise(12*m.GiB-1,50*m.GiB,['available_windows_ram'])
 def test_disk_failure_is_durable(self):self.exercise(12*m.GiB,50*m.GiB-1,['D_free_space'])
 def test_both_failures_are_durable(self):self.exercise(12*m.GiB-1,50*m.GiB-1,['available_windows_ram','D_free_space'])
 def test_exact_runtime_floors_still_pass(self):self.exercise(12*m.GiB,50*m.GiB,[])
if __name__=='__main__':unittest.main(verbosity=2)
