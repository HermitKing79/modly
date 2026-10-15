"""Focused CPU-only containment checks; no genuine extraction."""
import importlib.util
from pathlib import Path
import sys,tempfile,unittest
import io
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('brick_mesh_host',HERE/'mesh_host.py')
host=importlib.util.module_from_spec(spec);sys.modules[spec.name]=host;spec.loader.exec_module(host)

class HostTests(unittest.TestCase):
    def test_exact_cli_and_fixed_resource_envelope(self):
        args=host.extraction_argv()
        self.assertEqual(Path(args[0]),host.CLI)
        self.assertEqual(args[args.index('--run-contract')+1],str(host.SOURCE.parent/'run-contract.json'))
        self.assertEqual(args[args.index('--source')+1],str(host.SOURCE))
        self.assertEqual(args[args.index('--output')+1],str(host.OUTPUT))
        self.assertEqual((host.PROCESS_CAP,host.RAM_FLOOR,host.DISK_FLOOR,host.DEADLINE),(6*1024**3,12*1024**3,2*1024**3,180))
    def test_resource_rejection_keeps_actual_values(self):
        for ram,disk in ((host.RAM_FLOOR-1,host.DISK_FLOOR),(host.RAM_FLOOR,host.DISK_FLOOR-1)):
            row=host.admission(ram,disk)
            self.assertFalse(row['admitted']);self.assertEqual(row['available_ram_bytes'],ram)
            self.assertEqual(row['disk_free_bytes'],disk)
        self.assertTrue(host.admission(host.RAM_FLOOR,host.DISK_FLOOR)['admitted'])
    def test_gated_owned_fixture_denies_gpu_network_and_children(self):
        with tempfile.TemporaryDirectory(prefix='host-fixture-',dir=host.PRIVATE) as temp:
            record=host.supervise(Path(temp),fixture=True)
            self.assertEqual(record['classification'],'PASS',record)
            self.assertEqual(record['worker_returncode'],0)
            self.assertTrue(record['job_attached_before_GO'])
            self.assertFalse(record['job_cleanup']['forced_job_cleanup'])
            self.assertTrue(record['job_cleanup']['owned_tree_exit_verified'])
            self.assertGreater(record['peak_process_committed_bytes'],0)
            self.assertGreater(record['process_memory']['peak_working_set_bytes'],0)
            import json
            fixture=json.loads((Path(temp)/'fixture-result.json').read_text())
            self.assertEqual(fixture['denied'],['torch','subprocess','network'])
            self.assertTrue(fixture['CPU_only'])
    def test_invalid_authorization_rejected_before_any_worker_write(self):
        for token in (b'',b'NO\n',b'GO\r\n',b'GO\nx',b'GO'):
            with self.subTest(token=token),patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(token))),\
                    patch.object(host,'save',side_effect=AssertionError('Unexpected write')):
                with self.assertRaisesRegex(RuntimeError,'Exact GO'):
                    host.worker(HERE,fixture=True)

if __name__=='__main__':unittest.main(verbosity=2)
