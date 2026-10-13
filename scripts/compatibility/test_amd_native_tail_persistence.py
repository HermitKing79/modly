"""CPU-only file persistence fixtures; never a replacement operator observation."""
import json
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_native_python_tail_cpu as runner

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.base=runner.EV/'persistence-tests'/str(uuid.uuid4());self.base.mkdir(parents=True)
        self.output=self.base/'cpu-probe';(self.output/'scratch').mkdir(parents=True)
        self.result=dict(passed=True,worker_exit_verified=True,job_cleanup=dict(owned_tree_exit_verified=True,forced_job_cleanup=False))
    def test_byte_preserving_exclusive_copy(self):
        payload=b'{"synthetic_persistence_fixture":true}\n'
        (self.output/'scratch/observation.json').write_bytes(payload)
        with patch.object(runner,'EV',self.base):
            self.assertTrue(runner.preserve_capture(self.output,self.result))
            self.assertEqual((self.output/'observation.json').read_bytes(),payload)
            with self.assertRaises(FileExistsError):runner.preserve_capture(self.output,self.result)
        self.assertEqual((self.output/'scratch/observation.json').read_bytes(),payload)
    def test_missing_capture_never_synthesizes_success(self):
        with patch.object(runner,'EV',self.base):self.assertFalse(runner.preserve_capture(self.output,self.result))
        self.assertFalse((self.output/'observation.json').exists())
    def test_failed_or_unclean_workers_rejected(self):
        with patch.object(runner,'EV',self.base):
            for result in (dict(passed=False),dict(passed=True,worker_exit_verified=False),
                           dict(passed=True,worker_exit_verified=True,job_cleanup=dict(owned_tree_exit_verified=True,forced_job_cleanup=True))):
                with self.assertRaises(ValueError):runner.preserve_capture(self.output,result)
        self.assertFalse((self.output/'observation.json').exists())
    def test_foreign_directory_and_large_capture_rejected(self):
        (self.output/'scratch/observation.json').write_bytes(b'0'*262145)
        with patch.object(runner,'EV',self.base):
            with self.assertRaises(ValueError):runner.preserve_capture(self.output,self.result)
            with self.assertRaises(ValueError):runner.preserve_capture(self.base/'foreign',self.result)
        self.assertFalse((self.output/'observation.json').exists())

if __name__=='__main__':unittest.main()
