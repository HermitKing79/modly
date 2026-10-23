"""Native Linux exec-observer contracts, simulated procfs only; no WSL launch."""
from pathlib import Path
import unittest
from unittest.mock import patch
from services import amd_wsl_session as broker


class NativeExecObservationTests(unittest.TestCase):
    def setUp(self):
        self.leader=dict(pid=11,ppid=10,uid=1000,start_ticks=100,session=11,pgrp=11,state='S',rss_pages=5)
        self.argv=['/usr/bin/python3','-B','-I','-S','/mnt/c/owned.py']
        self.blob=b'\0'.join(x.encode() for x in self.argv)+b'\0'
        self.limits='Max address space         6442450944           6442450944           bytes\n'
        self.identity=patch.object(broker,'_read_identity',return_value=self.leader).start()
        patch.object(broker.os,'pidfd_open',return_value=17,create=True).start()
        self.closed=patch.object(broker.os,'close').start()
        patch.object(broker.os,'readlink',return_value='/usr/bin/python3.12').start()
        patch.object(broker.os.path,'realpath',return_value='/usr/bin/python3.12').start()
        self.bytes=patch.object(Path,'read_bytes',side_effect=lambda _:self.blob,autospec=True).start()
        self.text=patch.object(Path,'read_text',side_effect=lambda _:self.limits,autospec=True).start()
        self.select=patch.object(broker.select,'select',return_value=([],[],[])).start()
        self.addCleanup(patch.stopall)

    def observe(self):
        return broker._execution_observation({'argv':self.argv},self.leader)

    def test_native_exec_arguments_and_inherited_limits_are_observed(self):
        result=self.observe()
        self.assertEqual(result['executable'],'/usr/bin/python3.12')
        self.assertEqual(result['argv'],['/usr/bin/python3','-B','-I','-S','/mnt/c/owned.py'])
        self.assertEqual(result['rlimit_as_bytes'],[6442450944,6442450944])
        self.assertEqual(result['leader']['start_ticks'],100)

    def test_preexec_or_foreign_arguments_cannot_publish_execution(self):
        self.blob=b'/usr/bin/python3\0-B\0broker.py\0'
        self.assertIsNone(self.observe())

    def test_wrong_native_executable_cannot_publish_execution(self):
        with patch.object(broker.os,'readlink',return_value='/foreign/python3'):
            self.assertIsNone(self.observe())

    def test_incoherent_or_reused_leader_cannot_publish_execution(self):
        self.identity.side_effect=[dict(self.leader,start_ticks=999)]
        with self.assertRaises(RuntimeError):self.observe()
        self.identity.side_effect=[self.leader,dict(self.leader,ppid=99)]
        self.assertIsNone(self.observe())

    def test_exited_pidfd_cannot_publish_execution(self):
        self.select.return_value=([17],[],[])
        self.assertIsNone(self.observe())

    def test_wrong_limit_fails_instead_of_substituting_expected_value(self):
        self.limits='Max address space         unlimited           unlimited           bytes\n'
        with self.assertRaises(RuntimeError):self.observe()

    def test_disappeared_procfs_cannot_publish_execution(self):
        self.bytes.side_effect=FileNotFoundError('synthetic missing proc')
        self.assertIsNone(self.observe())


if __name__=='__main__':unittest.main(verbosity=2)
