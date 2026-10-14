"""Focused stdlib CPU tests; real directory metadata, simulated Linux fd transport."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).with_name('amd_wsl_native_import_policy.py')

class OwnedCleanup(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('cleanup_policy',SCRIPT)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.temp=tempfile.TemporaryDirectory(dir=SCRIPT.parents[2])
        self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name)
        self.owned=self.parent/'owned';self.owned.mkdir()
        self.foreign=self.parent/'foreign';self.foreign.mkdir()
        self.policy=self.module.ImportPolicy(self.owned)
        self.metadata=self.owned.stat()

    def attest(self, link=None, metadata=None):
        # Windows has no /proc/self/fd. Only this platform transport is mocked;
        # the production audit, basename/root checks and resolved stat are real.
        link=str(self.owned) if link is None else link
        metadata=self.metadata if metadata is None else metadata
        return (patch.object(self.module.os,'readlink',return_value=link),
                patch.object(self.module.os,'fstat',return_value=metadata))

    def test_owned_unlink_and_rmdir_are_admitted(self):
        a,b=self.attest()
        with a,b:
            for event in ('os.remove','os.rmdir'):
                self.policy.audit(event,('child',17))
                self.policy.audit(event,(b'child',17))
        self.assertEqual(self.policy.snapshot()['rejected_calls'],0)

    def test_unsafe_names_and_descriptor_types_rejected(self):
        a,b=self.attest()
        with a,b:
            for name in ('','.', '..', '../escape', 'a/b', 'a\\b', '\x00', str(self.owned/'child'),17):
                for event in ('os.remove','os.rmdir'):
                    with self.subTest(event=event,name=name),self.assertRaises(self.module.ImportPolicyError):
                        self.policy.audit(event,(name,17))
            for fd in (True,False,-2,1.5,'17'):
                with self.subTest(fd=fd),self.assertRaises(self.module.ImportPolicyError):
                    self.policy.audit('os.remove',('child',fd))

    def test_foreign_prefix_neighbor_deleted_and_unresolved_rejected(self):
        sibling=self.parent/'owned-neighbor';sibling.mkdir()
        for target in (self.foreign,sibling,str(self.owned)+' (deleted)',self.parent/'missing','relative'):
            a,b=self.attest(link=str(target))
            with a,b,self.subTest(target=target),self.assertRaises(self.module.ImportPolicyError):
                self.policy.audit('os.remove',('child',17))
        with patch.object(self.module.os,'readlink',side_effect=OSError('closed fd')),self.assertRaises(self.module.ImportPolicyError):
            self.policy.audit('os.rmdir',('child',17))
        with patch.object(self.module.os,'fstat',side_effect=OSError('closed fd')),self.assertRaises(self.module.ImportPolicyError):
            self.policy.audit('os.remove',('child',17))

    def test_nondirectory_and_inconsistent_identity_rejected(self):
        file=self.owned/'file';file.write_bytes(b'x')
        wrong=list(self.metadata);wrong[1]+=1
        deleted=list(self.metadata);deleted[3]=0
        for metadata in (file.stat(),os.stat_result(wrong),os.stat_result(deleted)):
            a,b=self.attest(metadata=metadata)
            with a,b,self.assertRaises(self.module.ImportPolicyError):
                self.policy.audit('os.remove',('child',17))

    def test_changed_descriptor_is_rejected(self):
        with patch.object(self.module.os,'fstat',return_value=self.metadata),patch.object(self.module.os,'readlink',side_effect=[str(self.owned),str(self.foreign)]),self.assertRaises(self.module.ImportPolicyError):
            self.policy.audit('os.remove',('child',17))

    def test_noncleanup_fd_operations_remain_rejected(self):
        cases=[('os.mkdir',('child',0o700,17)),('os.chmod',('child',0o600,17)),
            ('os.chown',('child',1000,1000,17)),('os.utime',('child',None,None,17)),
            ('os.rename',('a','b',17,17)),('os.link',('a','b',17,17)),('os.symlink',('a','b',17))]
        a,b=self.attest()
        with a,b:
            for event,args in cases:
                with self.subTest(event=event),self.assertRaises(self.module.ImportPolicyError):self.policy.audit(event,args)
        with self.assertRaises(self.module.ImportPolicyError):self.policy._write_path('child',17)

    def test_existing_absolute_path_rules_and_other_restrictions_preserved(self):
        for event in ('os.remove','os.rmdir'):
            self.policy.audit(event,(str(self.owned/'child'),-1))
            with self.assertRaises(self.module.ImportPolicyError):self.policy.audit(event,(str(self.foreign/'child'),-1))
        for event,args in [('socket.connect',()),('subprocess.Popen',()),('os.system',('x',))]:
            with self.assertRaises(self.module.ImportPolicyError):self.policy.audit(event,args)

if __name__=='__main__':unittest.main()
