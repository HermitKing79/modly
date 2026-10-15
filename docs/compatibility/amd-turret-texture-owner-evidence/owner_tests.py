"""Focused writer tests; simulated mapped UID here, real WSL acceptance separate."""
import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
if '--old' in sys.argv:
    sys.argv.remove('--old')
    artifact=types.ModuleType('artifact_before_owner_fix')
    sys.modules[artifact.__name__]=artifact
    exec(subprocess.check_output(['git','show','HEAD:scripts/compatibility/amd_texture_field_artifact.py'],cwd=ROOT),artifact.__dict__)
else:
    import amd_texture_field_artifact as artifact


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.info=self.root.lstat()
        self.raw=np.array([[-3,-2,-1,0,1,2]],dtype=np.float32)
        self.features=self.raw*np.float32(.5)+np.float32(.5)
        self.coords=np.array([[0,511,0,1]],dtype=np.int32)

    @contextlib.contextmanager
    def mapped(self):
        with contextlib.ExitStack() as stack:
            for name in ('getuid','geteuid','getgid','getegid'):
                stack.enter_context(patch.object(artifact.os,name,return_value=1000,create=True))
            stack.enter_context(patch.object(artifact,'_drvfs_mount',return_value='pinned-mount'))
            yield artifact._DrvfsOwnership(self.root,
                (self.info.st_dev,self.info.st_ino,0,0),(1000,1000,1000,1000),
                'pinned-mount','S-1-5-21-test','a'*64)

    def write(self, **kw):
        return artifact.write(self.root,self.raw,self.features,self.coords,{'mode':'fixture'},**kw)

    def test_explicit_capability_factory_exists(self):
        self.assertTrue(callable(getattr(artifact,'drvfs_ownership',None)))

    def test_default_uid_rejection_unchanged(self):
        with patch.object(artifact.os,'getuid',return_value=1000,create=True):
            with self.assertRaisesRegex(ValueError,'owned by'):
                self.write()
        self.assertEqual(list(self.root.iterdir()),[])

    def test_foreign_capability_rejected(self):
        with self.assertRaisesRegex(ValueError,'Foreign artifact'):
            self.write(authority=object())

    def test_whole_writer_reader_with_mapped_context(self):
        with self.mapped() as context:
            result=self.write(authority=context)
            self.assertTrue(result['verified'])
            self.assertTrue(artifact.verify(result['artifact_path'],self.root,result['metadata'],authority=context)['verified'])
            with self.assertRaisesRegex(ValueError,'owned by'):
                artifact.verify(result['artifact_path'],self.root)
            with self.assertRaisesRegex(ValueError,'owned by'):
                artifact._bounded_read(Path(result['artifact_path']),artifact.MAX_FILE_BYTES)

    def test_foreign_root_and_filename_rejected(self):
        with self.mapped() as context:
            other=self.root/'other';other.mkdir()
            with self.assertRaisesRegex(ValueError,'confined'):
                artifact.approved_root(other,authority=context)
            foreign=self.root/'foreign';foreign.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'confined'):
                artifact._bounded_read(foreign,100,authority=context)

    def test_changed_root_identity_rejected_before_creation(self):
        with self.mapped() as context:
            bad=artifact._DrvfsOwnership(context.root,(self.info.st_dev,self.info.st_ino+1,0,0),
                context.worker_identity,context.mount,context.windows_owner_sid,context.attestation_sha256)
            with self.assertRaisesRegex(ValueError,'identity changed'):
                self.write(authority=bad)
            self.assertEqual(list(self.root.iterdir()),[])

    def test_unexpected_file_owner_and_unattested_root_rejected(self):
        with self.mapped() as context:
            bad=types.SimpleNamespace(st_dev=self.info.st_dev,st_uid=999,st_gid=0)
            with self.assertRaisesRegex(ValueError,'Unexpected DrvFS'):
                artifact._owner(self.root/artifact.ARTIFACT_NAME,bad,context)
        with self.assertRaisesRegex(ValueError,'attestation'):
            artifact.drvfs_ownership(self.root,{'format':'modly-owned-drvfs-v1','roots':[]},'a'*64)


if __name__=='__main__':unittest.main(verbosity=2)
