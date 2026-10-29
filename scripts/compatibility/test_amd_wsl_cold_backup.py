"""Offline/synthetic Windows fixtures only. Never open the real VHDX or E:."""
import contextlib
import hashlib
import importlib
import io
import json
import os
import struct
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
try:
    import amd_wsl_cold_backup as backup
except ModuleNotFoundError:
    backup = None
TEMP = Path(__file__).resolve().parents[3] / 'artifacts'


class Implementation(unittest.TestCase):
    def test_backup_is_implemented(self):
        self.assertIsNotNone(backup, 'Cold backup implementation missing')


@unittest.skipIf(backup is None, 'Implementation missing')
class Streaming(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source.bin'
        self.src.write_bytes(b'0123456789abcdef')
        self.dest = self.root / 'backup.bin'
        self.events = []

    def identity(self):
        s = self.src.stat()
        return dict(size=s.st_size, mtime=s.st_mtime_ns, inode=s.st_ino)

    def independent(self, path):
        return dict(size=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def run_copy(self, source=None, identity=None, health=lambda: None,
                 independent=None):
        with self.src.open('rb') as original:
            return backup.stream_backup(
                source or original, self.dest, identity or self.identity, health,
                independent or self.independent,
                lambda phase, **kw: self.events.append((phase, kw)), block=3)

    def test_complete_bytes_and_three_hashes_agree(self):
        row = self.run_copy()
        self.assertEqual(self.dest.read_bytes(), b'0123456789abcdef')
        self.assertEqual(row['source_sha256'],
                         '9f9f5111f7b27a781f1f1ddde5ebc2dd2b796bfc7365c9c28b548e564176929f')
        self.assertEqual(row['destination_sha256'], row['source_sha256'])
        self.assertEqual(row['copy_sha256'], row['source_sha256'])
        self.assertEqual(row['bytes'], 16)

    def test_existing_destination_is_never_overwritten(self):
        self.dest.write_bytes(b'KEEP')
        with self.assertRaises(FileExistsError): self.run_copy()
        self.assertEqual(self.dest.read_bytes(), b'KEEP')

    def test_changed_source_identity_fails(self):
        original = self.identity(); calls = []
        def identity():
            calls.append(1)
            return original if len(calls) == 1 else dict(original, mtime=0)
        with self.assertRaisesRegex(RuntimeError, 'identity'): self.run_copy(identity=identity)
        self.assertTrue(self.dest.exists())

    def test_source_payload_change_without_metadata_change_fails(self):
        class Changing(io.BytesIO):
            def seek(self, offset, whence=0):
                if offset == 0:
                    self.getbuffer()[0] = ord('X')
                return super().seek(offset, whence)
        with self.assertRaisesRegex(RuntimeError, 'source hash'):
            self.run_copy(source=Changing(b'0123456789abcdef'))

    def test_read_failure_preserves_partial_file(self):
        class Broken(io.BytesIO):
            def read(self, n=-1):
                if self.tell() >= 3: raise OSError('read failure')
                return super().read(n)
        with self.assertRaisesRegex(OSError, 'read failure'):
            self.run_copy(source=Broken(b'0123456789abcdef'))
        self.assertEqual(self.dest.read_bytes(), b'012')

    def test_interruption_preserves_partial_without_resume(self):
        calls = []
        def health():
            calls.append(1)
            if len(calls) == 3: raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.run_copy(health=health)
        self.assertTrue(self.dest.exists())
        self.assertLess(self.dest.stat().st_size, 16)
        with self.assertRaises(FileExistsError): self.run_copy()

    def test_hash_disagreement_preserves_complete_failed_copy(self):
        with self.assertRaisesRegex(RuntimeError, 'destination hash'):
            self.run_copy(independent=lambda _: dict(size=16, sha256='0' * 64))
        self.assertEqual(self.dest.stat().st_size, 16)

    def test_destination_length_disagreement_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'destination hash'):
            self.run_copy(independent=lambda p: dict(self.independent(p), size=15))

    def test_short_read_is_not_a_complete_backup(self):
        with self.assertRaisesRegex(RuntimeError, 'length'):
            self.run_copy(source=io.BytesIO(b'012'))


@unittest.skipIf(backup is None, 'Implementation missing')
class Guards(unittest.TestCase):
    def test_helper_inventory_contains_partition_identity_and_owner_gate(self):
        helper=backup.HELPER.read_text()
        self.assertIn('Select-Object DiskNumber,PartitionNumber,DriveLetter,Offset,Size',helper)
        self.assertIn('Exact owned inventory GO plus EOF required',helper)
        self.assertLess(helper.index('Exact owned inventory GO plus EOF required'),helper.index("$key='HKCU:"))
        self.assertIn('-WaitForOwner',backup.helper_command('inventory'))

    def test_actual_destination_volume_must_match_pinned_e_volume(self):
        with self.assertRaisesRegex(RuntimeError,'volume'):
            backup.require_volume({'volume_serial':'WRONG'},'QUALIFIED')
        backup.require_volume({'volume_serial':'QUALIFIED'},'QUALIFIED')

    def test_setup_failure_retains_fail_receipt_and_closes_pins(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            case=Path(d)/'case';dest=Path(d)/'fresh-backup';pins=MagicMock()
            def add(path):
                if path==case:raise OSError('injected setup failure')
            pins.add.side_effect=add
            with patch.object(backup,'CASE',case),patch.object(backup,'BACKUP',dest),patch.object(backup,'DESTINATION',dest/'ext4.vhdx'),patch.object(backup,'Pins',return_value=pins),patch.object(backup,'published'),patch.object(backup,'digest',return_value=backup.PYTHON_SHA),patch.object(backup,'authority',return_value={}):
                self.assertEqual(backup.run('a'*40),1)
            self.assertEqual(json.loads((case/'final-result.json').read_text())['classification'],'FAIL')
            pins.close.assert_called_once()

    def test_reparse_ancestor_is_rejected(self):
        with patch.object(Path,'lstat',return_value=SimpleNamespace(st_file_attributes=0x400)):
            with self.assertRaisesRegex(RuntimeError,'Reparse'):backup.plain(Path('C:/redirected/file'))

    def test_private_authority_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p=Path(d)/'authority.json';p.write_bytes(b'{"verified":true}')
            with patch.object(backup,'AUTHORITY',p),patch.object(backup,'AUTHORITY_SHA',hashlib.sha256(p.read_bytes()).hexdigest()):
                self.assertEqual(backup.authority(),{'verified':True})
                p.write_bytes(b'{"verified":false}')
                with self.assertRaisesRegex(RuntimeError,'authority'):backup.authority()

    def test_valid_inventory_and_each_identity_change(self):
        import copy
        a=dict(registration={'name':'qualified'},logical=[{'DeviceID':'E:','FileSystem':'NTFS','VolumeSerialNumber':'qualified'}],
               disks=[{'Index':1,'SerialNumber':'qualified'}],partitions=[{'DiskNumber':1,'DriveLetter':'E'}])
        row=dict(copy.deepcopy(a),running=[],vm_processes=[])
        backup.authenticate_inventory(row,a)
        for field,key in [('registration','name'),('logical','VolumeSerialNumber'),('disks','SerialNumber'),('partitions','DiskNumber')]:
            bad=copy.deepcopy(row)
            if field=='registration':bad[field][key]='wrong'
            else:bad[field][0][key]='wrong'
            with self.assertRaises(RuntimeError):backup.authenticate_inventory(bad,a)

    def test_default_dry_run_never_queries_or_copies(self):
        with patch.object(backup, 'run', side_effect=AssertionError('live')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(backup.main([]), 0)

    def test_missing_approval_stops_before_run(self):
        with patch.object(backup, 'run', side_effect=AssertionError('live')):
            with self.assertRaises(SystemExit): backup.main(['--run', '--commit', 'a' * 40])

    def test_space_accounts_for_full_file_and_reserves(self):
        required = backup.required_space(29_581_377_536)
        self.assertGreaterEqual(required, 31_728_861_184 + 2 * 1024**3 + 4 * 1024**2)
        with self.assertRaises(RuntimeError): backup.require_space(required - 1, required)
        backup.require_space(required, required)

    def test_cpu_live_floor_and_memory_ceiling_fail_closed(self):
        good = dict(available_bytes=12*1024**3, private_commit_bytes=1)
        backup.check_memory(good)
        with self.assertRaises(RuntimeError):
            backup.check_memory(dict(good, available_bytes=12*1024**3-1))
        with self.assertRaises(RuntimeError):
            backup.check_memory(dict(good, private_commit_bytes=512*1024**2+1))
        with self.assertRaises((RuntimeError, KeyError)): backup.check_memory({})

    def test_only_fixed_read_only_helper_commands(self):
        commands = [backup.helper_command('inventory'), backup.helper_command('hash', Path('C:/fixture.bin'))]
        for cmd in commands:
            joined = ' '.join(cmd).lower()
            for forbidden in ('--shutdown', '--resize', '--terminate', '--exec', 'robocopy', 'download'):
                self.assertNotIn(forbidden, joined)
            self.assertIn('-File', cmd)
            self.assertNotIn('-Command', cmd)
        with self.assertRaises(ValueError): backup.helper_command('resize')

    def test_running_or_lingering_vm_activity_rejected(self):
        a = {'registration': {}, 'running': ['Modly-AMD-Ubuntu-24.04'], 'vm_processes': []}
        with self.assertRaisesRegex(RuntimeError, 'activity'): backup.require_quiet(a)
        a['running'] = []; a['vm_processes'] = [dict(name='VmmemWSL')]
        with self.assertRaisesRegex(RuntimeError, 'activity'): backup.require_quiet(a)

    def test_wrong_drive_or_registry_identity_rejected(self):
        a = dict(registration=dict(name='wrong'), running=[], vm_processes=[], logical=[], disks=[], partitions=[])
        with self.assertRaises(RuntimeError): backup.authenticate_inventory(a, {})

    def test_timeout_and_job_aggregate_limits_are_bounded(self):
        self.assertEqual(backup.DEADLINE, 1800)
        self.assertEqual(backup.CAP, 512*1024**2)
        self.assertEqual(backup.BUFFER, 4*1024**2)


@unittest.skipIf(backup is None or os.name != 'nt', 'Windows native fixture')
class WindowsHandles(unittest.TestCase):
    def test_inventory_without_owner_go_stops_before_queries(self):
        import subprocess
        p=subprocess.run(backup.helper_command('inventory'),input=b'',capture_output=True,timeout=15)
        self.assertNotEqual(p.returncode,0)
        self.assertIn(b'Exact owned inventory GO plus EOF required',p.stderr)

    def test_full_worker_on_synthetic_container_without_any_inventory_query(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);source=root/'source.vhdx';case=root/'case';case.mkdir();dest=root/'backup'
            data=bytearray(2*1024**2);data[:8]=b'vhdxfile'
            region=196608;data[region:region+4]=b'regi';struct.pack_into('<I',data,region+8,1)
            struct.pack_into('<16sQII',data,region+16,uuid.UUID('8b7ca206-4790-4b9a-b8fe-575f050f886e').bytes_le,1048576,1048576,0)
            data[1048576:1048584]=b'metadata';struct.pack_into('<H',data,1048576+10,1)
            struct.pack_into('<16sIIII',data,1048576+32,uuid.UUID('2fa54224-cd1b-4876-b211-5dbed83bf4b8').bytes_le,65536,8,0,0)
            struct.pack_into('<Q',data,1048576+65536,32*1024**3);source.write_bytes(data)
            with backup.native_file(source,'source') as f:identity=backup.file_identity(f);meta=backup.metadata(f)
            a=dict(source_identity=identity,vhdx_metadata=meta,registration={'name':'fixture'},logical=[{'DeviceID':'E:','VolumeSerialNumber':identity['volume_serial']}],disks=[],partitions=[])
            inventory=dict(a,running=[],vm_processes=[])
            payload={'token':'a'*32,'commit':'b'*40}
            (case/'worker-owner.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),token=payload['token'])))
            stdin=io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()+b'\n'))
            fake=SimpleNamespace(stdout=json.dumps(inventory).encode(),stderr=b'')
            with patch.object(backup,'SOURCE',source),patch.object(backup,'CASE',case),patch.object(backup,'BACKUP',dest),patch.object(backup,'DESTINATION',dest/'ext4.vhdx'),patch.object(backup,'authority',return_value=a),patch.object(backup.shutil,'disk_usage',return_value=SimpleNamespace(free=1000*1024**3)),patch.object(backup.subprocess,'run',return_value=fake) as run,patch.object(sys,'stdin',stdin):
                self.assertEqual(backup.worker(),0)
                kwargs=run.call_args.kwargs
                self.assertEqual(kwargs['input'],b'MODLY_COLD_BACKUP_INVENTORY_GO\n')
                self.assertNotIn('stdin',kwargs)
            stdin.close()
            result=json.loads((case/'worker-result.json').read_text())
            self.assertEqual(result['classification'],'PASS')
            self.assertTrue(result['handles_closed'])
            self.assertEqual(source.read_bytes(),data)
            self.assertEqual((dest/'ext4.vhdx').read_bytes(),data)
            self.assertEqual(result['source_sha256'],hashlib.sha256(data).hexdigest())

    def test_native_stream_and_independent_dotnet_hash(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);src=root/'source.bin';dest=root/'dest.bin'
            src.write_bytes(b'0123456789abcdef')
            made=[]
            @contextlib.contextmanager
            def create(path):
                with backup.native_file(path,'create') as f:
                    yield f
                    made.append(backup.file_identity(f))
            def independent(path):
                with backup.native_file(path,'verify') as f:
                    backup.verify_destination_identity(made[0],backup.file_identity(f))
                    import subprocess
                    p=subprocess.run(backup.helper_command('hash',path),capture_output=True,timeout=15,check=True)
                    return json.loads(p.stdout.decode('utf-8-sig'))
            with backup.native_file(src,'source') as f:
                row=backup.stream_backup(f,dest,lambda:backup.file_identity(f),lambda:None,independent,lambda *a,**kw:None,create=create)
            self.assertEqual(row['source_sha256'],row['destination_sha256'])
            self.assertEqual(src.read_bytes(),dest.read_bytes())

    def test_source_handle_cannot_write(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p=Path(d)/'source';p.write_bytes(b'KEEP')
            with backup.native_file(p,'source') as f:
                with self.assertRaises((OSError,io.UnsupportedOperation)):f.write(b'X')
            self.assertEqual(p.read_bytes(),b'KEEP')

    def test_hardlink_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p=Path(d)/'file';q=Path(d)/'alias';p.write_bytes(b'KEEP');os.link(p,q)
            with self.assertRaisesRegex(RuntimeError,'single-link'):
                with backup.native_file(p,'source'):pass

    def test_pinned_directory_cannot_be_renamed(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p=Path(d)/'original';p.mkdir();pins=backup.Pins()
            try:
                pins.add(p)
                with self.assertRaises(OSError):p.rename(Path(d)/'moved')
            finally:pins.close()

    def test_exclusive_source_rejects_second_reader_and_writer(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p = Path(d) / 'synthetic.bin'; p.write_bytes(b'known bytes')
            with backup.native_file(p, 'source') as first:
                self.assertEqual(first.read(), b'known bytes')
                for mode in ('source', 'create'):
                    with self.assertRaises(OSError):
                        with backup.native_file(p, mode): self.fail('conflicting handle')
            self.assertEqual(p.read_bytes(), b'known bytes')

    def test_create_new_never_truncates_and_metadata_tracks_changes(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p = Path(d) / 'fixture.bin'
            with backup.native_file(p, 'create') as f:
                f.write(b'abc'); f.flush(); os.fsync(f.fileno())
                identity = backup.file_identity(f)
                self.assertEqual(identity['size'], 3)
            with self.assertRaises(OSError):
                with backup.native_file(p, 'create'): pass
            self.assertEqual(p.read_bytes(), b'abc')


@unittest.skipIf(backup is None or os.name != 'nt', 'Windows native fixture')
class OwnedCleanup(unittest.TestCase):
    def stage(self, body, timeout, attach_failure=False):
        import subprocess
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            case=Path(d);(case/'tmp').mkdir()
            cmd=[sys.executable,'-B','-I','-S','-c','import sys;sys.stdin.buffer.read();'+body]
            controller=backup.Controller()
            with patch.object(backup,'CASE',case),patch.object(backup,'helper_command',return_value=cmd),patch.object(controller,'health'):
                if attach_failure:
                    with patch.object(backup.OwnedJob,'attach',side_effect=RuntimeError('attachment failed')):
                        with self.assertRaisesRegex(RuntimeError,'attachment failed'):controller.child(cmd,'fixture',timeout)
                elif timeout < 1:
                    with self.assertRaises(TimeoutError):controller.child(cmd,'fixture',timeout)
                else:
                    self.assertEqual(controller.child(cmd,'fixture',timeout).strip(),'17')
                return json.loads((case/'fixture-process.json').read_text())

    def test_harmless_child_natural_exit_and_commit_measurement(self):
        row=self.stage('print(17)',10)
        self.assertTrue(row['cleanup']['owned_tree_exit_verified'])
        self.assertFalse(row['cleanup']['forced_job_cleanup'])
        self.assertGreater(row['peak_job_commit_bytes'],0)

    def test_deadline_cleans_only_owned_job(self):
        row=self.stage('import time;time.sleep(10)',.01)
        self.assertTrue(row['cleanup']['owned_tree_exit_verified'])
        self.assertTrue(row['cleanup']['forced_job_cleanup'])

    def test_unattached_owned_handle_is_reaped_on_attach_failure(self):
        row=self.stage('import time;time.sleep(10)',10,True)
        self.assertTrue(row['cleanup']['owned_tree_exit_verified'])
        self.assertFalse(row['cleanup']['forced_job_cleanup'])


if __name__ == '__main__': unittest.main(verbosity=2)
