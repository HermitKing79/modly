"""Mocked filesystem and host-entry regressions; no Jobs or processes."""
import errno
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/compatibility/amd-windows-lifecycle-outside-evidence/windows_qualification.py'
spec = importlib.util.spec_from_file_location('writer_reporting_contract', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Sink:
    def __init__(self, failure=None): self.failure, self.writes = failure, []
    def write(self, value):
        if self.failure == 'json.write': raise PermissionError(errno.EACCES, 'write blocked')
        self.writes.append(value)
    def flush(self):
        if self.failure == 'flush': raise PermissionError(errno.EACCES, 'flush blocked')
    def close(self):
        if self.failure == 'close': raise PermissionError(errno.EACCES, 'close blocked')
    def fileno(self): return 123


class ReservedRoot:
    reserved = False
    def mkdir(self, **kwargs): self.reserved = True
    def __truediv__(self, name): return Path('C:/synthetic-owned') / name


class PrelaunchErrors(unittest.TestCase):
    def setUp(self): m._PRELAUNCH_PHASE = 'test.prelaunch'

    def test_each_publication_failure_keeps_exact_operation(self):
        for operation in ('staging.open-exclusive', 'json.write', 'flush', 'os.fsync', 'close', 'os.link', 'staging.unlink'):
            with self.subTest(operation=operation):
                sink = Sink(operation)
                failure = PermissionError(errno.EACCES, 'blocked')
                with patch.object(Path, 'open', side_effect=failure if operation == 'staging.open-exclusive' else None, return_value=sink), \
                     patch.object(m.os, 'fsync', side_effect=failure if operation == 'os.fsync' else None), \
                     patch.object(m.os, 'link', side_effect=failure if operation == 'os.link' else None), \
                     patch.object(Path, 'unlink', side_effect=failure if operation == 'staging.unlink' else None):
                    with self.assertRaises(PermissionError) as caught: m.save(Path('C:/synthetic-owned/attempt.json'), {'x': 1})
                self.assertEqual(caught.exception.modly_publication['operation'], operation)
                self.assertEqual(caught.exception.errno, errno.EACCES)

    def test_real_host_route_reports_failure_after_directory_reservation(self):
        root = ReservedRoot()
        error = PermissionError(errno.EACCES, 'link blocked')
        error.winerror = 5
        rows = []
        job = Mock(side_effect=AssertionError('No Job permitted'))
        fake_process = types.SimpleNamespace(_OwnedJob=job, _host_health=lambda p: {'available_ram_bytes': 20*1024**3, 'free_disk_bytes': 3*1024**3}, _error_detail=Mock(), _timing_record=Mock())
        with patch.dict(sys.modules, {'diagnostic_process': fake_process}), \
             patch.object(m.sys, 'path', list(sys.path)), \
             patch.object(m, 'OUT', root), \
             patch.object(m, 'preflight', return_value=({'python_executable_sha256': 'a'*64}, {})), \
             patch.object(m, 'source_pins', return_value={}), patch.object(m, 'sha', return_value='b'*64), \
             patch.object(Path, 'open', return_value=Sink()), patch.object(m.os, 'fsync'), \
             patch.object(m.os, 'link', side_effect=error), \
             patch.object(m, '_path_snapshot', side_effect=lambda path: {'exists': str(path).endswith('.writing')}), \
             patch.object(m, '_write_error_row', side_effect=lambda row: rows.append(row)), \
             patch.object(m.subprocess, 'Popen', side_effect=AssertionError('No process permitted')):
            with self.assertRaises(PermissionError) as caught: m.host_entry()
        self.assertIs(caught.exception, error)
        self.assertTrue(root.reserved)
        job.assert_not_called()
        self.assertEqual(rows[0]['filesystem_operation'], 'os.link')
        self.assertEqual(rows[0]['errno'], errno.EACCES)
        self.assertEqual(rows[0]['winerror'], 5)
        self.assertTrue(rows[0]['staging']['exists'])
        self.assertFalse(rows[0]['final']['exists'])
        self.assertIn('PermissionError', rows[0]['traceback'])
        self.assertEqual(rows[0]['prelaunch_phase'], 'host.attempt-publication')

    def test_diagnostic_failure_falls_back_to_full_stderr_without_replacement(self):
        original = OSError(errno.EIO, 'original fs failure')
        text = io.StringIO()
        with patch.object(m, 'host', side_effect=original), \
             patch.object(m, '_path_snapshot', return_value={'exists': None}), \
             patch.object(m, '_write_error_row', side_effect=PermissionError('diagnostic denied')), \
             patch.object(m.sys, 'stderr', text):
            with self.assertRaises(OSError) as caught: m.host_entry()
        self.assertIs(caught.exception, original)
        self.assertIn('original fs failure', text.getvalue())
        self.assertIn('diagnostic denied', text.getvalue())
        self.assertIn('traceback', text.getvalue())

    def test_absent_error_codes_remain_unknown(self):
        rows = []
        with patch.object(m, 'host', side_effect=ValueError('policy failure')), \
             patch.object(m, '_path_snapshot', return_value={'exists': False}), \
             patch.object(m, '_write_error_row', side_effect=lambda row: rows.append(row)):
            with self.assertRaises(ValueError): m.host_entry()
        self.assertIsNone(rows[0]['errno'])
        self.assertIsNone(rows[0]['winerror'])
        self.assertIsNone(rows[0]['filesystem_operation'])

    def test_close_failure_cannot_replace_json_write_failure(self):
        original = PermissionError(errno.EACCES, 'first write error')
        sink = Sink('close')
        sink.write = Mock(side_effect=original)
        with patch.object(Path, 'open', return_value=sink):
            with self.assertRaises(PermissionError) as caught: m.save(Path('C:/synthetic-owned/attempt.json'), {})
        self.assertIs(caught.exception, original)
        self.assertEqual(original.modly_publication['operation'], 'json.write')
        self.assertEqual(original.modly_publication['close_error']['type'], 'PermissionError')

    def test_explicit_preflight_entry_reports_original_read_error(self):
        original = FileNotFoundError(errno.ENOENT, 'missing admitted source')
        rows = []
        def failing_preflight():
            return m.read('C:/synthetic/missing.json')
        with patch.object(m, 'preflight', side_effect=failing_preflight), \
             patch.object(Path, 'read_bytes', side_effect=original), \
             patch.object(m, '_path_snapshot', return_value={'exists': False}), \
             patch.object(m, '_write_error_row', side_effect=lambda row: rows.append(row)):
            with self.assertRaises(FileNotFoundError) as caught: m.preflight_entry()
        self.assertIs(caught.exception, original)
        self.assertEqual(rows[0]['filesystem_operation'], 'Path.read_bytes')
        self.assertEqual(rows[0]['prelaunch_phase'], 'host.preflight')


if __name__ == '__main__': unittest.main()
