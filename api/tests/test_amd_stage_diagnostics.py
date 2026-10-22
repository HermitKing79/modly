"""Native attribution contracts: Windows APIs are replaced, never executed."""
import ctypes
import unittest
from unittest.mock import patch

from services import amd_stage_process as stage


class Api:
    def __init__(self):
        self.error = 0
        self.fail = None
        self.code = 5
        self.alive = 259
        for name in ('CreateJobObjectW', 'SetInformationJobObject', 'AssignProcessToJobObject',
                     'QueryInformationJobObject', 'TerminateJobObject', 'CloseHandle', 'OpenProcess',
                     'GetProcessTimes', 'QueryFullProcessImageNameW', 'IsProcessInJob', 'GetExitCodeProcess'):
            setattr(self, name, self.function(name))

    def function(self, name):
        def call(*args):
            if name == self.fail:
                self.error = self.code
                self.alive = 0
                return 0
            if name == 'GetExitCodeProcess':
                args[1]._obj.value = self.alive
                # A diagnostic probe may clobber last-error. Original must survive.
                self.error = 87
            if name == 'IsProcessInJob': args[2]._obj.value = True
            if name == 'GetProcessTimes': args[1]._obj.dwLowDateTime = 12345
            if name == 'QueryFullProcessImageNameW': args[2].value = 'C:\\pinned\\python.exe'
            return 123 if name in ('OpenProcess', 'CreateJobObjectW') else 1
        return call


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(stage._OwnedJob, '_call'), 'Native operation attribution missing')
        self.api = Api()
        for p in (patch.object(stage.ctypes, 'WinDLL', return_value=self.api),
                  patch.object(stage.ctypes, 'get_last_error', side_effect=lambda: self.api.error),
                  patch.object(stage.ctypes, 'set_last_error', side_effect=lambda n: setattr(self.api, 'error', n))):
            p.start(); self.addCleanup(p.stop)
        self.job = stage._OwnedJob()
        self.job._identity(42, 123)

    def test_each_checked_native_api_is_attributed_without_cleanup_overwrite(self):
        actions = {
            'CreateJobObjectW': lambda: stage._OwnedJob(),
            'SetInformationJobObject': lambda: stage._OwnedJob(),
            'AssignProcessToJobObject': lambda: self.job.attach(type('P', (), {'pid': 42, '_handle': 123})()),
            'IsProcessInJob': lambda: self.job._identity(42, 123),
            'GetProcessTimes': lambda: self.job._identity(42, 123),
            'QueryFullProcessImageNameW': lambda: self.job._identity(42, 123),
            'QueryInformationJobObject': self.job.snapshot,
            'OpenProcess': lambda: self.job._call('OpenProcess', 0x1000, False, 42, pid=42),
            'TerminateJobObject': self.job.terminate,
            'CloseHandle': self.job.close,
        }
        for name, action in actions.items():
            with self.subTest(api=name):
                self.api.fail, self.api.alive = name, 259
                self.job.handle = 123
                with self.assertRaises(OSError) as caught: action()
                record = caught.exception.amd_native_failure
                self.assertEqual(record['api'], name)
                self.assertEqual(record['winerror'], 5)
                self.assertTrue(record['native_error_verified'])
                self.assertTrue(record['job_owner'])

    def test_identity_and_liveness_do_not_reinterpret_access_denied_as_exit(self):
        self.api.fail = 'QueryFullProcessImageNameW'
        try:
            self.job._identity(42, 123)
        except OSError as error:
            row = error.amd_native_failure
            detail = stage._error_detail(error, 'normal.observe')
        else:
            self.fail('Access denied was suppressed')
        self.assertEqual(row['pid'], 42)
        self.assertEqual(row['authenticated_identity']['creation_filetime'], 12345)
        self.assertTrue(row['alive_before']['alive'])
        self.assertFalse(row['alive_after']['alive'])
        self.assertEqual(row['winerror'], 5)  # after-probe writes87, never use that.
        self.assertIn('QueryFullProcessImageNameW', detail['traceback'])

    def test_missing_native_error_cannot_be_called_winerror5(self):
        self.api.fail, self.api.code = 'GetProcessTimes', 0
        with self.assertRaises(RuntimeError) as caught: self.job._identity(42, 123)
        row = caught.exception.amd_native_failure
        self.assertEqual(row['winerror'], 0)
        self.assertFalse(row['native_error_verified'])

    def test_negative_native_code_is_not_verified_windows_error(self):
        self.api.fail, self.api.code = 'GetProcessTimes', -1
        with self.assertRaises(RuntimeError) as caught: self.job._identity(42, 123)
        self.assertFalse(caught.exception.amd_native_failure['native_error_verified'])

    def test_non_access_denied_code_is_recorded_exactly(self):
        self.api.fail, self.api.code = 'GetProcessTimes', 87
        with self.assertRaises(OSError) as caught: self.job._identity(42, 123)
        self.assertEqual(caught.exception.amd_native_failure['winerror'], 87)

    def test_failed_open_does_not_claim_cached_pid_incarnation_is_current(self):
        self.api.fail = 'OpenProcess'
        with self.assertRaises(OSError) as caught: self.job._call('OpenProcess', 0x1000, False, 42, pid=42)
        row = caught.exception.amd_native_failure
        self.assertIsNone(row['authenticated_identity'])
        self.assertEqual(row['last_authenticated_identity']['creation_filetime'], 12345)
        self.assertFalse(row['identity_from_current_handle'])

    def test_query_categories_are_preserved_in_failure(self):
        original = self.api.QueryInformationJobObject
        for category in (1, 9, 3):
            def fail_category(*args):
                if args[1] == category: self.api.error = 5; return 0
                return original(*args)
            self.api.QueryInformationJobObject = fail_category
            with self.assertRaises(OSError) as caught: self.job.snapshot()
            self.assertEqual(caught.exception.amd_native_failure['category'], category)

    def test_process_handle_close_cannot_replace_identity_failure(self):
        def pids(*args):
            if args[1] == 3:
                args[2]._obj.Count = 1
                args[2]._obj.Pids[0] = 42
            return 1
        self.api.QueryInformationJobObject = pids
        self.api.fail = 'QueryFullProcessImageNameW'
        def no_close(*args): self.api.error = 6; return 0
        self.api.CloseHandle = no_close
        with self.assertRaises(OSError) as caught: self.job.snapshot()
        self.assertEqual(caught.exception.amd_native_failure['api'], 'QueryFullProcessImageNameW')
        self.assertEqual(caught.exception.amd_handle_cleanup['native']['api'], 'CloseHandle')

    def test_membership_false_is_not_stale_winerror5(self):
        def not_member(*args):
            args[2]._obj.value = False
            self.api.error = 5
            return 1
        self.api.IsProcessInJob = not_member
        with self.assertRaises(RuntimeError) as caught: self.job._identity(42, 123)
        self.assertFalse(hasattr(caught.exception, 'amd_native_failure'))

    def test_probe_failure_is_unknown_and_never_masks_primary_failure(self):
        def no_probe(*args): self.api.error = 6; return 0
        self.api.GetExitCodeProcess = no_probe
        self.api.fail = 'QueryFullProcessImageNameW'
        with self.assertRaises(OSError) as caught: self.job._identity(42, 123)
        row = caught.exception.amd_native_failure
        self.assertEqual(row['winerror'], 5)
        self.assertIsNone(row['alive_before']['alive'])
        self.assertEqual(row['alive_after']['probe_winerror'], 6)

    def test_python_error_without_native_evidence_does_not_use_ctypes_last_error(self):
        self.api.error = 5
        try:
            raise PermissionError('file or pipe operation failed')
        except PermissionError as exc:
            row = stage._error_detail(exc, 'cleanup.control')
        self.assertEqual(row['phase'], 'cleanup.control')
        self.assertIsNone(row['winerror'])
        self.assertIsNone(row['native'])
        self.assertIn('test_python_error_without_native_evidence', row['traceback'])


if __name__ == '__main__': unittest.main()
