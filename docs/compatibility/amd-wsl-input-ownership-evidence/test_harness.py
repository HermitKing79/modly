"""Stdlib-only harness checks; these never launch WSL or import NumPy/Torch."""
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('test_owned_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host = load('host')
worker = load('worker')


class HarnessTests(unittest.TestCase):
    def test_diagnosis_command_is_cpu_gated_and_anchored(self):
        command = host.build_worker_command('diagnose', {'worker.py': 'a', 'host.py': 'b'})
        self.assertEqual(command[:7], ['wsl.exe', '-d', host.DISTRO, '-u', 'modly', '--exec', '/usr/bin/env'])
        self.assertIn('/opt/modly-wsl/venv/bin/python', command)
        self.assertIn('90s', command)
        self.assertEqual(command[command.index('--worker') - 1], host.guest_path(host.WORKER))
        self.assertIn('-I', command)
        self.assertIn('-B', command)
        self.assertNotIn('--manifest', command)
        self.assertTrue(all(key + '=0' in command for key in host.TUNABLE_ENV))
        self.assertEqual(host.SCRIPTS, host.ROOT / 'scripts/compatibility')

    def test_binary_go_rejects_extra_or_translated_bytes(self):
        worker.await_go(io.BytesIO(b'GO\n'))
        for blob in (b'', b'GO\r\n', b'GO\nx', b'GO', b'NO\n'):
            with self.subTest(blob=blob), self.assertRaises(RuntimeError):
                worker.await_go(io.BytesIO(blob))

    def test_ast_extraction_runs_actual_owner_predicate(self):
        namespace, provenance = worker.pinned_predicates()
        self.assertEqual(provenance['reader_sha256'], worker.READER_SHA256)
        fake_path = SimpleNamespace(lstat=lambda: SimpleNamespace(st_mode=0o40755, st_uid=0))
        # Exercise the original function body with measured metadata. Root UID
        # mismatch must remain a rejection; the diagnostic cannot weaken it.
        with patch.object(worker.os, 'getuid', return_value=1000, create=True):
            with self.assertRaisesRegex(ValueError, 'owned by the current user'):
                namespace['_plain'](fake_path, directory=True)

    def test_verdict_demands_clean_real_containment(self):
        raw = {'classification': 'UNKNOWN', 'worker_returncode': 0, 'observer_healthy': True,
               'windows_job_cleanup': {'owned_tree_exit_verified': True, 'forced_job_cleanup': False},
               'windows_job_grace': {'empty': True}, 'distro_termination': {'returncode': 0},
               'elapsed_seconds': 1, 'worker': {'classification': 'CPU_DIAGNOSED',
                   'gate_authorized': True, 'CPU_only': True, 'forbidden_import_attempts': [], 'forbidden_modules': []}}
        self.assertEqual(host.cpu_verdict('diagnose', 1, raw)['classification'], 'PASS')
        raw['windows_job_cleanup']['forced_job_cleanup'] = True
        self.assertEqual(host.cpu_verdict('diagnose', 1, raw)['classification'], 'UNKNOWN')

    def test_full_mode_requires_hashed_attestation_inputs(self):
        with self.assertRaises(ValueError):
            host.build_worker_command('acceptance', {'host.py': 'a', 'worker.py': 'b'})

    def test_security_result_requires_nonempty_passing_checks(self):
        self.assertTrue(worker.security_tests_passed({'all_passed': True, 'tests': {'negative-owner': True}}))
        self.assertTrue(worker.security_tests_passed({'all_passed': True, 'tests': [{'passed': True}]}))
        for evidence in ({'all_passed': True, 'tests': []}, {'all_passed': True, 'tests': {'unsafe': False}},
                         {'all_passed': False, 'tests': {'negative-owner': True}}, {}):
            with self.subTest(evidence=evidence):
                self.assertFalse(worker.security_tests_passed(evidence))

    def test_exact_private_attestation_translation_only(self):
        self.assertEqual(host.guest_path(host.PRIVATE_ATTESTATION),
                         '/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-wsl-input-ownership-48ad081/windows-input-attestation.json')
        with self.assertRaises(ValueError):
            host.guest_path(host.PRIVATE_ATTESTATION.with_name('other.json'))

    def test_readonly_audit_allows_only_exact_exclusive_result_write(self):
        output = Path('/mnt/c/owned/guest')
        guard = worker.audit_guard({'forbidden_import_attempts': []}, worker.FORBIDDEN, output, False)
        guard('open', ('/mnt/c/input', 'r', os.O_RDONLY))
        guard('open', (str(output / 'results.json'), 'x', os.O_WRONLY | os.O_CREAT | os.O_EXCL))
        for path, mode, flags in (('/mnt/c/input', 'w', os.O_WRONLY | os.O_TRUNC),
                                  (str(output / 'results.json'), 'w', os.O_WRONLY | os.O_TRUNC),
                                  (str(output / 'other'), 'x', os.O_WRONLY | os.O_CREAT | os.O_EXCL)):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                guard('open', (path, mode, flags))

    def test_readonly_audit_blocks_network_mutation_and_gpu_but_allows_numpy_loader(self):
        guard = worker.audit_guard({'forbidden_import_attempts': []}, worker.FORBIDDEN, Path('/owned'), False)
        guard('ctypes.dlopen', (None,))
        for event, arguments in (('socket.connect', ()), ('os.remove', ('/input', -1)),
                                 ('os.mkdir', ('/input', 0o777, -1)), ('ctypes.dlopen', ('libamdhip64.so',)),
                                 ('import', ('torch', None, None, None, None))):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                guard(event, arguments)


if __name__ == '__main__':
    unittest.main()
