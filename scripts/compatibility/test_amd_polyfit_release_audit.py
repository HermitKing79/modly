"""Offline release qualification; never import the downloaded native extension."""
import base64
import csv
from contextlib import contextmanager
import hashlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import amd_polyfit_release_audit as audit


@contextmanager
def fixture_directory():
    # Retain tiny test inputs in owned private evidence, never system TEMP.
    path = Path(tempfile.mkdtemp(prefix='offline-fixture-', dir=audit.BASE))
    assert path.resolve().parent == audit.BASE.resolve()
    yield path


def fixture(path, extra=None, bad_record=False):
    payload = {'polyfit/math.dll': b'MZ\x00Soplex 3.1.1\x00',
               'polyfit-1.6.0.dist-info/WHEEL': b'Tag: py3-none-any\n'}
    if extra:
        payload.update(extra)
    rows = []
    for name, data in payload.items():
        h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        rows.append((name, 'sha256=' + ('X' * 43 if bad_record else h), str(len(data))))
    record = 'polyfit-1.6.0.dist-info/RECORD'
    out = io.StringIO(); csv.writer(out).writerows(rows + [(record, '', '')])
    payload[record] = out.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in payload.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseAudit(unittest.TestCase):
    def test_payload_hashes_and_old_solver_marker(self):
        with fixture_directory() as d:
            p = Path(d) / 'sample.whl'; sha = fixture(p)
            result = audit.inspect_wheel(p, sha)
            self.assertEqual(result['file_count'], 3)
            self.assertTrue(result['old_soplex_marker_present'])
            self.assertFalse(result['native_code_executed'])

    def test_whole_archive_tampering_rejected(self):
        with fixture_directory() as d:
            p = Path(d) / 'sample.whl'; sha = fixture(p)
            with p.open('ab') as f: f.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'Whole-wheel'):
                audit.inspect_wheel(p, sha)

    def test_embedded_record_mismatch_rejected(self):
        with fixture_directory() as d:
            p = Path(d) / 'sample.whl'; sha = fixture(p, bad_record=True)
            with self.assertRaisesRegex(ValueError, 'RECORD'):
                audit.inspect_wheel(p, sha)

    def test_unsafe_and_case_colliding_archive_paths_rejected(self):
        for extra in ({'../escape': b'x'}, {'C:/escape': b'x'},
                      {'polyfit\\escape': b'x'}, {'POLYFIT/MATH.DLL': b'x'}):
            with self.subTest(extra=extra), fixture_directory() as d:
                p = Path(d) / 'sample.whl'; sha = fixture(p, extra)
                # Windows ZipInfo normalises backslashes before storage, so
                # the inconsistent original RECORD is then the rejection.
                with self.assertRaisesRegex(ValueError, 'path|collision|RECORD'):
                    audit.inspect_wheel(p, sha)

    def test_license_gate_never_admits_unestablished_grant(self):
        result = audit.license_gate(old_soplex=True, applicable_grant_evidenced=False)
        self.assertEqual(result['decision'], 'NO_GO_BEFORE_INSTALL_OR_NATIVE_EXECUTION')
        self.assertFalse(result['example_authorized'])
        self.assertEqual(audit.license_gate(False, False)['decision'],
                         'UNQUALIFIED_NOT_AN_EXECUTION_AUTHORIZATION')

    def test_default_inert_and_owned_GO_required(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        self.assertEqual(audit.main([]), 0)
        with patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'NO\n'))):
            with self.assertRaisesRegex(RuntimeError, 'GO'):
                audit.main(['--worker', 'release-audit'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
