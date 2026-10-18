"""Focused CPU tests for the Windows exact-input ACL boundary."""
import hashlib
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("windows_attest", Path(__file__).with_name("windows_attest.py"))
attest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attest)


class WindowsAttestationPolicyTests(unittest.TestCase):
    def row(self, *, file=True, sid="S-1-5-21-foreign-1003", rights=1179817,
            propagation="None", owner=None):
        return {
            "path": r"C:\exact\file" if file else "C:\\",
            "owner_sid": owner or attest.CURRENT_SID,
            "reparse": False, "link_type": None, "target": [],
            "attributes": 32 if file else stat.FILE_ATTRIBUTE_DIRECTORY,
            "sddl": "nonempty observed DACL",
            "rules": [{"type": "Allow", "sid": sid, "rights": rights,
                       "propagation": propagation}],
        }

    def test_foreign_read_only_file_acl_is_safe(self):
        self.assertEqual(attest.inspect_acl(self.row(), is_file=True), [])

    def test_each_foreign_file_mutation_is_rejected(self):
        for right in (0x2, 0x4, 0x10, 0x100, 0x10000, 0x40000, 0x80000,
                      0x10000000, 0x40000000):
            with self.subTest(right=right), self.assertRaises(ValueError):
                attest.inspect_acl(self.row(rights=right), is_file=True)

    def test_inherit_only_rule_is_not_applied_to_ancestor(self):
        row = self.row(file=False, rights=-536805376, propagation="InheritOnly")
        self.assertEqual(attest.inspect_acl(row, is_file=False), [])

    def test_root_create_directory_grant_cannot_replace_existing_child(self):
        row = self.row(file=False, rights=0x4, owner=attest.TRUSTED_INSTALLER_SID)
        grants = attest.inspect_acl(row, is_file=False)
        self.assertEqual(grants[0]["rights"], 4)

    def test_foreign_ancestor_delete_or_protection_change_is_rejected(self):
        for right in (0x40, 0x10000, 0x40000, 0x80000, 0x10000000, 0x40000000):
            with self.subTest(right=right), self.assertRaises(ValueError):
                attest.inspect_acl(self.row(file=False, rights=right), is_file=False)

    def test_windows_system_ancestor_owners_are_allowed(self):
        for owner in (attest.SYSTEM_SID, attest.ADMINISTRATORS_SID,
                      attest.TRUSTED_INSTALLER_SID):
            with self.subTest(owner=owner):
                attest.inspect_acl(self.row(file=False, owner=owner), is_file=False)

    def test_file_owner_must_be_actual_current_user(self):
        for owner in (attest.SYSTEM_SID, "S-1-5-21-foreign-1003"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                attest.inspect_acl(self.row(owner=owner), is_file=True)

    def test_reparse_and_link_evidence_are_rejected(self):
        for field, value in (("reparse", True), ("link_type", "SymbolicLink"),
                             ("target", [r"C:\different"]),
                             ("attributes", stat.FILE_ATTRIBUTE_REPARSE_POINT)):
            row = self.row()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                attest.inspect_acl(row, is_file=True)

    def test_missing_or_untyped_acl_evidence_is_rejected(self):
        for field, value in (("rules", []), ("sddl", ""), ("attributes", True)):
            row = self.row()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                attest.inspect_acl(row, is_file=True)

    def test_snapshot_output_cannot_escape_private_exact_filename(self):
        for path in (attest.PRIVATE / "other.json", attest.WORKSPACE / "public.json"):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                attest.fixed_output(path, attest.ATTESTATION)

    def test_growth_during_read_consumes_at_most_expected_bytes_plus_one(self):
        with tempfile.TemporaryDirectory(prefix="modly-attestation-bound-", dir=attest.PRIVATE) as directory:
            path = Path(directory) / "fixture.bin"
            original = b"DATA"
            path.write_bytes(original)
            limits, consumed = [], []
            with path.open("rb") as actual:
                class GrowingReader:
                    def __enter__(self):
                        return self

                    def __exit__(self, *_):
                        return False

                    def fileno(self):
                        return actual.fileno()

                    def read(self, limit):
                        if not limits:
                            # Grow only this stdlib fixture after the initial
                            # path/open identity checks and before its read.
                            with open(path, "ab") as writer:
                                writer.write(b"x" * 4096)
                        limits.append(limit)
                        chunk = actual.read(limit)
                        consumed.append(len(chunk))
                        return chunk

                with mock.patch.object(Path, "open", return_value=GrowingReader()):
                    with self.assertRaisesRegex(ValueError, "input byte count mismatch"):
                        attest.read_original(path, hashlib.sha256(original).hexdigest(), len(original))
            self.assertEqual(limits, [len(original) + 1])
            self.assertEqual(sum(consumed), len(original) + 1)

    def test_bounded_read_still_accepts_exact_original_fixture(self):
        with tempfile.TemporaryDirectory(prefix="modly-attestation-exact-", dir=attest.PRIVATE) as directory:
            path = Path(directory) / "fixture.bin"
            original = b"original CPU fixture"
            path.write_bytes(original)
            _, evidence = attest.read_original(path, hashlib.sha256(original).hexdigest(), len(original))
            self.assertTrue(evidence["stable"])


if __name__ == "__main__":
    unittest.main()

