"""Mocked only: these tests never run the host filesystem probe."""
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import types
import unittest
from unittest.mock import patch

import writer_probe


class FakeStream(io.StringIO):
    def __init__(self, fs, path, text="", *, mode="x", newline=None):
        super().__init__(text)
        self.fs, self.path, self.mode, self.newline = fs, path, mode, newline

    def write(self, value):
        if self.path.name.endswith(".writing"):
            self.fs.fail("json_write")
        if self.mode == "x" and self.newline is None:
            value = value.replace("\n", os.linesep)
        return super().write(value)

    def read(self, size=-1):
        result = super().read(size)
        return result.encode("utf-8") if "b" in self.mode else result

    def flush(self):
        if self.path.name.endswith(".writing"):
            self.fs.fail("flush")
        elif self.fs.report_flush_failure:
            self.fs.report_flush_failure = False
            raise OSError(errno.EIO, "primary report flush")
        super().flush()

    def fileno(self):
        return self

    def close(self):
        if not self.closed:
            self.fs.files[self.path]["text"] = self.getvalue()
            if self.path.name.endswith(".writing"):
                self.fs.fail("close")
        super().close()
        if self.path.name.endswith("-report.json") and self.fs.report_close_failure:
            self.fs.report_close_failure = False
            raise OSError(errno.EPERM, "secondary report close")


class FakeFS:
    """An in-memory filesystem; no Path I/O or real file descriptors."""
    def __init__(self, failure=None, cleanup_failure=False):
        self.files = {}
        self.directories = {}
        self.counter = 100
        self.failure, self.cleanup_failure = failure, cleanup_failure
        self.failed = False
        self.report_flush_failure = self.report_close_failure = False
        self.published_bytes = None
        self.unlinked = []
        self.root = Path("C:/private-new")
        for parent in reversed((self.root, *self.root.parents)):
            self.directories[parent] = self.new_id()

    def new_id(self):
        self.counter += 1
        return self.counter

    def fail(self, label):
        if label == self.failure and not self.failed:
            self.failed = True
            error = OSError(errno.EACCES, "injected " + label)
            error.winerror = 5
            raise error

    def mkdir(self, path):
        if path in self.directories:
            raise FileExistsError(errno.EEXIST, "exists", str(path))
        self.directories[path] = self.new_id()

    def lstat(self, path):
        if path in self.directories:
            return types.SimpleNamespace(st_dev=1, st_ino=self.directories[path],
                st_mode=stat.S_IFDIR | 0o700, st_nlink=1, st_size=0,
                st_file_attributes=0)
        if path not in self.files:
            raise FileNotFoundError(errno.ENOENT, "missing", str(path))
        row = self.files[path]
        links = sum(x["id"] == row["id"] for x in self.files.values())
        return types.SimpleNamespace(st_dev=1, st_ino=row["id"],
            st_mode=row.get("mode", stat.S_IFREG | 0o600), st_nlink=links,
            st_size=len(row["text"].encode("utf-8")),
            st_file_attributes=row.get("attributes", 0))

    def fstat(self, stream):
        return self.lstat(stream.path)

    def open(self, path, mode, **kwargs):
        if mode == "x":
            if path.name.endswith(".writing"):
                self.fail("exclusive_staging_open")
            elif path.name.endswith("-report.json"):
                self.fail("report_write")
            if path in self.files:
                raise FileExistsError(errno.EEXIST, "exists", str(path))
            self.files[path] = {"id": self.new_id(), "text": ""}
            return FakeStream(self, path, mode=mode, newline=kwargs.get("newline"))
        if path.name == "attempt.json":
            self.fail("final_readback")
        return FakeStream(self, path, self.files[path]["text"], mode=mode, newline=kwargs.get("newline"))

    def fsync(self, descriptor):
        self.fail("fsync")

    def link(self, source, destination):
        self.fail("link_no_overwrite")
        if destination in self.files:
            raise FileExistsError(errno.EEXIST, "exists", str(destination))
        self.files[destination] = self.files[source]
        self.published_bytes = self.files[source]["text"].encode("utf-8")

    def unlink(self, path):
        if self.cleanup_failure:
            raise OSError(errno.EPERM, "injected cleanup")
        self.unlinked.append(path)
        del self.files[path]

    def read_bytes(self, path):
        return b"mock source and interpreter bytes\n"


class WriterProbeTests(unittest.TestCase):
    def run_mock(self, fs=None):
        fs = fs or FakeFS()
        stderr = io.StringIO()
        report = writer_probe._run_probe(fs.root, fs=fs, token="mock", stderr=stderr)
        return fs, report, stderr

    def test_each_boundary_has_its_own_timed_operation(self):
        fs, report, _ = self.run_mock()
        self.assertEqual([row["operation"] for row in report["operations"]], [
            "exclusive_staging_open", "json_write", "flush", "fsync", "close",
            "link_no_overwrite", "final_readback", "cleanup"])
        self.assertEqual(report["classification"], "PASS")
        for row in report["operations"]:
            self.assertTrue(row["ok"])
            self.assertGreaterEqual(row["end_monotonic_ns"], row["start_monotonic_ns"])
            self.assertGreaterEqual(row["end_perf_counter_ns"], row["start_perf_counter_ns"])
            self.assertEqual(row["duration_ns"], row["end_perf_counter_ns"] - row["start_perf_counter_ns"])
        self.assertEqual(len(fs.unlinked), 2)
        self.assertEqual(len(fs.files), 1)  # Only the ordinary report survives.
        saved = next(iter(fs.files.values()))["text"]
        self.assertEqual(len(saved.encode("utf-8")), report["reporting"]["size_bytes"])
        self.assertEqual(json.loads(saved)["classification"], "PASS")

    def test_first_failure_stops_forward_pipeline_and_keeps_errno(self):
        forward = ["exclusive_staging_open", "json_write", "flush", "fsync",
                   "close", "link_no_overwrite", "final_readback"]
        api_names = ["Path.open", "json.dump", "stream.flush", "os.fsync",
                     "stream.close", "os.link", "Path.open"]
        for index, label in enumerate(forward):
            with self.subTest(label=label):
                _, report, _ = self.run_mock(FakeFS(failure=label))
                self.assertEqual(report["classification"], "BLOCKED")
                self.assertEqual([row["operation"] for row in report["operations"]], forward[:index + 1] + ["cleanup"])
                error = report["first_exception"]
                self.assertEqual(error["operation"], label)
                self.assertEqual(error["errno"], errno.EACCES)
                self.assertEqual(error["winerror"], 5)
                self.assertEqual(error["type"], "PermissionError")
                self.assertEqual(error["actual_api"]["name"], api_names[index])
                self.assertIn("injected " + label, error["traceback"])

    def test_cleanup_failure_cannot_replace_original_exception(self):
        _, report, _ = self.run_mock(FakeFS("fsync", cleanup_failure=True))
        self.assertEqual(report["first_exception"]["operation"], "fsync")
        self.assertEqual(report["first_exception"]["errno"], errno.EACCES)
        self.assertFalse(report["operations"][-1]["ok"])
        self.assertTrue(report["cleanup"]["errors"])

    def test_existing_final_is_never_overwritten_or_removed(self):
        fs = FakeFS()
        final = fs.root / "probe-mock" / "attempt.json"
        fs.files[final] = {"id": fs.new_id(), "text": "historical"}
        _, report, _ = self.run_mock(fs)
        self.assertEqual(report["first_exception"]["operation"], "link_no_overwrite")
        self.assertEqual(report["first_exception"]["errno"], errno.EEXIST)
        self.assertEqual(fs.files[final]["text"], "historical")
        self.assertNotIn(final, fs.unlinked)

    def test_foreign_hardlink_alias_prevents_cleanup(self):
        fs = FakeFS()
        original = fs.link
        def link_with_foreign_alias(source, destination):
            original(source, destination)
            fs.files[fs.root / "foreign-alias"] = fs.files[source]
        fs.link = link_with_foreign_alias
        _, report, _ = self.run_mock(fs)
        self.assertEqual(report["classification"], "BLOCKED")
        self.assertFalse(report["operations"][-1]["ok"])
        self.assertEqual(fs.unlinked, [])

    def test_report_failure_emits_full_json_to_stderr_without_recursion(self):
        _, report, stderr = self.run_mock(FakeFS("report_write"))
        fallback = json.loads(stderr.getvalue())
        self.assertEqual(fallback["reporting"]["error"]["operation"], "report_write")
        self.assertEqual(fallback["operations"], report["operations"])
        self.assertEqual(fallback["source_identity"], report["source_identity"])

    def test_cli_does_nothing_without_explicit_run(self):
        with patch.object(writer_probe, "_run_probe") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(writer_probe.main([]), 0)
            run.assert_not_called()

    def test_reparse_root_forces_stderr_without_creating_report(self):
        fs = FakeFS()
        original = fs.lstat
        def reparse_root(path):
            row = original(path)
            if path == fs.root:
                row.st_file_attributes = 0x400
            return row
        fs.lstat = reparse_root
        _, report, stderr = self.run_mock(fs)
        self.assertEqual(report["first_exception"]["operation"], "setup")
        self.assertEqual(fs.files, {})
        self.assertFalse(json.loads(stderr.getvalue())["reporting"]["write_succeeded"])

    def test_replaced_or_redirected_files_are_never_removed(self):
        for change in ({"mode": stat.S_IFLNK | 0o600}, {"attributes": 0x400}, {"id": 99999}):
            with self.subTest(change=change):
                fs = FakeFS()
                original = fs.link
                def replaced_link(source, destination):
                    original(source, destination)
                    fs.files[destination] = dict(fs.files[destination], **change)
                fs.link = replaced_link
                _, report, _ = self.run_mock(fs)
                self.assertEqual(report["classification"], "BLOCKED")
                self.assertNotIn(fs.root / "probe-mock" / "attempt.json", fs.unlinked)

    def test_existing_report_survives_exclusive_write_failure(self):
        fs = FakeFS()
        path = fs.root / "probe-mock-report.json"
        fs.files[path] = {"id": fs.new_id(), "text": "original report"}
        _, report, stderr = self.run_mock(fs)
        self.assertEqual(fs.files[path]["text"], "original report")
        self.assertEqual(report["reporting"]["error"]["errno"], errno.EEXIST)
        self.assertEqual(json.loads(stderr.getvalue())["classification"], "BLOCKED")

    def test_staging_uses_original_platform_newlines_and_binary_readback_sizes(self):
        fs, report, _ = self.run_mock()
        self.assertEqual(report["classification"], "PASS")
        self.assertEqual(fs.published_bytes.count(os.linesep.encode()), 5)
        self.assertEqual(report["final_size_bytes"], len(fs.published_bytes))
        self.assertEqual(report["payload_sha256"], hashlib.sha256(fs.published_bytes).hexdigest())

    def test_report_close_error_does_not_mask_report_flush_error(self):
        fs = FakeFS()
        fs.report_flush_failure = fs.report_close_failure = True
        _, report, stderr = self.run_mock(fs)
        self.assertEqual(report["first_exception"]["errno"], errno.EIO)
        self.assertEqual(report["first_exception"]["actual_api"]["name"], "stream.flush")
        self.assertIn("primary report flush", report["first_exception"]["traceback"])
        self.assertIn("secondary report close", report["first_exception"]["traceback"])
        self.assertEqual(json.loads(stderr.getvalue())["first_exception"]["errno"], errno.EIO)

    def test_stderr_failure_cannot_replace_original_exception(self):
        class BrokenStderr:
            def write(self, _text):
                raise OSError(errno.EPIPE, "broken diagnostic stream")
            def flush(self):
                raise AssertionError("Do not flush after failed write")
        fs = FakeFS("fsync")
        fs.report_flush_failure = True
        report = writer_probe._run_probe(fs.root, fs=fs, token="mock", stderr=BrokenStderr())
        self.assertEqual(report["first_exception"]["operation"], "fsync")
        self.assertEqual(report["first_exception"]["errno"], errno.EACCES)
        self.assertEqual(report["reporting"]["stderr_fallback_error"]["errno"], errno.EPIPE)

    def test_link_guard_failure_does_not_claim_os_link_was_called(self):
        fs = FakeFS()
        original = fs.lstat
        def replaced_after_close(path):
            row = original(path)
            if path.name == "probe-mock" and any(p.name.endswith(".writing") for p in fs.files):
                row.st_ino = 99999
            return row
        fs.lstat = replaced_after_close
        _, report, _ = self.run_mock(fs)
        row = next(row for row in report["operations"] if row["operation"] == "link_no_overwrite")
        self.assertFalse(row["primary_api_attempted"])
        self.assertEqual(row["last_suboperation"], "scratch_ownership_guards")
        self.assertIsNone(row["exception"]["actual_api"])
        self.assertIsNone(fs.published_bytes)


if __name__ == "__main__":
    unittest.main()
