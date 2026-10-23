"""Tiny stdlib evidence-writer qualification. Importing this module does no I/O.

Only ``--run`` executes the host probe. It creates a fresh scratch directory in
the new private evidence root, never retries a forward operation, and retains
the directory and an exclusive ordinary JSON report beside it. This is not a
Job, subprocess, WSL, GPU, package, or native lifecycle qualification.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback
import uuid


BASE = "d784e4b89e0a05096d2503755c7e931895895d45"
PRIVATE_ROOT = Path(__file__).absolute().parents[4] / "artifacts/amd-evidence-writer-d784e4b"
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class FileSystem:
    """Small I/O boundary, replaced entirely by in-memory test doubles."""
    @staticmethod
    def open(path, mode, **kwargs):
        return path.open(mode, **kwargs)

    mkdir = staticmethod(os.mkdir)
    lstat = staticmethod(os.lstat)
    fstat = staticmethod(os.fstat)
    fsync = staticmethod(os.fsync)
    link = staticmethod(os.link)
    unlink = staticmethod(os.unlink)

    @staticmethod
    def read_bytes(path):
        return path.read_bytes()


def _api_call(name, action, arguments=()):
    """Annotate the first throwing API immediately, before any diagnostics."""
    try:
        return action()
    except BaseException as error:
        if not hasattr(error, "_writer_probe_actual_api"):
            error._writer_probe_actual_api = {"name": name, "arguments": [str(arg) for arg in arguments]}
        raise


def _fs_call(fs, name, *args, **kwargs):
    api = "Path." + name if name in ("open", "read_bytes") else "os." + name
    return _api_call(api, lambda: getattr(fs, name)(*args, **kwargs), args)


def _identity(row):
    if not row.st_ino:
        raise ValueError("A nonzero filesystem creation identity is required")
    return {"device": row.st_dev, "inode": row.st_ino}


def _reject_alias(row, directory=False):
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(row.st_mode) or getattr(row, "st_file_attributes", 0) & REPARSE_POINT:
        raise ValueError("Symlink, reparse point, or unexpected file type rejected")


def _check_directory(fs, path, identity=None):
    row = _fs_call(fs, "lstat", path)
    _reject_alias(row, directory=True)
    found = _identity(row)
    if identity is not None and found != identity:
        raise ValueError("Owned scratch directory creation identity changed")
    return found


def _check_scratch(fs, state):
    root = state["root"]
    for path in reversed((root, *root.parents)):
        _check_directory(fs, path)
    _check_directory(fs, root, state["root_identity"])
    _check_directory(fs, state["scratch"], state["directory_identity"])


def _close_preserving(stream, primary_error):
    try:
        _api_call("stream.close", stream.close)
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note("Secondary close exception preserved separately:\n" +
                               "".join(traceback.format_exception(type(close_error), close_error, close_error.__traceback__)))


def _check_file(fs, path, identity, links):
    """Verify creation identity with both lstat and a temporary open handle."""
    before = _fs_call(fs, "lstat", path)
    _reject_alias(before)
    if _identity(before) != identity or before.st_nlink != links:
        raise ValueError("Owned file identity or hardlink count changed")
    stream = _fs_call(fs, "open", path, "rb")
    primary_error = None
    try:
        descriptor = _api_call("stream.fileno", stream.fileno)
        opened = _fs_call(fs, "fstat", descriptor)
        _reject_alias(opened)
        if _identity(opened) != identity or opened.st_nlink != links:
            raise ValueError("Open handle differs from owned file identity")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_preserving(stream, primary_error)
    after = _fs_call(fs, "lstat", path)
    _reject_alias(after)
    if _identity(after) != identity or after.st_nlink != links:
        raise ValueError("Owned file changed during identity verification")


def _exception(error, operation):
    return {
        "operation": operation, "type": type(error).__name__,
        "module": type(error).__module__, "message": str(error),
        "repr": repr(error), "args_repr": repr(error.args),
        "errno": getattr(error, "errno", None),
        "winerror": getattr(error, "winerror", None),
        "strerror": getattr(error, "strerror", None),
        "filename": str(error.filename) if getattr(error, "filename", None) is not None else None,
        "filename2": str(error.filename2) if getattr(error, "filename2", None) is not None else None,
        "actual_api": getattr(error, "_writer_probe_actual_api", None),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def _timed(operation, action):
    row = {"operation": operation, "start_monotonic_ns": time.monotonic_ns(),
           "start_perf_counter_ns": time.perf_counter_ns()}
    try:
        action()
    except BaseException as error:
        end_perf, end_mono = time.perf_counter_ns(), time.monotonic_ns()
        row.update(ok=False, exception=_exception(error, operation))
    else:
        end_perf, end_mono = time.perf_counter_ns(), time.monotonic_ns()
        row["ok"] = True
    row["end_perf_counter_ns"], row["end_monotonic_ns"] = end_perf, end_mono
    row["duration_ns"] = end_perf - row["start_perf_counter_ns"]
    return row


def _clock_info(name):
    info = time.get_clock_info(name)
    return {key: getattr(info, key) for key in ("implementation", "resolution", "monotonic", "adjustable")}


def _source_identity(fs):
    result = {}
    for role, path in (("probe", Path(__file__).absolute()), ("python_executable", Path(sys.executable).absolute())):
        data = _fs_call(fs, "read_bytes", path)
        result[role] = {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return result


def _cleanup(fs, state):
    """At most one close retry and two verified unlinks; no directory removal."""
    result = {"actions": [], "errors": [], "removed": [], "retained_unverified": []}

    def attempt(label, action):
        row = _timed(label, action)
        result["actions"].append(row)
        if not row["ok"]:
            result["errors"].append(row["exception"])
        return row["ok"]

    stream = state.get("stream")
    if stream is not None:
        if not attempt("cleanup.close", lambda: _api_call("stream.close", stream.close)):
            return result
        state["stream"] = None
    owned = state["owned"]
    result["retained_unverified"] = [str(path) for path in state["created"] if path not in owned]
    if not owned:
        return result
    if not attempt("cleanup.directory_identity", lambda: _check_scratch(fs, state)):
        return result
    # The one intentional staging/final hardlink pair is allowed. Any additional
    # alias, missing path, reparse point, or changed creation identity retains all.
    remaining = len(owned)
    for path, identity in owned.items():
        if not attempt("cleanup.verify." + path.name, lambda p=path, i=identity: _check_file(fs, p, i, remaining)):
            return result
    for path, identity in reversed(tuple(owned.items())):
        def remove_owned(p=path, i=identity):
            _check_scratch(fs, state)
            _check_file(fs, p, i, remaining)
            _fs_call(fs, "unlink", p)
        if not attempt("cleanup.unlink." + path.name, remove_owned):
            return result
        result["removed"].append(str(path))
        remaining -= 1
    return result


def _report_text(report):
    # A fixed point records the actual UTF-8 report size without rewriting it.
    report["reporting"]["size_bytes"] = 0
    while True:
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        size = len(text.encode("utf-8"))
        if size == report["reporting"]["size_bytes"]:
            return text
        report["reporting"]["size_bytes"] = size


def _run_probe(private_root, *, fs=None, token=None, stderr=None):
    """Internal runner; main gates real execution on an explicit --run.

    Tests supply a wholly in-memory filesystem; this helper never launches any
    process and never imports a package outside the Python standard library.
    """
    fs = fs if fs is not None else FileSystem()
    stderr = stderr if stderr is not None else sys.stderr
    token = token if token is not None else uuid.uuid4().hex
    private_root = Path(private_root)
    scratch = private_root / ("probe-" + token)
    report_path = private_root / ("probe-" + token + "-report.json")
    staging, final = scratch / "attempt.json.writing", scratch / "attempt.json"
    state = {"root": private_root, "scratch": scratch, "stream": None, "owned": {}, "created": []}
    report = {
        "schema": "amd-evidence-writer-probe-v1", "base_commit": BASE,
        "utc_anchor": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "classification": "BLOCKED", "private_root": str(private_root),
        "scratch_directory": str(scratch), "report_path": str(report_path),
        "clocks": {name: _clock_info(name) for name in ("monotonic", "perf_counter")},
        "measurement_scope": "Boundary durations include ownership guards. Check primary_api_attempted and last_suboperation before attributing a failure to the named primary API. Exception formatting is outside the measured interval.",
        "source_identity": {}, "operations": [], "first_exception": None,
        "reporting": {"path": str(report_path), "mode": "exclusive ordinary open('x')", "size_bytes": 0},
    }

    def setup():
        if not private_root.is_absolute() or not token.isascii() or not token.isalnum():
            raise ValueError("Absolute private root and alphanumeric fresh token required")
        # Reject redirected ancestors before creating anything. No resolve(),
        # mkdir(parents=True), existing scratch reuse, or historical file access.
        for path in reversed((private_root, *private_root.parents)):
            _check_directory(fs, path)
        state["root_identity"] = _check_directory(fs, private_root)
        report["root_creation_identity"] = state["root_identity"]
        report["source_identity"] = _source_identity(fs)
        _fs_call(fs, "mkdir", scratch)  # Exclusive reservation; never exist_ok.
        state["directory_identity"] = _check_directory(fs, scratch)
        report["scratch_creation_identity"] = state["directory_identity"]

    report["setup"] = _timed("setup", setup)
    if not report["setup"]["ok"]:
        report["first_exception"] = report["setup"]["exception"]

    current = {}

    def substep(name, action, primary=False):
        current["last_suboperation"] = name
        if primary:
            current["primary_api_attempted"] = True
        return action()

    def open_staging():
        state["stream"] = substep("staging.open('x')", lambda: _fs_call(fs, "open", staging, "x", encoding="utf-8"), True)
        state["created"].append(staging)
        descriptor = substep("staging.fileno", lambda: _api_call("stream.fileno", state["stream"].fileno))
        opened = substep("staging.fstat", lambda: _fs_call(fs, "fstat", descriptor))
        _reject_alias(opened)
        identity = _identity(opened)
        named = substep("staging.lstat", lambda: _fs_call(fs, "lstat", staging))
        _reject_alias(named)
        if _identity(named) != identity or opened.st_nlink != 1 or named.st_nlink != 1:
            raise ValueError("Exclusive staging open did not create one unaliased file")
        state["owned"][staging] = identity
        report["staging_creation_identity"] = identity

    payload = {"probe": "evidence writer only", "base_commit": BASE, "token": token}

    def write_json():
        substep("json.dump", lambda: _api_call("json.dump", lambda: json.dump(payload, state["stream"], indent=2, allow_nan=False)), True)
        substep("trailing_newline.write", lambda: _api_call("stream.write", lambda: state["stream"].write("\n")))

    def close_staging():
        substep("staging.close", lambda: _api_call("stream.close", state["stream"].close), True)
        state["stream"] = None

    def link_final():
        substep("scratch_ownership_guards", lambda: _check_scratch(fs, state))
        identity = state["owned"][staging]
        substep("staging_ownership_guards", lambda: _check_file(fs, staging, identity, 1))
        report["staging_size_bytes"] = substep("staging.lstat", lambda: _fs_call(fs, "lstat", staging)).st_size
        substep("os.link", lambda: _fs_call(fs, "link", staging, final), True)  # No replace.
        state["created"].append(final)
        named = substep("final.lstat", lambda: _fs_call(fs, "lstat", final))
        _reject_alias(named)
        if _identity(named) != identity:
            raise ValueError("Published hardlink does not match the created source")
        state["owned"][final] = identity

    def readback():
        substep("scratch_ownership_guards", lambda: _check_scratch(fs, state))
        substep("final_ownership_guards", lambda: _check_file(fs, final, state["owned"][final], 2))
        stream = substep("final.open('rb')", lambda: _fs_call(fs, "open", final, "rb"), True)
        primary_error = None
        try:
            raw = substep("final.read", lambda: _api_call("stream.read", lambda: stream.read(65537), (65537,)))
        except BaseException as error:
            primary_error = error
            raise
        finally:
            substep("final_readback.close", lambda: _close_preserving(stream, primary_error))
        if len(raw) > 65536 or json.loads(raw.decode("utf-8")) != payload:
            raise ValueError("Published JSON readback differs from the written payload")
        expected = (json.dumps(payload, indent=2, allow_nan=False) + "\n").replace("\n", os.linesep).encode("utf-8")
        if raw != expected:
            raise ValueError("Published bytes differ from indent=2 JSON plus newline")
        report["final_size_bytes"] = len(raw)
        report["payload_sha256"] = hashlib.sha256(raw).hexdigest()

    forward = (
        ("exclusive_staging_open", open_staging), ("json_write", write_json),
        ("flush", lambda: substep("staging.flush", lambda: _api_call("stream.flush", state["stream"].flush), True)),
        ("fsync", lambda: substep("os.fsync", lambda: _fs_call(fs, "fsync", _api_call("stream.fileno", state["stream"].fileno)), True)),
        ("close", close_staging), ("link_no_overwrite", link_final),
        ("final_readback", readback),
    )
    if report["first_exception"] is None:
        for label, action in forward:
            current.clear()
            current["primary_api_attempted"] = False
            row = _timed(label, action)
            row.update(current)
            report["operations"].append(row)
            if not row["ok"]:
                row["exception"]["suboperation"] = current.get("last_suboperation")
                report["first_exception"] = row["exception"]
                break

    def cleanup():
        report["cleanup"] = _cleanup(fs, state)
    cleanup_row = _timed("cleanup", cleanup)
    errors = report.get("cleanup", {}).get("errors", [])
    if errors:
        cleanup_row.update(ok=False, exception=errors[0])
    report["operations"].append(cleanup_row)
    if not cleanup_row["ok"] and report["first_exception"] is None:
        report["first_exception"] = cleanup_row["exception"]
    report["classification"] = "PASS" if report["first_exception"] is None else "BLOCKED"

    try:
        if "root_identity" not in state:
            raise ValueError("Private root was not admitted; report goes to stderr only")
        for path in reversed((private_root, *private_root.parents)):
            _check_directory(fs, path)
        _check_directory(fs, private_root, state["root_identity"])
        text = _report_text(report)
        # Deliberately no hardlink publication, staging, retry, or recursive save.
        stream = _fs_call(fs, "open", report_path, "x", encoding="utf-8", newline="\n")
        primary_error = None
        try:
            _api_call("stream.write", lambda: stream.write(text))
            _api_call("stream.flush", stream.flush)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            _close_preserving(stream, primary_error)
        report["reporting"]["write_succeeded"] = True  # CLI receipt, after close.
    except BaseException as error:
        detail = _exception(error, "report_write")
        report["reporting"].update(write_succeeded=False, error=detail)
        if report["first_exception"] is None:
            report["first_exception"] = detail
        report["classification"] = "BLOCKED"
        try:
            stderr.write(_report_text(report))
            stderr.flush()
        except BaseException as fallback_error:
            report["reporting"]["stderr_fallback_error"] = _exception(fallback_error, "stderr_fallback")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute exactly one fresh filesystem probe")
    parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT, help="existing new private evidence root")
    args = parser.parse_args(argv)
    if not args.run:
        parser.print_help()
        return 0
    report = _run_probe(args.private_root)
    print(json.dumps({"classification": report["classification"],
                      "report_path": report["report_path"],
                      "report_write_succeeded": report["reporting"].get("write_succeeded", False),
                      "first_failing_operation": (report["first_exception"] or {}).get("operation")}, allow_nan=False))
    return 0 if report["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
