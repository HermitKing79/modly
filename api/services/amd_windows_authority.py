"""Private, in-process Windows invocation registry; no serialized issuer.

Only the trusted runner calls the private seal after its local lifecycle checks.
This is not protection against hostile code modifying this Python interpreter.
"""
import json
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
import threading
import time
from types import MappingProxyType

from services.amd_completed_result import FilePin, _canonical, _identity, _need, _plain

_LOCK = threading.RLock()
_AUTHORITIES = {}


def _wsl_client_path():
    return _plain(Path(os.environ['SystemRoot']) / 'System32' / 'wsl.exe')


@dataclass(frozen=True)
class _WslClientSnapshot:
    """Only the exact OS client permits its original Windows servicing hardlink."""
    path: Path
    sha256: str
    bytes: int
    identity: tuple

    def document(self):
        return dict(path=str(self.path), sha256=self.sha256, bytes=self.bytes, identity=list(self.identity))

    def recheck(self):
        _need(_wsl_client_snapshot(self.path) == self, 'Original WSL client identity changed')


def _wsl_client_snapshot(path):
    path = _plain(path)
    _need(path == _wsl_client_path(), 'Only the exact System32 WSL client has the OS link exception')
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_nlink in (1, 2)
          and 0 < before.st_size <= 64*1024**2, 'Bounded original WSL OS client required')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        _need(_identity(before) == _identity(opened), 'WSL client replaced during open')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        _need(_identity(os.fstat(stream.fileno())) == _identity(opened)
              and _identity(_plain(path).stat()) == _identity(before), 'WSL client changed during read')
    return _WslClientSnapshot(path, digest, before.st_size, _identity(before))


def reopen_sources(binding):
    """Reauthenticate the private sealed source list, preserving the narrow OS pin."""
    from services.amd_upstream_job import _snapshot
    rows = tuple(_wsl_client_snapshot(Path(row['path']))
        if binding.get('backendkind') == 'wsl' and Path(row['path']) == _wsl_client_path()
        else _snapshot(Path(row['path']), 64*1024**2, expected=row['sha256'])
        for row in binding['sources'])
    _need([row.document() for row in rows] == binding['sources'], 'Original supervised source identity changed')
    return rows


def _guest_entrypoint(value):
    _need(type(value) is str and value.startswith('/mnt/c/') and '\\' not in value,
          'Exact C-volume guest entrypoint required')
    path = _plain('C:/' + value[7:])
    _need(value == '/mnt/c/' + path.as_posix()[3:], 'Canonical guest entrypoint required')
    return path


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def document(value):
    """A detached JSON copy of private immutable evidence for the host receipt."""
    if isinstance(value, (dict, MappingProxyType)):
        return {key: document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [document(item) for item in value]
    return value


def _command_document(command):
    return dict(stage_id=command.stage_id, kind=command.kind, argv=list(command.argv),
                cwd=str(command.cwd), context=dict(command.context), timeout_s=command.timeout_s,
                control_nonce=command.control_nonce)


def _policy(command):
    from services import amd_stage_process as stage
    actual = (stage.MEMORY_LIMIT, stage.RAM_FLOOR, stage.DISK_FLOOR)
    _need(actual == (6 * 1024**3, 12 * 1024**3, 2 * 1024**3), 'Original Windows resource policy required')
    policy = (*actual, command.timeout_s)
    if command.kind == 'wsl':
        from services import amd_wsl_session as session
        linux = (stage.DISTRO, stage.WSL_EXECUTABLE, stage.WSL_REAL_EXECUTABLE, session.MEMORY_LIMIT)
        _need(linux == ('Modly-AMD-Ubuntu-24.04', '/usr/bin/python3', '/usr/bin/python3.12', 6*1024**3),
              'Original WSL execution and resource policy required')
        policy += linux
    return policy


class _Invocation:
    __slots__ = ('_job', '_ticket', '_runner', '_token', '_original', '_original_bytes',
                 '_command', '_command_bytes', '_cancel', '_context_bytes', '_predecessor',
                 '_sources', '_policy', '_phase', '_deadline', '_authority', '_summary', '_receipt', '_evidence_dir', '_interpreter')

    def __init__(self, job, ticket, runner, original, command, sources):
        self._job, self._ticket, self._runner = job, ticket, runner
        self._token = object()
        self._original, self._original_bytes = original, _canonical(_command_document(original))
        self._command, self._command_bytes = command, _canonical(_command_document(command))
        self._cancel = job._cancel_event
        self._interpreter = sys.executable
        self._evidence_dir = _plain(runner.evidence_dir)
        self._context_bytes = _canonical(job._context.to_dict())
        self._predecessor = ticket.predecessor_sha256
        self._sources = sources
        self._policy = _policy(command)
        self._phase = 'bound'
        self._deadline = time.monotonic() + command.timeout_s
        self._authority = self._summary = self._receipt = None

    @property
    def authority(self):
        return self._authority

    @property
    def kind(self):
        return self._command.kind

    @property
    def wsl_client_path(self):
        _need(self.kind == 'wsl', 'WSL invocation required')
        return next(row.path for row in self._sources if type(row) is _WslClientSnapshot)

    def fail(self):
        with _LOCK:
            if self._authority is not None:
                _AUTHORITIES.pop(self._authority, None)
            self._authority = None
            self._phase = 'failed'
            self._job._blocked = True

    def check_cancel(self):
        try:
            _need(self._phase in ('bound', 'running', 'sealed'), 'Windows invocation is permanently consumed or blocked')
            _need(self._job._cancel_event is self._cancel and self._cancel is not None,
                  'Exact shared product cancellation source required')
            self._job._check(self._cancel)
            _need(time.monotonic() < self._deadline, 'Windows invocation deadline expired')
        except BaseException:
            self.fail()
            raise

    def recheck(self):
        try:
            self.check_cancel()
            _need(self._job._pending is self._ticket and self._job._bound_ticket is self._ticket,
                  'Exact pending Windows ticket required')
            _need(_canonical(self._job._context.to_dict()) == self._context_bytes
                  and self._ticket.predecessor_sha256 == self._predecessor,
                  'Windows invocation product context or predecessor changed')
            _need(_canonical(_command_document(self._original)) == self._original_bytes
                  and _canonical(_command_document(self._command)) == self._command_bytes,
                  'Host-bound Windows command changed')
            _need(_plain(self._command.cwd) == self._command.cwd and _policy(self._command) == self._policy,
                  'Windows invocation path or resource policy changed')
            _need(_plain(self._runner.evidence_dir) == self._evidence_dir
                  and self._interpreter == sys.executable
                  and self._command.argv[0] == (sys.executable if self.kind == 'windows' else '/usr/bin/python3'),
                  'Bound runner directory or interpreter selection changed')
            _need(not self._runner.unsettled, 'Unsettled Windows runner cannot issue authority')
            self._job._reauthenticate()
            for snapshot in self._sources:
                snapshot.recheck()
            if self._receipt is not None:
                self._receipt.recheck()
            self.check_cancel()
        except BaseException:
            self.fail()
            raise

    def start(self, runner):
        with _LOCK:
            try:
                _need(self._phase == 'bound' and runner is self._runner
                      and runner._bindings.get(self._token) is self, 'One registered Windows execution required')
                self.recheck()
                stage = self._job._stages[len(self._job._accepted)]
                _need(all(not spec.path.exists() and not spec.path.is_symlink() for spec in stage.outputs),
                      'Windows outputs exist before invocation start')
                self._phase = 'running'
                return self._command, self._cancel
            except BaseException:
                self.fail()
                raise

    def seal(self, evidence):
        """Called only by the private runner after local native settlement checks."""
        from services.amd_upstream_job import _snapshot
        with _LOCK:
            try:
                _need(self._phase == 'running' and self._authority is None, 'One running invocation may seal once')
                self.recheck()
                _need(type(evidence) is dict, 'Private runner-local lifecycle evidence required')
                expected = dict(terminal='succeeded', returncode=0, bootstrap_returncode=0,
                                natural=True, settled=True, forced=False, diagnostic_errors=0)
                if self.kind == 'wsl':
                    expected.update(backendkind='wsl', linux_client_returncode=0, linux_settled=True)
                else:
                    _need(evidence.get('backendkind', 'windows') == 'windows', 'Foreign backend lifecycle evidence')
                _need(all(evidence.get(key) == value and type(evidence.get(key)) is type(value)
                          for key, value in expected.items()), 'Successful unforced Windows lifecycle required')
                receipt_path = _plain(evidence['receipt_path'])
                _need(receipt_path == self._evidence_dir / self._ticket.stage_id / 'receipt.json',
                      'Exact original runner receipt path required')
                self._receipt = _snapshot(receipt_path, 4 * 1024**2, self._ticket.created_ns)
                summary = dict(binding=dict(stage_id=self._ticket.stage_id, nonce=self._ticket.nonce,
                    product_context=self._job._context.to_dict(), predecessor_sha256=self._predecessor,
                    command=json.loads(self._command_bytes),
                    sources=[row.document() for row in self._sources], deadline_monotonic_s=self._deadline,
                    runner_evidence_directory=str(self._evidence_dir),
                    resource_policy=dict(windows_process_bytes=self._policy[0], windows_job_bytes=self._policy[0],
                        available_ram_floor_bytes=self._policy[1], free_disk_floor_bytes=self._policy[2],
                        timeout_s=self._policy[3])),
                    evidence=json.loads(_canonical(evidence)))
                if self.kind == 'wsl':
                    summary['binding'].update(backendkind='wsl', linux_policy=dict(
                        distro=self._policy[4], executable=self._policy[5], real_executable=self._policy[6],
                        uid=1000, user='modly', rlimit_as_bytes=[self._policy[7]]*2,
                        aggregate_rss_limit_bytes=self._policy[7], windows_client=str(self.wsl_client_path)))
                _need(len(_canonical(summary)) <= 4 * 1024**2, 'Windows authority summary exceeds bound')
                self._summary = _freeze(summary)
                self.check_cancel()
                self._phase = 'sealed'
                self._authority = object()
                _AUTHORITIES[self._authority] = self
                return self._authority
            except BaseException:
                self.fail()
                raise


def bind(job, ticket, runner, command, *, backendkind='windows'):
    from services import amd_stage_process as stage
    from services.amd_upstream_job import UpstreamJob, _snapshot
    _need(type(job) is UpstreamJob and ticket is job._pending and ticket is not None,
          'Exact host-owned pending ticket required')
    _need(type(runner) is stage.OwnedStageRunner and type(command) is stage.StageCommand,
          'Original Windows runner and command required')
    _need(backendkind in ('windows', 'wsl') and command.kind == backendkind and command.stage_id == ticket.stage_id
          and dict(command.context) == job._context.to_dict(), 'Exact Windows stage/product context required')
    _need(job._cancel_event is not None and callable(getattr(job._cancel_event, 'is_set', None)),
          'Shared product cancellation event required')
    _need(command.control_nonce in (None, ticket.nonce), 'Foreign Windows control nonce')
    _need(len(command.argv) >= 5 and command.argv[0] == (sys.executable if backendkind == 'windows' else '/usr/bin/python3')
          and command.argv[1:4] == ('-B', '-I', '-S'), 'Explicit original interpreter and isolated stdlib entrypoint required')
    entrypoint = _plain(command.argv[4]) if backendkind == 'windows' else _guest_entrypoint(command.argv[4])
    _need(any(row.path == entrypoint for _, row in job._sources), 'Entrypoint must be an exact host-selected source pin')
    cwd = _plain(command.cwd)
    if backendkind == 'wsl':
        _need(cwd.drive.upper() == 'C:', 'WSL cwd must remain on the original C volume')
    _need(cwd.is_dir() and not (_plain(runner.evidence_dir) / ticket.stage_id).exists(),
          'Existing canonical cwd and fresh runner stage required')
    _need(type(getattr(runner, '_bindings', None)) is dict and not runner.unsettled,
          'Fresh original runner binding registry required')
    job._check()
    job._reauthenticate()
    fixed = tuple(_snapshot(path, limit) for path, limit in (
        (_plain(stage.__file__), 4 * 1024**2),
        (_plain(Path(stage.__file__).with_name('amd_stage_bootstrap.py')), 4 * 1024**2),
        (_plain(__file__), 4 * 1024**2),
        (_plain(Path(__file__).with_name('amd_upstream_job.py')), 4 * 1024**2),
        (_plain(sys.executable), 64 * 1024**2)))
    if backendkind == 'wsl':
        fixed += (_snapshot(_plain(Path(stage.__file__).with_name('amd_wsl_session.py')), 4*1024**2),
                  _wsl_client_snapshot(_wsl_client_path()))
    detached = stage.StageCommand(command.stage_id, backendkind, tuple(command.argv), cwd,
        json.loads(_canonical(dict(command.context))), command.timeout_s, control_nonce=ticket.nonce)
    invocation = _Invocation(job, ticket, runner, command, detached, fixed)
    runner._bindings[invocation._token] = invocation
    return invocation._token


def consume(authority, job, ticket, runner, *, backendkind='windows'):
    with _LOCK:
        _need(type(authority) is object, 'Opaque registered Windows authority required')
        invocation = _AUTHORITIES.get(authority)
        _need(type(invocation) is _Invocation, 'Unregistered or consumed Windows authority')
        # An unrelated owner cannot consume the rightful invocation as a denial
        # of service. Matching requests consume before mutable-file rechecks.
        _need(invocation._job is job and invocation._ticket is ticket and invocation._runner is runner
              and invocation.kind == backendkind,
              'Windows authority belongs to another owner, runner or ticket')
        _AUTHORITIES.pop(authority)
        try:
            _need(invocation._authority is authority and invocation._phase == 'sealed', 'Windows authority is no longer sealed')
            invocation.recheck()
            invocation._phase = 'consumed'
            invocation._authority = None
            receipt = invocation._receipt
            return invocation._summary, FilePin(receipt.path, receipt.sha256)
        except BaseException:
            invocation.fail()
            raise
