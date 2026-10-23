"""Private, host-selected AMD stages. Importing this module launches nothing.

Windows Jobs contain the bootstrap and every Windows descendant. WSL additionally
requires the Linux broker's terminal receipt; a Windows client exit proves no
Linux settlement. Callers retain an unverified runner and must block the job.
"""
import ctypes
from ctypes import wintypes as wt
from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import threading
import traceback
import uuid

MEMORY_LIMIT = 6 * 1024**3
RAM_FLOOR = 12 * 1024**3
DISK_FLOOR = 2 * 1024**3
DISTRO = "Modly-AMD-Ubuntu-24.04"
_HERE = Path(__file__).resolve().parent  # Diagnostic copy only; production remains pinned.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


@dataclass(frozen=True)
class StageCommand:
    stage_id: str
    kind: str
    argv: tuple[str, ...]
    cwd: Path
    context: dict
    timeout_s: float = 180.0

    def __post_init__(self):
        if not isinstance(self.stage_id, str) or not _NAME.fullmatch(self.stage_id):
            raise ValueError("A single safe stage identifier is required")
        if self.kind not in ("windows", "wsl"):
            raise ValueError("Stage kind must be windows or wsl")
        if not isinstance(self.argv, tuple) or not self.argv or any(
                not isinstance(arg, str) or not arg or "\0" in arg for arg in self.argv):
            raise ValueError("An explicit host-pinned argument tuple is required")
        if not isinstance(self.context, dict) or not isinstance(self.context.get("nonce"), str):
            raise ValueError("A completion context with nonce is required")
        if isinstance(self.timeout_s, bool) or not 0 < self.timeout_s <= 180:
            raise ValueError("Stage deadline must be within 180 seconds")


class _Basic(ctypes.Structure):
    _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64), ("Flags", wt.DWORD),
                ("MinWorkingSet", ctypes.c_size_t), ("MaxWorkingSet", ctypes.c_size_t),
                ("ActiveLimit", wt.DWORD), ("Affinity", ctypes.c_size_t),
                ("Priority", wt.DWORD), ("Scheduling", wt.DWORD)]


class _IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]


class _Limits(ctypes.Structure):
    _fields_ = [("Basic", _Basic), ("Counters", _IO), ("ProcessMemory", ctypes.c_size_t),
                ("JobMemory", ctypes.c_size_t), ("PeakProcessMemory", ctypes.c_size_t),
                ("PeakJobMemory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("User", "Kernel", "PeriodUser", "PeriodKernel")] + [
        (name, wt.DWORD) for name in ("Faults", "Total", "Active", "Terminated")]


class _OwnedJob:
    """An unnamed non-inheritable Job, without either breakaway flag."""
    def __init__(self):
        _require_windows()
        if ctypes.sizeof(_Limits) != 144 or ctypes.sizeof(_Accounting) != 48:
            raise RuntimeError("Unsupported Windows Job ABI")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.handle = None
        self.identities = {}
        self.creation_identities = {}
        self.job_owner = uuid.uuid4().hex
        self.phase = 'job-create'
        self.native_events = []
        definitions = {
            "CreateJobObjectW": ([ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
            "SetInformationJobObject": ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
            "AssignProcessToJobObject": ([wt.HANDLE, wt.HANDLE], wt.BOOL),
            "QueryInformationJobObject": ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p], wt.BOOL),
            "TerminateJobObject": ([wt.HANDLE, wt.UINT], wt.BOOL),
            "CloseHandle": ([wt.HANDLE], wt.BOOL),
            "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
            "GetProcessTimes": ([wt.HANDLE] + [ctypes.POINTER(wt.FILETIME)] * 4, wt.BOOL),
            "QueryFullProcessImageNameW": ([wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)], wt.BOOL),
            "IsProcessInJob": ([wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)], wt.BOOL),
            "GetExitCodeProcess": ([wt.HANDLE, ctypes.POINTER(wt.DWORD)], wt.BOOL),
        }
        for name, (arguments, result) in definitions.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self._call('CreateJobObjectW', None, None)
        limits = _Limits()
        limits.Basic.Flags = 0x2000 | 0x100 | 0x200  # kill-on-close, process + job memory.
        limits.ProcessMemory = limits.JobMemory = MEMORY_LIMIT
        try:
            self._call('SetInformationJobObject', self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), category=9)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.amd_creation_cleanup = _error_detail(cleanup, 'cleanup.job-create')
            raise

    def _alive(self, handle):
        # Diagnostic only: never substitutes for membership or creation checks.
        if handle is None:
            return {'alive': None, 'reason': 'no process handle'}
        value = wt.DWORD()
        try:
            ctypes.set_last_error(0)
            ok = self.api.GetExitCodeProcess(handle, ctypes.byref(value))
            code = ctypes.get_last_error() if not ok else None
        except Exception as error:
            return {'alive': None, 'probe_exception': repr(error)}
        return {'alive': value.value == 259 if ok else None,
                'exit_code': value.value if ok else None, 'probe_winerror': code}

    def _call(self, name, *args, pid=None, process_handle=None, category=None, identity_current=False):
        before = self._alive(process_handle)
        caller = sys._getframe(1)
        row = {'api': name, 'phase': self.phase, 'job_owner': self.job_owner,
               'job_handle': self.handle, 'pid': pid, 'category': category,
               'process_handle': process_handle, 'closing_handle': args[0] if name == 'CloseHandle' else None,
               'access_mask': args[0] if name == 'OpenProcess' else None,
               'last_authenticated_identity': self.creation_identities.get(pid),
               'authenticated_identity': self.creation_identities.get(pid) if identity_current else None,
               'identity_from_current_handle': identity_current,
               'python_caller': {'file': caller.f_code.co_filename, 'function': caller.f_code.co_name,
                                 'line': caller.f_lineno},
               'alive_before': before, 'utc': _utc(), 'monotonic_s': time.monotonic(),
               'thread_id': threading.get_ident()}
        ctypes.set_last_error(0)
        value = getattr(self.api, name)(*args)
        code = ctypes.get_last_error() if not value else None  # Before ANY other API/probe.
        verified = type(code) is int and code > 0
        row.update(ok=bool(value), winerror=code, native_error_verified=verified)
        self.native_events.append(row)
        del self.native_events[:-256]
        if not value:
            row['alive_after'] = self._alive(process_handle)
            error = ctypes.WinError(code) if verified else RuntimeError(name + ' failed without a Windows error code')
            error.amd_native_failure = row
            error.amd_recent_native_events = list(self.native_events)
            raise error
        return value

    def attach(self, proc):
        self._call('AssignProcessToJobObject', self.handle, int(proc._handle), pid=proc.pid, process_handle=int(proc._handle))
        self._identity(proc.pid, int(proc._handle))

    def _identity(self, pid, handle):
        if not self.handle or not handle:
            raise RuntimeError('Current process and exact private Job handles are required')
        member = wt.BOOL()
        self._call('IsProcessInJob', handle, self.handle, ctypes.byref(member), pid=pid, process_handle=handle)
        if not member.value:
            raise RuntimeError("Process membership in private Job could not be verified")
        values = [wt.FILETIME() for _ in range(4)]
        self._call('GetProcessTimes', handle, *(ctypes.byref(value) for value in values), pid=pid, process_handle=handle)
        created = values[0].dwLowDateTime | values[0].dwHighDateTime << 32
        self.creation_identities[pid] = {'pid': pid, 'creation_filetime': created, 'job_membership_verified': True,
                                         'observed_utc': _utc(), 'queried_process_handle': int(handle)}
        status = wt.DWORD()
        self._call('GetExitCodeProcess', handle, ctypes.byref(status), pid=pid, process_handle=handle, identity_current=True)
        observed = _utc()
        exit_observation = {'api': 'GetExitCodeProcess', 'succeeded': True, 'exit_code': status.value,
                            'termination_verified': status.value != 259, 'observed_utc': observed,
                            'process_handle': int(handle)}
        if status.value != 259:
            cached = self.identities.get((pid, created), {})
            source = cached.get('image_authentication', {})
            if not (cached.get('pid') == pid and cached.get('creation_filetime') == created
                    and cached.get('job_membership_verified') is True
                    and cached.get('job_owner') == self.job_owner and cached.get('job_handle') == int(self.handle)
                    and isinstance(cached.get('image'), str) and cached['image']
                    and source.get('api') == 'QueryFullProcessImageNameW' and source.get('succeeded') is True
                    and source.get('pid') == pid and source.get('creation_filetime') == created
                    and source.get('job_owner') == self.job_owner and source.get('job_handle') == int(self.handle)
                    and source.get('image') == cached['image'] and isinstance(source.get('observed_utc'), str)
                    and source['observed_utc']):
                raise RuntimeError('Terminated member lacks prior native image authentication for this exact Job and creation identity')
            self.identities[(pid, created)] = dict(cached, observed_utc=observed,
                observed_alive={'alive': False, 'exit_code': status.value, 'probe_winerror': None},
                exit_observation=exit_observation, identity_source='verified-retired-cache',
                retired_identity_reuse={'job_owner': self.job_owner, 'job_handle': int(self.handle),
                    'pid': pid, 'creation_filetime': created, 'process_handle': int(handle),
                    'source_query_utc': source['observed_utc'], 'observed_utc': observed})
            return
        # 259 is ambiguous and never authorizes cache reuse. Native image
        # authentication remains mandatory; any failure is retained unchanged.
        size = wt.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        self._call('QueryFullProcessImageNameW', handle, 0, image, ctypes.byref(size), pid=pid, process_handle=handle, identity_current=True)
        authenticated = _utc()
        self.identities[(pid, created)] = {"pid": pid, "creation_filetime": created,
                                          "image": image.value, "job_membership_verified": True,
            'job_owner': self.job_owner, 'job_handle': int(self.handle), 'identity_source': 'native-image-query',
            'image_authentication': {'api': 'QueryFullProcessImageNameW', 'succeeded': True,
                'pid': pid, 'creation_filetime': created, 'image': image.value,
                'job_owner': self.job_owner, 'job_handle': int(self.handle),
                'process_handle': int(handle), 'observed_utc': authenticated},
            'exit_observation': exit_observation, 'retired_identity_reuse': None,
            'observed_utc': authenticated, 'observed_alive': self._alive(handle)}

    def snapshot(self):
        accounting, limits = _Accounting(), _Limits()
        for category, value in ((1, accounting), (9, limits)):
            self._call('QueryInformationJobObject', self.handle, category, ctypes.byref(value), ctypes.sizeof(value), None, category=category)
        # A bounded list is sufficient for this small, pinned stage chain. Overflow blocks it.
        class Pids(ctypes.Structure):
            _fields_ = [("Assigned", wt.DWORD), ("Count", wt.DWORD), ("Pids", ctypes.c_size_t * 256)]
        ids = Pids()
        self._call('QueryInformationJobObject', self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None, category=3)
        for pid in ids.Pids[:ids.Count]:
            # A failed open is not proof of exit. Retain the same limited mask,
            # attribute the failure, and fail closed instead of silently skipping it.
            handle = self._call('OpenProcess', 0x1000, False, pid, pid=pid)
            try:
                self._identity(pid, handle)
            finally:
                primary = sys.exc_info()[1]
                try:
                    self._call('CloseHandle', handle, pid=pid)
                except BaseException as cleanup:
                    if primary is not None:
                        primary.amd_handle_cleanup = _error_detail(cleanup, 'cleanup.process-handle')
                    else:
                        raise
        return {"active": accounting.Active, "total_processes": accounting.Total,
                "peak_process_bytes": limits.PeakProcessMemory, "peak_job_bytes": limits.PeakJobMemory,
                "process_identities": list(self.identities.values()), "job_owner": self.job_owner,
                "enumerated_pids": list(ids.Pids[:ids.Count])}

    def terminate(self):
        self._call('TerminateJobObject', self.handle, 1)

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self._call('CloseHandle', handle)


def _error_detail(error, phase):
    """Private evidence only. Never infer native last-error from a Python error."""
    return {'type': type(error).__name__, 'message': str(error), 'repr': repr(error),
            'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None),
            'filename': getattr(error, 'filename', None), 'filename2': getattr(error, 'filename2', None),
            'phase': phase, 'utc': _utc(), 'monotonic_s': time.monotonic(),
            'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
            'python_frames': [{'file': row.filename, 'line': row.lineno, 'function': row.name}
                              for row in traceback.extract_tb(error.__traceback__)],
            'native': getattr(error, 'amd_native_failure', None),
            'recent_native_events': getattr(error, 'amd_recent_native_events', []),
            'creation_cleanup': getattr(error, 'amd_creation_cleanup', None),
            'handle_cleanup': getattr(error, 'amd_handle_cleanup', None)}


def _require_windows():
    if sys.platform != "win32":
        raise RuntimeError("AMD stage supervision requires Windows")


class _HostResourceError(RuntimeError):
    def __init__(self, observation):
        self.observation = dict(observation)
        super().__init__("Stage host resource floor crossed: " + json.dumps({**self.observation,
            "required_available_ram_bytes": RAM_FLOOR, "required_free_disk_bytes": DISK_FLOOR}))


def _host_health(path):
    class Memory(ctypes.Structure):
        _fields_ = [("length", wt.DWORD), ("load", wt.DWORD)] + [
            (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                                "virtual_total", "virtual_available", "extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    api.GlobalMemoryStatusEx.restype = wt.BOOL
    ctypes.set_last_error(0)
    if not api.GlobalMemoryStatusEx(ctypes.byref(memory)):
        code = ctypes.get_last_error()
        error = ctypes.WinError(code) if code else RuntimeError('GlobalMemoryStatusEx failed without a Windows error code')
        error.amd_native_failure = {'api': 'GlobalMemoryStatusEx', 'winerror': code,
                                    'native_error_verified': bool(code), 'pid': None}
        raise error
    observation = {"available_ram_bytes": memory.available, "free_disk_bytes": shutil.disk_usage(path).free}
    if observation["available_ram_bytes"] < RAM_FLOOR or observation["free_disk_bytes"] < DISK_FLOOR:
        raise _HostResourceError(observation)
    return observation


def _guest_path(value):
    text = str(value).replace("\\", "/")
    if text.startswith("/") and not text.startswith("//"):
        return text
    if not re.match(r"[Cc]:/", text):
        raise ValueError("WSL shared files must be explicit C: paths")
    return "/mnt/c/" + text[3:]


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _linux_ready(event):
    if not isinstance(event, dict):
        return False
    leader, broker = event.get("leader", {}), event.get("broker", {})
    return (event.get("uid") == 1000 and leader.get("uid") == broker.get("uid") == 1000
            and type(leader.get("pid")) is int and leader["pid"] > 0
            and leader.get("session") == leader["pid"] == leader.get("pgrp")
            and type(broker.get("pid")) is int and broker["pid"] > 0
            and leader["pid"] != broker["pid"]
            and leader.get("start_ticks", 0) > 0 and broker.get("start_ticks", 0) > 0
            and event.get("memory", {}).get("rlimit_as_bytes") == MEMORY_LIMIT)


def _linux_settled(terminal):
    return (_linux_ready(terminal) and terminal.get("linux_settled") is True
            and terminal.get("reaped_all") is True and terminal.get("owned_remaining") == []
            )


def _timing_record(stats, phase, began_ns, ended_ns):
    row = stats.setdefault(phase, dict(count=0, total_ns=0, max_ns=0,
                                     first_begin_ns=began_ns, last_end_ns=ended_ns))
    elapsed = ended_ns - began_ns
    row['count'] += 1
    row['total_ns'] += elapsed
    row['max_ns'] = max(row['max_ns'], elapsed)
    row['last_end_ns'] = ended_ns


def _timing_summary(stats):
    return {phase: dict(row) for phase, row in stats.items()}


class OwnedStageRunner:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = Path(evidence_dir)
        self.unsettled = False
        self.last_windows_observation = None

    def run(self, command: StageCommand, cancel_event, tick=None):
        if self.unsettled:
            raise RuntimeError("Prior owned stage settlement is unverified; runner is blocked")
        began = time.monotonic()
        directory = self.evidence_dir / command.stage_id
        directory.mkdir(parents=True, exist_ok=False)
        receipt_path = directory / "receipt.json"
        protocol_path = directory / "protocol.jsonl"
        spec = {"stage_id": command.stage_id, "kind": command.kind, "argv": list(command.argv),
                "cwd": str(command.cwd), "context": json.loads(json.dumps(command.context)),
                "nonce": command.context["nonce"], "timeout_s": command.timeout_s,
                "memory_limit": MEMORY_LIMIT, "stdout_file": str(directory / "stage.stdout.txt"),
                "stderr_file": str(directory / "stage.stderr.txt")}
        result = {"stage_id": command.stage_id, "context": spec["context"], "nonce": spec["nonce"],
                  "terminal": "failed", "returncode": None, "windows_settled": True,
                  "linux_settled": None if command.kind == "windows" else False,
                  "natural": False, "forced": False, "started_utc": _utc(),
                  "limits": {"windows_process_bytes": MEMORY_LIMIT, "windows_job_bytes": MEMORY_LIMIT,
                             "windows_job_caps_linux_memory": False, "timeout_s": command.timeout_s,
                             "available_ram_floor_bytes": RAM_FLOOR, "free_disk_floor_bytes": DISK_FLOOR},
                  "logs": {"stdout": spec["stdout_file"], "stderr": spec["stderr_file"],
                           "protocol": str(protocol_path)}, "receipt_file": str(receipt_path),
                  "observations": [], "windows": {}}
        timing_stats = {}
        result.update(windows_history=[], control_messages=[], diagnostic_errors=[],
                      timing=dict(anchor=dict(before_ns=time.monotonic_ns(), utc_ns=time.time_ns(),
                                              after_ns=time.monotonic_ns()), observer=timing_stats))
        proc = job = None
        attached = cancelled = gate_open = False
        terminal = None
        linux_ready = None
        deadline = began + command.timeout_s
        cursor = 0
        phase = 'normal.initial-admission'
        shutdown = False

        def capture(exc):
            detail = _error_detail(exc, phase)
            detail.update(stage_id=command.stage_id, context=spec['context'],
                          supervisor_job_owner=getattr(job, 'job_owner', None),
                          bootstrap_pid=getattr(proc, 'pid', None), during_shutdown=shutdown)
            result['diagnostic_errors'].append(detail)
            return detail

        def observe():
            nonlocal phase
            observation_begin_ns = time.monotonic_ns()
            timing_phase = 'cleanup.observe' if shutdown else 'normal.observe'
            try:
                phase = ('cleanup' if shutdown else 'normal') + '.observe'
                if tick is not None:
                    tick()
                try:
                    health = _host_health(directory)
                except _HostResourceError as exc:
                    result["observations"].append({"elapsed_s": time.monotonic() - began,
                                                   "admission": "blocked", **exc.observation})
                    raise
                result["observations"].append({"elapsed_s": time.monotonic() - began, **health})
                if job is not None:
                    job.phase = phase
                    result["windows"] = job.snapshot()
                    snapshot = {'utc': _utc(), 'monotonic_s': time.monotonic(), 'elapsed_s': time.monotonic() - began,
                                'phase': phase, **result['windows']}
                    self.last_windows_observation = snapshot
                    result['windows_history'].append(snapshot)
            finally:
                _timing_record(timing_stats, timing_phase, observation_begin_ns, time.monotonic_ns())

        def control(action):
            nonlocal phase
            phase = ('cleanup' if shutdown else 'normal') + '.control.' + action
            entry = {'action': action, 'phase': phase, 'utc': _utc(), 'monotonic_s': time.monotonic(), 'flushed': False, 'dispatch_begin_ns': time.monotonic_ns()}
            result['control_messages'].append(entry)
            proc.stdin.write((json.dumps({"action": action, "stage_id": command.stage_id,
                                          "nonce": spec["nonce"]}) + "\n").encode())
            proc.stdin.flush()
            entry['flushed'] = True
            entry['flush_return_ns'] = time.monotonic_ns()

        def read_events():
            nonlocal cursor, terminal, linux_ready, phase
            phase = ('cleanup' if shutdown else 'normal') + '.read-protocol'
            if protocol_path.stat().st_size > 1024 * 1024:
                raise RuntimeError("Stage protocol exceeded bounded size")
            with protocol_path.open("rb") as stream:
                stream.seek(cursor)
                while True:
                    line = stream.readline(65537)
                    if not line or not line.endswith(b"\n"):
                        break
                    if len(line) > 65536:
                        raise RuntimeError("Oversized stage protocol record")
                    cursor = stream.tell()
                    event = json.loads(line)
                    if (event.get("stage_id"), event.get("nonce"), event.get("context")) != (
                            command.stage_id, spec["nonce"], spec["context"]):
                        raise RuntimeError("Stage protocol context/nonce mismatch")
                    if event.get("event") == "ready" and command.kind == "wsl":
                        if not attached or not _linux_ready(event) or linux_ready is not None:
                            raise RuntimeError("Unverified Linux launch identity")
                        linux_ready = event
                        if cancel_event.is_set():
                            control("CANCEL")
                        else:
                            control("START")
                    if event.get("event") == "terminal":
                        if terminal is not None:
                            raise RuntimeError("Duplicate terminal stage receipt")
                        terminal = event
                    if event.get("event") == "linux-client-exited":
                        result["linux_client_returncode"] = event.get("returncode")

        def linux_settled():
            return (_linux_settled(terminal) and linux_ready is not None
                    and all(terminal.get(key) == linux_ready.get(key) for key in ("broker", "leader", "boot_id")))

        try:
            _require_windows()
            if cancel_event.is_set():
                cancelled = True
                result.update(terminal="cancelled", natural=True, no_stage_launched=True)
                return result
            observe()
            phase = 'normal.write-launch'
            spec_path = directory / "launch.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            phase = 'normal.job-create'
            job = _OwnedJob()
            phase = 'normal.open-transport-files'
            with protocol_path.open("xb", buffering=0) as protocol, (directory / "bootstrap.stderr.txt").open("xb") as errors:
                phase = 'normal.subprocess.Popen'
                proc = subprocess.Popen([sys.executable, "-B", "-I", "-S", str(_HERE / "diagnostic_bootstrap.py"), str(spec_path)],
                                        stdin=subprocess.PIPE, stdout=protocol, stderr=errors,
                                        cwd=str(command.cwd) if command.kind == "windows" else None,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                result["windows_settled"] = False
                phase = 'normal.job-attach'
                job.phase = phase
                job.attach(proc)
                attached = True
                observe()
                if cancel_event.is_set():
                    cancelled = True
                    control("CANCEL")
                else:
                    control("GO")
                    gate_open = True
                last_ping = time.monotonic()
                cancel_sent = None
                while True:
                    phase = 'normal.subprocess.poll'
                    exited = proc.poll() is not None
                    read_events()
                    if exited:
                        break
                    now = time.monotonic()
                    if cancel_event.is_set() and cancel_sent is None:
                        cancelled = True
                        control("CANCEL")
                        cancel_sent = now
                    if now >= deadline:
                        raise TimeoutError("Owned stage deadline exceeded")
                    if cancel_sent is not None and now - cancel_sent > 7:
                        raise TimeoutError("Owned stage cancellation settlement unverified")
                    if now - last_ping >= .5:
                        control("PING")
                        last_ping = now
                    observe()
                    time.sleep(.025)
                result["bootstrap_returncode"] = proc.returncode
                if terminal is None:
                    raise RuntimeError("Owned stage terminal receipt missing")
                result["forced"] = terminal.get("forced") is True
                result["natural"] = terminal.get("natural") is True and not result["forced"]
                if command.kind == "wsl":
                    result["linux"] = terminal
                    result["linux_settled"] = linux_settled()
                result["returncode"] = terminal.get("returncode")
                cancelled = cancelled or cancel_event.is_set() or terminal.get("terminal") == "cancelled"
                if cancelled and not terminal.get("cooperative_cancelled") and not terminal.get("no_stage_launched"):
                    result["natural"] = False
                result["terminal"] = ("cancelled" if cancelled else "succeeded"
                    if terminal.get("terminal") == "succeeded" and result["returncode"] == 0
                    and result["bootstrap_returncode"] == 0
                    and result.get("linux_client_returncode", 0) == 0 and result["natural"] else "failed")
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            capture(exc)
            if isinstance(exc, _HostResourceError):
                result["terminal"] = "blocked"
        finally:
            shutdown = True
            phase = 'cleanup.begin'
            if proc is not None and proc.stdin is not None and not proc.stdin.closed:
                try:
                    phase = 'cleanup.subprocess.poll'
                    if proc.poll() is None:
                        control("CANCEL")
                        # Linux owns its cancellation deadline. Keep its broker alive for acknowledgement.
                        cleanup_end = min(deadline + 8, time.monotonic() + 7)
                        while proc.poll() is None and time.monotonic() < cleanup_end:
                            read_events()
                            try:
                                observe()
                            except Exception as exc:
                                capture(exc)
                                result.setdefault("cleanup_observer_error", str(exc))
                            time.sleep(.025)
                        read_events()
                except Exception as exc:
                    capture(exc)
                    result["cleanup_control_error"] = str(exc)
                finally:
                    phase = 'cleanup.stdin.close'
                    try:
                        proc.stdin.close()
                    except Exception as exc:
                        capture(exc)
                        result['terminal'] = 'failed'
                        result['cleanup_stdin_close_error'] = str(exc)
            if job is not None:
                try:
                    phase = 'cleanup.job-settle'
                    job.phase = phase
                    grace_end = time.monotonic() + 1
                    while job.snapshot()["active"] and time.monotonic() < grace_end:
                        time.sleep(.025)
                    if job.snapshot()["active"]:
                        result.update(forced=True, natural=False)
                        job.terminate()
                    settle_end = time.monotonic() + 4
                    while True:
                        result["windows"] = job.snapshot()
                        if result["windows"]["active"] == 0:
                            result["windows_settled"] = True
                            break
                        if time.monotonic() >= settle_end:
                            break
                        time.sleep(.025)
                except Exception as exc:
                    capture(exc)
                    result["windows_settlement_error"] = str(exc)
                finally:
                    try:
                        phase = 'cleanup.job-close'
                        job.phase = phase
                        job.close()
                    except Exception as exc:
                        capture(exc)
                        result["windows_close_error"] = str(exc)
                        result["windows_settled"] = False
            if proc is not None and not attached:
                # This bootstrap has never received GO, so it cannot own stage descendants.
                try:
                    phase = 'cleanup.unattached-bootstrap'
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=2)
                    result["windows_settled"] = True
                except Exception as exc:
                    capture(exc)
                    result["bootstrap_settlement_error"] = str(exc)
            if command.kind == "wsl" and terminal is not None:
                result["linux"] = terminal
                result["linux_settled"] = linux_settled()
                result["forced"] = result["forced"] or terminal.get("forced") is True
            if command.kind == "wsl" and not gate_open:
                result["linux_settled"] = True  # No Linux client was ever launched.
            if not result["windows_settled"] or result["linux_settled"] is False:
                result["terminal"] = "blocked"
                self.unsettled = True
            elif result["forced"] and result["terminal"] == "succeeded":
                result["terminal"] = "failed"
            if result["forced"]:
                result["natural"] = False
            result.update(finished_utc=_utc(), elapsed_s=time.monotonic() - began)
            def timing_receipt(name, **extra):
                row = dict(schema='modly.fixture-timing.runner.v1', stage_id=command.stage_id,
                           context=spec['context'], timing=result['timing'], **extra)
                with (directory / name).open('x', encoding='utf-8') as stream:
                    json.dump(row, stream); stream.flush(); os.fsync(stream.fileno())
            timing_receipt('timing-before-raw.json', boundary='before final raw receipt serialization')
            raw_begin_ns = time.monotonic_ns()
            receipt_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raw_end_ns = time.monotonic_ns()
            timing_receipt('timing-after-raw.json', raw_serialization_begin_ns=raw_begin_ns,
                           raw_serialization_end_ns=raw_end_ns,
                           raw_serialization_elapsed_ns=raw_end_ns-raw_begin_ns)
        return result
