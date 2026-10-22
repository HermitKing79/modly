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

MEMORY_LIMIT = 6 * 1024**3
RAM_FLOOR = 12 * 1024**3
DISK_FLOOR = 2 * 1024**3
DISTRO = "Modly-AMD-Ubuntu-24.04"
_HERE = Path(__file__).resolve().parent
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
        }
        for name, (arguments, result) in definitions.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _Limits()
        limits.Basic.Flags = 0x2000 | 0x100 | 0x200  # kill-on-close, process + job memory.
        limits.ProcessMemory = limits.JobMemory = MEMORY_LIMIT
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, proc):
        if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        self._identity(proc.pid, int(proc._handle))

    def _identity(self, pid, handle):
        member = wt.BOOL()
        if not self.api.IsProcessInJob(handle, self.handle, ctypes.byref(member)) or not member.value:
            raise RuntimeError("Process membership in private Job could not be verified")
        values = [wt.FILETIME() for _ in range(4)]
        if not self.api.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = values[0].dwLowDateTime | values[0].dwHighDateTime << 32
        size = wt.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not self.api.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.identities[(pid, created)] = {"pid": pid, "creation_filetime": created,
                                          "image": image.value, "job_membership_verified": True}

    def snapshot(self):
        accounting, limits = _Accounting(), _Limits()
        for category, value in ((1, accounting), (9, limits)):
            if not self.api.QueryInformationJobObject(self.handle, category, ctypes.byref(value), ctypes.sizeof(value), None):
                raise ctypes.WinError(ctypes.get_last_error())
        # A bounded list is sufficient for this small, pinned stage chain. Overflow blocks it.
        class Pids(ctypes.Structure):
            _fields_ = [("Assigned", wt.DWORD), ("Count", wt.DWORD), ("Pids", ctypes.c_size_t * 256)]
        ids = Pids()
        if not self.api.QueryInformationJobObject(self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
            raise ctypes.WinError(ctypes.get_last_error())
        for pid in ids.Pids[:ids.Count]:
            handle = self.api.OpenProcess(0x1000, False, pid)
            if not handle:  # A member may have exited between the two native queries.
                continue
            try:
                self._identity(pid, handle)
            finally:
                self.api.CloseHandle(handle)
        return {"active": accounting.Active, "total_processes": accounting.Total,
                "peak_process_bytes": limits.PeakProcessMemory, "peak_job_bytes": limits.PeakJobMemory,
                "process_identities": list(self.identities.values())}

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


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
    if not api.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
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


class OwnedStageRunner:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = Path(evidence_dir)
        self.unsettled = False

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
        proc = job = None
        attached = cancelled = gate_open = False
        terminal = None
        linux_ready = None
        deadline = began + command.timeout_s
        cursor = 0

        def observe():
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
                result["windows"] = job.snapshot()

        def control(action):
            proc.stdin.write((json.dumps({"action": action, "stage_id": command.stage_id,
                                          "nonce": spec["nonce"]}) + "\n").encode())
            proc.stdin.flush()

        def read_events():
            nonlocal cursor, terminal, linux_ready
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
            spec_path = directory / "launch.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            job = _OwnedJob()
            with protocol_path.open("xb", buffering=0) as protocol, (directory / "bootstrap.stderr.txt").open("xb") as errors:
                proc = subprocess.Popen([sys.executable, "-B", "-I", "-S", str(_HERE / "amd_stage_bootstrap.py"), str(spec_path)],
                                        stdin=subprocess.PIPE, stdout=protocol, stderr=errors,
                                        cwd=str(command.cwd) if command.kind == "windows" else None,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                result["windows_settled"] = False
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
            if isinstance(exc, _HostResourceError):
                result["terminal"] = "blocked"
        finally:
            if proc is not None and proc.stdin is not None and not proc.stdin.closed:
                try:
                    if proc.poll() is None:
                        control("CANCEL")
                        # Linux owns its cancellation deadline. Keep its broker alive for acknowledgement.
                        cleanup_end = min(deadline + 8, time.monotonic() + 7)
                        while proc.poll() is None and time.monotonic() < cleanup_end:
                            read_events()
                            try:
                                observe()
                            except Exception as exc:
                                result.setdefault("cleanup_observer_error", str(exc))
                            time.sleep(.025)
                        read_events()
                except Exception as exc:
                    result["cleanup_control_error"] = str(exc)
                finally:
                    proc.stdin.close()
            if job is not None:
                try:
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
                    result["windows_settlement_error"] = str(exc)
                finally:
                    try:
                        job.close()
                    except Exception as exc:
                        result["windows_close_error"] = str(exc)
                        result["windows_settled"] = False
            if proc is not None and not attached:
                # This bootstrap has never received GO, so it cannot own stage descendants.
                try:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=2)
                    result["windows_settled"] = True
                except Exception as exc:
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
            receipt_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
