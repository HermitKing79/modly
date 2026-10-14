"""One CPU-only WSL worker; read-only private Job identities and bounded exit grace."""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import datetime
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from_source = Path(__file__).with_name("amd_wsl_stdin_probe.py")
import importlib.util
spec = importlib.util.spec_from_file_location("stdin_probe", from_source)
stdin_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stdin_probe)
smoke = stdin_probe.load(from_source.with_name("amd_wsl_rocdxg_smoke.py"), "teardown_smoke")
HERE = Path(__file__).resolve()
OUTPUT = smoke.ROOT / "docs/compatibility/amd-wsl-single-worker-teardown-evidence/observation"


class JobPids(ct.Structure):
    _fields_ = [("assigned", wt.DWORD), ("count", wt.DWORD), ("pids", ct.c_size_t * 16)]


class ProcessEntry(ct.Structure):
    _fields_ = [("size", wt.DWORD), ("usage", wt.DWORD), ("pid", wt.DWORD),
                ("heap", ct.c_size_t), ("module", wt.DWORD), ("threads", wt.DWORD),
                ("parent", wt.DWORD), ("priority", wt.LONG), ("flags", wt.DWORD),
                ("name", wt.WCHAR * 260)]


def valid_pids(row):
    if row.count != row.assigned or row.count > 16:
        raise RuntimeError("Private Job PID list incomplete or exceeds 16-member bound")
    pids = list(row.pids[:row.count])
    if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
        raise RuntimeError("Invalid or duplicate Job process identities")
    return pids


def grace(empty, tick, seconds, clock=time.monotonic, sleep=time.sleep):
    began = clock()
    while True:
        tick()  # Even an already-empty Job needs a healthy monitored decision.
        if empty(): return {"empty": True, "seconds": clock() - began}
        if clock() - began >= seconds: return {"empty": False, "seconds": clock() - began}
        sleep(min(.1, max(0., seconds - (clock() - began))))


class JobObservation:
    """Read-only diagnostic handles; never kills or assigns any process."""
    def __init__(self, job):
        if not job.handle or ct.sizeof(JobPids) != 136 or ct.sizeof(ProcessEntry) != 568:
            raise RuntimeError("Missing private Job or unexpected Windows ABI")
        self.job, self.api, self.handles, self.identities = job, job.api, {}, {}
        signatures = {
            "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
            "IsProcessInJob": ([wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL),
            "QueryFullProcessImageNameW": ([wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)], wt.BOOL),
            "GetProcessTimes": ([wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4, wt.BOOL),
            "GetExitCodeProcess": ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
            "WaitForSingleObject": ([wt.HANDLE, wt.DWORD], wt.DWORD),
            "CreateToolhelp32Snapshot": ([wt.DWORD, wt.DWORD], wt.HANDLE),
            "Process32FirstW": ([wt.HANDLE, ct.POINTER(ProcessEntry)], wt.BOOL),
            "Process32NextW": ([wt.HANDLE, ct.POINTER(ProcessEntry)], wt.BOOL)}
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result

    def require(self, ok):
        if not ok: raise ct.WinError(ct.get_last_error())

    def parents(self):
        handle = self.api.CreateToolhelp32Snapshot(2, 0)
        if handle == ct.c_void_p(-1).value: raise ct.WinError(ct.get_last_error())
        result = {}
        try:
            row = ProcessEntry(); row.size = ct.sizeof(row)
            self.require(self.api.Process32FirstW(handle, ct.byref(row)))
            for _ in range(65536):
                result[row.pid] = {"parent_pid": row.parent, "name": row.name}
                if not self.api.Process32NextW(handle, ct.byref(row)):
                    if ct.get_last_error() != 18: raise ct.WinError(ct.get_last_error())
                    return result
            raise RuntimeError("Toolhelp snapshot exceeds bound")
        finally: self.require(self.api.CloseHandle(handle))

    def status(self, handle):
        wait = self.api.WaitForSingleObject(handle, 0)
        if wait not in (0, 258): raise RuntimeError("Process exit wait failed")
        times = [wt.FILETIME() for _ in range(4)]
        self.require(self.api.GetProcessTimes(handle, *(ct.byref(t) for t in times)))
        ticks = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime
        code = wt.DWORD(); self.require(self.api.GetExitCodeProcess(handle, ct.byref(code)))
        created = ticks(times[0])
        return {"created_filetime_100ns": created,
                "created_utc": (datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc)
                                + datetime.timedelta(microseconds=created // 10)).isoformat(),
                "exited": wait == 0, "exit_code": code.value if wait == 0 else None,
                "exit_filetime_100ns": ticks(times[1]) if wait == 0 else None}

    def capture(self, phase):
        row = JobPids()
        self.require(self.api.QueryInformationJobObject(self.job.handle, 3, ct.byref(row), ct.sizeof(row), None))
        pids = valid_pids(row)
        new = [pid for pid in pids if pid not in self.handles]
        if len(self.handles) + len(new) > 16:
            raise RuntimeError("Cumulative observed process identities exceed 16-member bound")
        parents = self.parents() if new else {}
        for pid in new:
            handle = self.api.OpenProcess(0x101000, False, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
            self.require(handle)
            try:
                member = wt.BOOL()
                self.require(self.api.IsProcessInJob(handle, self.job.handle, ct.byref(member)))
                if not member.value: raise RuntimeError("Observed PID no longer belongs to private Job")
                image = ct.create_unicode_buffer(32768); length = wt.DWORD(len(image))
                self.require(self.api.QueryFullProcessImageNameW(handle, 0, image, ct.byref(length)))
                if pid not in parents: raise RuntimeError("Observed Job PID absent from Toolhelp snapshot")
                self.identities[pid] = {"pid": pid, "image": image.value, "name": Path(image.value).name,
                                        "sampled_toolhelp": parents[pid],
                                        "first_observed_phase": phase}
                self.handles[pid] = handle
            except BaseException:
                self.api.CloseHandle(handle)
                raise
        return [{**self.identities[pid], **self.status(handle), "in_current_job_list": pid in pids}
                for pid, handle in self.handles.items()]

    def close(self):
        for handle in self.handles.values(): self.require(self.api.CloseHandle(handle))
        self.handles.clear()


def guest(output):
    sys.addaudithook(stdin_probe.forbid_ai)
    output = output.resolve(); output.relative_to(smoke.ROOT); output.mkdir(exist_ok=False)
    smoke.await_go(sys.stdin.buffer)
    smoke.journal(output, "cpu-authorized", gpu_allowed=False)
    root = Path('/opt/modly-wsl/venv'); rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name; s = p.lstat()
            rows.append([p.relative_to(root).as_posix(), s.st_size, s.st_mtime_ns, s.st_mode])
    import platform
    package_query = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'],
                                   capture_output=True, text=True, timeout=15)
    if package_query.returncode: raise RuntimeError("Read-only package identity query failed")
    identity = {'python': platform.python_version(), 'kernel': platform.release(),
                'venv_entries': len(rows),
                'venv_metadata_sha256': hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest(),
                'python_packages': sorted([(d.metadata['Name'], d.version) for d in md.distributions(
                    path=[str(root / 'lib/python3.12/site-packages')])]),
                'dpkg_packages': package_query.stdout.splitlines(),
                'torch_imported': 'torch' in sys.modules, 'triton_imported': 'triton' in sys.modules}
    smoke.save_json(output / "results.json", {"accepted": True, "gpu_allowed": False,
                    "identity": identity, "guest_pid": os.getpid(), "finished_utc": smoke.utc_now()})
    return 0


def observe(output):
    sys.addaudithook(stdin_probe.forbid_ai)
    output.resolve().relative_to(smoke.ROOT)
    smoke.save_json(output.parent / "observation-started.json", {"utc": smoke.utc_now(),
                    "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest()})
    output.mkdir(exist_ok=False)
    began = time.monotonic(); since = smoke.utc_now()
    helper = smoke.load_sibling("amd_bounded_gpu_proof")
    monitor = smoke.HealthMonitor(lambda: helper.windows_observation(since), helper.healthy_observation)
    result = {"classification": "UNKNOWN", "snapshots": [], "observer_failed": False,
              "pid_limit": 16, "natural_grace_seconds": 10, "posttermination_grace_seconds": 5}
    job = smoke.load_sibling("amd_forward_owned_job").OwnedJob()
    observed = None; child = None; terminated = False
    def tick():
        smoke.check_deadlines(time.monotonic(), began, None)
        try: monitor.poll(time.monotonic())
        except BaseException:
            result["observer_failed"] = True
            raise
    def health(after):
        while not monitor.healthy_since(after): tick(); time.sleep(.025)
        tick()
        return monitor.latest()
    def snapshot(phase):
        if len(result["snapshots"]) >= 1024: raise RuntimeError("Process evidence bound exceeded")
        rows = observed.capture(phase)
        event = {"phase": phase, "elapsed_seconds": time.monotonic() - began,
                 "utc": smoke.utc_now(), "processes": rows}
        result["snapshots"].append(event)
        smoke.journal(output, "job-processes", detail=event, gpu_allowed=False)
        return not any(row["in_current_job_list"] for row in rows)
    def wait_cleanup(function, seconds):
        future = smoke.in_thread(function); deadline = time.monotonic() + seconds
        while not future.done():
            if time.monotonic() >= deadline: raise TimeoutError("Owned cleanup completion unverified")
            if not result["observer_failed"]:
                try: tick()
                except BaseException as exc: result["cleanup_monitor_error"] = str(exc)
            time.sleep(.025)
        return future.result()
    def terminate():
        nonlocal terminated
        result["distro_termination"] = wait_cleanup(smoke.terminate_owned_distro, 11)
        terminated = True
        smoke.journal(output, "named-distro-termination", capture=result["distro_termination"], gpu_allowed=False)
        if result["distro_termination"]["returncode"]: raise RuntimeError("Named-distro termination failed")
    try:
        before = health(began)
        observed = JobObservation(job)
        command = smoke.worker_command(smoke.guest_path(HERE), smoke.guest_path(output / "guest"))
        result["command"] = command
        with (output / "stdout.txt").open("xb") as out, (output / "stderr.txt").open("xb") as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err)
            result["client_pid"] = child.pid; job.attach(child)
            attached = time.monotonic(); snapshot("attached")
            after = health(attached); snapshot("before-go")
            smoke.authorize_go(child.stdin, True, before, after); child.stdin.close()
            go = time.monotonic()
            while child.poll() is None:
                tick(); snapshot("guest-running")
                if time.monotonic() - go >= 30: raise TimeoutError("CPU worker deadline")
                time.sleep(.1)
            result["client_exit_code"] = child.returncode; snapshot("client-exited")
            result["guest"] = json.loads((output / "guest/results.json").read_text(encoding="utf-8"))
            if child.returncode or not result["guest"].get("accepted"): raise RuntimeError("CPU guest did not exit normally")
            expected = json.loads((smoke.ROOT / "docs/compatibility/amd-wsl-stdin-evidence/linux-environment-after.json").read_text(encoding="utf-8"))["record"]
            result["linux_identity_preserved"] = result["guest"]["identity"] == expected
            if not result["linux_identity_preserved"]: raise RuntimeError("Linux environment identity mismatch")
            result["natural_grace"] = grace(lambda: snapshot("natural-grace"), tick, 10)
            if not result["natural_grace"]["empty"]:
                snapshot("before-distro-termination"); terminate()
                result["posttermination_grace"] = grace(lambda: snapshot("after-distro-termination"), tick, 5)
            result["before-job-cleanup_empty"] = snapshot("before-job-cleanup")
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:
            try: child.stdin.close()
            except BaseException as exc: result["stdin_close_error"] = str(exc)
        try:
            result["job_cleanup"] = wait_cleanup(job.finish, 6)
            if observed is not None:
                result["final_process_states"] = [{**observed.identities[pid], **observed.status(handle)}
                                                   for pid, handle in observed.handles.items()]
        except BaseException as exc: result["job_cleanup_error"] = str(exc)
        if child is not None:
            try:
                if not terminated: terminate()
                child.wait(timeout=2)
                health(time.monotonic())
                result["post_cleanup_healthy"] = True
            except BaseException as exc: result["distro_cleanup_error"] = str(exc)
        if observed is not None:
            try: observed.close()
            except BaseException as exc: result["handle_close_error"] = str(exc)
        result["observations"] = monitor.observations
        result["elapsed_seconds"] = time.monotonic() - began
        if (not any(key.endswith('error') for key in result) and not result["observer_failed"]
                and result["elapsed_seconds"] < smoke.HOST_DEADLINE
                and result.get("before-job-cleanup_empty") is True and result.get("post_cleanup_healthy") is True
                and result.get("job_cleanup", {}).get("forced_job_cleanup") is False
                and result.get("job_cleanup", {}).get("owned_tree_exit_verified") is True
                and all(row["exited"] for row in result.get("final_process_states", []))):
            result["classification"] = "PASS"
        smoke.save_json(output / "results.json", result)
    print(json.dumps({"classification": result["classification"], "job_cleanup": result.get("job_cleanup")}))
    return 0 if result["classification"] == "PASS" else 1


def confirm(output):
    """One CPU substitute through the actual production supervisor.

    Only script/marker selection and read-only Job instrumentation differ.
    Production GPU result validation is preserved; its UNKNOWN is expected.
    """
    sys.addaudithook(stdin_probe.forbid_ai)
    from unittest.mock import patch
    from types import SimpleNamespace
    real_load = smoke.load_sibling
    original_job = real_load("amd_forward_owned_job").OwnedJob
    events = []
    class RecordedJob(original_job):
        def __init__(self):
            super().__init__()
            self.observed = JobObservation(self)
        def capture(self, phase):
            if len(events) >= 1024: raise RuntimeError("Confirmation process evidence bound")
            events.append({"phase": phase, "utc": smoke.utc_now(),
                           "processes": self.observed.capture(phase)})
        def attach(self, child):
            super().attach(child)
            self.capture("attached")
        def active(self):
            count = super().active()
            # Diagnostics must not interfere with the real finish() query.
            # An incomplete identity record still rejects the CPU verdict.
            try: self.capture("production-active-query")
            except BaseException as exc: events.append({"identity_error": str(exc)})
            return count
        def finish(self, timeout=5.):
            try:
                self.capture("before-production-finish")
            except BaseException as exc:
                events.append({"identity_error": str(exc)})
            try:
                result = super().finish(timeout)
                events.append({"phase": "after-production-finish", "processes": [
                    {**self.observed.identities[pid], **self.observed.status(handle)}
                    for pid, handle in self.observed.handles.items()]})
                return result
            finally: self.observed.close()
    def sibling(name):
        return SimpleNamespace(OwnedJob=RecordedJob) if name == "amd_forward_owned_job" else real_load(name)
    marker = output.parent / "confirmation-started.json"
    with patch.object(smoke, "SCRIPT", HERE), patch.object(smoke, "ATTEMPT_MARKER", marker), \
            patch.object(smoke, "load_sibling", side_effect=sibling):
        returncode = smoke.supervise(output)
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    guest_result = result.get("worker", {})
    previous = json.loads((smoke.ROOT / "docs/compatibility/amd-wsl-stdin-evidence/linux-environment-after.json").read_text(encoding="utf-8"))["record"]
    clean = (returncode == 1 and result["classification"] == "UNKNOWN"
             and result.get("worker_returncode") == 0 and guest_result.get("accepted") is True
             and guest_result.get("gpu_allowed") is False and guest_result.get("identity") == previous
             and result.get("observer_healthy") is True
             and result.get("distro_termination", {}).get("returncode") == 0
             and result.get("windows_job_cleanup", {}).get("forced_job_cleanup") is False
             and result.get("windows_job_cleanup", {}).get("owned_tree_exit_verified") is True
             and result.get("windows_job_grace", {}).get("empty") is True
             and result["elapsed_seconds"] < smoke.HOST_DEADLINE
             and not any(key.endswith("error") for key in result)
             and not any("identity_error" in event for event in events)
             and bool(events) and all(row["exited"] for row in events[-1].get("processes", [])))
    verdict = {"classification": "PASS" if clean else "UNKNOWN", "production_supervisor_returncode": returncode,
               "production_classification": result["classification"],
               "production_unknown_reason": "CPU guest has no GPU computation/cleanup evidence; GPU acceptance is unchanged",
               "production_source_sha256": hashlib.sha256(smoke.SCRIPT.read_bytes()).hexdigest(),
               "probe_source_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(), "process_events": events}
    smoke.save_json(output / "cpu-confirmation.json", verdict)
    print(json.dumps({"cpu_classification": verdict["classification"]}))
    return 0 if clean else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.worker is not None: raise SystemExit(guest(args.worker))
    if sys.platform != "win32" or args.observe == args.confirm:
        parser.error("Select exactly one Windows --observe or --confirm")
    if args.confirm: raise SystemExit(confirm(OUTPUT.with_name("confirmation")))
    raise SystemExit(observe(OUTPUT))
