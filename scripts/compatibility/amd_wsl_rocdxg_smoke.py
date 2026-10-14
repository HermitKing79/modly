"""One owned WSL distro, one 256-element F32 addition, no retry or model work.

The Windows Job owns the wsl.exe client only. The guest timeout and explicit
termination of the named distro provide separate Linux-process containment.
Imports are stdlib-only until the Linux worker receives the host's GO.
"""
import argparse
from concurrent.futures import Future
import datetime
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import weakref


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
DISTRO = "Modly-AMD-Ubuntu-24.04"
VENV_PYTHON = "/opt/modly-wsl/venv/bin/python"
TORCH_RUNTIME_VERSION = "2.9.1+rocm7.2.1.gitff65f5bc"
TORCH_DISTRIBUTION_VERSION = "2.9.1+rocm7.2.1.lw.gitff65f5bc"
TORCH_GIT_VERSION = "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf"
HIP_VERSION = "7.2.53211-e1a6bc5663"
GiB = 1024 ** 3
SMOKE_MEMORY_LIMIT = 64 * 1024 ** 2
HOST_DEADLINE = 120.0
GPU_DEADLINE = 60.0
OBSERVATION_DEADLINE = 10.0
OBSERVATION_INTERVAL = 5.0
ATTEMPT_MARKER = ROOT / "docs/compatibility/amd-wsl-rocdxg-gpu-resume-evidence/gpu-attempt.marker"


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT.with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def healthy_observation(capture):
    return load_sibling("amd_bounded_gpu_proof").healthy_observation(capture)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal(output, phase, **detail):
    with (output / "phases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "utc": utc_now(),
                                 "monotonic_seconds": time.monotonic(), **detail}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_attempt(output, marker):
    marker.parent.mkdir(parents=True, exist_ok=True)
    save_json(marker, {"started_utc": utc_now(), "distro": DISTRO,
                       "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                       "output": str(output)})
    # Never remove the marker, even if directory creation or a later gate fails.
    output.mkdir(parents=True, exist_ok=False)


def worker_command(script_linux, output_linux):
    if not all(path.startswith("/mnt/c/") and "\n" not in path
               for path in (script_linux, output_linux)):
        raise ValueError("Worker script and evidence must be absolute /mnt/c paths")
    return ["wsl.exe", "-d", DISTRO, "-u", "modly", "--exec", "/usr/bin/env",
            "HSA_ENABLE_DXG_DETECTION=1",
            "LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:/opt/rocm/lib:/usr/lib/wsl/lib", "/usr/bin/timeout",
            "--signal=TERM", "--kill-after=5s", "90s", VENV_PYTHON, "-B", "-I",
            script_linux, "--worker", output_linux]


def guest_path(path):
    path = path.resolve()
    path.relative_to(ROOT)
    if path.drive.lower() != "c:":
        raise ValueError("This proof checkout must be on C:")
    return "/mnt/c/" + path.as_posix()[3:]


def await_go(stream):
    # The host closes stdin after its exact three-byte token. Reading one
    # extra byte rejects CRLF/trailing data and requires EOF for acceptance.
    if stream.read(4) != b"GO\n":
        raise RuntimeError("Host authorization absent or refused; no Torch import or GPU work")


def authorize_go(stream, job_attached, before, after):
    if not job_attached or not healthy_observation(before) or not healthy_observation(after):
        raise RuntimeError("GO requires Job attachment and healthy prelaunch/post-attachment observations")
    stream.write(b"GO\n")
    stream.flush()


def fixture():
    a = [(i % 33 - 16) / 2 for i in range(256)]
    b = [((i * 7) % 35 - 16) / 8 for i in range(256)]
    # Independent Python arithmetic; these small dyadic sums are exact in F32.
    return a, b, [left + right for left, right in zip(a, b)]


class NumericalError(RuntimeError):
    pass


class NativeGPUError(RuntimeError):
    pass


def validate_output(values, shape, dtype):
    expected = fixture()[2]
    if (shape != [256] or dtype != "float32" or not isinstance(values, list)
            or len(values) != 256 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
            or values != expected):
        raise NumericalError("CPU result differs from the exact 256-element F32 reference")


def validate_runtime_identity(record):
    expected = {"torch": TORCH_RUNTIME_VERSION, "distribution_version": TORCH_DISTRIBUTION_VERSION,
                "git_version": TORCH_GIT_VERSION, "hip": HIP_VERSION, "cuda": None,
                "prefix": "/opt/modly-wsl/venv"}
    if (any(key not in record or record[key] != value for key, value in expected.items())
            or not isinstance(record.get("python"), str) or not record["python"].startswith("3.12.")
            or record.get("gpu_initialized_before") is not False):
        raise RuntimeError("Unexpected Python/PyTorch/HIP provenance or premature GPU initialization")


def validate_cleanup(record):
    if record.get("owned_gpu_tensor_weakrefs_collected") is not True:
        raise RuntimeError("Owned GPU tensors have not been collected")
    for key in ("memory_before", "memory_release"):
        if record.get(key) != {"allocated": 0, "reserved": 0}:
            raise RuntimeError(key + " must prove zero allocated and reserved bytes")
    peak = record.get("memory_peak", {})
    if (set(peak) != {"allocated", "reserved"} or
            any(type(value) is not int or not 0 <= value <= SMOKE_MEMORY_LIMIT for value in peak.values())):
        raise RuntimeError("Measured allocated/reserved peak exceeds the 64 MiB smoke limit")


def check_deadlines(now, began, go):
    if now - began >= HOST_DEADLINE:
        raise TimeoutError("120-second host deadline")
    if go is not None and now - go >= GPU_DEADLINE:
        raise TimeoutError("60-second GPU-stage deadline")


def in_thread(function):
    future = Future()
    def run():
        try:
            future.set_result(function())
        except BaseException as exc:
            future.set_exception(exc)
    threading.Thread(target=run, daemon=True).start()
    return future


class HealthMonitor:
    """Existing ten-second observer, overlapping starts every five seconds."""
    def __init__(self, observer, healthy, submit=in_thread):
        self.observer, self.healthy, self.submit = observer, healthy, submit
        self.pending, self.observations = [], []
        self.last_started = None

    def poll(self, now):
        if self.last_started is not None and now - self.last_started > OBSERVATION_DEADLINE:
            raise TimeoutError("Windows observer start gap exceeded ten seconds")
        remaining = []
        for started, future in self.pending:
            if future.done():
                ended, capture = future.result()
                row = {"started_monotonic": started, "ended_monotonic": ended, "capture": capture}
                self.observations.append(row)
                if ended - started > OBSERVATION_DEADLINE:
                    raise TimeoutError("Windows observer completed late")
                if not self.healthy(capture):
                    raise RuntimeError("Windows driver observation unhealthy or incomplete")
            elif now - started >= OBSERVATION_DEADLINE:
                raise TimeoutError("Windows observer still pending at ten-second deadline")
            else:
                remaining.append((started, future))
        self.pending = remaining
        if self.last_started is None or now - self.last_started >= OBSERVATION_INTERVAL:
            def observe():
                capture = self.observer()  # Reuses the original helper's ten-second child timeout.
                return time.monotonic(), capture
            self.pending.append((now, self.submit(observe)))
            self.last_started = now

    def healthy_since(self, started):
        # Windows Python 3.12's monotonic clock can give adjacent events the
        # same reading. Equality cannot prove an observer started after a gate.
        return any(row["started_monotonic"] > started and self.healthy(row["capture"])
                   for row in self.observations)

    def latest(self):
        return max(self.observations, key=lambda row: row["started_monotonic"])["capture"]


def cpu_identity():
    models = []
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            value = line.split(":", 1)[1].strip()
            if value not in models:
                models.append(value)
    return {"model_names": models, "logical_cpus": os.cpu_count(),
            "uname": list(platform.uname()), "os_release": platform.freedesktop_os_release(),
            "python": platform.python_version(), "executable": sys.executable,
            "prefix": sys.prefix, "pid": os.getpid()}


def worker(output):
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(exist_ok=False)
    record = {"classification": "UNKNOWN", "started_utc": utc_now(), "gate_authorized": False,
              "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}
    torch = None
    a = b = c = None
    refs = []
    gpu_began = None
    def gpu_call(phase, function):
        if time.monotonic() - gpu_began >= GPU_DEADLINE:
            raise TimeoutError("Worker GPU-stage deadline")
        journal(output, phase + "-start")
        try:
            result = function()
        except Exception as exc:
            raise NativeGPUError(phase + ": " + str(exc)) from exc
        journal(output, phase + "-complete")
        return result
    def memory():
        return {"allocated": torch.cuda.memory_allocated(0), "reserved": torch.cuda.memory_reserved(0)}
    try:
        journal(output, "gate-wait", gpu_allowed=False)
        await_go(sys.stdin.buffer)
        record["gate_authorized"] = True
        journal(output, "gate-authorized", gpu_allowed=True)
        gpu_began = time.monotonic()
        record["cpu"] = cpu_identity()
        for name in ("hip-cache", "comgr-cache", "torch-extensions", "triton-cache"):
            (output / name).mkdir()
        os.environ.update(HSA_ENABLE_DXG_DETECTION="1", HIP_VISIBLE_DEVICES="0",
                          ROCR_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0",
                          LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib:/opt/rocm/lib:/usr/lib/wsl/lib",
                          HIP_CACHE_DIR=str(output / "hip-cache"),
                          AMD_COMGR_CACHE_DIR=str(output / "comgr-cache"),
                          TORCH_EXTENSIONS_DIR=str(output / "torch-extensions"),
                          TRITON_CACHE_DIR=str(output / "triton-cache"))
        record["linux_environment"] = {key: os.environ[key] for key in
                                       ("HSA_ENABLE_DXG_DETECTION", "LD_LIBRARY_PATH",
                                        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        import torch
        record["runtime"] = {"torch": torch.__version__, "hip": torch.version.hip,
                             "distribution_version": importlib.metadata.version("torch"),
                             "git_version": torch.version.git_version,
                             "version_file_sha256": hashlib.sha256(
                                 Path(torch.__file__).with_name("version.py").read_bytes()).hexdigest(),
                             "cuda": torch.version.cuda, "torch_file": torch.__file__,
                             "python": platform.python_version(), "prefix": sys.prefix,
                             "gpu_initialized_before": torch.cuda.is_initialized()}
        validate_runtime_identity(record["runtime"])
        count = gpu_call("device-count", torch.cuda.device_count)
        props = gpu_call("device-properties", lambda: torch.cuda.get_device_properties(0))
        free, total = gpu_call("free-vram", lambda: torch.cuda.mem_get_info(0))
        record["gpu"] = {"name": props.name, "architecture": getattr(props, "gcnArchName", None),
                         "device_count": count, "total_memory_bytes": props.total_memory,
                         "free_memory_bytes": free, "runtime_total_memory_bytes": total,
                         "multi_processor_count": props.multi_processor_count, "properties": str(props),
                         "allocator_cap_bytes": 12 * GiB}
        if (count != 1 or "RX 9070 XT" not in props.name
                or str(getattr(props, "gcnArchName", "")).split(":")[0] != "gfx1201"
                or free < 13 * GiB or total < 13 * GiB):
            raise RuntimeError("GPU identity/free-VRAM admission refused")
        gpu_call("allocator-cap", lambda: torch.cuda.set_per_process_memory_fraction(12 * GiB / total, 0))
        gpu_call("reset-memory-peak", lambda: torch.cuda.reset_peak_memory_stats(0))
        record["memory_before"] = gpu_call("memory-before", memory)
        if record["memory_before"] != {"allocated": 0, "reserved": 0}:
            raise RuntimeError("Unexpected GPU allocations before the fixture")
        left, right, expected = fixture()
        left_cpu = torch.tensor(left, dtype=torch.float32, device="cpu")
        right_cpu = torch.tensor(right, dtype=torch.float32, device="cpu")
        reference_cpu = torch.tensor(expected, dtype=torch.float32, device="cpu")
        record["fixture"] = {"left": left, "right": right, "expected": expected}
        a = gpu_call("copy-left-to-gpu", lambda: left_cpu.to("cuda:0"))
        refs.append(weakref.ref(a))
        b = gpu_call("copy-right-to-gpu", lambda: right_cpu.to("cuda:0"))
        refs.append(weakref.ref(b))
        operation_began = time.monotonic()
        c = gpu_call("one-vector-addition", lambda: a + b)
        refs.append(weakref.ref(c))
        gpu_call("addition-synchronize", lambda: torch.cuda.synchronize(0))
        record["addition_synchronized_seconds"] = time.monotonic() - operation_began
        actual_cpu = gpu_call("copy-result-to-cpu", c.cpu)
        gpu_call("copy-synchronize", lambda: torch.cuda.synchronize(0))
        values = actual_cpu.tolist()
        record["computation"] = {"operation": "vector-add", "gpu_arithmetic_operations": 1,
                                 "shape": list(actual_cpu.shape), "dtype": str(actual_cpu.dtype).removeprefix("torch."),
                                 "values": [value if math.isfinite(value) else str(value) for value in values],
                                 "torch_cpu_equal": bool(torch.equal(actual_cpu, reference_cpu))}
        validate_output(values, record["computation"]["shape"], record["computation"]["dtype"])
        if not record["computation"]["torch_cpu_equal"]:
            raise NumericalError("torch.equal rejected the independent CPU reference")
        record["memory_peak"] = gpu_call("memory-peak", lambda: {
            "allocated": torch.cuda.max_memory_allocated(0), "reserved": torch.cuda.max_memory_reserved(0)})
        record["classification"] = "PASS"
    except BaseException as exc:
        record.update(classification="FAIL" if isinstance(exc, (NumericalError, NativeGPUError)) else "UNKNOWN",
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        a = b = c = None
        gc.collect()
        record["owned_gpu_tensor_weakrefs_collected"] = len(refs) == 3 and all(ref() is None for ref in refs)
        if torch is not None and torch.cuda.is_initialized():
            try:
                journal(output, "memory-release-start")
                if "memory_peak" not in record:
                    record["memory_peak"] = {"allocated": torch.cuda.max_memory_allocated(0),
                                             "reserved": torch.cuda.max_memory_reserved(0)}
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
                record["memory_release"] = memory()
                validate_cleanup(record)
                journal(output, "memory-release-complete", **record["memory_release"])
            except BaseException as exc:
                record["cleanup_error"] = type(exc).__name__ + ": " + str(exc)
                if record["classification"] == "PASS":
                    record["classification"] = "UNKNOWN"
        elif record["classification"] == "PASS":
            record.update(classification="UNKNOWN", cleanup_error="GPU initialization unverified")
        record["gpu_stage_seconds"] = time.monotonic() - gpu_began if gpu_began is not None else None
        record["finished_utc"] = utc_now()
        journal(output, "worker-finished", classification=record["classification"])
        save_json(output / "results.json", record)
    return 0 if record["classification"] == "PASS" else 1


def terminate_owned_distro():
    command = ["wsl.exe", "--terminate", DISTRO]
    capture = subprocess.run(command, capture_output=True, timeout=10)
    return {"command": command, "returncode": capture.returncode,
            "stdout": capture.stdout.decode("utf-8", errors="replace"),
            "stderr": capture.stderr.decode("utf-8", errors="replace")}


def supervise(output):
    if sys.platform != "win32":
        raise RuntimeError("Supervisor requires Windows; Linux runs only the gated worker")
    output = output.resolve()
    output.relative_to(ROOT)
    reserve_attempt(output, ATTEMPT_MARKER)
    began = time.monotonic()
    since = utc_now()
    helper = load_sibling("amd_bounded_gpu_proof")
    monitor = HealthMonitor(lambda: helper.windows_observation(since), helper.healthy_observation)
    record = {"classification": "UNKNOWN", "distro": DISTRO, "started_utc": since,
              "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              "containment_scope": "Windows Job: wsl.exe client; Linux: guest timeout and named-distro termination"}
    child = job = None
    go = None
    observer_failed = False
    def tick(gpu_allowed=True):
        nonlocal observer_failed
        now = time.monotonic()
        check_deadlines(now, began, go if gpu_allowed else None)
        try:
            monitor.poll(now)
        except BaseException:
            observer_failed = True
            raise
    def wait_health(after):
        while not monitor.healthy_since(after):
            tick()
            time.sleep(0.025)
        tick()
        return monitor.latest()
    journal(output, "supervisor-start", gpu_allowed=False)
    try:
        before = wait_health(began)
        record["observation_before"] = before
        command = worker_command(guest_path(SCRIPT), guest_path(output / "guest"))
        record["worker_command"] = command
        job = load_sibling("amd_forward_owned_job").OwnedJob()
        with (output / "worker.stdout.txt").open("xb") as stdout, (output / "worker.stderr.txt").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            journal(output, "wsl-client-started", pid=child.pid, gpu_allowed=False)
            job.attach(child)
            attached = time.monotonic()
            journal(output, "windows-job-attached", gpu_allowed=False)
            after = wait_health(attached)
            record["observation_after_attachment"] = after
            tick()
            if child.poll() is not None:
                raise RuntimeError("Worker exited before GO")
            authorize_go(child.stdin, True, before, after)
            go = time.monotonic()
            child.stdin.close()
            journal(output, "go-sent", gpu_allowed=True)
            while child.poll() is None:
                tick()
                time.sleep(0.025)
            record["worker_returncode"] = child.returncode
            tick()
            journal(output, "wsl-client-exited", returncode=child.returncode, gpu_allowed=False)
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        # Continue monitoring during cleanup. A failed observer cannot restore PASS.
        def await_cleanup(future, seconds):
            deadline = time.monotonic() + seconds
            while not future.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned cleanup completion unverified")
                if not observer_failed_local[0]:
                    try:
                        monitor.poll(time.monotonic())
                    except BaseException as exc:
                        record["cleanup_observer_error"] = str(exc)
                        observer_failed_local[0] = True
                time.sleep(0.025)
            return future.result()
        observer_failed_local = [observer_failed]
        if child is not None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()  # EOF keeps the Linux launch gate closed.
        if job is not None:
            # A normal WSL client can exit milliseconds before its owned
            # wslhost/conhost helpers. Give them a bounded monitored grace;
            # unchanged job.finish() remains the last-resort containment.
            if child is not None and child.poll() == 0 and not observer_failed_local[0]:
                grace_began = time.monotonic()
                try:
                    while job.active():
                        if time.monotonic() - grace_began >= 5.0:
                            break
                        check_deadlines(time.monotonic(), began, None)
                        try:
                            monitor.poll(time.monotonic())
                        except BaseException:
                            observer_failed_local[0] = True
                            raise
                        time.sleep(0.025)
                    record["windows_job_grace"] = {"empty": job.active() == 0,
                        "seconds": time.monotonic() - grace_began, "limit_seconds": 5.0}
                except BaseException as exc:
                    record["windows_job_grace_error"] = str(exc)
            try:
                record["windows_job_cleanup"] = await_cleanup(in_thread(job.finish), 6.0)
            except BaseException as exc:
                record["windows_job_cleanup_error"] = str(exc)
        if child is not None:
            try:
                record["distro_termination"] = await_cleanup(in_thread(terminate_owned_distro), 11.0)
                if child.poll() is None:
                    child.wait(timeout=1)
                record.setdefault("worker_returncode", child.returncode)
                termination_finished = time.monotonic()
                journal(output, "owned-distro-terminated", capture=record["distro_termination"])
                if not observer_failed_local[0]:
                    while not monitor.healthy_since(termination_finished):
                        check_deadlines(time.monotonic(), began, None)
                        monitor.poll(time.monotonic())
                        time.sleep(0.025)
                    record["observation_after_exit"] = monitor.latest()
            except BaseException as exc:
                record["distro_cleanup_or_post_observer_error"] = str(exc)
        record["observations"] = monitor.observations
        record["observer_healthy"] = not observer_failed_local[0] and "observation_after_exit" in record
        try:
            result = json.loads((output / "guest/results.json").read_text(encoding="utf-8"))
            record["worker"] = result
            if result.get("classification") == "FAIL":
                record["classification"] = "FAIL"
            if result.get("classification") == "PASS":
                computation = result.get("computation", {})
                validate_output(computation.get("values"), computation.get("shape"), computation.get("dtype"))
                validate_cleanup(result)
                if (result.get("gate_authorized") is not True or computation.get("torch_cpu_equal") is not True
                        or computation.get("gpu_arithmetic_operations") != 1):
                    raise RuntimeError("Incomplete gated worker computation evidence")
                if ("error" not in record and record.get("worker_returncode") == 0 and record["observer_healthy"]
                        and time.monotonic() - began < HOST_DEADLINE
                        and record.get("distro_termination", {}).get("returncode") == 0
                        and record.get("windows_job_cleanup", {}).get("owned_tree_exit_verified") is True
                        and record["windows_job_cleanup"].get("forced_job_cleanup") is False
                        and "windows_job_grace_error" not in record
                        and "distro_cleanup_or_post_observer_error" not in record):
                    record["classification"] = "PASS"
        except BaseException as exc:
            record["worker_evidence_error"] = type(exc).__name__ + ": " + str(exc)
        record.update(elapsed_seconds=time.monotonic() - began, finished_utc=utc_now())
        journal(output, "supervisor-finished", classification=record["classification"])
        save_json(output / "results.json", record)
    print("WSL_ROCDXG_RESULT=" + record["classification"], flush=True)
    return 0 if record["classification"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker)
    if args.output is None:
        parser.error("--output is required for the single Windows-supervised attempt")
    return supervise(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
