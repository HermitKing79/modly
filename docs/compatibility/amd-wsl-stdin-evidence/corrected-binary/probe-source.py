"""Bounded CPU-only WSL stdin evidence; calls the genuine smoke gate functions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SMOKE = HERE.with_name("amd_wsl_rocdxg_smoke.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbid_ai(event, args):
    if event == "import" and str(args[0]).split(".")[0] in {
            "torch", "triton", "numpy", "trellis2", "flex_gemm", "o_voxel", "cumesh"}:
        raise RuntimeError("CPU stdin probe forbids AI/native imports")


def guest(output):
    sys.addaudithook(forbid_ai)
    smoke = load(SMOKE, "stdin_smoke_gate")
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(exist_ok=False)
    # Peek leaves the actual stdin stream untouched for the production gate.
    prefix = sys.stdin.buffer.peek(16)[:16]
    record = {"raw_prefix_hex": prefix.hex(), "raw_prefix_limit": 16,
              "smoke_sha256": hashlib.sha256(SMOKE.read_bytes()).hexdigest(),
              "executable": sys.executable, "python": sys.version,
              "accepted": False, "gate_reads": []}

    class RecordedInput:
        def readline(self):
            value = sys.stdin.readline(16)
            record["gate_reads"].append({"method": "readline", "repr": repr(value),
                                         "utf8_hex": value.encode("utf-8").hex()})
            return value

        def read(self, count):
            if count != 4:
                raise RuntimeError("Unexpected authorization read bound")
            value = sys.stdin.buffer.read(count)
            record["gate_reads"].append({"method": "read", "bound": count,
                                         "hex": value.hex(), "bytes": len(value)})
            return value

    try:
        smoke.await_go(RecordedInput())
        record["accepted"] = True
    except RuntimeError as exc:
        record["gate_error"] = str(exc)
    record["torch_imported"] = "torch" in sys.modules
    record["triton_imported"] = "triton" in sys.modules
    smoke.save_json(output / "received.json", record)
    return 0  # An observed strict rejection is a successful CPU diagnostic.


def capture_pipe_writes(stream, text_mode, written):
    buffer = stream.buffer if text_mode else stream
    original = buffer.write
    def write(data):
        if not isinstance(data, bytes) or len(written) + len(data) > 32:
            raise RuntimeError("Authorization pipe evidence exceeded its bound")
        count = original(data)
        written.extend(data[:count])
        return count
    buffer.write = write


def host(output, original):
    if sys.platform != "win32":
        raise RuntimeError("CPU transport supervisor requires Windows")
    sys.addaudithook(forbid_ai)
    smoke = load(SMOKE, "stdin_smoke_gate")
    helper = smoke.load_sibling("amd_bounded_gpu_proof")
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(exist_ok=False)
    since = smoke.utc_now()
    monitor = smoke.HealthMonitor(lambda: helper.windows_observation(since), helper.healthy_observation)
    began = time.monotonic()
    job = smoke.load_sibling("amd_forward_owned_job").OwnedJob()
    result = {"mode": "original-text" if original else "corrected-binary", "cases": [],
              "classification": "UNKNOWN", "smoke_sha256": hashlib.sha256(SMOKE.read_bytes()).hexdigest()}
    child = None
    failed = False

    def tick():
        if time.monotonic() - began >= 120:
            raise TimeoutError("CPU transport batch deadline")
        monitor.poll(time.monotonic())

    def health_after(start):
        while not monitor.healthy_since(start):
            tick()
            time.sleep(.025)
        tick()
        return monitor.latest()

    cases = [("original-go", None, None)] if original else [
        ("production-go", None, True), ("wrong-token", b"NO\n", False),
        ("crlf", b"GO\r\n", False), ("missing-newline", b"GO", False),
        ("extra-data", b"GO\nX", False), ("empty-line", b"\n", False),
        ("eof", b"", False)]
    try:
        before = health_after(began)
        for name, payload, expected in cases:
            case = output / name
            case.mkdir()
            command = smoke.worker_command(smoke.guest_path(HERE), smoke.guest_path(case / "guest"))
            written = bytearray()
            options = {"text": True, "encoding": "utf-8"} if original else {}
            with (case / "stdout.txt").open("xb") as out, (case / "stderr.txt").open("xb") as err:
                child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, **options)
                job.attach(child)
                after = health_after(time.monotonic())
                capture_pipe_writes(child.stdin, original, written)
                if payload is None:
                    smoke.authorize_go(child.stdin, True, before, after)
                else:
                    child.stdin.write(payload)
                    child.stdin.flush()
                child.stdin.close()
                case_began = time.monotonic()
                while child.poll() is None:
                    tick()
                    if time.monotonic() - case_began >= 30:
                        raise TimeoutError("CPU transport case deadline")
                    time.sleep(.025)
                received = json.loads((case / "guest/received.json").read_text(encoding="utf-8"))
                row = {"case": name, "command": command, "host_mode": options,
                       "logical_request_utf8_hex": "474f0a" if payload is None else payload.hex(),
                       "actual_pipe_write_hex": written.hex(), "child_exit": child.returncode,
                       "received": received}
                smoke.save_json(case / "host.json", row)
                result["cases"].append(row)
                if child.returncode != 0 or received["torch_imported"] or received["triton_imported"]:
                    raise RuntimeError("CPU guest or import-isolation failure")
                if expected is not None and received["accepted"] is not expected:
                    raise RuntimeError("Actual production authorization result differs from contract")
                if not original and received["raw_prefix_hex"] != written.hex():
                    raise RuntimeError("Host pipe bytes differ from guest prefix")
                child = None
        result["classification"] = "OBSERVED" if original else "PASS"
    except BaseException as exc:
        failed = True
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        # Every owned cleanup is attempted even when an earlier one fails.
        # Reuse the production monitored-future pattern; no new process manager.
        def cleanup_call(function, seconds):
            nonlocal failed
            future = smoke.in_thread(function)
            deadline = time.monotonic() + seconds
            while not future.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU probe cleanup completion unverified")
                if not failed:
                    try:
                        monitor.poll(time.monotonic())
                    except BaseException as exc:
                        failed = True
                        result["cleanup_monitor_error"] = str(exc)
                time.sleep(.025)
            return future.result()

        if child is not None and child.stdin is not None and not child.stdin.closed:
            try:
                child.stdin.close()
            except BaseException as exc:
                failed = True
                result["stdin_close_error"] = str(exc)
        try:
            result["job_cleanup"] = cleanup_call(job.finish, 6)
        except BaseException as exc:
            failed = True
            result["job_cleanup_error"] = str(exc)
        if child is not None:
            try:
                child.wait(timeout=2)
            except BaseException as exc:
                failed = True
                result["child_wait_error"] = str(exc)
        try:
            result["distro_termination"] = cleanup_call(smoke.terminate_owned_distro, 11)
        except BaseException as exc:
            failed = True
            result["distro_termination_error"] = str(exc)
        try:
            result["post_termination_observer"] = health_after(time.monotonic())
        except BaseException as exc:
            failed = True
            result["monitor_error"] = str(exc)
        result["observations"] = monitor.observations
        result["elapsed_seconds"] = time.monotonic() - began
        if (failed or result.get("job_cleanup", {}).get("forced_job_cleanup", True)
                or result.get("distro_termination", {}).get("returncode") != 0):
            result["classification"] = "UNKNOWN"
        smoke.save_json(output / "results.json", result)
    print(json.dumps({"classification": result["classification"], "cases": len(result["cases"])}))
    return 0 if result["classification"] in {"OBSERVED", "PASS"} else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original-text", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        raise SystemExit(guest(args.worker))
    if args.output is None:
        parser.error("--output required")
    raise SystemExit(host(args.output, args.original_text))
