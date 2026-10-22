"""Stdlib-only Windows launch gate, invoked with -I -S by the private runner."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback


def guest_path(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/") and not value.startswith("//"):
        return value
    if not value.lower().startswith("c:/"):
        raise ValueError("WSL shared files must remain on the C: volume")
    return "/mnt/c/" + value[3:]


def valid_control(message, spec):
    if not isinstance(message, dict) or (message.get("stage_id"), message.get("nonce")) != (
            spec["stage_id"], spec["nonce"]):
        raise ValueError("Control message does not match stage/nonce")
    if message.get("action") not in ("GO", "START", "PING", "CANCEL"):
        raise ValueError("Unknown stage control action")
    return message["action"]


def emit(spec, **fields):
    print(json.dumps({"stage_id": spec["stage_id"], "nonce": spec["nonce"],
                      "context": spec["context"], **fields}), flush=True)


def read_control(stream, spec):
    line = stream.readline(65537)
    if not line or len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("Stage control closed or exceeded limit")
    return valid_control(json.loads(line), spec)


def native_stage(spec):
    controls = queue.Queue(maxsize=64)

    def reader():
        try:
            while True:
                controls.put(read_control(sys.stdin.buffer.raw, spec))
        except Exception:
            controls.put("CANCEL")

    env = os.environ.copy()
    env.update(MODLY_COMPLETION_CONTEXT_JSON=json.dumps(spec["context"]), MODLY_STAGE_ID=spec["stage_id"],
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cancel_ack = str(Path(spec["stdout_file"]).with_name("cancel-ack.json"))
    env["MODLY_STAGE_CANCEL_ACK"] = cancel_ack
    forced = cancelled = False
    cancel_at = None
    last_host = time.monotonic()
    deadline = last_host + spec["timeout_s"]
    with open(spec["stdout_file"], "xb", buffering=0) as stdout, open(spec["stderr_file"], "xb", buffering=0) as stderr:
        child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env, stdin=subprocess.PIPE,
                                 stdout=stdout, stderr=stderr, close_fds=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=reader, daemon=True).start()
        emit(spec, event="stage-started", pid=child.pid, timestamp_ns=time.time_ns())
        while child.poll() is None:
            now = time.monotonic()
            try:
                action = controls.get(timeout=.025)
            except queue.Empty:
                action = None
            if action is not None:
                emit(spec, event='control-received', action=action, child_pid=child.pid, timestamp_ns=time.time_ns())
                last_host = now
            if now >= deadline or now - last_host > 15:
                action = "CANCEL"
            if action == "CANCEL" and cancel_at is None:
                cancelled, cancel_at = True, now
                try:
                    child.stdin.write((json.dumps({"action": "CANCEL", "stage_id": spec["stage_id"],
                                                  "nonce": spec["nonce"]}) + "\n").encode())
                    child.stdin.flush()
                except (BrokenPipeError, OSError) as error:
                    # Private stderr/protocol, without hiding the primary error.
                    traceback.print_exc()
                    emit(spec, event='control-error', operation='child.stdin.CANCEL.write_flush',
                         child_pid=child.pid, winerror=getattr(error, 'winerror', None), timestamp_ns=time.time_ns())
            if cancel_at is not None and now - cancel_at >= 3:
                forced = True
                child.kill()
                child.wait(timeout=2)
        emit(spec, event='child-exited', child_pid=child.pid, returncode=child.returncode,
             cancellation_sent=cancelled, forced=forced, timestamp_ns=time.time_ns())
        child.stdin.close()
        returncode = child.wait(timeout=2)
    cooperative = False
    if cancelled and not forced:
        try:
            ack = json.loads(Path(cancel_ack).read_text(encoding="utf-8"))
            cooperative = (ack.get("terminal") == "cancelled" and ack.get("context") == spec["context"]
                           and ack.get("stage_id") == spec["stage_id"] and ack.get("nonce") == spec["nonce"])
        except (OSError, ValueError) as error:
            traceback.print_exc()
            emit(spec, event='ack-error', operation='cancel-ack.read_json',
                 winerror=getattr(error, 'winerror', None), timestamp_ns=time.time_ns())
        emit(spec, event='ack-checked', cooperative=cooperative, timestamp_ns=time.time_ns())
    emit(spec, event="terminal", terminal="cancelled" if cancelled else "succeeded" if returncode == 0 else "failed",
         returncode=returncode, forced=forced, natural=not forced and (not cancelled or cooperative),
         cooperative_cancelled=cooperative, timestamp_ns=time.time_ns())
    return 0


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    action = read_control(sys.stdin.buffer.raw, spec)
    emit(spec, event='control-received', action=action, timestamp_ns=time.time_ns())
    if action != "GO":
        emit(spec, event="terminal", terminal="cancelled", returncode=None, forced=False, natural=True,
             linux_settled=True, reaped_all=True, owned_remaining=[], no_stage_launched=True)
        return 0
    if spec["kind"] == "windows":
        return native_stage(spec)
    if spec["kind"] != "wsl":
        raise ValueError("Unknown stage backend")
    helper = guest_path(Path(__file__).with_name("amd_wsl_session.py"))
    # The broker creates a second, genuine Linux session gate. Do not buffer or
    # consume further stdin here: the broker receives START/PING/CANCEL directly.
    wsl = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/wsl.exe")
    child = subprocess.Popen([wsl, "-d", "Modly-AMD-Ubuntu-24.04", "-u", "modly", "--exec",
                              "/usr/bin/python3", "-B", "-I", "-S", helper, guest_path(spec_path)],
                             stdin=sys.stdin.buffer, stdout=sys.stdout.buffer, stderr=sys.stderr.buffer,
                             close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = child.wait()
    emit(spec, event="linux-client-exited", returncode=result, pid=child.pid, timestamp_ns=time.time_ns())
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
